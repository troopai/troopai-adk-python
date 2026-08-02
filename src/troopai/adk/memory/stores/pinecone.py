"""Pinecone (hosted) vector store.

Namespace is stored as record metadata and filtered with Pinecone's filter DSL
(not Pinecone's native namespaces) so ``get`` is namespace-free.  The index must
be created out-of-band with the ``cosine`` metric and matching dimension.  The
sync client is wrapped with ``asyncio.to_thread``.

Counts: Pinecone delete operations do not report a count, so ``delete`` returns
the number of ids requested and ``clear`` returns 0 (documented approximation).
``clear`` deletes by metadata filter; ensure your Pinecone index supports
metadata-filtered deletes (consult Pinecone's documentation for your index type).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

try:
    import pinecone
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PineconeVectorStore requires pinecone: pip install 'troopai-adk-python[memory-pinecone]'"
    ) from exc

from troopai.adk.memory.memory_types import (
    MemoryKind,
    MemoryMetadata,
    MemorySearchFilter,
    MemorySource,
)
from troopai.adk.memory.vector_store import VectorQueryResult, VectorRecord

logger = logging.getLogger(__name__)


def _to_meta(record: VectorRecord) -> dict[str, Any]:
    meta = record.metadata
    result: dict[str, Any] = {
        "namespace": record.namespace,
        "content": record.content,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "source": meta.source.value,
        "importance": meta.importance,
        "kind": meta.kind.value,
        "agent_name": meta.agent_name or "",  # None -> ""; "" reads back as None
        "session_id": meta.session_id or "",  # None -> ""; "" reads back as None
        "custom": json.dumps(meta.custom),
    }
    # Pinecone rejects an empty-list metadata value; omit the key when there are
    # no categories (an absent key reads back as an empty tuple in _from_meta).
    if len(meta.categories) > 0:
        result["categories"] = list(meta.categories)
    return result


def _from_meta(rid: str, values: list[float], meta: dict[str, Any]) -> VectorRecord:
    for required in ("namespace", "content", "created_at", "updated_at"):
        if meta.get(required) is None:
            raise RuntimeError(f"PineconeVectorStore: record {rid!r} is missing required metadata field {required!r}")
    metadata = MemoryMetadata(
        source=MemorySource(meta.get("source", "manual")),
        importance=int(meta.get("importance", 3)),
        categories=tuple(meta.get("categories", [])),
        session_id=meta.get("session_id") or None,
        agent_name=meta.get("agent_name") or None,
        kind=MemoryKind(meta.get("kind", "episodic")),
        custom=json.loads(meta.get("custom", "{}")),
    )
    return VectorRecord(
        id=rid,
        vector=tuple(float(x) for x in values),
        namespace=str(meta["namespace"]),
        content=str(meta["content"]),
        metadata=metadata,
        created_at=float(meta["created_at"]),
        updated_at=float(meta["updated_at"]),
    )


def _build_filter(namespace: str, filter: MemorySearchFilter | None) -> dict[str, Any]:
    clauses: list[dict[str, Any]] = [{"namespace": {"$eq": namespace}}]
    if filter is not None:
        if filter.kind is not None:
            clauses.append({"kind": {"$eq": filter.kind.value}})
        if filter.importance is not None:
            clauses.append({"importance": {"$gte": filter.importance}})
        if filter.agent_name is not None:
            clauses.append({"agent_name": {"$eq": filter.agent_name}})
        if filter.after is not None:
            clauses.append({"created_at": {"$gt": filter.after}})
        if filter.before is not None:
            clauses.append({"created_at": {"$lt": filter.before}})
        if filter.categories is not None:
            clauses.append({"categories": {"$in": list(filter.categories)}})
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


def _sync_upsert(index: Any, vectors: list[dict[str, Any]]) -> None:
    index.upsert(vectors=vectors)


def _sync_query(
    index: Any,
    vector: list[float],
    top_k: int,
    filter: dict[str, Any],
) -> list[dict[str, Any]]:
    res = index.query(
        vector=vector,
        top_k=top_k,
        filter=filter,
        include_metadata=True,
        include_values=True,
    )
    return list(res["matches"])


def _sync_fetch(index: Any, ids: list[str]) -> dict[str, Any]:
    res = index.fetch(ids=ids)
    return dict(res["vectors"])


def _sync_delete_ids(index: Any, ids: list[str]) -> None:
    index.delete(ids=ids)


def _sync_delete_filter(index: Any, filter: dict[str, Any]) -> None:
    index.delete(filter=filter)


class PineconeVectorStore:
    """Pinecone-backed vector store.

    Args:
        index: Existing Pinecone index name (cosine metric, matching dimension).
        api_key: Optional key; ``None`` uses Pinecone's env resolution.
    """

    def __init__(self, *, index: str, api_key: str | None = None) -> None:
        client = pinecone.Pinecone(api_key=api_key)
        # Pinecone's response objects (QueryResponse | ApplyResult, FetchResponse) don't
        # expose the dict-subscript fields we consume as typed attributes; using Index
        # would force coupling to its generated response classes across every helper.
        self._index: Any = client.Index(index)

    async def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or replace records in the Pinecone index.

        Namespace and all metadata are stored in the Pinecone record
        payload so that ``get`` remains namespace-free.

        Args:
            records: Records to insert or replace.
        """
        if len(records) == 0:
            return
        vectors = [{"id": r.id, "values": list(r.vector), "metadata": _to_meta(r)} for r in records]
        await asyncio.to_thread(_sync_upsert, self._index, vectors)
        logger.debug("PineconeVectorStore: upserted %d records", len(records))

    async def query(
        self,
        vector: tuple[float, ...],
        *,
        namespace: str,
        k: int = 5,
        filter: MemorySearchFilter | None = None,
    ) -> list[VectorQueryResult]:
        """Return the ``k`` nearest records from Pinecone.

        Namespace and filter facets are translated to Pinecone's filter DSL.

        Args:
            vector: Query embedding to compare against stored records.
            namespace: Namespace to search within (filtered via metadata).
            k: Maximum number of results to return.
            filter: Optional metadata filters.

        Returns:
            List of :class:`VectorQueryResult` ordered by descending score.

        Raises:
            RuntimeError: If a match is returned without vector values.
        """
        matches = await asyncio.to_thread(
            _sync_query,
            self._index,
            list(vector),
            k,
            _build_filter(namespace, filter),
        )
        out: list[VectorQueryResult] = []
        for match in matches:
            values = match.get("values")
            if values is None:
                raise RuntimeError(
                    f"PineconeVectorStore: query match {match['id']!r} returned no vector values "
                    "(ensure the index/SDK returns values with include_values=True)"
                )
            record = _from_meta(str(match["id"]), list(values), dict(match["metadata"]))
            score = max(0.0, min(1.0, float(match["score"])))
            out.append(VectorQueryResult(record=record, score=score))
        return out

    async def get(self, record_id: str) -> VectorRecord | None:
        """Fetch a record by its global id.

        Args:
            record_id: The unique record identifier (namespace-free).

        Returns:
            The matching :class:`VectorRecord`, or ``None`` if not found.
        """
        vectors = await asyncio.to_thread(_sync_fetch, self._index, [record_id])
        if record_id not in vectors:
            return None
        hit = vectors[record_id]
        return _from_meta(record_id, list(hit["values"]), dict(hit["metadata"]))

    async def delete(self, ids: list[str]) -> int:
        """Delete records by id.

        Pinecone does not report a delete count, so the number of ids
        requested is returned as an approximation.

        Args:
            ids: Record identifiers to delete.

        Returns:
            Number of ids requested (approximate; Pinecone does not confirm existence).
        """
        if len(ids) == 0:
            return 0
        await asyncio.to_thread(_sync_delete_ids, self._index, list(ids))
        return len(ids)

    async def clear(self, *, namespace: str) -> int:
        """Delete all records in a namespace.

        Pinecone does not report a delete count, so always returns 0.

        Args:
            namespace: The namespace whose records should be deleted.

        Returns:
            Always ``0`` (Pinecone delete-by-filter reports no count).
        """
        await asyncio.to_thread(_sync_delete_filter, self._index, _build_filter(namespace, None))
        return 0

    async def close(self) -> None:
        """No-op; the Pinecone client holds no long-lived connection to close."""

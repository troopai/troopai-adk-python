"""Chroma (embedded) vector store.

Stores caller-supplied embeddings (no Chroma-side embedding function).
Cosine space via the collection metadata ``{"hnsw:space": "cosine"}``.  The
sync client is wrapped with ``asyncio.to_thread``.  Category filtering is not
expressible in Chroma's metadata ``where`` and is skipped (logged); namespace
and kind are always honored.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, cast

try:
    import chromadb
    from chromadb.api.models.Collection import Collection as ChromaCollection
    from chromadb.api.types import GetResult, PyEmbedding, QueryResult, Where
except ImportError as exc:  # pragma: no cover
    raise ImportError("ChromaVectorStore requires chromadb: pip install 'troopai-adk-python[memory-chroma]'") from exc

from troopai.adk.memory.memory_types import (
    MemoryKind,
    MemoryMetadata,
    MemorySearchFilter,
    MemorySource,
)
from troopai.adk.memory.vector_store import VectorQueryResult, VectorRecord

logger = logging.getLogger(__name__)


def _to_chroma_meta(record: VectorRecord) -> dict[str, Any]:
    meta = record.metadata
    return {
        "namespace": record.namespace,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "source": meta.source.value,
        "importance": meta.importance,
        "kind": meta.kind.value,
        "agent_name": meta.agent_name or "",  # None -> ""; "" reads back as None
        "session_id": meta.session_id or "",  # None -> ""; "" reads back as None
        "categories": json.dumps(list(meta.categories)),
        "custom": json.dumps(meta.custom),
    }


def _from_chroma(rid: str, vector: Any, document: str, meta: dict[str, Any]) -> VectorRecord:
    # chromadb returns embeddings as numpy arrays; convert element-by-element to float
    for required in ("namespace", "created_at", "updated_at"):
        if meta.get(required) is None:
            raise RuntimeError(f"ChromaVectorStore: record {rid!r} is missing required metadata field {required!r}")
    metadata = MemoryMetadata(
        source=MemorySource(meta.get("source", "manual")),
        importance=int(meta.get("importance", 3)),
        categories=tuple(json.loads(meta.get("categories", "[]"))),
        session_id=meta.get("session_id") or None,
        agent_name=meta.get("agent_name") or None,
        kind=MemoryKind(meta.get("kind", "episodic")),
        custom=json.loads(meta.get("custom", "{}")),
    )
    return VectorRecord(
        id=rid,
        vector=tuple(float(x) for x in vector),
        namespace=str(meta["namespace"]),
        content=document,
        metadata=metadata,
        created_at=float(meta["created_at"]),
        updated_at=float(meta["updated_at"]),
    )


def _build_where(namespace: str, filter: MemorySearchFilter | None) -> Where:
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
            logger.warning(
                "ChromaVectorStore: category filter is not supported by Chroma metadata WHERE; "
                "results are unconstrained by category"
            )
    if len(clauses) == 1:
        return cast("Where", clauses[0])
    return cast("Where", {"$and": list(clauses)})


def _sync_upsert(col: ChromaCollection, records: list[VectorRecord]) -> None:
    vecs: list[PyEmbedding] = [list(r.vector) for r in records]
    col.upsert(
        ids=[r.id for r in records],
        embeddings=vecs,
        metadatas=[_to_chroma_meta(r) for r in records],
        documents=[r.content for r in records],
    )


def _sync_query(col: ChromaCollection, vector: list[float], k: int, where: Where) -> QueryResult:
    query_vecs: list[PyEmbedding] = [list(vector)]
    return col.query(
        query_embeddings=query_vecs,
        n_results=k,
        where=where,
        include=["embeddings", "metadatas", "documents", "distances"],
    )


def _sync_get(col: ChromaCollection, ids: list[str]) -> GetResult:
    return col.get(ids=ids, include=["embeddings", "metadatas", "documents"])


def _sync_get_where(col: ChromaCollection, where: Where) -> GetResult:
    return col.get(where=where)


def _sync_delete_ids(col: ChromaCollection, ids: list[str]) -> None:
    col.delete(ids=ids)


def _sync_delete_where(col: ChromaCollection, where: Where) -> None:
    col.delete(where=where)


class ChromaVectorStore:
    """Embedded Chroma vector store.

    Args:
        collection: Collection name.
        path: Optional on-disk path; ``None`` uses an ephemeral in-process DB.
    """

    def __init__(self, *, collection: str = "memory_vectors", path: str | None = None) -> None:
        client = chromadb.PersistentClient(path=path) if path is not None else chromadb.EphemeralClient()
        self._col = client.get_or_create_collection(name=collection, metadata={"hnsw:space": "cosine"})

    async def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or replace records in the Chroma collection.

        Args:
            records: Records to insert or replace.
        """
        if len(records) == 0:
            return
        await asyncio.to_thread(_sync_upsert, self._col, records)
        logger.debug("ChromaVectorStore: upserted %d records", len(records))

    async def query(
        self,
        vector: tuple[float, ...],
        *,
        namespace: str,
        k: int = 5,
        filter: MemorySearchFilter | None = None,
    ) -> list[VectorQueryResult]:
        """Return the ``k`` nearest records using Chroma HNSW cosine distance.

        Score is computed as ``1 - chroma_distance`` (Chroma uses cosine distance,
        not similarity).  Category filtering is not supported by Chroma's
        metadata ``where`` DSL and is silently skipped.

        Args:
            vector: Query embedding to compare against stored records.
            namespace: Namespace to search within (filtered via metadata).
            k: Maximum number of results to return.
            filter: Optional metadata filters (categories are silently ignored).

        Returns:
            List of :class:`VectorQueryResult` ordered by descending score.

        Raises:
            RuntimeError: If Chroma returns ``None`` for a requested include field.
        """
        res: QueryResult = await asyncio.to_thread(
            _sync_query, self._col, list(vector), k, _build_where(namespace, filter)
        )
        out: list[VectorQueryResult] = []
        ids: list[str] = res["ids"][0]
        distances = res["distances"]
        embeddings = res["embeddings"]
        documents = res["documents"]
        metadatas = res["metadatas"]
        if distances is None or embeddings is None or documents is None or metadatas is None:
            raise RuntimeError("ChromaVectorStore: query returned None for a requested include field")
        for i, rid in enumerate(ids):
            score = max(0.0, 1.0 - float(distances[0][i]))
            record = _from_chroma(rid, embeddings[0][i], documents[0][i], dict(metadatas[0][i]))
            out.append(VectorQueryResult(record=record, score=score))
        return out

    async def get(self, record_id: str) -> VectorRecord | None:
        """Fetch a record by its global id.

        Args:
            record_id: The unique record identifier (namespace-free).

        Returns:
            The matching :class:`VectorRecord`, or ``None`` if not found.

        Raises:
            RuntimeError: If Chroma returns ``None`` for a requested include field.
        """
        res: GetResult = await asyncio.to_thread(_sync_get, self._col, [record_id])
        if len(res["ids"]) == 0:
            return None
        embeddings = res["embeddings"]
        documents = res["documents"]
        metadatas = res["metadatas"]
        if embeddings is None or documents is None or metadatas is None:
            raise RuntimeError("ChromaVectorStore: get returned None for a requested include field")
        return _from_chroma(res["ids"][0], embeddings[0], documents[0], dict(metadatas[0]))

    async def delete(self, ids: list[str]) -> int:
        """Delete records by id and return the count found before deletion.

        The count is determined by a pre-delete ``get``; it is accurate unless
        a concurrent upsert or delete races the window between count and delete.

        Args:
            ids: Record identifiers to delete.

        Returns:
            Number of records that existed at the time of the delete call.
        """
        if len(ids) == 0:
            return 0
        # count before delete: accurate unless a concurrent upsert/delete races this window
        existing: GetResult = await asyncio.to_thread(_sync_get, self._col, list(ids))
        found = len(existing["ids"])
        await asyncio.to_thread(_sync_delete_ids, self._col, list(ids))
        return found

    async def clear(self, *, namespace: str) -> int:
        """Delete all records in a namespace.

        The count is determined by a pre-delete ``get``; it is accurate unless
        a concurrent upsert or delete races the window between count and delete.

        Args:
            namespace: The namespace whose records should be deleted.

        Returns:
            Number of records that existed at the time of the clear call.
        """
        where = _build_where(namespace, None)
        # count before delete: accurate unless a concurrent upsert/delete races this window
        existing: GetResult = await asyncio.to_thread(_sync_get_where, self._col, where)
        found = len(existing["ids"])
        await asyncio.to_thread(_sync_delete_where, self._col, where)
        return found

    async def close(self) -> None:
        """No-op; the embedded client needs no explicit close."""

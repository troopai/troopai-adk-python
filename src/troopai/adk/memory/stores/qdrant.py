"""Qdrant vector store (native async client).

Namespace is stored in the point payload and filtered (not a separate
collection) so ``get`` is namespace-free.  Cosine distance via collection
config.  Point ids must be UUIDs or unsigned ints — pass UUID-string ids.

Counts: Qdrant delete does not report a count, so ``delete`` returns the number
of ids requested.  ``clear`` issues a ``count`` before deleting so it can
return the number of points removed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import cast

try:
    from qdrant_client import AsyncQdrantClient, models
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "QdrantVectorStore requires qdrant-client: pip install 'troopai-adk-python[memory-qdrant]'"
    ) from exc

from troopai.adk.memory.memory_types import (
    MemoryKind,
    MemoryMetadata,
    MemorySearchFilter,
    MemorySource,
)
from troopai.adk.memory.vector_store import VectorQueryResult, VectorRecord

logger = logging.getLogger(__name__)


def _payload(record: VectorRecord) -> dict[str, object]:
    meta = record.metadata
    return {
        "namespace": record.namespace,
        "content": record.content,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "source": meta.source.value,
        "importance": meta.importance,
        "kind": meta.kind.value,
        "agent_name": meta.agent_name,
        "session_id": meta.session_id,
        "categories": list(meta.categories),
        "custom": meta.custom,
    }


def _to_record(point: models.Record | models.ScoredPoint) -> VectorRecord:
    payload = point.payload or {}
    for required in ("namespace", "content", "created_at", "updated_at"):
        if payload.get(required) is None:
            raise RuntimeError(f"QdrantVectorStore: point {point.id!r} is missing required payload field {required!r}")
    metadata = MemoryMetadata(
        source=MemorySource(payload.get("source", "manual")),
        importance=int(payload.get("importance", 3)),
        categories=tuple(cast("list[str]", payload.get("categories", []))),
        session_id=cast("str", payload["session_id"]) if payload.get("session_id") is not None else None,
        agent_name=cast("str", payload["agent_name"]) if payload.get("agent_name") is not None else None,
        kind=MemoryKind(payload.get("kind", "episodic")),
        custom=cast("dict[str, str]", payload.get("custom", {})),
    )
    raw = point.vector
    if not isinstance(raw, list):
        raise RuntimeError(
            f"QdrantVectorStore: point {point.id} returned a non-list vector "
            f"({type(raw).__name__}); expected an unnamed dense vector"
        )
    vector: tuple[float, ...] = tuple(float(x) for x in cast("list[float]", raw))
    return VectorRecord(
        id=str(point.id),
        vector=vector,
        namespace=str(payload["namespace"]),
        content=str(payload["content"]),
        metadata=metadata,
        created_at=float(payload["created_at"]),
        updated_at=float(payload["updated_at"]),
    )


def _build_qfilter(namespace: str, filter: MemorySearchFilter | None) -> models.Filter:
    # models.Condition is the union of all condition types; typing list[Condition]
    # satisfies the invariant required by Filter.must (list[FieldCondition] does not).
    must: list[models.Condition] = [models.FieldCondition(key="namespace", match=models.MatchValue(value=namespace))]
    if filter is not None:
        if filter.kind is not None:
            must.append(models.FieldCondition(key="kind", match=models.MatchValue(value=filter.kind.value)))
        if filter.importance is not None:
            must.append(models.FieldCondition(key="importance", range=models.Range(gte=filter.importance)))
        if filter.agent_name is not None:
            must.append(models.FieldCondition(key="agent_name", match=models.MatchValue(value=filter.agent_name)))
        if filter.after is not None:
            must.append(models.FieldCondition(key="created_at", range=models.Range(gt=filter.after)))
        if filter.before is not None:
            must.append(models.FieldCondition(key="created_at", range=models.Range(lt=filter.before)))
        if filter.categories is not None:
            must.append(models.FieldCondition(key="categories", match=models.MatchAny(any=list(filter.categories))))
    return models.Filter(must=must)


class QdrantVectorStore:
    """Qdrant-backed vector store.

    Args:
        collection: Collection name (created on first use with cosine distance).
        dimensions: Embedding dimension.
        url: Qdrant server URL (omit when using ``location``).
        api_key: Optional Qdrant Cloud API key.
        location: In-process location such as ``":memory:"`` (for tests).
    """

    _ready: bool  # annotated so type checkers don't narrow to Literal[False] from __init__

    def __init__(
        self,
        *,
        collection: str,
        dimensions: int,
        url: str | None = None,
        api_key: str | None = None,
        location: str | None = None,
    ) -> None:
        if dimensions <= 0:
            raise ValueError(f"QdrantVectorStore dimensions must be > 0, got {dimensions}")
        self._collection = collection
        self._dim = dimensions
        self._ready = False
        self._init_lock = asyncio.Lock()
        if location is not None:
            self._client = AsyncQdrantClient(location=location)
        else:
            self._client = AsyncQdrantClient(url=url, api_key=api_key)

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        async with self._init_lock:
            if bool(self._ready):  # bool() defeats Literal[False] narrowing checkers infer inside the lock
                return
            if not await self._client.collection_exists(self._collection):
                await self._client.create_collection(
                    self._collection,
                    vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE),
                )
                logger.info("QdrantVectorStore: created collection %r (dim=%d)", self._collection, self._dim)
            self._ready = True

    async def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or replace records in the Qdrant collection.

        Args:
            records: Records to insert or replace.
        """
        await self._ensure_ready()
        points = [models.PointStruct(id=r.id, vector=list(r.vector), payload=_payload(r)) for r in records]
        await self._client.upsert(self._collection, points=points)
        logger.debug("QdrantVectorStore: upserted %d records", len(records))

    async def query(
        self,
        vector: tuple[float, ...],
        *,
        namespace: str,
        k: int = 5,
        filter: MemorySearchFilter | None = None,
    ) -> list[VectorQueryResult]:
        """Return the ``k`` nearest records using Qdrant cosine similarity.

        Args:
            vector: Query embedding to compare against stored records.
            namespace: Namespace to search within (filtered via payload).
            k: Maximum number of results to return.
            filter: Optional metadata filters applied as Qdrant conditions.

        Returns:
            List of :class:`VectorQueryResult` ordered by descending score.
        """
        await self._ensure_ready()
        response = await self._client.query_points(
            self._collection,
            query=list(vector),
            limit=k,
            query_filter=_build_qfilter(namespace, filter),
            with_payload=True,
            with_vectors=True,
        )
        return [
            VectorQueryResult(record=_to_record(point), score=max(0.0, min(1.0, float(point.score))))
            for point in response.points
        ]

    async def get(self, record_id: str) -> VectorRecord | None:
        """Fetch a record by its global id.

        Args:
            record_id: The unique record identifier (UUID string; namespace-free).

        Returns:
            The matching :class:`VectorRecord`, or ``None`` if not found.
        """
        await self._ensure_ready()
        points = await self._client.retrieve(self._collection, ids=[record_id], with_payload=True, with_vectors=True)
        return _to_record(points[0]) if len(points) > 0 else None

    async def delete(self, ids: list[str]) -> int:
        """Delete records by id.

        Qdrant does not report a delete count, so the number of ids
        requested is returned as an approximation.

        Args:
            ids: Record identifiers to delete (UUID strings).

        Returns:
            Number of ids requested (approximate; Qdrant does not confirm existence).
        """
        await self._ensure_ready()
        if len(ids) == 0:
            return 0
        await self._client.delete(
            self._collection,
            # ExtendedPointId is Annotated[int, Strict] | Annotated[str, Strict] | UUID;
            # plain str satisfies the runtime contract — cast to match the invariant type.
            points_selector=models.PointIdsList(points=cast("list[models.ExtendedPointId]", ids)),
        )
        return len(ids)

    async def clear(self, *, namespace: str) -> int:
        """Delete all records in a namespace and return the count removed.

        A ``count`` query is issued before deletion so an accurate count can
        be returned; this is a best-effort snapshot unless a concurrent
        upsert or delete races the window.

        Args:
            namespace: The namespace whose records should be deleted.

        Returns:
            Number of records removed (exact Qdrant count at deletion time).
        """
        await self._ensure_ready()
        q_filter = _build_qfilter(namespace, None)
        # count before delete: accurate unless a concurrent upsert/delete races this window
        count_result = await self._client.count(self._collection, count_filter=q_filter, exact=True)
        removed = count_result.count
        await self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(filter=q_filter),
        )
        logger.debug("QdrantVectorStore: cleared %d records from namespace %r", removed, namespace)
        return removed

    async def close(self) -> None:
        """Close the Qdrant async client and release its connections."""
        await self._client.close()

"""Pure-Python in-memory vector store (prototyping/testing baseline).

Brute-force O(N) cosine over a dict.  No external dependencies.  Thread-safe.
For production use a persistent backend (pgvector/Pinecone/Chroma/Qdrant).
"""

from __future__ import annotations

import logging
import math
import threading

from troopai.adk.memory.memory_types import MemorySearchFilter
from troopai.adk.memory.vector_store import VectorQueryResult, VectorRecord

logger = logging.getLogger(__name__)


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 for a zero vector)."""
    if len(a) != len(b):
        raise ValueError(f"vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _matches_filter(record: VectorRecord, filter: MemorySearchFilter | None) -> bool:
    """Apply the non-namespace facets of a filter (namespace handled by caller)."""
    if filter is None:
        return True
    meta = record.metadata
    if filter.kind is not None and meta.kind != filter.kind:
        return False
    if filter.importance is not None and meta.importance < filter.importance:
        return False
    if filter.agent_name is not None and meta.agent_name != filter.agent_name:
        return False
    if filter.categories is not None and len(set(meta.categories) & set(filter.categories)) == 0:
        return False
    if filter.after is not None and record.created_at <= filter.after:
        return False
    return not (filter.before is not None and record.created_at >= filter.before)


class InMemoryVectorStore:
    """Dict-backed vector store with brute-force cosine search."""

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}
        self._dim: int | None = None
        self._lock = threading.Lock()

    async def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or replace records in the in-process store.

        Validates that every record's vector dimension matches those already
        stored (enforcing a single consistent embedding model per store instance).

        Args:
            records: Records to insert or replace.

        Raises:
            ValueError: If any record's vector dimension differs from the
                dimension of records already in the store.
        """
        with self._lock:
            expected = self._dim
            for record in records:
                if expected is None:
                    expected = len(record.vector)
                elif len(record.vector) != expected:
                    raise ValueError(
                        f"InMemoryVectorStore: vector dimension {len(record.vector)} does not "
                        f"match store dimension {expected} (did the embedding model change?)"
                    )
            self._dim = expected
            for record in records:
                self._records[record.id] = record
        logger.debug("InMemoryVectorStore: upserted %d records", len(records))

    async def query(
        self,
        vector: tuple[float, ...],
        *,
        namespace: str,
        k: int = 5,
        filter: MemorySearchFilter | None = None,
    ) -> list[VectorQueryResult]:
        """Return the ``k`` nearest records by cosine similarity.

        Args:
            vector: Query embedding to compare against stored records.
            namespace: Namespace to search within.
            k: Maximum number of results to return.
            filter: Optional metadata filters.

        Returns:
            List of :class:`VectorQueryResult` ordered by descending score.

        Raises:
            ValueError: If the query vector dimension differs from the
                dimension of records in the store.
        """
        with self._lock:
            if self._dim is not None and len(vector) != self._dim:
                raise ValueError(
                    f"InMemoryVectorStore: query dimension {len(vector)} does not match store dimension {self._dim}"
                )
            candidates = list(self._records.values())
        scored: list[VectorQueryResult] = []
        for record in candidates:
            if record.namespace != namespace:
                continue
            if filter is not None and filter.namespace is not None and record.namespace != filter.namespace:
                continue
            if not _matches_filter(record, filter):
                continue
            scored.append(VectorQueryResult(record=record, score=max(0.0, min(1.0, _cosine(vector, record.vector)))))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    async def get(self, record_id: str) -> VectorRecord | None:
        """Fetch a record by its global id.

        Args:
            record_id: The unique record identifier.

        Returns:
            The matching :class:`VectorRecord`, or ``None`` if not found.
        """
        with self._lock:
            return self._records.get(record_id)

    async def delete(self, ids: list[str]) -> int:
        """Delete records by id and return the exact count removed.

        Args:
            ids: Record identifiers to delete.

        Returns:
            Number of records actually removed (exact for this backend).
        """
        count = 0
        with self._lock:
            for record_id in ids:
                if record_id in self._records:
                    del self._records[record_id]
                    count += 1
        return count

    async def clear(self, *, namespace: str) -> int:
        """Delete all records in a namespace.

        Args:
            namespace: The namespace whose records should be deleted.

        Returns:
            Number of records removed.
        """
        with self._lock:
            to_delete = [rid for rid, rec in self._records.items() if rec.namespace == namespace]
            for rid in to_delete:
                del self._records[rid]
        logger.info("InMemoryVectorStore: cleared %d records (namespace=%s)", len(to_delete), namespace)
        return len(to_delete)

    async def close(self) -> None:
        """No-op; nothing to release."""

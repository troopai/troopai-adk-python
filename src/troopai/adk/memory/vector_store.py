"""Vector-store abstraction for semantic memory.

A low-level, framework-owned (client-side) interface for storing and querying
embedding vectors.  Distinct from provider-hosted vector stores referenced by
``FileSearchTool.vector_store_ids`` (those are searched server-side by the
provider; this Protocol is queried directly by the framework).

Namespace is modeled as a metadata filter, not a physical partition: ids are
global, so ``get`` needs no namespace (matching ``Memory.get(memory_id)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from troopai.adk.memory.memory_types import MemoryMetadata, MemorySearchFilter


@dataclass(frozen=True)
class VectorRecord:
    """A stored embedding plus the content and metadata it represents.

    Attributes:
        id: Unique record identifier.
        vector: The embedding components.
        namespace: Scoping key (e.g. ``"user:123"``).
        content: The original text.
        metadata: Memory metadata (source, importance, kind, ...).
        created_at: Unix timestamp of creation.
        updated_at: Unix timestamp of last update.
    """

    id: str
    """Unique record identifier."""

    vector: tuple[float, ...]
    """The embedding components."""

    namespace: str
    """Scoping key (e.g. ``"user:123"``)."""

    content: str
    """The original text."""

    metadata: MemoryMetadata
    """Memory metadata (source, importance, kind, ...)."""

    created_at: float
    """Unix timestamp of creation."""

    updated_at: float
    """Unix timestamp of last update."""


@dataclass(frozen=True)
class VectorQueryResult:
    """A record with its similarity score.

    Attributes:
        record: The matching record.
        score: Cosine similarity normalized to 0.0-1.0.
    """

    record: VectorRecord
    """The matching record."""

    score: float
    """Cosine similarity normalized to 0.0-1.0."""


@runtime_checkable
class VectorStore(Protocol):
    """Client-side vector store. Namespace is a metadata filter (see module doc)."""

    async def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or replace records.

        Namespace travels on each record's ``namespace`` field.

        Args:
            records: Records to insert or replace.
        """
        ...

    async def query(
        self,
        vector: tuple[float, ...],
        *,
        namespace: str,
        k: int = 5,
        filter: MemorySearchFilter | None = None,
    ) -> list[VectorQueryResult]:
        """Return the ``k`` most similar records in ``namespace`` (filtered).

        ``k`` defaults to 5 to match ``Memory.search(limit=5)``; ``VectorMemory``
        always passes ``k`` explicitly.

        Args:
            vector: Query embedding to compare against stored records.
            namespace: Namespace to search within (applied as a metadata filter).
            k: Maximum number of results to return.
            filter: Optional metadata filters (kind, importance, agent, time range).

        Returns:
            List of :class:`VectorQueryResult` ordered by descending similarity score.
        """
        ...

    async def get(self, record_id: str) -> VectorRecord | None:
        """Fetch a record by its global id.

        Args:
            record_id: The unique record identifier (namespace-free).

        Returns:
            The matching :class:`VectorRecord`, or ``None`` if not found.
        """
        ...

    async def delete(self, ids: list[str]) -> int:
        """Delete records by id and return the number removed.

        Backends that cannot report an exact delete count (e.g. Pinecone,
        Qdrant) return the number of ids *requested* as an approximation,
        so callers must not rely on the return value to determine existence.

        Args:
            ids: Record identifiers to delete.

        Returns:
            Number of records removed (may be approximate for some backends).
        """
        ...

    async def clear(self, *, namespace: str) -> int:
        """Delete all records in ``namespace`` and return the number removed.

        Args:
            namespace: The namespace whose records should be deleted.

        Returns:
            Number of records removed.
        """
        ...

    async def close(self) -> None:
        """Release resources held by the store (connections, pools, etc.)."""
        ...

"""VectorMemory — an embedding-backed Memory.

Composes an :class:`Embedder` and a :class:`VectorStore` to satisfy the
existing :class:`Memory` ABC.  ``add`` embeds content as a document and upserts;
``search`` embeds the query and queries the store.  Because it is a ``Memory``,
the Runner, ``MemoryConfig``, and the existing ``RecallMemoryTool`` use it
unchanged.
"""

from __future__ import annotations

import logging
from typing import override

from troopai.adk.llms.embedder import Embedder
from troopai.adk.memory.memory import Memory
from troopai.adk.memory.memory_types import (
    MemoryEntry,
    MemoryMetadata,
    MemorySearchFilter,
    MemorySearchResult,
    MemorySource,
)
from troopai.adk.memory.vector_store import VectorRecord, VectorStore

logger = logging.getLogger(__name__)


def _record_to_entry(record: VectorRecord) -> MemoryEntry:
    """Project a VectorRecord onto a MemoryEntry (drop the vector)."""
    return MemoryEntry(
        id=record.id,
        content=record.content,
        namespace=record.namespace,
        metadata=record.metadata,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


class VectorMemory(Memory):
    """Embedding-backed memory that composes an :class:`Embedder` and a :class:`VectorStore`.

    ``add`` embeds content as a document and upserts the record.
    ``search`` embeds the query and issues a nearest-neighbour query.

    Args:
        store: The vector store backend.
        embedder: The embedder that converts content and queries into vectors.
    """

    def __init__(self, *, store: VectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    @override
    async def add(
        self,
        content: str,
        *,
        namespace: str,
        metadata: MemoryMetadata | None = None,
    ) -> MemoryEntry:
        """Embed and store a new memory entry.

        Args:
            content: The knowledge to remember.
            namespace: Scoping key (e.g. ``"user:123"``).
            metadata: Optional metadata.  If ``None``, uses default
                metadata with ``MemorySource.MANUAL``.

        Returns:
            The created :class:`MemoryEntry`.

        Raises:
            RuntimeError: If the embedder returns no embedding for the content.
        """
        meta = metadata or MemoryMetadata(source=MemorySource.MANUAL)
        try:
            embeddings = await self._embedder.aembed_documents([content])
        except Exception as exc:
            logger.error("VectorMemory.add: embedder error for namespace=%s: %s", namespace, exc)
            raise
        if len(embeddings) == 0:
            raise RuntimeError("VectorMemory.add: embedder returned no embedding")
        now = self._now()
        record = VectorRecord(
            id=self._generate_id(),
            vector=embeddings[0].vector,
            namespace=namespace,
            content=content,
            metadata=meta,
            created_at=now,
            updated_at=now,
        )
        try:
            await self._store.upsert([record])
        except Exception as exc:
            logger.error("VectorMemory.add: store upsert error for namespace=%s: %s", namespace, exc)
            raise
        logger.debug("VectorMemory: added entry %s (namespace=%s)", record.id, namespace)
        return _record_to_entry(record)

    @override
    async def search(
        self,
        query: str,
        *,
        namespace: str,
        limit: int = 5,
        filter: MemorySearchFilter | None = None,
    ) -> list[MemorySearchResult]:
        """Embed the query and return the nearest memory entries.

        Args:
            query: Search query text.  Returns empty list if blank.
            namespace: Namespace to search within.
            limit: Maximum results to return.
            filter: Optional filters on metadata fields.

        Returns:
            List of :class:`MemorySearchResult` ordered by descending relevance.
        """
        if len(query.strip()) == 0:
            return []
        try:
            embedding = await self._embedder.aembed_query(query)
        except Exception as exc:
            logger.error(
                "VectorMemory.search: embedder error for query=%r namespace=%s: %s",
                query[:50],
                namespace,
                exc,
            )
            raise
        try:
            results = await self._store.query(
                embedding.vector,
                namespace=namespace,
                k=limit,
                filter=filter,
            )
        except Exception as exc:
            logger.error(
                "VectorMemory.search: store query error for namespace=%s: %s",
                namespace,
                exc,
            )
            raise
        return [MemorySearchResult(entry=_record_to_entry(r.record), score=r.score) for r in results]

    @override
    async def get(self, memory_id: str) -> MemoryEntry | None:
        """Retrieve a specific memory entry by ID.

        Args:
            memory_id: The entry's unique identifier.

        Returns:
            The entry, or ``None`` if not found.
        """
        record = await self._store.get(memory_id)
        return _record_to_entry(record) if record is not None else None

    @override
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by id.

        Returns ``True`` when the store reports a removal. Note: backends that
        cannot report exact delete counts (e.g. Pinecone, Qdrant) treat the
        request as removed, so ``True`` does not guarantee the id existed.
        """
        return await self._store.delete([memory_id]) > 0

    @override
    async def clear(self, *, namespace: str) -> int:
        """Delete all entries in a namespace.

        Args:
            namespace: The namespace to clear.

        Returns:
            Number of entries deleted.
        """
        return await self._store.clear(namespace=namespace)

    @override
    async def close(self) -> None:
        """Close the underlying vector store and release its resources."""
        await self._store.close()

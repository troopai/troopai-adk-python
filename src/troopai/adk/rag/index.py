"""DocumentIndex — chunk, embed, store, and search a document corpus.

The retrieval core of the RAG layer. It composes an :class:`Embedder` and a
:class:`VectorStore` (the same primitives :class:`VectorMemory` uses, so any of
the framework's vector backends works) with a :class:`TextChunker`. Documents
are split into chunks, embedded in batches, and upserted; a query is embedded
and matched by cosine similarity. Chunks are stored under an explicit namespace
so an index can share a backend with conversation memory without collision.

This is independently useful: a developer can build and query an index
directly, without the agent-facing ``DocumentSearchTool``.
"""

from __future__ import annotations

import logging
import time
import uuid

from troopai.adk.llms.embedder import Embedder
from troopai.adk.memory.memory_types import MemoryKind, MemoryMetadata, MemorySource
from troopai.adk.memory.stores.in_memory import InMemoryVectorStore
from troopai.adk.memory.vector_store import VectorRecord, VectorStore
from troopai.adk.rag.chunking import TextChunker
from troopai.adk.rag.document import DocumentSearchHit, LoadedDocument

logger = logging.getLogger(__name__)

SOURCE_FACET = "source"
"""Reserved ``custom`` metadata key holding a chunk's originating source."""


class DocumentIndex:
    """Chunk, embed, store, and semantically search a document corpus.

    Args:
        embedder: Converts chunk and query text into vectors.
        store: Vector backend. Defaults to an ephemeral
            :class:`InMemoryVectorStore`.
        chunker: Splitter applied before embedding. Defaults to a
            :class:`TextChunker` with standard bounds.
        namespace: Scoping key for this index's chunks. Defaults to
            ``"documents"``.
        batch_size: Maximum chunks embedded per provider call. Must be > 0.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore | None = None,
        chunker: TextChunker | None = None,
        namespace: str = "documents",
        batch_size: int = 128,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"DocumentIndex.batch_size must be > 0, got {batch_size}")
        if len(namespace) == 0:
            raise ValueError("DocumentIndex.namespace must be non-empty")
        self._embedder = embedder
        self._store: VectorStore = store if store is not None else InMemoryVectorStore()
        self._chunker = chunker if chunker is not None else TextChunker()
        self._namespace = namespace
        self._batch_size = batch_size

    async def add_documents(self, documents: list[LoadedDocument]) -> int:
        """Chunk, embed (in batches), and store ``documents``.

        Args:
            documents: The documents to index.

        Returns:
            The number of chunks stored.

        Raises:
            RuntimeError: If the embedder returns a count mismatched to its
                input batch.
        """
        chunks: list[LoadedDocument] = []
        for document in documents:
            chunks.extend(self._chunker.split_document(document))
        if len(chunks) == 0:
            return 0
        stored = 0
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            embeddings = await self._embedder.aembed_documents([chunk.content for chunk in batch])
            if len(embeddings) != len(batch):
                raise RuntimeError(f"Embedder returned {len(embeddings)} vectors for {len(batch)} chunks")
            now = time.time()
            records = [
                self._build_record(chunk, vector=embedding.vector, now=now)
                for chunk, embedding in zip(batch, embeddings, strict=True)
            ]
            await self._store.upsert(records)
            stored += len(records)
        logger.debug("DocumentIndex: stored %d chunk(s) in namespace=%s", stored, self._namespace)
        return stored

    async def search(self, query: str, *, limit: int = 5) -> list[DocumentSearchHit]:
        """Embed ``query`` and return the nearest stored chunks.

        Args:
            query: The search text. A blank query returns no hits.
            limit: Maximum number of hits to return.

        Returns:
            Matching chunks ordered by descending relevance.
        """
        if len(query.strip()) == 0:
            return []
        embedding = await self._embedder.aembed_query(query)
        results = await self._store.query(embedding.vector, namespace=self._namespace, k=limit)
        return [_result_to_hit(result.record, result.score) for result in results]

    async def clear(self) -> int:
        """Delete every chunk in this index's namespace.

        Returns:
            The number of chunks removed.
        """
        return await self._store.clear(namespace=self._namespace)

    async def close(self) -> None:
        """Release the underlying vector store's resources."""
        await self._store.close()

    def _build_record(self, chunk: LoadedDocument, *, vector: tuple[float, ...], now: float) -> VectorRecord:
        """Build a :class:`VectorRecord` carrying the chunk's provenance."""
        # Provenance goes last so a loader metadata key named "source" cannot
        # overwrite the reserved facet that _result_to_hit reports as the hit
        # source.
        custom = {**chunk.metadata, SOURCE_FACET: chunk.source}
        metadata = MemoryMetadata(source=MemorySource.TOOL, kind=MemoryKind.SEMANTIC, custom=custom)
        return VectorRecord(
            id=uuid.uuid4().hex,
            vector=vector,
            namespace=self._namespace,
            content=chunk.content,
            metadata=metadata,
            created_at=now,
            updated_at=now,
        )


def _result_to_hit(record: VectorRecord, score: float) -> DocumentSearchHit:
    """Project a stored record + score back onto a developer-facing hit."""
    custom = dict(record.metadata.custom)
    source = custom.pop(SOURCE_FACET, "")
    return DocumentSearchHit(content=record.content, source=source, score=score, metadata=custom)

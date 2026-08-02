"""Tests for DocumentIndex (chunk -> embed -> store -> search)."""

from __future__ import annotations

import hashlib
from typing import override

import pytest

from troopai.adk.llms import Embedder, Embedding
from troopai.adk.memory.stores.in_memory import InMemoryVectorStore
from troopai.adk.rag.chunking import TextChunker
from troopai.adk.rag.document import LoadedDocument
from troopai.adk.rag.index import DocumentIndex

_DIM = 64


class _HashEmbedder(Embedder):
    """Deterministic bag-of-hashed-words embedder (offline, no API)."""

    @override
    async def aembed_documents(self, texts: list[str]) -> list[Embedding]:
        out: list[Embedding] = []
        for text in texts:
            vector = [0.0] * _DIM
            for token in text.lower().split():
                index = int(hashlib.md5(token.encode()).hexdigest(), 16) % _DIM
                vector[index] += 1.0
            out.append(Embedding(vector=tuple(vector), model="hash"))
        return out

    @property
    @override
    def dimensions(self) -> int | None:
        return _DIM


def _docs() -> list[LoadedDocument]:
    return [
        LoadedDocument(content="apples oranges bananas are fruits", source="fruit.txt"),
        LoadedDocument(content="python rust go are programming languages", source="lang.txt"),
    ]


async def test_add_documents_returns_chunk_count() -> None:
    index = DocumentIndex(embedder=_HashEmbedder())
    assert await index.add_documents(_docs()) == 2


async def test_empty_documents_store_nothing() -> None:
    index = DocumentIndex(embedder=_HashEmbedder())
    assert await index.add_documents([]) == 0
    assert await index.add_documents([LoadedDocument(content="   ", source="blank.txt")]) == 0


async def test_search_returns_hit_with_provenance() -> None:
    index = DocumentIndex(embedder=_HashEmbedder())
    await index.add_documents(_docs())
    hits = await index.search("programming languages", limit=1)
    assert len(hits) == 1
    assert hits[0].source == "lang.txt"
    assert hits[0].score > 0.0


async def test_blank_query_returns_no_hits() -> None:
    index = DocumentIndex(embedder=_HashEmbedder())
    await index.add_documents(_docs())
    assert await index.search("  ") == []


async def test_clear_empties_the_index() -> None:
    index = DocumentIndex(embedder=_HashEmbedder())
    await index.add_documents(_docs())
    removed = await index.clear()
    assert removed == 2
    assert await index.search("fruits") == []


async def test_chunk_metadata_round_trips() -> None:
    index = DocumentIndex(embedder=_HashEmbedder(), chunker=TextChunker(chunk_size=20, chunk_overlap=5))
    await index.add_documents([LoadedDocument(content="alpha " * 30, source="a.txt", metadata={"page": "2"})])
    hits = await index.search("alpha", limit=1)
    assert hits[0].metadata["page"] == "2"
    assert "chunk" in hits[0].metadata


async def test_namespaces_isolate_within_one_store() -> None:
    store = InMemoryVectorStore()
    a = DocumentIndex(embedder=_HashEmbedder(), store=store, namespace="A")
    b = DocumentIndex(embedder=_HashEmbedder(), store=store, namespace="B")
    await a.add_documents([LoadedDocument(content="alpha topic", source="a.txt")])
    await b.add_documents([LoadedDocument(content="beta topic", source="b.txt")])
    hits = await a.search("alpha topic", limit=5)
    assert {hit.source for hit in hits} == {"a.txt"}


async def test_small_batch_size_indexes_all_chunks() -> None:
    index = DocumentIndex(embedder=_HashEmbedder(), batch_size=1)
    docs = [LoadedDocument(content=f"doc number {i}", source=f"{i}.txt") for i in range(5)]
    assert await index.add_documents(docs) == 5


@pytest.mark.parametrize(("batch", "namespace"), [(0, "n"), (-1, "n"), (4, "")])
def test_invalid_construction_raises(batch: int, namespace: str) -> None:
    with pytest.raises(ValueError):
        DocumentIndex(embedder=_HashEmbedder(), batch_size=batch, namespace=namespace)


async def test_loader_source_metadata_cannot_override_provenance() -> None:
    # A loader metadata key named "source" must not masquerade as the hit's
    # originating source; the real provenance wins.
    index = DocumentIndex(embedder=_HashEmbedder())
    await index.add_documents(
        [LoadedDocument(content="alpha topic here", source="real.txt", metadata={"source": "spoofed"})]
    )
    hits = await index.search("alpha topic", limit=1)
    assert len(hits) == 1
    assert hits[0].source == "real.txt"
    assert hits[0].source != "spoofed"

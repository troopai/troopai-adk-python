"""Tests for VectorMemory (Embedder + VectorStore composed into a Memory)."""

from __future__ import annotations

from typing import override

import pytest

from troopai.adk.llms import Embedder, Embedding
from troopai.adk.memory import MemoryMetadata, MemorySource
from troopai.adk.memory.stores.in_memory import InMemoryVectorStore
from troopai.adk.memory.vector_memory import VectorMemory


class _BagEmbedder(Embedder):
    """Deterministic 3-d embedder: [len, count('a'), count('e')]."""

    @override
    async def aembed_documents(self, texts: list[str]) -> list[Embedding]:
        return [Embedding(vector=(float(len(t)), float(t.count("a")), float(t.count("e"))), model="bag") for t in texts]

    @property
    @override
    def dimensions(self) -> int | None:
        return 3


def _mem() -> VectorMemory:
    return VectorMemory(store=InMemoryVectorStore(), embedder=_BagEmbedder())


async def test_add_then_search_round_trip() -> None:
    mem = _mem()
    entry = await mem.add("banana", namespace="u1")
    results = await mem.search("banana", namespace="u1", limit=5)
    assert results[0].entry.id == entry.id
    assert results[0].score == pytest.approx(1.0)


async def test_get_delete_clear() -> None:
    mem = _mem()
    entry = await mem.add("apple", namespace="u1", metadata=MemoryMetadata(source=MemorySource.MANUAL))
    fetched = await mem.get(entry.id)
    assert fetched is not None
    assert fetched.content == "apple"
    assert await mem.delete(entry.id) is True
    assert await mem.get(entry.id) is None
    await mem.add("pear", namespace="u1")
    assert await mem.clear(namespace="u1") == 1
    await mem.close()


async def test_add_from_session_pipeline_reused() -> None:
    """The inherited add_from_session pipeline works over VectorMemory."""
    from troopai.adk.memory.extractor import ExtractionResult, MemoryExtractor

    class _FakeExtractor(MemoryExtractor):
        @override
        async def extract(self, messages: list[object], *, namespace: str) -> list[ExtractionResult]:
            return [ExtractionResult(content="user likes tea")]

    mem = _mem()
    entries = await mem.add_from_session(["hi"], namespace="u1", extractor=_FakeExtractor())
    assert len(entries) == 1
    results = await mem.search("tea", namespace="u1")
    assert results[0].entry.content == "user likes tea"


async def test_search_empty_query_returns_empty() -> None:
    mem = _mem()
    await mem.add("banana", namespace="u1")
    assert await mem.search("   ", namespace="u1") == []


async def test_search_namespace_isolation() -> None:
    mem = _mem()
    await mem.add("banana", namespace="u1")
    assert await mem.search("banana", namespace="other") == []


def test_add_from_session_annotation_is_typed() -> None:
    """add_from_session messages param must not be list[Any]."""
    import inspect

    from troopai.adk.memory.memory import Memory

    hints = {}
    for name, param in inspect.signature(Memory.add_from_session).parameters.items():
        if name == "messages":
            hints["messages"] = str(param.annotation)
    assert "Any" not in hints.get("messages", "Any"), (
        "add_from_session messages should not be list[Any]; got: " + hints.get("messages", "")
    )


async def test_add_embedder_error_logs_and_reraises(caplog) -> None:
    """VectorMemory.add must logger.error and re-raise on embedder failure."""
    import logging

    class _ErrorEmbedder(Embedder):
        @override
        async def aembed_documents(self, texts: list[str]) -> list[Embedding]:
            raise RuntimeError("embedder exploded on add")

        @property
        @override
        def dimensions(self) -> int | None:
            return 2

    mem = VectorMemory(store=InMemoryVectorStore(), embedder=_ErrorEmbedder())
    with (
        caplog.at_level(logging.ERROR, logger="troopai.adk.memory.vector_memory"),
        pytest.raises(RuntimeError, match="embedder exploded on add"),
    ):
        await mem.add("test", namespace="u1")
    assert any("VectorMemory.add" in r.message for r in caplog.records if r.levelno >= logging.ERROR)


async def test_search_embedder_error_logs_and_reraises() -> None:
    """VectorMemory.search must logger.error and re-raise on embedder failure."""

    class _ErrorQueryEmbedder(Embedder):
        @override
        async def aembed_documents(self, texts: list[str]) -> list[Embedding]:
            return [Embedding(vector=(1.0, 0.0, 0.0), model="err")]

        @override
        async def aembed_query(self, text: str) -> Embedding:
            raise RuntimeError("embedder exploded on search")

        @property
        @override
        def dimensions(self) -> int | None:
            return 3

    mem = VectorMemory(store=InMemoryVectorStore(), embedder=_ErrorQueryEmbedder())
    await mem.add("something", namespace="u1")
    with pytest.raises(RuntimeError, match="embedder exploded on search"):
        await mem.search("query", namespace="u1")

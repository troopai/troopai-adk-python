"""Tests for the Embedder ABC, Embedding value type, and EmbeddingLRUCache."""

from __future__ import annotations

from typing import override

import pytest

from troopai.adk.llms import Embedder, Embedding, EmbeddingLRUCache


class _CountingEmbedder(Embedder):
    """Records how many texts it embedded; deterministic 2-d vectors."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @override
    async def aembed_documents(self, texts: list[str]) -> list[Embedding]:
        self.calls.append(list(texts))
        return [Embedding(vector=(float(len(t)), 1.0), model="fake") for t in texts]

    @property
    @override
    def dimensions(self) -> int | None:
        return 2


def test_embedding_dimensions() -> None:
    assert Embedding(vector=(0.1, 0.2, 0.3), model="m").dimensions == 3


async def test_aembed_query_delegates_to_documents() -> None:
    emb = _CountingEmbedder()
    result = await emb.aembed_query("hello")
    assert result.vector == (5.0, 1.0)
    assert emb.calls == [["hello"]]


def test_cache_hit_and_eviction() -> None:
    cache = EmbeddingLRUCache(max_size=2)
    e1 = Embedding(vector=(1.0,), model="m")
    e2 = Embedding(vector=(2.0,), model="m")
    e3 = Embedding(vector=(3.0,), model="m")
    cache.put(e1, text="a")
    cache.put(e2, text="b")
    assert cache.get("m", "a") is e1  # hit, marks 'a' recent
    cache.put(e3, text="c")  # evicts least-recent ('b')
    assert cache.get("m", "b") is None
    assert cache.get("m", "a") is e1
    assert cache.get("m", "c") is e3


def test_cache_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError):
        EmbeddingLRUCache(max_size=0)


class _EmptyEmbedder(Embedder):
    @override
    async def aembed_documents(self, texts: list[str]) -> list[Embedding]:
        return []

    @property
    @override
    def dimensions(self) -> int | None:
        return None


async def test_aembed_query_raises_on_empty_result() -> None:
    with pytest.raises(RuntimeError, match="no embedding"):
        await _EmptyEmbedder().aembed_query("hello")

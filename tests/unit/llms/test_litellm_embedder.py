"""Tests for LiteLLMEmbedder (litellm.aembedding mocked — no network)."""

from __future__ import annotations

from typing import Any

import pytest

from troopai.adk.llms import EmbeddingLRUCache
from troopai.adk.llms.litellm.litellm_embedder import LiteLLMEmbedder


class _FakeAembedding:
    """Stand-in for litellm.aembedding; counts calls, returns 2-d vectors."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def __call__(self, *, model: str, input: list[str], **kwargs: Any) -> dict[str, Any]:
        self.calls.append(list(input))
        return {"data": [{"embedding": [float(len(t)), 1.0], "index": i} for i, t in enumerate(input)]}


async def test_embeds_batch_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeAembedding()
    monkeypatch.setattr("litellm.aembedding", fake)
    emb = LiteLLMEmbedder(model="text-embedding-3-small")
    out = await emb.aembed_documents(["ab", "cdef"])
    assert [e.vector for e in out] == [(2.0, 1.0), (4.0, 1.0)]
    assert fake.calls == [["ab", "cdef"]]


async def test_cache_avoids_second_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeAembedding()
    monkeypatch.setattr("litellm.aembedding", fake)
    emb = LiteLLMEmbedder(model="m", cache=EmbeddingLRUCache(max_size=8))
    await emb.aembed_documents(["x"])
    await emb.aembed_documents(["x"])
    assert fake.calls == [["x"]]


async def test_empty_input_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeAembedding()
    monkeypatch.setattr("litellm.aembedding", fake)
    emb = LiteLLMEmbedder(model="m")
    assert await emb.aembed_documents([]) == []
    assert fake.calls == []


def test_dimensions_returns_hint() -> None:
    assert LiteLLMEmbedder(model="m", dimensions_hint=768).dimensions == 768
    assert LiteLLMEmbedder(model="m").dimensions is None


async def test_reorders_embeddings_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    async def out_of_order(*, model: str, input: list[str], **kwargs: Any) -> dict[str, Any]:
        # Return items in reversed order, but tagged with their true index.
        items = [{"embedding": [float(len(t)), 1.0], "index": i} for i, t in enumerate(input)]
        return {"data": list(reversed(items))}

    monkeypatch.setattr("litellm.aembedding", out_of_order)
    emb = LiteLLMEmbedder(model="m")
    out = await emb.aembed_documents(["a", "bbb"])
    # index 0 -> "a" (len 1), index 1 -> "bbb" (len 3): order must follow input, not the reversed response
    assert [e.vector for e in out] == [(1.0, 1.0), (3.0, 1.0)]


async def test_missing_data_key_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """aembed_documents must raise RuntimeError (not KeyError) when response has no 'data' key."""

    async def bad_response(*, model: str, input: list[str], **kwargs: Any) -> dict[str, Any]:
        return {}  # missing 'data' key

    monkeypatch.setattr("litellm.aembedding", bad_response)
    emb = LiteLLMEmbedder(model="text-embedding-bad")
    with pytest.raises(RuntimeError, match="text-embedding-bad"):
        await emb.aembed_documents(["hello"])


async def test_partial_cache_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeAembedding()
    monkeypatch.setattr("litellm.aembedding", fake)
    emb = LiteLLMEmbedder(model="m", cache=EmbeddingLRUCache(max_size=8))
    await emb.aembed_documents(["alpha", "gamma"])  # warm the cache
    fake.calls.clear()
    out = await emb.aembed_documents(["alpha", "beta", "gamma"])
    assert fake.calls == [["beta"]]  # only the miss hit the network
    # results reassembled in original order (alpha=5, beta=4, gamma=5 chars):
    assert [e.vector for e in out] == [(5.0, 1.0), (4.0, 1.0), (5.0, 1.0)]

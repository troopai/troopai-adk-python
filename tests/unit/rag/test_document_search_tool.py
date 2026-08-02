"""Tests for DocumentSearchTool and its typed *SearchTool wrappers."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import override

import pytest

from troopai.adk.llms import Embedder, Embedding
from troopai.adk.rag.loaders import DocumentLoader, MarkdownLoader, PDFLoader, TextLoader
from troopai.adk.tools.builtin.document_search_tool import (
    DirectorySearchTool,
    DocumentSearchTool,
    MarkdownSearchTool,
    PDFSearchTool,
    TXTSearchTool,
)

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


def _write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


async def test_invoke_returns_formatted_hits(tmp_path: Path) -> None:
    source = _write(tmp_path, "bio.txt", "Mitochondria is the powerhouse of the cell.")
    tool = TXTSearchTool(sources=[source], embedder=_HashEmbedder())
    result = await tool.on_invoke(None, json.dumps({"query": "powerhouse of the cell"}))
    assert "powerhouse" in result
    assert "result(s)" in result


async def test_search_returns_hits_with_source(tmp_path: Path) -> None:
    source = _write(tmp_path, "a.txt", "alpha beta gamma delta")
    tool = DocumentSearchTool(sources=[source], embedder=_HashEmbedder())
    hits = await tool.search("alpha beta")
    assert hits[0].source == source


async def test_indexing_is_lazy_and_idempotent(tmp_path: Path) -> None:
    source = _write(tmp_path, "a.txt", "alpha beta gamma")
    tool = DocumentSearchTool(sources=[source], embedder=_HashEmbedder())
    first = await tool.index()
    second = await tool.index()
    assert first == second == 1


async def test_clear_resets_and_reindexes(tmp_path: Path) -> None:
    source = _write(tmp_path, "a.txt", "alpha beta gamma")
    tool = DocumentSearchTool(sources=[source], embedder=_HashEmbedder())
    await tool.index()
    assert await tool.clear() == 1
    assert await tool.index() == 1


async def test_empty_query_message(tmp_path: Path) -> None:
    source = _write(tmp_path, "a.txt", "alpha")
    tool = DocumentSearchTool(sources=[source], embedder=_HashEmbedder())
    assert "No query" in await tool.on_invoke(None, json.dumps({"query": "   "}))


async def test_no_hits_message() -> None:
    tool = DocumentSearchTool(sources=[], embedder=_HashEmbedder())
    result = await tool.on_invoke(None, json.dumps({"query": "anything"}))
    assert "No relevant content" in result


async def test_invalid_json_args_reported(tmp_path: Path) -> None:
    source = _write(tmp_path, "a.txt", "alpha")
    tool = DocumentSearchTool(sources=[source], embedder=_HashEmbedder())
    assert "JSON parse error" in await tool.on_invoke(None, "{not json")


async def test_limit_is_clamped_to_ceiling(tmp_path: Path) -> None:
    for i in range(8):
        _write(tmp_path, f"f{i}.txt", f"common topic document number {i}")
    tool = DirectorySearchTool(sources=[str(tmp_path)], embedder=_HashEmbedder())
    result = await tool.on_invoke(None, json.dumps({"query": "common topic", "limit": 9999}))
    # 8 docs indexed; an over-large limit must not error and returns at most all of them.
    assert "result(s)" in result


async def test_non_string_query_is_coerced_not_crashed(tmp_path: Path) -> None:
    # A misbehaving LLM may emit a non-string ``query`` (e.g. an int) despite
    # the schema. ``on_invoke`` must coerce it rather than raise AttributeError
    # on ``.strip()`` (which the executor would turn into an Error result that
    # burns the tool's retry budget).
    source = _write(tmp_path, "a.txt", "alpha beta gamma")
    tool = DocumentSearchTool(sources=[source], embedder=_HashEmbedder())
    # Before the fix this raised AttributeError on ``int.strip()``; the call
    # must now complete and return a formatted digest, not blow up.
    result = await tool.on_invoke(None, '{"query": 123}')
    assert "Error" not in result
    assert "result(s) for '123'" in result


async def test_non_integer_limit_returns_validation_message(tmp_path: Path) -> None:
    # A non-integer ``limit`` must yield a clean validation message, not a
    # ValueError from ``int(...)`` that the executor converts to an Error
    # result and counts against the retry budget.
    source = _write(tmp_path, "a.txt", "alpha beta gamma")
    tool = DocumentSearchTool(sources=[source], embedder=_HashEmbedder())
    result = await tool.on_invoke(None, json.dumps({"query": "alpha", "limit": "many"}))
    assert "Invalid 'limit'" in result


def test_default_limit_validation_raises() -> None:
    with pytest.raises(ValueError):
        DocumentSearchTool(sources=[], embedder=_HashEmbedder(), default_limit=0)
    with pytest.raises(ValueError):
        DocumentSearchTool(sources=[], embedder=_HashEmbedder(), default_limit=999)


def test_wrappers_pin_their_loader() -> None:
    embedder = _HashEmbedder()
    assert isinstance(TXTSearchTool(embedder=embedder).loader, TextLoader)
    assert isinstance(MarkdownSearchTool(embedder=embedder).loader, MarkdownLoader)
    assert isinstance(PDFSearchTool(embedder=embedder).loader, PDFLoader)


def test_wrapper_accepts_explicit_loader_override() -> None:
    custom = TextLoader()
    tool = TXTSearchTool(embedder=_HashEmbedder(), loader=custom)
    assert tool.loader is custom


async def test_generic_tool_auto_dispatches_mixed_sources(tmp_path: Path) -> None:
    txt = _write(tmp_path, "a.txt", "alpha content here")
    md = _write(tmp_path, "b.md", "# Beta\n\nbeta content here")
    tool = DocumentSearchTool(sources=[txt, md], embedder=_HashEmbedder())
    assert await tool.index() == 2
    hits = await tool.search("beta content", limit=2)
    assert any(hit.source == md for hit in hits)


async def test_names_are_distinct() -> None:
    embedder = _HashEmbedder()
    names = {
        DocumentSearchTool(embedder=embedder).name,
        TXTSearchTool(embedder=embedder).name,
        MarkdownSearchTool(embedder=embedder).name,
        PDFSearchTool(embedder=embedder).name,
    }
    assert len(names) == 4


class _FlakyTextLoader(DocumentLoader):
    """Wraps a real TextLoader but fails once for a chosen source, then succeeds.

    Reproduces a transient loader/store error partway through indexing so the
    retry path can be exercised.
    """

    def __init__(self, fail_source: str) -> None:
        self._inner = TextLoader()
        self._fail_source = fail_source
        self._failed = False

    @override
    async def load(self, source: str):
        if source == self._fail_source and not self._failed:
            self._failed = True
            raise RuntimeError("transient load error")
        return await self._inner.load(source)


async def test_partial_index_failure_does_not_duplicate_on_retry(tmp_path: Path) -> None:
    f1 = _write(tmp_path, "a.txt", "alpha beta gamma")
    f2 = _write(tmp_path, "b.txt", "delta epsilon zeta")

    # Reference count: both files indexed cleanly, once each.
    clean = DocumentSearchTool(sources=[f1, f2], embedder=_HashEmbedder())
    expected = await clean.index()

    # f2 fails on the first attempt; f1 has already been indexed by then.
    tool = DocumentSearchTool(sources=[f1, f2], embedder=_HashEmbedder(), loader=_FlakyTextLoader(fail_source=f2))
    with pytest.raises(RuntimeError):
        await tool.index()

    # Retry must resume at f2, not re-index f1 (which would duplicate chunks).
    total = await tool.index()
    assert total == expected


def test_index_lock_survives_reuse_across_separate_sync_runs(tmp_path: Path) -> None:
    source = _write(tmp_path, "a.txt", "alpha beta gamma")
    tool = DocumentSearchTool(sources=[source], embedder=_HashEmbedder())

    # First run binds the indexing lock to its (throwaway) event loop.
    first = asyncio.run(tool.index())
    assert first == 1

    # A second sync run spins up a fresh loop; the lock must not be bound to
    # the first, now-closed loop.
    hits = asyncio.run(tool.search("alpha"))
    assert isinstance(hits, list)

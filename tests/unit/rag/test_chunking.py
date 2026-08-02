"""Tests for the recursive TextChunker."""

from __future__ import annotations

import pytest

from troopai.adk.rag.chunking import TextChunker
from troopai.adk.rag.document import LoadedDocument


def test_blank_text_yields_no_chunks() -> None:
    assert TextChunker().split_text("   \n\t ") == []


def test_short_text_is_one_chunk() -> None:
    chunks = TextChunker(chunk_size=100, chunk_overlap=20).split_text("a short sentence")
    assert chunks == ["a short sentence"]


def test_long_text_splits_into_bounded_chunks() -> None:
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = ". ".join(f"sentence number {i} has some words" for i in range(40))
    chunks = chunker.split_text(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_overlap_repeats_context() -> None:
    chunker = TextChunker(chunk_size=40, chunk_overlap=15, separators=(" ", ""))
    chunks = chunker.split_text(" ".join(f"word{i}" for i in range(30)))
    # Consecutive chunks should share at least one token because of overlap.
    first_tail = set(chunks[0].split()[-2:])
    second_head = set(chunks[1].split()[:3])
    assert len(first_tail & second_head) > 0


def test_split_document_propagates_source_and_chunk_index() -> None:
    chunker = TextChunker(chunk_size=30, chunk_overlap=5)
    doc = LoadedDocument(content="x" * 200, source="a.txt", metadata={"k": "v"})
    pieces = chunker.split_document(doc)
    assert len(pieces) > 1
    assert all(piece.source == "a.txt" for piece in pieces)
    assert all(piece.metadata["k"] == "v" for piece in pieces)
    assert [piece.metadata["chunk"] for piece in pieces] == [str(i) for i in range(len(pieces))]


def test_split_document_single_chunk_keeps_original() -> None:
    doc = LoadedDocument(content="tiny", source="a.txt")
    pieces = TextChunker().split_document(doc)
    assert pieces == [doc]
    assert "chunk" not in pieces[0].metadata


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-1, 0), (10, 10), (10, 20)],
)
def test_invalid_bounds_raise(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        TextChunker(chunk_size=size, chunk_overlap=overlap)


def test_empty_separators_raise() -> None:
    with pytest.raises(ValueError):
        TextChunker(separators=())


def test_separators_without_empty_terminator_raise() -> None:
    # A separators tuple lacking the per-character "" fallback cannot guarantee
    # character-level splitting, so a piece with no boundary would silently
    # exceed chunk_size; reject the misconfiguration at construction time.
    with pytest.raises(ValueError):
        TextChunker(chunk_size=5, chunk_overlap=1, separators=("\n", "|"))


@pytest.mark.parametrize(
    ("text", "size", "overlap"),
    [
        ("aa bbbb cc dddddd ee", 12, 8),
        ("a bb ccc dddd eeeee", 10, 7),
        ("xx yyyy zz wwwwww vv", 12, 9),
        ("aaa bbbbb ccc ddddddd eee", 14, 10),
    ],
)
def test_overlap_retention_does_not_emit_oversized_chunk(text: str, size: int, overlap: int) -> None:
    # Regression: when the gap (chunk_size - chunk_overlap) is small, the
    # overlap retained from the previous window plus the next piece could
    # exceed chunk_size. Every emitted chunk must stay within chunk_size.
    chunks = TextChunker(chunk_size=size, chunk_overlap=overlap).split_text(text)
    assert all(len(chunk) <= size for chunk in chunks)

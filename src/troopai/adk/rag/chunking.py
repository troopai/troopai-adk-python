"""Recursive character text splitting for RAG ingestion.

Long documents are split into bounded, overlapping chunks before embedding so
that each vector represents a focused span (better retrieval granularity and
citation). The splitter is recursive: it tries the coarsest separator first
(paragraph breaks), falling through to finer ones (lines, sentences, words,
characters) only when a piece still exceeds ``chunk_size``. This keeps natural
boundaries intact wherever possible.

Pure-stdlib and dependency-free. Sizes are measured in characters, not tokens,
to avoid coupling ingestion to any tokenizer; the defaults (~1000 chars ≈ 250
tokens) sit comfortably inside every embedding model's context window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from troopai.adk.rag.document import LoadedDocument

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")
"""Separators tried coarsest-first; the empty string splits per character."""


@dataclass(frozen=True)
class TextChunker:
    """Splits text into bounded, overlapping chunks on natural boundaries.

    Attributes:
        chunk_size: Maximum characters per chunk. Must be > 0.
        chunk_overlap: Characters of trailing context repeated at the start
            of the next chunk, to preserve continuity across a boundary.
            Must be >= 0 and < ``chunk_size``.
        separators: Ordered boundary strings, tried coarsest-first. The
            final separator must be ``""`` (per-character) as a guaranteed
            fallback for text with no other boundary.
    """

    chunk_size: int = 1000
    """Maximum characters per chunk."""

    chunk_overlap: int = 100
    """Characters of trailing context repeated at the start of the next chunk."""

    separators: tuple[str, ...] = field(default_factory=lambda: DEFAULT_SEPARATORS)
    """Ordered boundary strings, tried coarsest-first."""

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"TextChunker.chunk_size must be > 0, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"TextChunker.chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"TextChunker.chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        if len(self.separators) == 0:
            raise ValueError("TextChunker.separators must be non-empty")
        if self.separators[-1] != "":
            raise ValueError(
                "TextChunker.separators must end with '' (empty string) so that text "
                "with no other boundary can always be split per character within chunk_size"
            )

    def split_text(self, text: str) -> list[str]:
        """Split a string into bounded, overlapping chunks.

        Args:
            text: The text to split.

        Returns:
            The chunks in document order. Empty input yields an empty list.
            Text already within ``chunk_size`` yields a single chunk.
        """
        if len(text.strip()) == 0:
            return []
        return self._split(text, self.separators)

    def split_document(self, document: LoadedDocument) -> list[LoadedDocument]:
        """Split a document, propagating its source and adding a chunk index.

        Args:
            document: The document to split.

        Returns:
            One :class:`LoadedDocument` per chunk. Each carries the original
            ``source`` and ``metadata`` plus a ``chunk`` facet (its 0-based
            index). A document already within ``chunk_size`` returns a
            single-element list with no ``chunk`` facet added.
        """
        chunks = self.split_text(document.content)
        if len(chunks) <= 1:
            return [document] if len(chunks) == 1 else []
        result: list[LoadedDocument] = []
        for index, chunk in enumerate(chunks):
            metadata = {**document.metadata, "chunk": str(index)}
            result.append(LoadedDocument(content=chunk, source=document.source, metadata=metadata))
        return result

    def _split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        """Recursively split ``text`` using the first viable separator."""
        separator = separators[-1]
        remaining: tuple[str, ...] = ()
        for index, candidate in enumerate(separators):
            if candidate == "":
                separator = candidate
                break
            if candidate in text:
                separator = candidate
                remaining = separators[index + 1 :]
                break
        pieces = list(text) if separator == "" else text.split(separator)
        final: list[str] = []
        accumulated: list[str] = []
        for piece in pieces:
            if len(piece) > self.chunk_size and len(remaining) > 0:
                final.extend(self._merge(accumulated, separator))
                accumulated = []
                final.extend(self._split(piece, remaining))
            else:
                accumulated.append(piece)
        final.extend(self._merge(accumulated, separator))
        return final

    def _merge(self, pieces: list[str], separator: str) -> list[str]:
        """Greedily pack ``pieces`` into ``chunk_size`` chunks with overlap."""
        sep_len = len(separator)
        chunks: list[str] = []
        window: list[str] = []
        length = 0
        for piece in pieces:
            addition = len(piece) + (sep_len if len(window) > 0 else 0)
            if length + addition > self.chunk_size and len(window) > 0:
                joined = separator.join(window).strip()
                if len(joined) > 0:
                    chunks.append(joined)
                length, window = self._shrink_for_overlap(window, length, sep_len)
                length, window = self._shrink_for_piece(window, length, sep_len, len(piece))
            window.append(piece)
            length += len(piece) + (sep_len if len(window) > 1 else 0)
        joined = separator.join(window).strip()
        if len(joined) > 0:
            chunks.append(joined)
        return chunks

    def _shrink_for_overlap(self, window: list[str], length: int, sep_len: int) -> tuple[int, list[str]]:
        """Drop leading pieces until the window fits within ``chunk_overlap``."""
        while length > self.chunk_overlap and len(window) > 0:
            removed = window.pop(0)
            length -= len(removed) + (sep_len if len(window) > 0 else 0)
        return length, window

    def _shrink_for_piece(self, window: list[str], length: int, sep_len: int, piece_len: int) -> tuple[int, list[str]]:
        """Drop leading overlap pieces until the retained window plus the next
        piece still fits within ``chunk_size``.

        ``_shrink_for_overlap`` only bounds the retained window by
        ``chunk_overlap``; when the incoming piece is large relative to the gap
        ``chunk_size - chunk_overlap`` the retained overlap plus that piece can
        still exceed ``chunk_size``. This trims further so the next emitted
        chunk stays within bounds.
        """
        while len(window) > 0 and length + piece_len + sep_len > self.chunk_size:
            removed = window.pop(0)
            length -= len(removed) + (sep_len if len(window) > 0 else 0)
        return length, window

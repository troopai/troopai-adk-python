"""Core RAG data types — loaded documents and search hits.

A :class:`LoadedDocument` is the unit a loader returns: a contiguous span of
extracted text plus provenance metadata (the source it came from, and
loader-specific facets such as a PDF page number or a website title). The
chunker splits a document into smaller ``LoadedDocument`` pieces before
embedding. A :class:`DocumentSearchHit` is one retrieval result: the matching
chunk, its provenance, and a relevance score.

These types are deliberately decoupled from the memory module's
``MemoryMetadata`` (which carries conversation-memory semantics like
importance and episodic/semantic kind). The document-search layer owns its
own provenance shape; the bridge to ``VectorRecord`` lives in
:mod:`troopai.adk.rag.index`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoadedDocument:
    """A span of text extracted from a source, with provenance.

    Attributes:
        content: The extracted text.
        source: The path or URL the text was loaded from.
        metadata: Loader-specific provenance facets as string key-value
            pairs (e.g. ``{"page": "3"}`` for a PDF, ``{"title": "..."}``
            for a website). String-valued to round-trip cleanly through
            vector-store metadata backends.
    """

    content: str
    """The extracted text."""

    source: str
    """The path or URL the text was loaded from."""

    metadata: dict[str, str] = field(default_factory=dict)
    """Loader-specific provenance facets as string key-value pairs."""


@dataclass(frozen=True)
class DocumentSearchHit:
    """A single semantic-search result over an indexed corpus.

    Attributes:
        content: The matching chunk's text.
        source: The path or URL the chunk was loaded from.
        score: Cosine similarity normalized to 0.0-1.0 (higher is closer).
        metadata: Provenance facets carried from the originating
            :class:`LoadedDocument` (e.g. page number, chunk index).
    """

    content: str
    """The matching chunk's text."""

    source: str
    """The path or URL the chunk was loaded from."""

    score: float
    """Cosine similarity normalized to 0.0-1.0 (higher is closer)."""

    metadata: dict[str, str] = field(default_factory=dict)
    """Provenance facets carried from the originating document."""

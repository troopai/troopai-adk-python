"""DocumentSearchTool — agent-facing semantic search over a document corpus.

A built-in tool that gives an agent a ``search(query)`` capability over a
developer-curated set of documents (files, directories, or URLs). It composes
the RAG primitives — format loaders, the :class:`TextChunker`, and a
:class:`DocumentIndex` over any :class:`VectorStore` — behind the
:class:`ExecutableBuiltinTool` contract, so it sits in ``Agent.tools`` and is
dispatched like any other tool.

Design choices that follow the framework's cost and safety posture:

- **Explicit embedder, no default.** Embedding spends tokens, so the developer
  must supply an :class:`Embedder`; the framework never picks one implicitly.
- **Ephemeral store by default.** Absent an explicit ``vector_store``, an
  in-memory store is used — nothing is written to disk without opt-in.
- **Sources are bound at construction.** The corpus is the developer's, fixed
  when the tool is built; the LLM supplies only the *query*, never a new path
  or URL. This keeps the tool off the attacker-controlled file/SSRF surface.
- **Lazy, idempotent indexing.** Loading + chunking + embedding runs on the
  first search (or an explicit :meth:`DocumentSearchTool.index` call) and once
  only — the embedding cost is paid when the corpus is first queried.

The thin ``*SearchTool`` subclasses (``PDFSearchTool``, ``WebsiteSearchTool``,
…) pin a loader and tailor the description; they add no pipeline of their own.
See ``docs/rag/`` and ``examples/rag/``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from troopai.adk.llms.embedder import Embedder
from troopai.adk.memory.vector_store import VectorStore
from troopai.adk.rag.chunking import TextChunker
from troopai.adk.rag.document import DocumentSearchHit
from troopai.adk.rag.index import DocumentIndex
from troopai.adk.rag.loaders import (
    CSVLoader,
    DirectoryLoader,
    DocumentLoader,
    DOCXLoader,
    GithubLoader,
    JSONLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    WebsiteLoader,
    YoutubeChannelLoader,
    YoutubeVideoLoader,
    resolve_loader,
)
from troopai.adk.tools.builtin.builtin_tool import ExecutableBuiltinTool

logger = logging.getLogger(__name__)

MAX_LIMIT_CEILING = 50
"""Hard upper bound on results per query, surfaced to the LLM in the schema."""


class DocumentSearchInput(BaseModel):
    """Input schema for the ``document_search`` tool family."""

    query: str = Field(description="What to look for. A natural-language search query over the documents.")
    limit: int | None = Field(
        default=None,
        ge=1,
        le=MAX_LIMIT_CEILING,
        description="Maximum number of passages to return. Defaults to the tool's configured limit.",
    )


@dataclass(kw_only=True)
class DocumentSearchTool(ExecutableBuiltinTool):
    """Semantic search over a developer-curated document corpus.

    Attributes:
        name: Tool name shown to the LLM (``"document_search"``).
        description: Tool description shown to the LLM.
        schema: Input schema (:class:`DocumentSearchInput`).
        embedder: Embedder used for chunks and queries (required).
        sources: Paths / directories / URLs to index. Routed by extension or
            URL shape when ``loader`` is ``None``.
        vector_store: Backend for chunk vectors. Defaults to an ephemeral
            in-memory store.
        chunker: Splitter applied before embedding. Defaults to a standard
            :class:`TextChunker`.
        loader: Pins a single loader for every source (set by the typed
            subclasses). ``None`` means dispatch per source by type.
        namespace: Scoping key for this corpus's chunks.
        default_limit: Results returned when the model omits ``limit``.
    """

    name: str = "document_search"
    description: str = (
        "Search the indexed documents for passages relevant to a query and "
        "return them with their source. Use this to ground answers in the "
        "provided document corpus."
    )
    schema: type[BaseModel] | dict[str, Any] = DocumentSearchInput

    embedder: Embedder
    sources: Sequence[str] = ()
    vector_store: VectorStore | None = None
    chunker: TextChunker | None = None
    loader: DocumentLoader | None = None
    namespace: str = "documents"
    default_limit: int = 5
    _index: DocumentIndex = field(init=False, repr=False)
    _indexed: bool = field(default=False, init=False, repr=False)
    _indexed_source_count: int = field(default=0, init=False, repr=False)
    _chunk_count: int = field(default=0, init=False, repr=False)
    _lock: asyncio.Lock | None = field(default=None, init=False, repr=False)
    _lock_loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.default_limit <= 0 or self.default_limit > MAX_LIMIT_CEILING:
            raise ValueError(f"default_limit must be 1-{MAX_LIMIT_CEILING}, got {self.default_limit}")
        self._index = DocumentIndex(
            embedder=self.embedder,
            store=self.vector_store,
            chunker=self.chunker,
            namespace=self.namespace,
        )
        if self.on_invoke is None:
            self.on_invoke = _make_search_invoke(self)

    def _get_lock(self) -> asyncio.Lock:
        """Return the indexing lock bound to the current running loop.

        An ``asyncio.Lock`` binds to whichever loop first acquires it and then
        rejects acquisition from any other loop. A tool instance reused across
        separate synchronous runs (each spinning up its own throwaway loop)
        would raise ``RuntimeError`` on the second run, so the lock is created
        lazily and rebuilt whenever the running loop changes. Sequential runs
        never overlap, so a fresh lock per loop still guarantees mutual
        exclusion within any single run.
        """
        loop = asyncio.get_running_loop()
        lock = self._lock
        if lock is None or self._lock_loop is not loop:
            lock = asyncio.Lock()
            self._lock = lock
            self._lock_loop = loop
        return lock

    async def index(self) -> int:
        """Load, chunk, and embed the configured sources (idempotent).

        Runs once; later calls return the cached chunk count without
        re-embedding. Called automatically on the first search, or directly to
        pre-warm the corpus and pay the embedding cost up front.

        Returns:
            The number of chunks indexed.
        """
        async with self._get_lock():
            if self._indexed:
                return self._chunk_count
            # Resume from the first source not yet successfully indexed. If a
            # prior attempt raised partway through (a bad path, a network
            # blip), the sources it already added stay in the index and are
            # skipped here — re-loading them would embed and store every chunk
            # a second time, so a retried search would return duplicates.
            for source in self.sources[self._indexed_source_count :]:
                loader = self.loader if self.loader is not None else resolve_loader(source)
                documents = await loader.load(source)
                self._chunk_count += await self._index.add_documents(documents)
                self._indexed_source_count += 1
            self._indexed = True
            logger.debug("DocumentSearchTool[%s]: indexed %d chunk(s)", self.name, self._chunk_count)
            return self._chunk_count

    async def search(self, query: str, *, limit: int | None = None) -> list[DocumentSearchHit]:
        """Index on first use, then return the passages most relevant to ``query``.

        Args:
            query: The search text.
            limit: Maximum hits; defaults to ``default_limit``.

        Returns:
            Matching passages ordered by descending relevance.
        """
        await self.index()
        resolved = self.default_limit if limit is None else max(1, min(MAX_LIMIT_CEILING, limit))
        return await self._index.search(query, limit=resolved)

    async def clear(self) -> int:
        """Drop every indexed chunk and reset, so the next search re-indexes.

        Returns:
            The number of chunks removed.
        """
        async with self._get_lock():
            removed = await self._index.clear()
            self._indexed = False
            self._indexed_source_count = 0
            self._chunk_count = 0
            return removed


def _make_search_invoke(tool: DocumentSearchTool):
    """Create the ``on_invoke`` callable for a document-search tool.

    Args:
        tool: The tool whose corpus the callable searches.

    Returns:
        An async callable matching the built-in ``on_invoke`` contract
        (``(ctx, raw_args) -> str``).
    """

    async def _invoke(ctx: Any, raw_args: str) -> str:  # noqa: ARG001 - ctx unused; matches builtin contract
        try:
            args = json.loads(raw_args) if len(raw_args) > 0 else {}
        except json.JSONDecodeError as exc:
            return f"Invalid tool arguments (JSON parse error): {exc}"
        query = args.get("query", "")
        if not isinstance(query, str):
            query = str(query)
        if len(query.strip()) == 0:
            return "No query provided."
        # ``limit`` is declared with ``ge=1, le=MAX_LIMIT_CEILING`` on the
        # schema, but the executor calls ``on_invoke`` directly without
        # re-running Pydantic validation, so a misbehaving LLM could submit a
        # non-integer ``limit``. Coerce and clamp defensively, returning a
        # clean validation message instead of raising — a raise here would be
        # turned into an error result and burn the tool's retry budget.
        raw_limit = args.get("limit")
        if raw_limit is None:
            limit = None
        else:
            try:
                limit = max(1, min(MAX_LIMIT_CEILING, int(raw_limit)))
            except (TypeError, ValueError):
                return f"Invalid 'limit' — must be an integer between 1 and {MAX_LIMIT_CEILING}."
        hits = await tool.search(query, limit=limit)
        return _format_hits(query, hits)

    return _invoke


def _provenance(hit: DocumentSearchHit) -> str:
    """Build a human-readable provenance label from a hit's metadata."""
    label = hit.source
    page = hit.metadata.get("page")
    path = hit.metadata.get("path")
    if page is not None:
        label += f" (page {page})"
    elif path is not None:
        label += f" :: {path}"
    return label


def _format_hits(query: str, hits: list[DocumentSearchHit]) -> str:
    """Render search hits as a Markdown digest for the LLM."""
    if len(hits) == 0:
        return f"No relevant content found for '{query}'."
    lines = [f"## {len(hits)} result(s) for '{query}'\n"]
    for hit in hits:
        lines.append(f"### {_provenance(hit)}  [score {hit.score:.2f}]\n{hit.content}\n")
    return "\n".join(lines).strip()


# =====================================================================
# Typed wrappers — pin a loader and tailor the description. No pipeline
# of their own: each delegates to DocumentSearchTool.
# =====================================================================


@dataclass(kw_only=True)
class PDFSearchTool(DocumentSearchTool):
    """Semantic search over the indexed PDF document(s)."""

    name: str = "pdf_search"
    description: str = "Search the content of the indexed PDF document(s) for passages relevant to a query."

    def __post_init__(self) -> None:
        if self.loader is None:
            self.loader = PDFLoader()
        super().__post_init__()


@dataclass(kw_only=True)
class DOCXSearchTool(DocumentSearchTool):
    """Semantic search over the indexed Word (.docx) document(s)."""

    name: str = "docx_search"
    description: str = "Search the content of the indexed Word (.docx) document(s) for relevant passages."

    def __post_init__(self) -> None:
        if self.loader is None:
            self.loader = DOCXLoader()
        super().__post_init__()


@dataclass(kw_only=True)
class CSVSearchTool(DocumentSearchTool):
    """Semantic search over the indexed CSV file(s), row by row."""

    name: str = "csv_search"
    description: str = "Search the rows of the indexed CSV file(s) for entries relevant to a query."

    def __post_init__(self) -> None:
        if self.loader is None:
            self.loader = CSVLoader()
        super().__post_init__()


@dataclass(kw_only=True)
class JSONSearchTool(DocumentSearchTool):
    """Semantic search over the indexed JSON file(s)."""

    name: str = "json_search"
    description: str = "Search the contents of the indexed JSON file(s) for entries relevant to a query."

    def __post_init__(self) -> None:
        if self.loader is None:
            self.loader = JSONLoader()
        super().__post_init__()


@dataclass(kw_only=True)
class TXTSearchTool(DocumentSearchTool):
    """Semantic search over the indexed plain-text file(s)."""

    name: str = "txt_search"
    description: str = "Search the content of the indexed text file(s) for passages relevant to a query."

    def __post_init__(self) -> None:
        if self.loader is None:
            self.loader = TextLoader()
        super().__post_init__()


@dataclass(kw_only=True)
class MarkdownSearchTool(DocumentSearchTool):
    """Semantic search over the indexed Markdown document(s)."""

    name: str = "markdown_search"
    description: str = "Search the content of the indexed Markdown document(s) for passages relevant to a query."

    def __post_init__(self) -> None:
        if self.loader is None:
            self.loader = MarkdownLoader()
        super().__post_init__()


@dataclass(kw_only=True)
class DirectorySearchTool(DocumentSearchTool):
    """Semantic search over every supported file under the indexed director(ies)."""

    name: str = "directory_search"
    description: str = "Search the content of all supported files under the indexed director(ies) for a query."

    def __post_init__(self) -> None:
        if self.loader is None:
            self.loader = DirectoryLoader()
        super().__post_init__()


@dataclass(kw_only=True)
class WebsiteSearchTool(DocumentSearchTool):
    """Semantic search over the readable text of the indexed web page(s)."""

    name: str = "website_search"
    description: str = "Search the readable text of the indexed web page(s) for passages relevant to a query."

    def __post_init__(self) -> None:
        if self.loader is None:
            self.loader = WebsiteLoader()
        super().__post_init__()


@dataclass(kw_only=True)
class GithubSearchTool(DocumentSearchTool):
    """Semantic search over the text files of the indexed GitHub repositor(ies)."""

    name: str = "github_search"
    description: str = "Search the text files of the indexed GitHub repositor(ies) for content relevant to a query."

    def __post_init__(self) -> None:
        if self.loader is None:
            self.loader = GithubLoader()
        super().__post_init__()


@dataclass(kw_only=True)
class YoutubeVideoSearchTool(DocumentSearchTool):
    """Semantic search over the transcript(s) of the indexed YouTube video(s)."""

    name: str = "youtube_video_search"
    description: str = "Search the transcript(s) of the indexed YouTube video(s) for passages relevant to a query."

    def __post_init__(self) -> None:
        if self.loader is None:
            self.loader = YoutubeVideoLoader()
        super().__post_init__()


@dataclass(kw_only=True)
class YoutubeChannelSearchTool(DocumentSearchTool):
    """Semantic search over the transcripts of the indexed YouTube channel(s)."""

    name: str = "youtube_channel_search"
    description: str = "Search the transcripts of the indexed YouTube channel's videos for content relevant to a query."

    def __post_init__(self) -> None:
        if self.loader is None:
            self.loader = YoutubeChannelLoader()
        super().__post_init__()

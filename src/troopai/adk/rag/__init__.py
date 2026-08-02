"""Retrieval-augmented generation (RAG) primitives.

A small, reusable layer for turning documents into a semantically searchable
index, built on the framework's existing :class:`Embedder` and
:class:`VectorStore` abstractions:

- :mod:`~troopai.adk.rag.loaders` — format loaders (PDF, DOCX, CSV, JSON, text,
  Markdown, directory, website, GitHub, YouTube) behind a common
  :class:`DocumentLoader` ABC, plus :func:`resolve_loader` dispatch.
- :class:`TextChunker` — recursive, bounded text splitting.
- :class:`DocumentIndex` — chunk → embed → store → search over any
  ``VectorStore`` backend.

The agent-facing ``DocumentSearchTool`` (and its ``PDFSearchTool`` /
``WebsiteSearchTool`` / … wrappers) live in
:mod:`troopai.adk.tools.builtin.document_search_tool` and compose these
primitives. See ``docs/rag/`` and ``examples/rag/``.
"""

from troopai.adk.rag.chunking import DEFAULT_SEPARATORS, TextChunker
from troopai.adk.rag.document import DocumentSearchHit, LoadedDocument
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
    is_url,
    resolve_loader,
)

__all__ = [
    "DEFAULT_SEPARATORS",
    "CSVLoader",
    "DOCXLoader",
    "DirectoryLoader",
    "DocumentIndex",
    "DocumentLoader",
    "DocumentSearchHit",
    "GithubLoader",
    "JSONLoader",
    "LoadedDocument",
    "MarkdownLoader",
    "PDFLoader",
    "TextChunker",
    "TextLoader",
    "WebsiteLoader",
    "YoutubeChannelLoader",
    "YoutubeVideoLoader",
    "is_url",
    "resolve_loader",
]

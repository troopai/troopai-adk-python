"""Dispatch a source (path or URL) to the loader that handles it.

Auto-dispatch powers the generic ``DocumentSearchTool`` and the directory
loader: given a heterogeneous corpus, each source is routed by URL shape
(YouTube / GitHub / website) or file extension (.pdf, .docx, .csv, .json,
.md, .txt) to a default-constructed loader. Named search tools bypass this by
pinning an explicit, pre-configured loader instead.

Loaders are constructed fresh per call; the stdlib loaders are stateless and
cheap, and the optional-dependency loaders (PDF/DOCX) only check for their
package at construction (no import), so resolving a source is side-effect-free
beyond that check.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from troopai.adk.exceptions.exceptions import UnsupportedDocumentSourceError
from troopai.adk.rag.loaders.base import DocumentLoader
from troopai.adk.rag.loaders.directory import DirectoryLoader
from troopai.adk.rag.loaders.docx import DOCXLoader
from troopai.adk.rag.loaders.github import GithubLoader
from troopai.adk.rag.loaders.pdf import PDFLoader
from troopai.adk.rag.loaders.plaintext import MarkdownLoader, TextLoader
from troopai.adk.rag.loaders.structured import CSVLoader, JSONLoader
from troopai.adk.rag.loaders.website import WebsiteLoader
from troopai.adk.rag.loaders.youtube import YoutubeChannelLoader, YoutubeVideoLoader

FILE_EXTENSIONS: dict[str, str] = {
    ".txt": "text",
    ".text": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdx": "markdown",
    ".csv": "csv",
    ".json": "json",
    ".pdf": "pdf",
    ".docx": "docx",
}
"""File extension → loader key. The key abstracts over the concrete class so
:func:`resolve_loader` stays a typed dispatch rather than dynamic lookup."""


def is_url(source: str) -> bool:
    """Return whether ``source`` is an ``http``/``https`` URL.

    Args:
        source: The source string to classify.

    Returns:
        ``True`` if ``source`` parses as an http(s) URL, else ``False``.
    """
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and len(parsed.netloc) > 0


def _resolve_url_loader(source: str) -> DocumentLoader:
    """Route an http(s) URL to the YouTube, GitHub, or website loader."""
    host = urlparse(source).netloc.lower()
    if "youtube.com" in host or "youtu.be" in host:
        lowered = source.lower()
        if any(marker in lowered for marker in ("/channel/", "/@", "/c/", "/user/")):
            return YoutubeChannelLoader()
        return YoutubeVideoLoader()
    if "github.com" in host:
        return GithubLoader()
    return WebsiteLoader()


def _resolve_file_loader(source: str, suffix: str) -> DocumentLoader:
    """Construct the loader for a recognised file ``suffix``."""
    key = FILE_EXTENSIONS.get(suffix)
    if key == "text":
        return TextLoader()
    if key == "markdown":
        return MarkdownLoader()
    if key == "csv":
        return CSVLoader()
    if key == "json":
        return JSONLoader()
    if key == "pdf":
        return PDFLoader()
    if key == "docx":
        return DOCXLoader()
    raise UnsupportedDocumentSourceError(
        source,
        f"No loader for '{suffix or source}'. Supported file types: "
        f"{', '.join(sorted(FILE_EXTENSIONS))}; or an http(s) URL.",
    )


def resolve_loader(source: str) -> DocumentLoader:
    """Return a default-constructed loader for ``source``.

    Args:
        source: A file path, directory path, or http(s) URL.

    Returns:
        The :class:`DocumentLoader` that handles ``source``.

    Raises:
        UnsupportedDocumentSourceError: If no loader matches ``source``.
        ImportError: If the matched loader needs an optional package that
            is not installed.
    """
    if is_url(source):
        return _resolve_url_loader(source)
    path = Path(source)
    if path.is_dir():
        return DirectoryLoader()
    return _resolve_file_loader(source, path.suffix.lower())

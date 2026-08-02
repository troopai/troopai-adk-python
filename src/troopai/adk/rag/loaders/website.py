"""Website loader — fetches a page and extracts its readable text.

Downloads an http(s) URL and strips it to visible text (scripts, styles, and
markup removed) so the embedding sees content rather than HTML. Uses
``requests`` + ``BeautifulSoup``; install the ``rag-web`` extra.

Sources are developer-supplied at tool construction — the LLM never injects a
URL — so this loader is not an attacker-controlled fetch surface. It is still
restricted to http(s) and bounded by a request timeout.
"""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar, override

from troopai.adk.exceptions.exceptions import DocumentLoadError
from troopai.adk.rag.document import LoadedDocument
from troopai.adk.rag.loaders.base import DocumentLoader

logger = logging.getLogger(__name__)

USER_AGENT = "troopai-adk-python-document-search/1.0"
"""Identifying User-Agent sent with page fetches."""


def _fetch_and_extract(source: str, timeout: float) -> tuple[str, str]:
    """Fetch ``source`` and return ``(title, text)`` (runs in a worker thread)."""
    import requests  # pyright: ignore[reportMissingImports]
    from bs4 import BeautifulSoup  # pyright: ignore[reportMissingImports]

    try:
        response = requests.get(source, timeout=timeout, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DocumentLoadError(source, f"Could not fetch {source}: {exc}") from exc
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title is not None and soup.title.string is not None else ""
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if len(line.strip()) > 0]
    return title, "\n".join(lines)


class WebsiteLoader(DocumentLoader):
    """Loads a web page's readable text (via requests + BeautifulSoup).

    Attributes:
        timeout: Per-request timeout in seconds. Defaults to 15.0.
    """

    requires_packages: ClassVar[tuple[str, ...]] = ("requests", "bs4")
    install_extra: ClassVar[str] = "rag-web"

    def __init__(self, *, timeout: float = 15.0) -> None:
        """
        Args:
            timeout: Per-request timeout in seconds.

        Raises:
            ImportError: If requests / BeautifulSoup are not installed.
            ValueError: If ``timeout`` is not positive.
        """
        if timeout <= 0:
            raise ValueError(f"WebsiteLoader.timeout must be > 0, got {timeout}")
        self.timeout = timeout
        self.ensure_dependencies()

    @override
    async def load(self, source: str) -> list[LoadedDocument]:
        """Fetch ``source`` and return its readable text as one document.

        Args:
            source: An http(s) page URL.

        Returns:
            A single-element list, or an empty list if the page has no text.

        Raises:
            DocumentLoadError: If the page cannot be fetched.
        """
        title, text = await asyncio.to_thread(_fetch_and_extract, source, self.timeout)
        if len(text.strip()) == 0:
            logger.debug("WebsiteLoader: %s has no extractable text", source)
            return []
        metadata = {"url": source}
        if len(title) > 0:
            metadata["title"] = title
        return [LoadedDocument(content=text, source=source, metadata=metadata)]

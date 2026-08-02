"""Stdlib loaders for plain-text and Markdown files.

Both read a UTF-8 file off disk (in a worker thread) and return its text as a
single document. Markdown is treated as text: the structure is left intact so
the chunker can split on its blank-line and heading boundaries, and so the raw
syntax remains searchable.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import override

from troopai.adk.exceptions.exceptions import DocumentLoadError
from troopai.adk.rag.document import LoadedDocument
from troopai.adk.rag.loaders.base import DocumentLoader

logger = logging.getLogger(__name__)


def _read_text_file(source: str) -> str:
    """Read a UTF-8 text file, raising :class:`DocumentLoadError` on failure."""
    path = Path(source)
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DocumentLoadError(source, f"File not found: {source}") from exc
    except OSError as exc:
        raise DocumentLoadError(source, f"Could not read {source}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(source, f"{source} is not valid UTF-8 text: {exc}") from exc


class TextLoader(DocumentLoader):
    """Loads a plain-text (``.txt``) file as a single document."""

    @override
    async def load(self, source: str) -> list[LoadedDocument]:
        """Read ``source`` and return its text as one document.

        Args:
            source: Path to a UTF-8 text file.

        Returns:
            A single-element list, or an empty list if the file is blank.

        Raises:
            DocumentLoadError: If the file cannot be read or decoded.
        """
        content = await asyncio.to_thread(_read_text_file, source)
        if len(content.strip()) == 0:
            logger.debug("TextLoader: %s is empty", source)
            return []
        return [LoadedDocument(content=content, source=source)]


class MarkdownLoader(DocumentLoader):
    """Loads a Markdown (``.md`` / ``.markdown`` / ``.mdx``) file as one document.

    Markdown is loaded verbatim — headings, lists, and code fences are kept so
    the chunker can split on their natural boundaries and the syntax stays
    searchable.
    """

    @override
    async def load(self, source: str) -> list[LoadedDocument]:
        """Read ``source`` and return its Markdown text as one document.

        Args:
            source: Path to a Markdown file.

        Returns:
            A single-element list, or an empty list if the file is blank.

        Raises:
            DocumentLoadError: If the file cannot be read or decoded.
        """
        content = await asyncio.to_thread(_read_text_file, source)
        if len(content.strip()) == 0:
            logger.debug("MarkdownLoader: %s is empty", source)
            return []
        return [LoadedDocument(content=content, source=source, metadata={"format": "markdown"})]

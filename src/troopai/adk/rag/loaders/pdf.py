"""PDF loader backed by PyMuPDF.

Extracts text one document per page so retrieval can cite a page number and so
a single oversized PDF does not collapse into one opaque span. PyMuPDF is an
optional dependency; install the ``rag-pdf`` extra.
"""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar, cast, override

from troopai.adk.exceptions.exceptions import DocumentLoadError
from troopai.adk.rag.document import LoadedDocument
from troopai.adk.rag.loaders.base import DocumentLoader

logger = logging.getLogger(__name__)


def _extract_pages(source: str) -> list[LoadedDocument]:
    """Extract per-page text from a PDF using PyMuPDF (runs in a worker thread)."""
    import pymupdf  # pyright: ignore[reportMissingImports]  # optional dependency, guarded at construction

    documents: list[LoadedDocument] = []
    try:
        with pymupdf.open(source) as doc:
            for number in range(doc.page_count):
                # get_text() with its default "text" option returns a str; the stub's
                # union return (str | list | dict) covers the other extraction modes.
                text = cast(str, doc.load_page(number).get_text())
                if len(text.strip()) == 0:
                    continue
                documents.append(LoadedDocument(content=text, source=source, metadata={"page": str(number + 1)}))
    except FileNotFoundError as exc:
        raise DocumentLoadError(source, f"PDF not found: {source}") from exc
    except (RuntimeError, ValueError) as exc:
        raise DocumentLoadError(source, f"Could not parse PDF {source}: {exc}") from exc
    return documents


class PDFLoader(DocumentLoader):
    """Loads a PDF file as one document per page (via PyMuPDF)."""

    requires_packages: ClassVar[tuple[str, ...]] = ("pymupdf",)
    install_extra: ClassVar[str] = "rag-pdf"

    def __init__(self) -> None:
        """Verify PyMuPDF is importable, failing fast if the extra is missing.

        Raises:
            ImportError: If PyMuPDF is not installed.
        """
        self.ensure_dependencies()

    @override
    async def load(self, source: str) -> list[LoadedDocument]:
        """Extract text from ``source``, one document per page.

        Args:
            source: Path to a PDF file.

        Returns:
            One document per page with extractable text (empty pages skipped).

        Raises:
            DocumentLoadError: If the PDF cannot be opened or parsed.
        """
        documents = await asyncio.to_thread(_extract_pages, source)
        logger.debug("PDFLoader: %s -> %d page document(s)", source, len(documents))
        return documents

"""Directory loader — fans every recognised file out to its format loader.

Walks a directory, routes each file with a supported extension through
:func:`resolve_loader`, and aggregates the results. Files with an unsupported
extension are skipped silently; files whose loader needs a missing optional
package, or that fail to parse, are skipped with a logged warning so one bad
file never aborts indexing the rest of the corpus.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import override

from troopai.adk.exceptions.exceptions import (
    DocumentLoadError,
    UnsupportedDocumentSourceError,
)
from troopai.adk.rag.document import LoadedDocument
from troopai.adk.rag.loaders.base import DocumentLoader

logger = logging.getLogger(__name__)


class DirectoryLoader(DocumentLoader):
    """Loads every supported file under a directory.

    Attributes:
        recursive: Whether to descend into subdirectories. Defaults to
            ``True``.
    """

    def __init__(self, *, recursive: bool = True) -> None:
        """
        Args:
            recursive: Whether to descend into subdirectories.
        """
        self.recursive = recursive

    @override
    async def load(self, source: str) -> list[LoadedDocument]:
        """Load every supported file under directory ``source``.

        Args:
            source: Path to a directory.

        Returns:
            The concatenated documents from all loadable files, in sorted
            path order.

        Raises:
            DocumentLoadError: If ``source`` is not an existing directory.
        """
        # Imported here (not at module scope) to break the directory <-> registry cycle.
        from troopai.adk.rag.loaders.registry import FILE_EXTENSIONS, resolve_loader

        root = Path(source)
        if not root.is_dir():
            raise DocumentLoadError(source, f"Not a directory: {source}")
        paths = sorted(root.rglob("*") if self.recursive else root.glob("*"))
        documents: list[LoadedDocument] = []
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in FILE_EXTENSIONS:
                continue
            entry = str(path)
            try:
                loader = resolve_loader(entry)
                documents.extend(await loader.load(entry))
            except (DocumentLoadError, UnsupportedDocumentSourceError, ImportError) as exc:
                logger.warning("DirectoryLoader: skipping %s: %s", entry, exc)
        logger.debug("DirectoryLoader: %s -> %d document(s)", source, len(documents))
        return documents

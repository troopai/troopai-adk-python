"""Stdlib loaders for structured CSV and JSON files.

Structured formats are flattened into one document per record so each row /
top-level element becomes an independently retrievable chunk, rather than
embedding the whole file as one opaque blob. CSV rows render as ``header:
value`` lines (carrying column names into the embedding); JSON renders as
indented text.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, override

from troopai.adk.exceptions.exceptions import DocumentLoadError
from troopai.adk.rag.document import LoadedDocument
from troopai.adk.rag.loaders.base import DocumentLoader

logger = logging.getLogger(__name__)


def _read_raw(source: str) -> str:
    """Read a UTF-8 file, raising :class:`DocumentLoadError` on failure."""
    try:
        return Path(source).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DocumentLoadError(source, f"File not found: {source}") from exc
    except OSError as exc:
        raise DocumentLoadError(source, f"Could not read {source}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(source, f"{source} is not valid UTF-8 text: {exc}") from exc


def _parse_csv_rows(source: str, raw: str) -> list[LoadedDocument]:
    """Render each CSV row as a ``header: value`` document."""
    reader = csv.DictReader(io.StringIO(raw))
    documents: list[LoadedDocument] = []
    for index, row in enumerate(reader):
        rendered = "\n".join(f"{key}: {value}" for key, value in row.items() if value is not None)
        if len(rendered.strip()) == 0:
            continue
        documents.append(LoadedDocument(content=rendered, source=source, metadata={"row": str(index)}))
    return documents


class CSVLoader(DocumentLoader):
    """Loads a ``.csv`` file as one document per data row.

    Each row renders as ``header: value`` lines using the file's header row
    for column names, so the column semantics are embedded alongside the
    values.
    """

    @override
    async def load(self, source: str) -> list[LoadedDocument]:
        """Read ``source`` and return one document per CSV row.

        Args:
            source: Path to a CSV file with a header row.

        Returns:
            One document per non-empty data row.

        Raises:
            DocumentLoadError: If the file cannot be read or parsed.
        """
        raw = await asyncio.to_thread(_read_raw, source)
        try:
            documents = await asyncio.to_thread(_parse_csv_rows, source, raw)
        except csv.Error as exc:
            raise DocumentLoadError(source, f"Malformed CSV {source}: {exc}") from exc
        logger.debug("CSVLoader: %s -> %d row document(s)", source, len(documents))
        return documents


def _render_json(source: str, data: Any) -> list[LoadedDocument]:
    """Render a top-level JSON list element-wise, else as one document."""
    if isinstance(data, list):
        documents: list[LoadedDocument] = []
        for index, element in enumerate(data):
            rendered = json.dumps(element, indent=2, ensure_ascii=False)
            documents.append(LoadedDocument(content=rendered, source=source, metadata={"index": str(index)}))
        return documents
    rendered = json.dumps(data, indent=2, ensure_ascii=False)
    return [LoadedDocument(content=rendered, source=source)]


class JSONLoader(DocumentLoader):
    """Loads a ``.json`` file as searchable text.

    A top-level JSON array becomes one document per element (each element is
    independently retrievable); any other top-level value becomes a single
    document. Values are re-serialized with indentation for readability.
    """

    @override
    async def load(self, source: str) -> list[LoadedDocument]:
        """Read ``source`` and return its JSON content as document(s).

        Args:
            source: Path to a JSON file.

        Returns:
            One document per top-level array element, or a single document
            for any other top-level value.

        Raises:
            DocumentLoadError: If the file cannot be read or is invalid JSON.
        """
        raw = await asyncio.to_thread(_read_raw, source)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DocumentLoadError(source, f"Invalid JSON in {source}: {exc}") from exc
        documents = _render_json(source, data)
        logger.debug("JSONLoader: %s -> %d document(s)", source, len(documents))
        return documents

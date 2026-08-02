"""Write generated deploy artifacts to disk.

The render functions in :mod:`troopai.adk.deploy.templates` (and each
target's ``generate``) return ``{relative_path: content}`` maps;
:func:`write_artifacts` materializes them, never clobbering an existing
file unless ``force`` is set.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def write_artifacts(files: dict[str, str], dest: Path, *, force: bool = False) -> tuple[list[Path], list[Path]]:
    """Write *files* under *dest*, skipping existing paths unless *force*.

    Args:
        files: Map of ``relative_path -> file content``.
        dest: Directory the relative paths are written under.
        force: Overwrite existing files when ``True``; otherwise skip them.

    Returns:
        A ``(written, skipped)`` tuple of absolute paths.
    """
    written: list[Path] = []
    skipped: list[Path] = []
    for relative_path, content in files.items():
        target = dest / relative_path
        if target.exists() and not force:
            skipped.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    logger.debug("wrote %d artifact(s), skipped %d under %s", len(written), len(skipped), dest)
    return written, skipped

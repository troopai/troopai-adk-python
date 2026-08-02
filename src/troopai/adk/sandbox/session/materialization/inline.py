"""Materializers for inline ``File`` and synthetic ``Dir`` entries.

These carry their content in the manifest itself, so they need no
host-filesystem or network access and have no path-grant or symlink
concerns beyond the workspace-relative key check. ``Dir`` children are
flattened by ``Manifest.iter_entries`` into their own ``(key, entry)``
pairs, so this materializer only creates the directory node — it does
NOT recurse (the orchestrator drives every child as a sibling entry).
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import TYPE_CHECKING

from troopai.adk.sandbox.session.materialization.metadata import apply_entry_metadata
from troopai.adk.sandbox.session.materialization.paths import normalize_workspace_key
from troopai.adk.types.sandbox.entries import MaterializedFile

if TYPE_CHECKING:
    from troopai.adk.sandbox.clients.session import BaseSandboxSession
    from troopai.adk.types.sandbox.entries import Dir, File

logger = logging.getLogger(__name__)

__all__ = ["materialize_dir", "materialize_file"]


async def materialize_file(session: BaseSandboxSession, key: str, entry: File) -> MaterializedFile:
    """Write ``entry.content`` to ``key`` and apply its metadata.

    Args:
        session: Live backend session.
        key: Workspace-relative destination path. Re-normalized
            defensively here too — these helpers are public
            (``__all__``) and independently callable, and the
            materializer is a filesystem-security boundary, so it
            never assumes the caller routed through ``Manifest``.
        entry: The ``File`` entry carrying inline bytes.
    """
    safe = normalize_workspace_key(key)
    content = entry.content
    await session.write(safe, BytesIO(content))
    await apply_entry_metadata(session, safe, entry)
    logger.debug("materialized File at %s (%d bytes)", safe, len(content))
    return MaterializedFile(
        path=safe,
        size_bytes=len(content),
        permissions=entry.permissions,
        is_directory=False,
    )


async def materialize_dir(session: BaseSandboxSession, key: str, entry: Dir) -> MaterializedFile:
    """Create the directory at ``key`` and apply its metadata.

    Children are materialized by the orchestrator as their own entries
    (``Manifest.iter_entries`` flattens the tree), so this does not
    recurse into ``entry.children``.

    Args:
        session: Live backend session.
        key: Workspace-relative destination path. Re-normalized
            defensively here too — these helpers are public
            (``__all__``) and independently callable, and the
            materializer is a filesystem-security boundary, so it
            never assumes the caller routed through ``Manifest``.
        entry: The ``Dir`` entry.
    """
    safe = normalize_workspace_key(key)
    await session.mkdir(safe, parents=True)
    await apply_entry_metadata(session, safe, entry)
    logger.debug("materialized Dir at %s", safe)
    return MaterializedFile(
        path=safe,
        size_bytes=0,
        permissions=entry.permissions,
        is_directory=True,
    )

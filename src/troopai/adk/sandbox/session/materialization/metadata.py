"""Apply POSIX ownership / mode to a materialized entry via the session.

OpenAI's sandbox attaches an ``_apply_metadata`` method to each entry
class; this project's manifest entries are Layer-1 behaviorless data,
so metadata application lives here and drives the typed session
``run`` primitive instead.

``chmod`` is always applied — the manifest ``Permissions`` default is
the cost-conservative ``rw-r--r--`` and the developer opted into
whatever bits the entry carries. ``chgrp`` is applied only when the
entry declares an owning user/group. A non-zero exit from either
command raises ``ExecNonZeroError`` (its docstring designates manifest
materialization as a sanctioned raise site) rather than silently
materializing a file with the wrong ownership.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from troopai.adk.exceptions.exceptions import ExecNonZeroError

if TYPE_CHECKING:
    from troopai.adk.sandbox.clients.session import BaseSandboxSession
    from troopai.adk.types.sandbox.entries import BaseEntry

logger = logging.getLogger(__name__)

__all__ = ["apply_entry_metadata"]

_STDERR_CAP = 300
"""Truncate captured stderr so a huge error does not bloat the exception."""


def _dir_traverse_mode(mode: int) -> int:
    """Add execute (traverse) wherever a directory grants read.

    The ``Permissions`` triplet is authored file-style; for a
    directory entry the read bit implies traverse — standard POSIX /
    ``chmod a+X`` semantics, and exactly the interpretation
    ``Permissions``'s ``directory`` field documents. Without this the
    cost-conservative default ``rw-r--r--`` (0o644) would leave a
    just-created directory non-traversable, so the materializer could
    not place the declared children into it.
    """
    result = mode
    for shift in (6, 3, 0):
        if result & (0o4 << shift):
            result |= 0o1 << shift
    return result


async def apply_entry_metadata(session: BaseSandboxSession, key: str, entry: BaseEntry) -> None:
    """Apply ``entry.permissions`` (and group, if any) to ``key``.

    Args:
        session: Live backend session to drive ``chmod`` / ``chgrp`` on.
        key: Workspace-relative path already materialized.
        entry: The manifest entry whose permissions/group to apply.

    Note:
        ``entry.group`` may be a ``User`` or a ``Group``; both are
        applied via ``chgrp <name>``. A ``User`` is applied as its
        same-name primary group — the upstream-faithful contract,
        valid on typically-provisioned Linux sandbox images where
        ``useradd`` creates a matching group.

    Raises:
        ExecNonZeroError: ``chmod`` or ``chgrp`` exited non-zero.
    """
    raw_mode = entry.permissions.to_mode()
    effective = _dir_traverse_mode(raw_mode) if entry.is_dir() else raw_mode
    mode = f"{effective:o}"
    chmod = await session.run("chmod", mode, key, shell=False)
    if chmod.exit_code != 0:
        detail = chmod.stderr.decode("utf-8", errors="replace")[:_STDERR_CAP]
        raise ExecNonZeroError(f"chmod {mode} {key!r} failed (exit={chmod.exit_code}): {detail}")
    group = entry.group
    if group is None:
        return
    chgrp = await session.run("chgrp", group.name, key, shell=False)
    if chgrp.exit_code != 0:
        detail = chgrp.stderr.decode("utf-8", errors="replace")[:_STDERR_CAP]
        raise ExecNonZeroError(f"chgrp {group.name!r} {key!r} failed (exit={chgrp.exit_code}): {detail}")
    logger.debug("applied metadata mode=%s group=%s to %s", mode, group.name, key)

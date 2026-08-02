"""Build shell-safe ``--exclude=`` argument lists for ``tar`` invocations.

Backends that capture workspace snapshots via shell-driven
``tar cf - .`` need to skip paths the framework knows shouldn't
be in the durable snapshot (e.g. the snapshot fingerprint cache,
the memory subsystem's transient JSONL). Generating the
``--exclude`` flags by hand risks shell-injection through
attacker-controlled relative paths; this helper quotes the
output via ``shlex.quote`` and emits both ``<rel>`` and
``./<rel>`` forms because backends differ on which one tar
recognises depending on how the workspace root was named.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable
from pathlib import Path

__all__ = ["shell_tar_exclude_args"]


def shell_tar_exclude_args(skip_relpaths: Iterable[Path]) -> list[str]:
    """Translate workspace-relative paths into ``--exclude=`` tar arguments.

    Args:
        skip_relpaths: Paths inside the workspace to exclude from
            the tar capture. Empty / root paths are silently
            dropped because they're no-ops at the tar layer.

    Returns:
        Ordered list of ``--exclude=...`` flags ready for a
        ``shlex.join``-friendly tar command. Each input path
        produces TWO flags (``<rel>`` and ``./<rel>``) — tar matches
        members against the literal path the archive emits, which
        depends on whether the capture was rooted at ``.`` or at
        a named subdirectory.
    """
    excludes: list[str] = []
    for rel in sorted(skip_relpaths, key=lambda p: p.as_posix()):
        rel_posix = rel.as_posix().lstrip("/")
        if len(rel_posix) == 0 or rel_posix in {".", "/"}:
            continue
        excludes.append(f"--exclude={shlex.quote(rel_posix)}")
        excludes.append(f"--exclude={shlex.quote(f'./{rel_posix}')}")
    return excludes

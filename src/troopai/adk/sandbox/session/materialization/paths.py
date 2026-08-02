"""Workspace-path helpers for manifest materialization.

Entry keys are already validated workspace-relative by ``Manifest``;
these helpers re-assert the invariant defensively (the materializer is
a filesystem-security boundary, so it must not trust that every caller
went through ``Manifest``) and detect ancestor/descendant overlap so
the orchestrator can serialize entries that would otherwise race on
the same subtree.
"""

from __future__ import annotations

from pathlib import PurePosixPath

__all__ = ["normalize_workspace_key", "paths_overlap"]


def normalize_workspace_key(key: str) -> str:
    """Return the POSIX-normalized workspace-relative form of ``key``.

    Defense-in-depth: ``Manifest`` already rejects absolute / ``..``
    keys, but the materializer is re-validated here so a future caller
    that constructs entries without going through ``Manifest`` cannot
    escape the workspace root.

    Args:
        key: Candidate workspace-relative key.

    Raises:
        ValueError: ``key`` is empty, absolute, or contains ``..``.
    """
    if len(key) == 0:
        raise ValueError("workspace key must be non-empty")
    pure = PurePosixPath(key)
    if pure.is_absolute():
        raise ValueError(f"workspace key must be relative, got {key!r}")
    if ".." in pure.parts:
        raise ValueError(f"workspace key must not contain '..': {key!r}")
    return pure.as_posix()


def paths_overlap(left: str, right: str) -> bool:
    """Return True iff ``left`` and ``right`` collide on the same subtree.

    Two keys collide when they are equal or one is an ancestor of the
    other — materializing both concurrently could race (e.g. one writes
    ``a/b`` while the other ``mkdir``\\ s ``a``). Such pairs are flushed
    to run sequentially.

    Args:
        left: First workspace-relative key.
        right: Second workspace-relative key.
    """
    pure_left = PurePosixPath(left)
    pure_right = PurePosixPath(right)
    return pure_left == pure_right or pure_left in pure_right.parents or pure_right in pure_left.parents

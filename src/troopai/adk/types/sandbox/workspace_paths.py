"""Workspace path policy — what absolute paths the sandbox may touch.

A manifest's ``extra_path_grants`` is a trust boundary: only paths
listed here are accessible via local materialization, ADK file APIs,
or shell execution where the backend can enforce filesystem policy.

Treat manifests containing grants as TRUSTED CONFIGURATION. Never
load grants from model output or other untrusted payloads.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = ["SandboxPathGrant", "WorkspacePathPolicy"]


_FILESYSTEM_ROOTS: frozenset[str] = frozenset({"/", "\\"})


def _is_filesystem_root(path: str) -> bool:
    """True iff ``path`` is a filesystem root (``"/"`` or single drive letter)."""
    if path in _FILESYSTEM_ROOTS:
        return True
    # Windows drive root, e.g. "C:\\" or "C:/".
    return os.name == "nt" and len(path) == 3 and path[1] == ":" and path[2] in ("\\", "/")


class SandboxPathGrant(BaseModel):
    """Grant access to a specific absolute path outside the workspace.

    Grants apply to local source materialization (``LocalFile`` /
    ``LocalDir`` sources outside the host process working directory),
    ADK file APIs, and shell execution where the backend enforces
    filesystem policy.

    Attributes:
        path: Absolute path being granted (no filesystem roots).
        read_only: When True, only read access is granted.
        description: Optional human-readable rationale for audit.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    path: str
    """Absolute path being granted (filesystem roots rejected)."""

    read_only: bool = False
    """When True, only read access is granted."""

    description: str | None = None
    """Optional human-readable rationale for audit."""

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        if len(value) == 0:
            raise ValueError("SandboxPathGrant.path must be non-empty")
        p = Path(value)
        if not p.is_absolute():
            raise ValueError(f"SandboxPathGrant.path must be absolute, got {value!r}")
        if _is_filesystem_root(value):
            raise ValueError(f"SandboxPathGrant.path may not be a filesystem root, got {value!r}")
        return value


class WorkspacePathPolicy(BaseModel):
    """Aggregate path-policy view derived from a manifest.

    Constructed once per session; the backend consults it before
    every file or shell operation that could touch a host path.

    Attributes:
        root: The workspace root (manifest ``root``).
        grants: All path grants declared on the manifest.
    """

    model_config: ClassVar = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    root: str
    """The workspace root."""

    grants: tuple[SandboxPathGrant, ...] = ()
    """All path grants declared on the manifest."""

    def _safe_resolve(self, path: str | Path) -> Path | None:
        """Resolve ``path`` lexically (no filesystem touch, no symlink follow).

        We deliberately do NOT call ``Path.resolve()`` which would (a)
        perform filesystem I/O — racy and capable of raising opaque
        ``OSError`` from inside a trust-boundary check — and (b) follow
        symlinks, breaking grants that target a host directory whose
        contents could be replaced by a malicious symlink at runtime.
        Pure lexical normalization via ``os.path.normpath`` is the
        safe choice for policy evaluation.
        """
        import os

        raw = str(path)
        if len(raw) == 0:
            return None
        normalized = os.path.normpath(raw)
        return Path(normalized)

    def is_under_workspace(self, path: str | Path) -> bool:
        """True iff ``path`` is the workspace root or a lexical descendant.

        Lexical-only check (no symlink follow). A symlink-based escape
        attempt is rejected by this method; the backend is still
        responsible for refusing to traverse symlinks at execution time.
        """
        p = self._safe_resolve(path)
        if p is None:
            return False
        root = self._safe_resolve(self.root)
        if root is None:
            return False
        try:
            p.relative_to(root)
            return True
        except ValueError:
            return False

    def matching_grant(self, path: str | Path) -> SandboxPathGrant | None:
        """Return the most-specific grant covering ``path`` (or ``None``).

        "Most specific" means the grant whose path is the deepest
        prefix of ``path`` after lexical normalization. With grants
        ``/tmp`` and ``/tmp/sub`` and target ``/tmp/sub/x``, the
        ``/tmp/sub`` grant wins. ``None`` means "no grant covers this
        path — deny" (NOT "policy couldn't decide"). Callers MUST
        treat ``None`` as a deny verdict.
        """
        target = self._safe_resolve(path)
        if target is None:
            return None
        best: SandboxPathGrant | None = None
        best_grant_depth = -1
        for grant in self.grants:
            grant_root = self._safe_resolve(grant.path)
            if grant_root is None:
                continue
            try:
                target.relative_to(grant_root)
            except ValueError:
                continue
            grant_depth = len(grant_root.parts)
            if grant_depth > best_grant_depth:
                best = grant
                best_grant_depth = grant_depth
        return best

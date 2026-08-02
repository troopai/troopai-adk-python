"""Editor protocol + typed operations for the apply_patch surface.

The framework defines:

* ``ApplyPatchOperation`` — one create/update/delete operation that
  the model emits or a caller assembles.
* ``ApplyPatchResult`` — the operation's outcome.
* ``ApplyPatchEditor`` — the runtime-checkable Protocol that any
  editor backend MUST satisfy (sandbox sessions implement it via
  ``WorkspaceEditor``).

We DELIBERATELY omit the ``ctx_wrapper`` field present in the
upstream OpenAI design: our framework passes execution context
through ``ToolContext`` already, and stapling a second wrapper
onto the operation dataclass duplicates that channel.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

__all__ = [
    "ApplyPatchEditor",
    "ApplyPatchOperation",
    "ApplyPatchOperationType",
    "ApplyPatchResult",
]

ApplyPatchOperationType = Literal["create_file", "update_file", "delete_file"]
"""The three operation kinds a V4A patch envelope can carry."""


@dataclass(slots=True)
class ApplyPatchOperation:
    """One create/update/delete operation against the workspace.

    Attributes:
        type: ``"create_file"`` / ``"update_file"`` / ``"delete_file"``.
        path: Workspace-relative path the operation targets.
        diff: V4A diff body. Required for create/update; MUST be
            ``None`` for delete.
        move_to: Optional workspace-relative destination — when set
            on an ``update_file`` operation, the edited contents
            land at ``move_to`` and the original ``path`` is
            removed (rename + edit in one shot).
    """

    type: ApplyPatchOperationType
    """The operation kind."""

    path: str
    """Workspace-relative path the operation targets."""

    diff: str | None = None
    """V4A diff body (None for delete operations)."""

    move_to: str | None = None
    """Optional rename destination on update operations."""


@dataclass(slots=True)
class ApplyPatchResult:
    """Operation outcome surfaced to the caller / tool result.

    Attributes:
        status: ``"completed"`` / ``"failed"`` / ``None`` (unknown).
        output: Human-readable summary line (e.g. ``"Updated src/x.py"``).
    """

    status: Literal["completed", "failed"] | None = None
    """Whether the operation completed successfully."""

    output: str | None = None
    """Optional human-readable summary."""

    metadata: dict[str, object] = field(default_factory=dict)
    """Free-form metadata bag for editor-specific extras (e.g. fuzz score)."""


@runtime_checkable
class ApplyPatchEditor(Protocol):
    """Editor that materializes ``ApplyPatchOperation`` against some backend.

    Concrete editors include ``WorkspaceEditor`` (sandbox session)
    and tests' in-memory fakes. Each method MAY return either an
    ``ApplyPatchResult`` directly or a coroutine producing one;
    callers awaitable-detect via ``inspect.isawaitable``.
    """

    def create_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult | Awaitable[ApplyPatchResult]:
        """Create ``operation.path`` from the diff (``+``-only lines)."""
        ...

    def update_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult | Awaitable[ApplyPatchResult]:
        """Rewrite ``operation.path`` per the diff; optionally rename to ``move_to``."""
        ...

    def delete_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult | Awaitable[ApplyPatchResult]:
        """Remove ``operation.path`` from the workspace."""
        ...

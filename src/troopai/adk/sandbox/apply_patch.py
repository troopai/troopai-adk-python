"""``WorkspaceEditor`` — applies V4A patches against a live sandbox session.

Wraps ``BaseSandboxSession`` so the apply_patch tool can call a
single ``await editor.apply_patch(...)`` regardless of which
concrete backend is underneath. The editor reads the current file
contents, runs the diff engine, and writes back through the
session's typed write/rm primitives.

The ``PatchFormat`` indirection lets us add v5a / other dialects
later without touching the editor; pass ``patch_format="v4a"``
(the default) or a custom callable implementing the protocol.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from troopai.adk.exceptions.exceptions import (
    ApplyPatchError,
    WorkspaceReadNotFoundError,
)
from troopai.adk.sandbox.apply_diff import ApplyDiffMode, apply_diff
from troopai.adk.sandbox.editor import (
    ApplyPatchOperation,
    ApplyPatchResult,
)

if TYPE_CHECKING:
    from troopai.adk.sandbox.clients.session import BaseSandboxSession
    from troopai.adk.types.sandbox.permissions import User

logger = logging.getLogger(__name__)

__all__ = ["PatchFormat", "V4AFormat", "WorkspaceEditor"]


@runtime_checkable
class PatchFormat(Protocol):
    """Pluggable diff dialect — implement to support v5a / other formats."""

    @staticmethod
    def apply_diff(input_text: str, diff: str, mode: ApplyDiffMode = "default") -> str:
        """Return the result of applying ``diff`` to ``input_text``."""
        ...


class V4AFormat:
    """OpenAI's V4A format — the default dialect."""

    @staticmethod
    def apply_diff(input_text: str, diff: str, mode: ApplyDiffMode = "default") -> str:
        return apply_diff(input_text, diff, mode=mode)


class WorkspaceEditor:
    """Apply V4A patches against a live ``BaseSandboxSession``.

    Construct one per session (or per session+user pair) and call
    ``await apply_patch(ops)`` for each model-emitted patch.
    """

    def __init__(
        self,
        session: BaseSandboxSession,
        *,
        user: str | User | None = None,
    ) -> None:
        self._session = session
        self._user = user

    async def apply_patch(
        self,
        operations: ApplyPatchOperation | dict[str, object] | list[ApplyPatchOperation | dict[str, object]],
        *,
        patch_format: PatchFormat | Literal["v4a"] = "v4a",
    ) -> str:
        """Apply one or more operations; return ``"Done!"`` on success."""
        format_impl = resolve_patch_format(patch_format)
        for operation in coerce_operations(operations):
            await self.apply_operation(operation, patch_format=format_impl)
        return "Done!"

    async def apply_operation(
        self,
        operation: ApplyPatchOperation,
        *,
        patch_format: PatchFormat | Literal["v4a"] = "v4a",
    ) -> ApplyPatchResult:
        """Apply a single operation; return its ``ApplyPatchResult``."""
        format_impl = resolve_patch_format(patch_format)
        relative_path = self._validate_path(operation.path)
        destination = self._session.normalize_path(relative_path)
        display_path = relative_path.as_posix()

        if operation.type == "delete_file":
            await self._ensure_exists(destination, display_path=display_path)
            await self._session.rm(destination, user=self._user)
            return ApplyPatchResult(status="completed", output=f"Deleted {display_path}")

        if operation.diff is None:
            raise ApplyPatchError(f"Missing diff for operation type {operation.type} on path {operation.path}")

        # ``diff`` is narrowed at the helper-signature boundary: the
        # caller already verified non-None at line 104, but explicit
        # parameter passing keeps the contract enforced by the type
        # system (R5: no assert as safety check; strippable under -O).
        diff = operation.diff

        if operation.type == "update_file":
            return await self._apply_update(
                operation,
                diff=diff,
                destination=destination,
                relative_path=relative_path,
                display_path=display_path,
                format_impl=format_impl,
            )

        if operation.type == "create_file":
            return await self._apply_create(
                operation,
                diff=diff,
                destination=destination,
                display_path=display_path,
                format_impl=format_impl,
            )

        raise ApplyPatchError(f"Unknown operation type: {operation.type}")

    async def _apply_update(
        self,
        operation: ApplyPatchOperation,
        *,
        diff: str,
        destination: Path,
        relative_path: Path,
        display_path: str,
        format_impl: PatchFormat,
    ) -> ApplyPatchResult:
        original_text = await self._read_text(destination, op_path=operation.path)
        try:
            updated_text = format_impl.apply_diff(original_text, diff, mode="default")
        except ValueError as exc:
            raise ApplyPatchError(f"apply_diff failed on {operation.path}: {exc}") from exc

        if operation.move_to is None:
            await self._write_text(destination, updated_text)
            return ApplyPatchResult(status="completed", output=f"Updated {display_path}")

        moved_relative_path = self._validate_path(operation.move_to)
        moved_destination = self._session.normalize_path(moved_relative_path)
        await self._write_text(moved_destination, updated_text)
        moved_display_path = moved_relative_path.as_posix()
        if moved_destination != destination:
            try:
                await self._session.rm(destination, user=self._user)
            except Exception as exc:
                # Defense-in-depth: write succeeded at the new path but the
                # source unlink failed. Surface BOTH paths so the operator
                # can clean up the orphan; never leave the caller thinking
                # nothing happened.
                raise ApplyPatchError(
                    f"apply_patch torn write: wrote {moved_display_path} but failed to remove "
                    f"original at {display_path}: {exc}. Workspace contains two copies of the file."
                ) from exc
        _ = relative_path  # rationale: surfaced through display_path below
        return ApplyPatchResult(
            status="completed",
            output=f"Updated {display_path}\nMoved {display_path} to {moved_display_path}",
        )

    async def _apply_create(
        self,
        operation: ApplyPatchOperation,
        *,
        diff: str,
        destination: Path,
        display_path: str,
        format_impl: PatchFormat,
    ) -> ApplyPatchResult:
        try:
            created_text = format_impl.apply_diff("", diff, mode="create")
        except ValueError as exc:
            raise ApplyPatchError(f"apply_diff failed on {operation.path}: {exc}") from exc
        await self._write_text(destination, created_text)
        return ApplyPatchResult(status="completed", output=f"Created {display_path}")

    def _validate_path(self, path: str | Path) -> Path:
        if isinstance(path, str):
            if len(path.strip()) == 0:
                raise ApplyPatchError("apply_patch: empty path is not allowed")
            normalized_path = Path(path)
        else:
            normalized_path = path

        # Best-effort: reject absolute paths + parent traversal here so the
        # backend never gets a chance to escape its root. Per-backend path
        # normalization happens later via ``session.normalize_path``.
        if normalized_path.is_absolute():
            raise ApplyPatchError(f"apply_patch: absolute path not allowed: {normalized_path}")
        if any(part == ".." for part in normalized_path.parts):
            raise ApplyPatchError(f"apply_patch: parent traversal not allowed: {normalized_path}")
        return normalized_path

    async def _ensure_exists(self, destination: Path, *, display_path: str) -> None:
        try:
            handle = await self._session.read(destination, user=self._user)
        except (FileNotFoundError, WorkspaceReadNotFoundError) as exc:
            raise ApplyPatchError(f"apply_patch: file not found: {display_path}") from exc
        else:
            handle.close()

    async def _read_text(self, destination: Path, *, op_path: str) -> str:
        try:
            handle = await self._session.read(destination, user=self._user)
        except (FileNotFoundError, WorkspaceReadNotFoundError) as exc:
            raise ApplyPatchError(f"apply_patch: file not found: {op_path}") from exc

        try:
            payload = handle.read()
        finally:
            handle.close()

        if isinstance(payload, str):
            return payload
        if isinstance(payload, bytes | bytearray):
            try:
                return bytes(payload).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ApplyPatchError(f"apply_patch: non-UTF-8 content at {op_path}") from exc
        raise ApplyPatchError(f"apply_patch read() returned non-text content: {type(payload).__name__}")

    async def _write_text(self, destination: Path, text: str) -> None:
        await self._session.mkdir(destination.parent, parents=True, user=self._user)
        await self._session.write(
            destination,
            io.BytesIO(text.encode("utf-8")),
            user=self._user,
        )


def coerce_operations(
    operations: ApplyPatchOperation | dict[str, object] | list[ApplyPatchOperation | dict[str, object]],
) -> list[ApplyPatchOperation]:
    if isinstance(operations, ApplyPatchOperation):
        return [operations]
    if isinstance(operations, dict):
        return [coerce_operation_mapping(operations)]
    if isinstance(operations, list):
        coerced: list[ApplyPatchOperation] = []
        for operation in operations:
            if isinstance(operation, ApplyPatchOperation):
                coerced.append(operation)
            elif isinstance(operation, dict):
                coerced.append(coerce_operation_mapping(operation))
            else:
                raise ApplyPatchError(f"Invalid apply_patch operation type: {type(operation).__name__}")
        return coerced
    raise ApplyPatchError(f"Invalid apply_patch operations payload: {type(operations).__name__}")


def coerce_operation_mapping(operation: dict[str, object]) -> ApplyPatchOperation:
    raw_type = operation.get("type")
    raw_path = operation.get("path")
    raw_diff = operation.get("diff")
    raw_move_to = operation.get("move_to")

    if raw_type not in {"create_file", "update_file", "delete_file"}:
        raise ApplyPatchError(f"Invalid apply_patch operation type: {raw_type!r}")
    if not isinstance(raw_path, str):
        raise ApplyPatchError(f"Invalid apply_patch path type: {type(raw_path).__name__}")
    if raw_diff is not None and not isinstance(raw_diff, str):
        raise ApplyPatchError(f"Invalid apply_patch diff type: {type(raw_diff).__name__}")
    if raw_move_to is not None and not isinstance(raw_move_to, str):
        raise ApplyPatchError(f"Invalid apply_patch move_to type: {type(raw_move_to).__name__}")
    return ApplyPatchOperation(
        type=raw_type,
        path=raw_path,
        diff=raw_diff,
        move_to=raw_move_to,
    )


def resolve_patch_format(
    patch_format: PatchFormat | Literal["v4a"],
) -> PatchFormat:
    if patch_format == "v4a":
        return V4AFormat
    if isinstance(patch_format, PatchFormat):
        return patch_format
    raise ApplyPatchError(f"Unsupported patch format: {patch_format!r}")

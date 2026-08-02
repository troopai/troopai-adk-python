"""Tests for ``troopai.adk.sandbox.apply_patch`` — WorkspaceEditor.

These cover engine semantics with an in-memory fake session — the
real backends are exercised in their per-backend test files.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from troopai.adk.exceptions.exceptions import ApplyPatchError
from troopai.adk.sandbox.apply_patch import (
    V4AFormat,
    WorkspaceEditor,
    coerce_operations,
    resolve_patch_format,
)
from troopai.adk.sandbox.editor import ApplyPatchOperation


class _FakeSession:
    """Minimal session double — only the methods WorkspaceEditor calls."""

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self._files: dict[str, str] = files or {}
        self.write = AsyncMock(side_effect=self._write_impl)
        self.mkdir = AsyncMock()
        self.rm = AsyncMock(side_effect=self._rm_impl)
        self.read = AsyncMock(side_effect=self._read_impl)

    def normalize_path(self, path: Path | str, *, for_write: bool = False) -> Path:
        _ = for_write
        return Path(path)

    async def _write_impl(self, path: Any, data: Any, *, user: Any = None) -> None:
        _ = user
        if hasattr(data, "read"):
            payload = data.read()
            if isinstance(payload, bytes):
                self._files[str(path)] = payload.decode("utf-8")
            elif isinstance(payload, str):
                self._files[str(path)] = payload

    async def _rm_impl(self, path: Any, *, user: Any = None) -> None:
        _ = user
        self._files.pop(str(path), None)

    async def _read_impl(self, path: Any, *, user: Any = None) -> Any:
        _ = user
        key = str(path)
        if key not in self._files:
            raise FileNotFoundError(key)
        return io.BytesIO(self._files[key].encode("utf-8"))

    @property
    def files(self) -> dict[str, str]:
        return self._files


class TestWorkspaceEditorOperations:
    @pytest.mark.asyncio
    async def test_create_file_writes_diff_content(self) -> None:
        session = _FakeSession()
        editor = WorkspaceEditor(session)  # type: ignore[arg-type]
        op = ApplyPatchOperation(
            type="create_file",
            path="hello.txt",
            diff="+hello\n+world",
        )
        result = await editor.apply_operation(op)
        assert result.status == "completed"
        assert session.files["hello.txt"] == "hello\nworld"

    @pytest.mark.asyncio
    async def test_update_file_applies_diff(self) -> None:
        session = _FakeSession(files={"src/app.py": "alpha\nbeta\ngamma"})
        editor = WorkspaceEditor(session)  # type: ignore[arg-type]
        op = ApplyPatchOperation(
            type="update_file",
            path="src/app.py",
            diff="@@\n alpha\n-beta\n+BETA\n gamma",
        )
        result = await editor.apply_operation(op)
        assert result.status == "completed"
        assert session.files["src/app.py"] == "alpha\nBETA\ngamma"

    @pytest.mark.asyncio
    async def test_update_file_with_move_to_renames(self) -> None:
        session = _FakeSession(files={"old.py": "alpha\nbeta\ngamma"})
        editor = WorkspaceEditor(session)  # type: ignore[arg-type]
        op = ApplyPatchOperation(
            type="update_file",
            path="old.py",
            diff="@@\n alpha\n-beta\n+BETA\n gamma",
            move_to="new.py",
        )
        result = await editor.apply_operation(op)
        assert result.status == "completed"
        assert "new.py" in session.files
        assert "old.py" not in session.files

    @pytest.mark.asyncio
    async def test_delete_file_removes_path(self) -> None:
        session = _FakeSession(files={"trash.txt": "bye"})
        editor = WorkspaceEditor(session)  # type: ignore[arg-type]
        op = ApplyPatchOperation(type="delete_file", path="trash.txt")
        result = await editor.apply_operation(op)
        assert result.status == "completed"
        assert "trash.txt" not in session.files

    @pytest.mark.asyncio
    async def test_delete_file_missing_raises(self) -> None:
        session = _FakeSession()
        editor = WorkspaceEditor(session)  # type: ignore[arg-type]
        op = ApplyPatchOperation(type="delete_file", path="missing.txt")
        with pytest.raises(ApplyPatchError, match="file not found"):
            await editor.apply_operation(op)


class TestPathValidation:
    @pytest.mark.asyncio
    async def test_rejects_absolute_path(self) -> None:
        editor = WorkspaceEditor(_FakeSession())  # type: ignore[arg-type]
        op = ApplyPatchOperation(type="delete_file", path="/etc/passwd")
        with pytest.raises(ApplyPatchError, match="absolute path"):
            await editor.apply_operation(op)

    @pytest.mark.asyncio
    async def test_rejects_parent_traversal(self) -> None:
        editor = WorkspaceEditor(_FakeSession())  # type: ignore[arg-type]
        op = ApplyPatchOperation(type="delete_file", path="../outside")
        with pytest.raises(ApplyPatchError, match="parent traversal"):
            await editor.apply_operation(op)

    @pytest.mark.asyncio
    async def test_rejects_empty_path(self) -> None:
        editor = WorkspaceEditor(_FakeSession())  # type: ignore[arg-type]
        op = ApplyPatchOperation(type="delete_file", path="  ")
        with pytest.raises(ApplyPatchError, match="empty path"):
            await editor.apply_operation(op)


class TestPayloadCoercion:
    def test_coerce_operations_single_op(self) -> None:
        op = ApplyPatchOperation(type="delete_file", path="x.txt")
        assert coerce_operations(op) == [op]

    def test_coerce_operations_dict(self) -> None:
        out = coerce_operations({"type": "delete_file", "path": "x.txt"})
        assert len(out) == 1
        assert out[0].path == "x.txt"

    def test_coerce_operations_list_mixed(self) -> None:
        op1 = ApplyPatchOperation(type="delete_file", path="a.txt")
        out = coerce_operations([op1, {"type": "delete_file", "path": "b.txt"}])
        assert len(out) == 2
        assert out[1].path == "b.txt"

    def test_coerce_operations_invalid_type(self) -> None:
        with pytest.raises(ApplyPatchError, match="Invalid"):
            coerce_operations({"type": "wat", "path": "x.txt"})

    def test_coerce_operations_invalid_payload(self) -> None:
        with pytest.raises(ApplyPatchError, match="Invalid"):
            coerce_operations(42)  # type: ignore[arg-type]


class TestPatchFormatResolution:
    def test_default_v4a(self) -> None:
        assert resolve_patch_format("v4a") is V4AFormat

    def test_custom_format(self) -> None:
        class _CustomFormat:
            @staticmethod
            def apply_diff(input_text: str, diff: str, mode: Any = "default") -> str:
                _ = mode
                return input_text + diff

        out = resolve_patch_format(_CustomFormat)
        assert out is _CustomFormat

    def test_unknown_format(self) -> None:
        with pytest.raises(ApplyPatchError, match="Unsupported"):
            resolve_patch_format("v999")  # type: ignore[arg-type]


class TestApplyPatchBatching:
    @pytest.mark.asyncio
    async def test_batch_returns_done(self) -> None:
        session = _FakeSession()
        editor = WorkspaceEditor(session)  # type: ignore[arg-type]
        ops = [
            ApplyPatchOperation(type="create_file", path="a.txt", diff="+hello"),
            ApplyPatchOperation(type="create_file", path="b.txt", diff="+world"),
        ]
        out = await editor.apply_patch(ops)
        assert out == "Done!"
        assert session.files == {"a.txt": "hello", "b.txt": "world"}

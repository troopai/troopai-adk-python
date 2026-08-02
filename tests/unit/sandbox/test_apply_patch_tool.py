"""Tests for ``apply_patch_tool`` — args validation + tool dispatch.

The factory ``make_apply_patch_tool`` runs when a capability builds
its tools, but the tool's invocation path (``ApplyPatchArgs``
validation + ``_on_invoke`` → ``session.apply_patch``) was
previously exercised by no test. These pin that path.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from troopai.adk.sandbox.tools.apply_patch_tool import (
    ApplyPatchArgs,
    make_apply_patch_tool,
)


class TestApplyPatchArgs:
    def test_valid(self) -> None:
        args = ApplyPatchArgs(patch="--- a\n+++ b\n")
        assert args.patch == "--- a\n+++ b\n"

    def test_patch_is_required(self) -> None:
        with pytest.raises(ValidationError):
            ApplyPatchArgs()  # type: ignore[call-arg]


def _session_with_apply_patch(result: Any = "applied: 1 file") -> Any:
    session = AsyncMock()
    session.apply_patch = AsyncMock(return_value=result)
    return session


class TestMakeApplyPatchTool:
    def test_tool_shape(self) -> None:
        tool = make_apply_patch_tool(session=_session_with_apply_patch())
        assert tool.name == "apply_patch"
        assert tool.schema is ApplyPatchArgs
        assert tool.on_invoke is not None

    async def test_invoke_dispatches_to_session_apply_patch(self) -> None:
        session = _session_with_apply_patch("applied: ok")
        tool = make_apply_patch_tool(session=session)
        raw = json.dumps({"patch": "--- a\n+++ b\n@@\n-x\n+y\n"})
        result = await tool.on_invoke(None, raw)  # type: ignore[arg-type]
        assert result == {
            "summary": "applied: ok",
            "patch_size_bytes": len(b"--- a\n+++ b\n@@\n-x\n+y\n"),
        }
        session.apply_patch.assert_awaited_once_with("--- a\n+++ b\n@@\n-x\n+y\n", user=None)

    async def test_invoke_patch_size_bytes_counts_utf8_bytes_not_codepoints(self) -> None:
        """``patch_size_bytes`` must report the UTF-8 byte size, not the
        Unicode code-point count, so multi-byte content is not undercounted."""
        session = _session_with_apply_patch()
        tool = make_apply_patch_tool(session=session)
        patch = "+# 🚀 日本語 café\n"
        result = await tool.on_invoke(None, json.dumps({"patch": patch}))  # type: ignore[arg-type]
        expected_bytes = len(patch.encode("utf-8"))
        assert result["patch_size_bytes"] == expected_bytes
        # Multi-byte content makes the byte count strictly exceed the
        # code-point count, which is exactly what the pre-fix code returned.
        assert expected_bytes > len(patch)

    async def test_invoke_forwards_user(self) -> None:
        session = _session_with_apply_patch()
        tool = make_apply_patch_tool(session=session, user="alice")
        await tool.on_invoke(None, json.dumps({"patch": "p"}))  # type: ignore[arg-type]
        session.apply_patch.assert_awaited_once_with("p", user="alice")

    async def test_invoke_bad_json_raises(self) -> None:
        tool = make_apply_patch_tool(session=_session_with_apply_patch())
        with pytest.raises(ValidationError):
            await tool.on_invoke(None, "{not valid json")  # type: ignore[arg-type]

    async def test_invoke_session_error_propagates(self) -> None:
        session = AsyncMock()
        session.apply_patch = AsyncMock(side_effect=RuntimeError("apply blew up"))
        tool = make_apply_patch_tool(session=session)
        with pytest.raises(RuntimeError, match="apply blew up"):
            await tool.on_invoke(None, json.dumps({"patch": "p"}))  # type: ignore[arg-type]

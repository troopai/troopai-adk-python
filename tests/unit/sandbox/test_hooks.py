"""Tests for the RunHooks sandbox lifecycle methods (P08)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from troopai.adk.hooks.hooks import CompositeRunHooks, RunHooks, compose_run_hooks
from troopai.adk.types.sandbox.exec_result import ExecResult
from troopai.adk.types.sandbox.snapshot import SnapshotMetadata, SnapshotRef
from troopai.adk.types.sandbox.usage import SandboxUsage


def _ref() -> SnapshotRef:
    return SnapshotRef(snapshot_id="s1", store_uri="file:///tmp/snaps")


def _meta() -> SnapshotMetadata:
    return SnapshotMetadata(
        ref=_ref(),
        created_at_iso="2025-01-01T00:00:00Z",
        size_bytes=128,
    )


class TestBaseHooksAreNoOp:
    @pytest.mark.parametrize(
        "method,args",
        [
            ("on_sandbox_start", (None, None, None)),
            ("on_sandbox_stop", (None, None, None, SandboxUsage())),
            ("on_sandbox_exec_start", (None, None, "ls")),
            (
                "on_sandbox_exec_end",
                (None, None, "ls", ExecResult(stdout=b"", stderr=b"", exit_code=0)),
            ),
            ("on_sandbox_snapshot", (None, None, _meta())),
            ("on_sandbox_error", (None, None, RuntimeError("boom"))),
        ],
    )
    @pytest.mark.asyncio
    async def test_default_returns_none(self, method: str, args: tuple[Any, ...]) -> None:
        hooks = RunHooks[Any]()
        result = await getattr(hooks, method)(*args)
        assert result is None


class TestCompositeFanout:
    @pytest.mark.asyncio
    async def test_fanout_to_every_member(self) -> None:
        m1 = AsyncMock(spec=RunHooks)
        m2 = AsyncMock(spec=RunHooks)
        composite = CompositeRunHooks([m1, m2])
        usage = SandboxUsage(exec_count=3)
        await composite.on_sandbox_stop(None, None, None, usage)
        m1.on_sandbox_stop.assert_awaited_once_with(None, None, None, usage)
        m2.on_sandbox_stop.assert_awaited_once_with(None, None, None, usage)

    @pytest.mark.asyncio
    async def test_fanout_preserves_error_collection(self) -> None:
        # One member raises; the other still runs; the first error
        # propagates after all members have run (existing semantic).
        m1 = AsyncMock(spec=RunHooks)
        m1.on_sandbox_exec_start.side_effect = RuntimeError("boom")
        m2 = AsyncMock(spec=RunHooks)
        composite = CompositeRunHooks([m1, m2])
        with pytest.raises(RuntimeError, match="boom"):
            await composite.on_sandbox_exec_start(None, None, "ls")
        m2.on_sandbox_exec_start.assert_awaited_once()


class TestComposeRunHooks:
    @pytest.mark.asyncio
    async def test_compose_single_returns_member_directly(self) -> None:
        m = AsyncMock(spec=RunHooks)
        composed = compose_run_hooks(m)
        assert composed is m

    @pytest.mark.asyncio
    async def test_compose_none_returns_noop(self) -> None:
        composed = compose_run_hooks(None, None)
        # No exceptions; default no-op base. ``None`` arguments are cast
        # via ``Any`` because the hook signatures are typed against
        # ``RunContext``/``Agent`` for documentation; the runtime
        # contract is "any value passed through".
        any_none: Any = None
        await composed.on_sandbox_start(any_none, any_none, any_none)

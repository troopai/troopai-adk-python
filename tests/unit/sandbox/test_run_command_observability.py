"""Tests for run_command observability emission + violation auditing."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from troopai.adk.exceptions.exceptions import SandboxCommandRejected
from troopai.adk.sandbox.guardrails.command_guardrail import SandboxCommandGuardrail
from troopai.adk.sandbox.observability.observability import SandboxObservability
from troopai.adk.sandbox.tools.run_command_tool import make_run_command_tool
from troopai.adk.types.sandbox.exec_result import ExecResult
from troopai.adk.types.sandbox.usage import SandboxUsage


def _session(exit_code: int = 0, duration_ms: int = 1000) -> AsyncMock:
    session = AsyncMock()
    session.run = AsyncMock(
        return_value=ExecResult(stdout=b"ok", stderr=b"", exit_code=exit_code, duration_ms=duration_ms)
    )
    return session


async def test_tool_records_usage_through_observability() -> None:
    obs = SandboxObservability(backend_id="local", tracing_enabled=False, usage=SandboxUsage())
    tool = make_run_command_tool(session=_session(), observability=obs)
    assert tool.on_invoke is not None
    out = await tool.on_invoke(None, json.dumps({"command": "ls"}))  # type: ignore[arg-type]
    assert out["exit_code"] == 0
    assert obs.usage.exec_count == 1


async def test_tool_emits_violation_then_raises_on_denied_command() -> None:
    sink = AsyncMock()
    obs = SandboxObservability(backend_id="local", tracing_enabled=False, usage=SandboxUsage(), audit_sink=sink)
    policy = SandboxCommandGuardrail(denylist=["rm -rf /"], pattern_mode="prefix")
    tool = make_run_command_tool(session=_session(), observability=obs, command_policy=policy)
    assert tool.on_invoke is not None
    with pytest.raises(SandboxCommandRejected):
        await tool.on_invoke(None, json.dumps({"command": "rm -rf /"}))  # type: ignore[arg-type]
    kinds = [call.args[0].event_type for call in sink.emit.await_args_list]
    assert "violation" in kinds


async def test_tool_without_observability_still_works() -> None:
    tool = make_run_command_tool(session=_session())
    assert tool.on_invoke is not None
    out = await tool.on_invoke(None, json.dumps({"command": "ls"}))  # type: ignore[arg-type]
    assert out["exit_code"] == 0

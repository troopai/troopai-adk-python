# tests/unit/sandbox/observability/test_observability.py
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.sandbox.observability.observability import SandboxObservability
from troopai.adk.types.sandbox.cost import SandboxCostDescriptor
from troopai.adk.types.sandbox.exec_result import ExecResult
from troopai.adk.types.sandbox.usage import SandboxUsage


def _exec_result(exit_code: int = 0, duration_ms: int = 60000) -> ExecResult:
    return ExecResult(stdout=b"", stderr=b"", exit_code=exit_code, duration_ms=duration_ms)


async def test_after_exec_records_usage_with_cost() -> None:
    obs = SandboxObservability(
        backend_id="e2b",
        tracing_enabled=False,
        usage=SandboxUsage(),
        cost=SandboxCostDescriptor(usd_per_minute=0.06),
    )
    await obs.after_exec("ls -la", _exec_result(duration_ms=60000))
    assert obs.usage.exec_count == 1
    assert obs.usage.computed_cost_usd == pytest.approx(0.06)


async def test_exec_emits_audit_events_when_sink_present() -> None:
    sink = AsyncMock()
    obs = SandboxObservability(
        backend_id="e2b",
        tracing_enabled=False,
        usage=SandboxUsage(),
        audit_sink=sink,
        session_id="sess-1",
    )
    await obs.after_exec("ls", _exec_result())
    await obs.on_violation("rm -rf /", "denied by policy")
    kinds = [call.args[0].event_type for call in sink.emit.await_args_list]
    assert "exec" in kinds
    assert "violation" in kinds


async def test_hooks_fire_when_present() -> None:
    hooks = AsyncMock()
    agent = MagicMock()
    agent.name = "a"
    obs = SandboxObservability(
        backend_id="local",
        tracing_enabled=False,
        usage=SandboxUsage(),
        hooks=hooks,
        context=MagicMock(),
        agent=agent,
    )
    await obs.before_exec("ls")
    await obs.after_exec("ls", _exec_result())
    hooks.on_sandbox_exec_start.assert_awaited_once()
    hooks.on_sandbox_exec_end.assert_awaited_once()


async def test_no_sink_no_hooks_is_noop() -> None:
    obs = SandboxObservability(backend_id="local", tracing_enabled=False, usage=SandboxUsage())
    await obs.before_exec("ls")
    await obs.after_exec("ls", _exec_result())
    await obs.on_violation("x", "y")  # must not raise
    assert obs.usage.exec_count == 1


async def test_emit_audit_suppresses_sink_errors() -> None:
    sink = AsyncMock()
    sink.emit.side_effect = RuntimeError("audit backend down")
    obs = SandboxObservability(
        backend_id="e2b",
        tracing_enabled=False,
        usage=SandboxUsage(),
        audit_sink=sink,
    )
    # A failing audit sink must never break the run; usage still accrues.
    await obs.after_exec("ls", _exec_result())
    assert obs.usage.exec_count == 1

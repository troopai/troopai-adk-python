"""Tests for audit-event emission from the tool executor.

Verifies that exactly one audit event is emitted per tool-call resolution
for both the ok and error outcomes in _execute_single_tool_call (covers
the sync path via execute_tool_calls and the streaming path which reuses
the same function).
"""

from __future__ import annotations

import asyncio

import pytest

from troopai.adk.audit import InMemoryAuditSink
from troopai.adk.exceptions import ToolTimeoutError
from troopai.adk.run.config import RunConfig
from troopai.adk.run.tools_executor import execute_tool_calls
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

# ── Helpers ──────────────────────────────────────────────────────────


def _make_agent(tools: list) -> object:
    from types import SimpleNamespace

    from troopai.adk.agents.middleware import Middleware

    return SimpleNamespace(
        name="test_agent",
        tools=tools,
        tool_use_behavior="run_llm_again",
        handoffs=None,
        llm=None,
        hooks=None,
        middleware=Middleware(),
    )


def _make_ctx(tenant_id: str | None = None) -> object:
    from troopai.adk.run.context import RunContext

    return RunContext(context=None, tenant_id=tenant_id)


def _make_hooks() -> object:
    from troopai.adk.hooks.hooks import RunHooks

    return RunHooks()


async def _echo_handler(_ctx: object, raw_args: str) -> str:
    import json

    args = json.loads(raw_args) if len(raw_args) > 0 else {}
    return args.get("text", "echo")


async def _boom_handler(_ctx: object, _raw_args: str) -> str:
    raise RuntimeError("intentional failure")


echo = FunctionTool(
    name="echo",
    description="Echo the text argument",
    schema={"type": "object", "properties": {"text": {"type": "string"}}},
    on_invoke=_echo_handler,
)

boom = FunctionTool(
    name="boom",
    description="Always raises",
    schema={"type": "object", "properties": {}},
    on_invoke=_boom_handler,
)


# ── Tests ────────────────────────────────────────────────────────────


async def test_executed_tool_emits_ok_event() -> None:
    """A successfully executed tool emits exactly one 'ok' audit event."""
    sink = InMemoryAuditSink()
    agent = _make_agent([echo])
    ctx = _make_ctx(tenant_id="t1")
    config = RunConfig(audit_sink=sink)
    call = LLMResponseFunctionToolCall(call_id="c1", name="echo", arguments='{"text": "hi"}')

    await execute_tool_calls(agent, [call], ctx, _make_hooks(), config)

    assert len(sink.events) == 1
    assert sink.events[0].outcome == "ok"
    assert sink.events[0].tool_name == "echo"
    assert sink.events[0].tenant_id == "t1"
    assert sink.events[0].result_hash is not None


async def test_tool_error_emits_error_event() -> None:
    """A tool that raises with fail_on_tool_error=False emits exactly one 'error' event.

    The converted-error branch must set the flag, reach the single on_tool_end
    emit point, and produce exactly one event — not an 'error' plus an 'ok'.
    """
    sink = InMemoryAuditSink()
    agent = _make_agent([boom])
    ctx = _make_ctx(tenant_id="t1")
    config = RunConfig(audit_sink=sink, fail_on_tool_error=False)
    call = LLMResponseFunctionToolCall(call_id="c1", name="boom", arguments="{}")

    await execute_tool_calls(agent, [call], ctx, _make_hooks(), config)

    assert len(sink.events) == 1
    assert sink.events[0].outcome == "error"
    assert sink.events[0].tool_name == "boom"
    assert sink.events[0].tenant_id == "t1"
    assert sink.events[0].result_hash is not None  # converted error message is hashed


async def test_tool_error_fail_on_error_emits_before_raise() -> None:
    """fail_on_tool_error=True: exactly one 'error' event is emitted before the raise propagates."""
    sink = InMemoryAuditSink()
    agent = _make_agent([boom])
    ctx = _make_ctx(tenant_id="t1")
    config = RunConfig(audit_sink=sink, fail_on_tool_error=True)
    call = LLMResponseFunctionToolCall(call_id="c1", name="boom", arguments="{}")

    with pytest.raises(RuntimeError, match="intentional failure"):
        await execute_tool_calls(agent, [call], ctx, _make_hooks(), config)

    assert len(sink.events) == 1
    assert sink.events[0].outcome == "error"


async def test_timeout_raise_emits_error_event() -> None:
    """A tool timeout with timeout_behavior='raise_exception' emits one 'error' event before raising."""

    async def _slow_handler(_ctx: object, _raw_args: str) -> str:
        await asyncio.sleep(1)
        return "done"

    slow = FunctionTool(
        name="slow",
        description="Sleeps past its timeout",
        schema={"type": "object", "properties": {}},
        on_invoke=_slow_handler,
        timeout=0.01,
        timeout_behavior="raise_exception",
    )
    sink = InMemoryAuditSink()
    agent = _make_agent([slow])
    ctx = _make_ctx(tenant_id="t1")
    config = RunConfig(audit_sink=sink, fail_on_tool_error=True)
    call = LLMResponseFunctionToolCall(call_id="c1", name="slow", arguments="{}")

    with pytest.raises(ToolTimeoutError):
        await execute_tool_calls(agent, [call], ctx, _make_hooks(), config)

    assert len(sink.events) == 1
    assert sink.events[0].outcome == "error"


async def test_no_sink_no_emit() -> None:
    """When audit_sink is None, no event is emitted (no-op)."""
    agent = _make_agent([echo])
    ctx = _make_ctx(tenant_id="t1")
    config = RunConfig(audit_sink=None)
    call = LLMResponseFunctionToolCall(call_id="c1", name="echo", arguments='{"text": "hi"}')

    # Just must not raise; nothing to assert on
    await execute_tool_calls(agent, [call], ctx, _make_hooks(), config)


async def test_exactly_one_event_per_call_multiple_tools() -> None:
    """With two tool calls, exactly two audit events are emitted."""
    sink = InMemoryAuditSink()
    echo2 = FunctionTool(
        name="echo2",
        description="Echo two",
        schema={"type": "object", "properties": {}},
        on_invoke=_echo_handler,
    )
    agent = _make_agent([echo, echo2])
    ctx = _make_ctx(tenant_id="t1")
    config = RunConfig(audit_sink=sink)
    calls = [
        LLMResponseFunctionToolCall(call_id="c1", name="echo", arguments='{"text": "a"}'),
        LLMResponseFunctionToolCall(call_id="c2", name="echo2", arguments="{}"),
    ]

    await execute_tool_calls(agent, calls, ctx, _make_hooks(), config)

    assert len(sink.events) == 2
    assert {e.outcome for e in sink.events} == {"ok"}


async def test_timeout_error_as_result_emits_error_event() -> None:
    """A timeout with timeout_behavior='error_as_result' audits as 'error', not 'ok'."""

    async def _slow(_ctx: object, _raw_args: str) -> str:
        await asyncio.sleep(1)
        return "done"

    slow = FunctionTool(
        name="slow_res",
        description="Sleeps past its timeout (returned as result)",
        schema={"type": "object", "properties": {}},
        on_invoke=_slow,
        timeout=0.01,
        timeout_behavior="error_as_result",
    )
    sink = InMemoryAuditSink()
    agent = _make_agent([slow])
    ctx = _make_ctx(tenant_id="t1")
    config = RunConfig(audit_sink=sink, fail_on_tool_error=False)
    call = LLMResponseFunctionToolCall(call_id="c1", name="slow_res", arguments="{}")

    results, _ = await execute_tool_calls(agent, [call], ctx, _make_hooks(), config)

    assert len(results) == 1  # timeout converted to a result, run continues
    assert len(sink.events) == 1
    assert sink.events[0].outcome == "error"  # not "ok"


async def test_audit_strict_failure_propagates_through_executor() -> None:
    """audit_strict=True: a sink failure on the success path propagates out of the executor."""

    class _Boom:
        async def record(self, event: object) -> None:
            raise RuntimeError("sink down")

    agent = _make_agent([echo])
    ctx = _make_ctx(tenant_id="t1")
    config = RunConfig(audit_sink=_Boom(), audit_strict=True)
    call = LLMResponseFunctionToolCall(call_id="c1", name="echo", arguments='{"text": "hi"}')

    with pytest.raises(RuntimeError, match="sink down"):
        await execute_tool_calls(agent, [call], ctx, _make_hooks(), config)

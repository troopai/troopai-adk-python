"""Tests for audit-event emission from the HITL resume path.

Verifies that exactly one audit event is emitted per tool-call resolution
in execute_approved_tool, for both ok and error outcomes.
"""

from __future__ import annotations

import pytest

from troopai.adk.audit import InMemoryAuditSink
from troopai.adk.run.config import RunConfig
from troopai.adk.run.tools_executor import execute_approved_tool
from troopai.adk.tools.deferred_tool import DeferredToolCall
from troopai.adk.tools.function_tool import FunctionTool

# ── Helpers ──────────────────────────────────────────────────────────


def _make_agent(tools: list) -> object:
    from types import SimpleNamespace

    from troopai.adk.agents.middleware import Middleware

    return SimpleNamespace(
        tools=tools,
        tool_use_behavior="run_llm_again",
        handoffs=None,
        llm=None,
        hooks=None,
        middleware=Middleware(),
        name="test_agent",
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


async def test_approved_tool_emits_ok_event() -> None:
    """An approved tool that executes successfully emits exactly one 'ok' event."""
    sink = InMemoryAuditSink()
    agent = _make_agent([echo])
    ctx = _make_ctx(tenant_id="t1")
    config = RunConfig(audit_sink=sink)
    approved = DeferredToolCall(
        tool_call_id="c1",
        tool_name="echo",
        tool_arguments={"text": "hi"},
        raw_arguments='{"text": "hi"}',
    )

    _result, success = await execute_approved_tool(agent, approved, ctx, _make_hooks(), config, None)

    assert success is True
    assert len(sink.events) == 1
    assert sink.events[0].outcome == "ok"
    assert sink.events[0].tool_name == "echo"
    assert sink.events[0].tenant_id == "t1"
    assert sink.events[0].result_hash is not None


async def test_approved_tool_error_emits_error_event() -> None:
    """An approved tool that raises with fail_on_tool_error=False emits one 'error' event."""
    sink = InMemoryAuditSink()
    agent = _make_agent([boom])
    ctx = _make_ctx(tenant_id="t1")
    config = RunConfig(audit_sink=sink, fail_on_tool_error=False)
    approved = DeferredToolCall(
        tool_call_id="c1",
        tool_name="boom",
        tool_arguments={},
        raw_arguments="{}",
    )

    _result, success = await execute_approved_tool(agent, approved, ctx, _make_hooks(), config, None)

    assert success is True  # error-as-result path, not a hard failure
    assert len(sink.events) == 1
    assert sink.events[0].outcome == "error"
    assert sink.events[0].tool_name == "boom"
    assert sink.events[0].tenant_id == "t1"
    assert sink.events[0].result_hash is not None  # converted error message is hashed


async def test_approved_tool_fail_on_error_emits_before_raise() -> None:
    """fail_on_tool_error=True on the HITL path: one 'error' event before the raise propagates."""
    sink = InMemoryAuditSink()
    agent = _make_agent([boom])
    ctx = _make_ctx(tenant_id="t1")
    config = RunConfig(audit_sink=sink, fail_on_tool_error=True)
    approved = DeferredToolCall(
        tool_call_id="c1",
        tool_name="boom",
        tool_arguments={},
        raw_arguments="{}",
    )

    with pytest.raises(RuntimeError, match="intentional failure"):
        await execute_approved_tool(agent, approved, ctx, _make_hooks(), config, None)

    assert len(sink.events) == 1
    assert sink.events[0].outcome == "error"


async def test_approved_tool_no_sink_no_emit() -> None:
    """When audit_sink is None, no event is emitted (no-op)."""
    agent = _make_agent([echo])
    ctx = _make_ctx(tenant_id="t1")
    config = RunConfig(audit_sink=None)
    approved = DeferredToolCall(
        tool_call_id="c1",
        tool_name="echo",
        tool_arguments={"text": "hi"},
        raw_arguments='{"text": "hi"}',
    )

    _result, success = await execute_approved_tool(agent, approved, ctx, _make_hooks(), config, None)

    assert success is True

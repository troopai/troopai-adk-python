"""Tests for the per-tenant tool allowlist gate in the tool executor.

Verifies that the gate fires BEFORE tool lookup so builtins are gated too,
that hard mode raises ToolNotPermittedForTenant, that soft mode returns a
denial result without running the tool body, and that a permitted tenant
executes normally and records an 'ok' audit event.
"""

from __future__ import annotations

import pytest

from troopai.adk.audit import InMemoryAuditSink
from troopai.adk.exceptions import ToolNotPermittedForTenant
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


# Records each invocation so tests can assert the tool body did / didn't run.
_secret_calls: list[str] = []


async def _secret_handler(_ctx: object, raw_args: str) -> str:
    import json

    _secret_calls.append(raw_args)
    args = json.loads(raw_args) if len(raw_args) > 0 else {}
    return f"secret:{args.get('text', '')}"


secret = FunctionTool(
    name="secret",
    description="Returns secret data",
    schema={"type": "object", "properties": {"text": {"type": "string"}}},
    on_invoke=_secret_handler,
)


# ── Tests ────────────────────────────────────────────────────────────


async def test_hard_deny_raises_before_execution() -> None:
    """deny-all for t1 => calling 'secret' raises, tool body never runs."""
    _secret_calls.clear()
    agent = _make_agent([secret])
    ctx = _make_ctx("t1")
    config = RunConfig(tenant_tool_allowlist={"t1": set()})
    call = LLMResponseFunctionToolCall(call_id="c1", name="secret", arguments='{"text":"x"}')

    with pytest.raises(ToolNotPermittedForTenant):
        await execute_tool_calls(agent, [call], ctx, _make_hooks(), config)

    assert len(_secret_calls) == 0  # tool body never ran


async def test_soft_deny_returns_denial_not_tool_output() -> None:
    """Soft deny: executor returns a result but the tool body never ran."""
    _secret_calls.clear()
    agent = _make_agent([secret])
    ctx = _make_ctx("t1")
    config = RunConfig(tenant_tool_allowlist={"t1": set()}, tenant_allowlist_soft_deny=True)
    call = LLMResponseFunctionToolCall(call_id="c1", name="secret", arguments='{"text":"x"}')

    results, _deferred = await execute_tool_calls(agent, [call], ctx, _make_hooks(), config)

    assert len(results) == 1
    assert "secret:x" not in str(results[0].output)
    assert len(_secret_calls) == 0  # tool body never ran


async def test_permitted_tenant_runs_and_audits_ok() -> None:
    """A tenant with the tool in its allowlist runs normally and emits an 'ok' event."""
    sink = InMemoryAuditSink()
    agent = _make_agent([secret])
    ctx = _make_ctx("t1")
    config = RunConfig(tenant_tool_allowlist={"t1": {"secret"}}, audit_sink=sink)
    call = LLMResponseFunctionToolCall(call_id="c1", name="secret", arguments='{"text":"x"}')

    _secret_calls.clear()
    results, _ = await execute_tool_calls(agent, [call], ctx, _make_hooks(), config)

    assert "secret:x" in str(results[0].output)
    assert len(_secret_calls) == 1  # permitted tenant's tool body ran
    assert sink.events[-1].outcome == "ok"


async def test_untenanted_run_with_default_deny_executes() -> None:
    """An untenanted run is never tenant-governed, even under default_deny (executor-level)."""
    _secret_calls.clear()
    agent = _make_agent([secret])
    ctx = _make_ctx(None)  # untenanted
    config = RunConfig(tenant_tool_allowlist={"t1": set()}, tenant_allowlist_default_deny=True)
    call = LLMResponseFunctionToolCall(call_id="c1", name="secret", arguments='{"text": "x"}')

    results, _ = await execute_tool_calls(agent, [call], ctx, _make_hooks(), config)

    assert "secret:x" in str(results[0].output)
    assert len(_secret_calls) == 1  # ran despite default_deny — untenanted bypasses

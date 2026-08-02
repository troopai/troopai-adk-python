"""Tests for the per-tenant tool allowlist gate on the HITL resume path.

Verifies that execute_approved_tool is NOT bypassable via human approval:
a forbidden tenant's tool still raises (hard mode) or returns a soft-denial
result (soft mode) even after a human has "approved" the HITL call.
"""

from __future__ import annotations

import pytest

from troopai.adk.exceptions import ToolNotPermittedForTenant
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


async def _secret_handler(_ctx: object, raw_args: str) -> str:
    import json

    args = json.loads(raw_args) if len(raw_args) > 0 else {}
    return f"secret:{args.get('text', '')}"


secret = FunctionTool(
    name="secret",
    description="Returns secret data",
    schema={"type": "object", "properties": {"text": {"type": "string"}}},
    on_invoke=_secret_handler,
)


# ── Tests ────────────────────────────────────────────────────────────


async def test_hitl_hard_deny_raises() -> None:
    """The allowlist must NOT be bypassable via human approval (hard mode)."""
    agent = _make_agent([secret])
    ctx = _make_ctx("t1")
    config = RunConfig(tenant_tool_allowlist={"t1": set()})
    approved = DeferredToolCall(
        tool_call_id="c1",
        tool_name="secret",
        tool_arguments={"text": "x"},
        raw_arguments='{"text":"x"}',
    )

    with pytest.raises(ToolNotPermittedForTenant):
        await execute_approved_tool(agent, approved, ctx, _make_hooks(), config, None)


async def test_hitl_soft_deny_returns_failure() -> None:
    """Soft deny on HITL path: returns (denial_message, False), tool body never ran."""
    agent = _make_agent([secret])
    ctx = _make_ctx("t1")
    config = RunConfig(tenant_tool_allowlist={"t1": set()}, tenant_allowlist_soft_deny=True)
    approved = DeferredToolCall(
        tool_call_id="c1",
        tool_name="secret",
        tool_arguments={"text": "x"},
        raw_arguments='{"text":"x"}',
    )

    result, success = await execute_approved_tool(agent, approved, ctx, _make_hooks(), config, None)

    assert success is False
    assert "secret:x" not in result

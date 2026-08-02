"""Tests for HITL (Human-in-the-Loop) through the tools executor.

Covers:
- Bug 1: DeferredToolCall construction uses correct keyword (tool_call_id)
- End-to-end HITL deferral flow through execute_tool_calls
"""

import pytest

from troopai.adk.run.config import RunConfig
from troopai.adk.run.tools_executor import execute_tool_calls
from troopai.adk.tools.deferred_tool import DeferredToolCall
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

# ── Helpers ──────────────────────────────────────────────────────────


def _make_agent(tools):
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


def _make_tool_call(call_id, name, args="{}"):
    return LLMResponseFunctionToolCall(call_id=call_id, name=name, arguments=args)


def _make_ctx():
    from troopai.adk.run.context import RunContext

    return RunContext(context=None)


def _make_hooks():
    from troopai.adk.hooks.hooks import RunHooks

    return RunHooks()


async def _echo_handler(_ctx, _raw_args):
    return "executed"


# ── Tests ────────────────────────────────────────────────────────────


class TestHITLExecutor:
    @pytest.mark.asyncio
    async def test_requires_approval_returns_deferred(self) -> None:
        """Tool with requires_approval=True returns DeferredToolCall (Bug 1 regression)."""
        tool = FunctionTool(
            name="delete_user",
            description="Delete a user",
            schema={"type": "object", "properties": {"user_id": {"type": "string"}}},
            on_invoke=_echo_handler,
            requires_approval=True,
        )
        agent = _make_agent([tool])
        config = RunConfig(fail_on_tool_error=False)
        tc = _make_tool_call("call_1", "delete_user", '{"user_id": "123"}')

        results, deferred = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=config,
            model="gpt-4o-mini",
        )

        # No completed results — the tool was deferred
        assert len(results) == 0
        assert deferred is not None
        assert len(deferred.approvals) == 1

        approval = deferred.approvals[0]
        assert isinstance(approval, DeferredToolCall)
        assert approval.tool_call_id == "call_1"
        assert approval.tool_name == "delete_user"
        assert approval.tool_arguments == {"user_id": "123"}

    @pytest.mark.asyncio
    async def test_no_approval_executes_normally(self) -> None:
        """Tool without requires_approval executes immediately."""
        tool = FunctionTool(
            name="read_data",
            description="Read data",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
            requires_approval=False,
        )
        agent = _make_agent([tool])
        config = RunConfig(fail_on_tool_error=False)
        tc = _make_tool_call("call_1", "read_data")

        results, deferred = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=config,
            model="gpt-4o-mini",
        )

        assert len(results) == 1
        assert results[0].output == "executed"
        assert deferred is None

    @pytest.mark.asyncio
    async def test_mixed_approval_and_normal(self) -> None:
        """Mix of approved and non-approved tools splits correctly."""
        safe_tool = FunctionTool(
            name="read",
            description="Read",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
            requires_approval=False,
        )
        dangerous_tool = FunctionTool(
            name="delete",
            description="Delete",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
            requires_approval=True,
        )
        agent = _make_agent([safe_tool, dangerous_tool])
        config = RunConfig(fail_on_tool_error=False)

        results, deferred = await execute_tool_calls(
            agent=agent,
            tool_calls=[
                _make_tool_call("c1", "read"),
                _make_tool_call("c2", "delete"),
            ],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=config,
            model="gpt-4o-mini",
        )

        assert len(results) == 1
        assert results[0].output == "executed"
        assert deferred is not None
        assert len(deferred.approvals) == 1
        assert deferred.approvals[0].tool_call_id == "c2"

    @pytest.mark.asyncio
    async def test_callable_requires_approval(self) -> None:
        """requires_approval can be a callable that returns True."""
        tool = FunctionTool(
            name="conditional_tool",
            description="Maybe needs approval",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
            requires_approval=lambda _ctx: True,
        )
        agent = _make_agent([tool])
        config = RunConfig(fail_on_tool_error=False)
        tc = _make_tool_call("c1", "conditional_tool")

        results, deferred = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=config,
            model="gpt-4o-mini",
        )

        assert len(results) == 0
        assert deferred is not None
        assert deferred.approvals[0].tool_call_id == "c1"

    @pytest.mark.asyncio
    async def test_deferred_tool_call_has_request_time(self) -> None:
        """DeferredToolCall gets a request_time via default_factory."""
        tool = FunctionTool(
            name="t",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
            requires_approval=True,
        )
        agent = _make_agent([tool])
        config = RunConfig(fail_on_tool_error=False)
        tc = _make_tool_call("c1", "t")

        _, deferred = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=config,
            model="gpt-4o-mini",
        )

        from datetime import datetime

        approval = deferred.approvals[0]
        assert isinstance(approval.request_time, datetime)
        # .isoformat() must not crash
        assert isinstance(approval.request_time.isoformat(), str)

"""Tests for RunConfig.can_use_tool permission callback."""

import pytest

from troopai.adk.run.config import RunConfig
from troopai.adk.run.tools_executor import execute_tool_calls
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


async def _echo_handler(_ctx, _raw_args):
    return "ok"


def _make_ctx(context=None):
    from troopai.adk.run.context import RunContext

    return RunContext(context=context)


def _make_hooks():
    from troopai.adk.hooks.hooks import RunHooks

    return RunHooks()


# ── Tests ────────────────────────────────────────────────────────────


class TestCanUseTool:
    @pytest.mark.asyncio
    async def test_denied_returns_error(self) -> None:
        """When can_use_tool returns False, the tool is denied."""
        tool = FunctionTool(
            name="admin_delete",
            description="Delete stuff",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
        )
        agent = _make_agent([tool])

        def deny_all(_agent, _tool_name, _ctx):
            return False

        config = RunConfig(can_use_tool=deny_all, fail_on_tool_error=False)
        tc = _make_tool_call("c1", "admin_delete")

        results, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=config,
            model="gpt-4o-mini",
        )
        assert len(results) == 1
        assert "Permission denied" in results[0].output

    @pytest.mark.asyncio
    async def test_allowed_executes_normally(self) -> None:
        """When can_use_tool returns True, the tool executes."""
        tool = FunctionTool(
            name="safe_tool",
            description="Safe",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
        )
        agent = _make_agent([tool])

        def allow_all(_agent, _tool_name, _ctx):
            return True

        config = RunConfig(can_use_tool=allow_all, fail_on_tool_error=False)
        tc = _make_tool_call("c1", "safe_tool")

        results, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=config,
            model="gpt-4o-mini",
        )
        assert results[0].output == "ok"

    @pytest.mark.asyncio
    async def test_selective_permission(self) -> None:
        """can_use_tool can allow some tools and deny others."""
        tool_a = FunctionTool(
            name="read_data",
            description="Read",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
        )
        tool_b = FunctionTool(
            name="admin_delete",
            description="Delete",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
        )
        agent = _make_agent([tool_a, tool_b])

        def no_admin(_agent, tool_name, _ctx):
            return not tool_name.startswith("admin_")

        config = RunConfig(can_use_tool=no_admin, fail_on_tool_error=False)

        results, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[
                _make_tool_call("c1", "read_data"),
                _make_tool_call("c2", "admin_delete"),
            ],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=config,
            model="gpt-4o-mini",
        )
        assert results[0].output == "ok"
        assert "Permission denied" in results[1].output

    @pytest.mark.asyncio
    async def test_async_callback(self) -> None:
        """can_use_tool can be async."""
        tool = FunctionTool(
            name="t",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
        )
        agent = _make_agent([tool])

        async def async_deny(_agent, _tool_name, _ctx):
            return False

        config = RunConfig(can_use_tool=async_deny, fail_on_tool_error=False)
        tc = _make_tool_call("c1", "t")

        results, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=config,
            model="gpt-4o-mini",
        )
        assert "Permission denied" in results[0].output

    @pytest.mark.asyncio
    async def test_none_means_no_check(self) -> None:
        """When can_use_tool is None, all tools are allowed."""
        tool = FunctionTool(
            name="t",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
        )
        agent = _make_agent([tool])
        config = RunConfig(can_use_tool=None, fail_on_tool_error=False)
        tc = _make_tool_call("c1", "t")

        results, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=config,
            model="gpt-4o-mini",
        )
        assert results[0].output == "ok"

"""Tests for HITL resumption security checks (Bug 8).

Verifies that resume_from_state() re-runs Layer 0 (can_use_tool) and
enabled check on approved tools, matching documented behavior:
"Deferred Tool Resumption → Re-runs Layer 0 (permissions apply even after human approval)"
"""

from unittest.mock import patch

import pytest

from troopai.adk.run.config import RunConfig
from troopai.adk.run.resumption import resume_from_state
from troopai.adk.tools.deferred_tool import (
    DeferredToolCall,
)
from troopai.adk.tools.function_tool import FunctionTool

# ── Helpers ──────────────────────────────────────────────────────────


def _make_agent(tools, name="test_agent"):
    from types import SimpleNamespace

    from troopai.adk.agents.agent_guardrails import AgentGuardrails
    from troopai.adk.agents.middleware import Middleware

    return SimpleNamespace(
        name=name,
        tools=tools,
        tool_use_behavior="run_llm_again",
        handoffs=None,
        llm=None,
        system_prompt=None,
        llm_config=None,
        output_type=None,
        guardrails=AgentGuardrails(),
        hooks=None,
        middleware=Middleware(),
    )


async def _echo_handler(_ctx, _raw_args):
    return "executed"


def _make_deferred_tool(tool_name="admin_delete", call_id="tc_1"):
    return DeferredToolCall(
        tool_call_id=call_id,
        tool_name=tool_name,
        tool_arguments={"id": "123"},
        raw_arguments='{"id": "123"}',
    )


def _make_state(approved_tools, context=None):
    """Create a minimal RunState with approved tools."""
    from troopai.adk.run.state import RunState
    from troopai.adk.tools.deferred_tool import DeferredToolRequests
    from troopai.adk.types.items import ItemHelpers

    raw_messages = [
        {"role": "user", "content": "test"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": approved.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": approved.tool_name,
                        "arguments": approved.raw_arguments,
                    },
                }
                for approved in approved_tools
            ],
        },
    ]

    state = RunState(
        conversation_history=list(ItemHelpers.messages_to_run_items(raw_messages)),
        context=context,
        deferred_tool_requests=DeferredToolRequests(approvals=approved_tools),
        original_user_prompt="test",
        current_agent_name="test_agent",
        turn_count=1,
    )
    # Manually approve all tools
    for tool in approved_tools:
        state.approve(tool)
    return state


# ── Tests ────────────────────────────────────────────────────────────


class TestResumptionPermissionCheck:
    @pytest.mark.asyncio
    async def test_can_use_tool_denies_on_resumption(self) -> None:
        """can_use_tool returning False blocks execution even after human approval (Bug 8)."""
        tool = FunctionTool(
            name="admin_delete",
            description="Delete",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
        )
        agent = _make_agent([tool])

        deferred = _make_deferred_tool("admin_delete", "tc_1")
        state = _make_state([deferred])

        def deny_all(_agent, _tool_name, _ctx):
            return False

        config = RunConfig(can_use_tool=deny_all, fail_on_tool_error=False)

        # Patch run_agent_loop to avoid needing a real LLM
        with patch("troopai.adk.run.loop.run_agent_loop") as mock_loop:
            # Make the loop return a minimal result
            from troopai.adk.run.context import RunContext
            from troopai.adk.types.run import RunResult

            mock_loop.return_value = RunResult(
                final_output="done",
                user_prompt="test",
                new_items=[],
                context=RunContext(context=None),
                last_agent=agent,
            )

            _ = await resume_from_state(
                agent=agent,
                state=state,
                config=config,
            )

            # The loop should have been called with the permission denied message
            call_args = mock_loop.call_args
            messages = call_args.kwargs.get("initial_messages", [])
            # Find the tool result (Layer 1: type="function_call_output")
            tool_msgs = [m for m in messages if m.get("type") == "function_call_output"]
            assert any("Permission denied" in str(m.get("output", "")) for m in tool_msgs)


class TestResumptionEnabledCheck:
    @pytest.mark.asyncio
    async def test_disabled_tool_blocked_on_resumption(self) -> None:
        """Tool disabled between deferral and approval is blocked (Bug 8)."""
        tool = FunctionTool(
            name="admin_delete",
            description="Delete",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
            enabled=False,  # Disabled after deferral
        )
        agent = _make_agent([tool])

        deferred = _make_deferred_tool("admin_delete", "tc_1")
        state = _make_state([deferred])

        config = RunConfig(fail_on_tool_error=False)

        with patch("troopai.adk.run.loop.run_agent_loop") as mock_loop:
            from troopai.adk.run.context import RunContext
            from troopai.adk.types.run import RunResult

            mock_loop.return_value = RunResult(
                final_output="done",
                user_prompt="test",
                new_items=[],
                context=RunContext(context=None),
                last_agent=agent,
            )

            _ = await resume_from_state(
                agent=agent,
                state=state,
                config=config,
            )

            call_args = mock_loop.call_args
            messages = call_args.kwargs.get("initial_messages", [])
            tool_msgs = [m for m in messages if m.get("type") == "function_call_output"]
            assert any("currently disabled" in str(m.get("output", "")) for m in tool_msgs)

"""Tests for guardrail exception handling (Bug 6).

Verifies that guardrail exceptions propagate when config.fail_on_tool_error=True
and are silently skipped when False.
"""

import pytest

from troopai.adk.run.config import RunConfig
from troopai.adk.run.tools_executor import execute_tool_calls
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tools.tool_guardrails import (
    ToolGuardrailFunctionOutput,
    ToolGuardrails,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    tool_input_guardrail,
    tool_output_guardrail,
)
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
    return "ok"


@tool_input_guardrail()
async def _crashing_input_guardrail(_data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    raise RuntimeError("Input guardrail service unavailable")


@tool_output_guardrail()
async def _crashing_output_guardrail(_data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    raise RuntimeError("Output guardrail service unavailable")


# ── Tests ────────────────────────────────────────────────────────────


class TestGuardrailFailurePropagation:
    @pytest.mark.asyncio
    async def test_input_guardrail_crash_propagates_when_fail_on_error(self) -> None:
        """Input guardrail crash raises when fail_on_tool_error=True (Bug 6 fix)."""
        tool = FunctionTool(
            name="t",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
            guardrails=ToolGuardrails(input=[_crashing_input_guardrail]),
        )
        agent = _make_agent([tool])
        config = RunConfig(fail_on_tool_error=True)
        tc = _make_tool_call("c1", "t")

        with pytest.raises(RuntimeError, match="Input guardrail service unavailable"):
            await execute_tool_calls(
                agent=agent,
                tool_calls=[tc],
                ctx_wrapper=_make_ctx(),
                hooks=_make_hooks(),
                config=config,
                model="gpt-4o-mini",
            )

    @pytest.mark.asyncio
    async def test_input_guardrail_crash_skipped_when_no_fail(self) -> None:
        """Input guardrail crash is skipped when fail_on_tool_error=False."""
        tool = FunctionTool(
            name="t",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
            guardrails=ToolGuardrails(input=[_crashing_input_guardrail]),
        )
        agent = _make_agent([tool])
        config = RunConfig(fail_on_tool_error=False)
        tc = _make_tool_call("c1", "t")

        results, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=config,
            model="gpt-4o-mini",
        )
        # Tool executes because guardrail was skipped
        assert results[0].output == "ok"

    @pytest.mark.asyncio
    async def test_output_guardrail_crash_propagates_when_fail_on_error(self) -> None:
        """Output guardrail crash raises when fail_on_tool_error=True (Bug 6 fix)."""
        tool = FunctionTool(
            name="t",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
            guardrails=ToolGuardrails(output=[_crashing_output_guardrail]),
        )
        agent = _make_agent([tool])
        config = RunConfig(fail_on_tool_error=True)
        tc = _make_tool_call("c1", "t")

        with pytest.raises(RuntimeError, match="Output guardrail service unavailable"):
            await execute_tool_calls(
                agent=agent,
                tool_calls=[tc],
                ctx_wrapper=_make_ctx(),
                hooks=_make_hooks(),
                config=config,
                model="gpt-4o-mini",
            )

    @pytest.mark.asyncio
    async def test_output_guardrail_crash_skipped_when_no_fail(self) -> None:
        """Output guardrail crash is skipped when fail_on_tool_error=False."""
        tool = FunctionTool(
            name="t",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
            guardrails=ToolGuardrails(output=[_crashing_output_guardrail]),
        )
        agent = _make_agent([tool])
        config = RunConfig(fail_on_tool_error=False)
        tc = _make_tool_call("c1", "t")

        results, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=config,
            model="gpt-4o-mini",
        )
        # Tool result passes through because guardrail was skipped
        assert results[0].output == "ok"


class TestArgsValidatorRemoved:
    """args_validator was removed — input guardrails cover this use case."""

    def test_no_args_validator_attribute(self) -> None:
        tool = FunctionTool(
            name="t",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
        )
        assert not hasattr(tool, "args_validator")

"""Tests that tool guardrail dispatch records audit entries on the run context."""

from __future__ import annotations

import contextlib
from typing import Any

from troopai.adk.agents.agent import Agent
from troopai.adk.audit.event import hash_payload
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.config import RunConfig
from troopai.adk.run.context import RunContext
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
from troopai.adk.types.guardrails import GuardrailAction
from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall


async def _echo(_ctx: Any, _raw_args: Any) -> str:
    return "tool-result"


def _agent(tool: FunctionTool) -> Agent:
    return Agent(name="a", system_prompt="t", tools=[tool])


def _call() -> LLMResponseFunctionToolCall:
    return LLMResponseFunctionToolCall(call_id="c1", name="t", arguments="{}")


def _tool(
    *,
    input_g: list | None = None,
    output_g: list | None = None,
) -> FunctionTool:
    return FunctionTool(
        name="t",
        description="d",
        schema={"type": "object", "properties": {}},
        on_invoke=_echo,
        guardrails=ToolGuardrails(input=input_g or [], output=output_g or []),
    )


@tool_input_guardrail()
async def _allow_in(_data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    return ToolGuardrailFunctionOutput.allow()


@tool_input_guardrail()
async def _reject_in(_data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    return ToolGuardrailFunctionOutput.reject_content("blocked input")


@tool_input_guardrail()
async def _raise_in(_data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    return ToolGuardrailFunctionOutput.raise_exception()


@tool_output_guardrail()
async def _reject_out(_data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    return ToolGuardrailFunctionOutput.reject_content("blocked output")


async def _run(tool: FunctionTool, ctx: RunContext) -> None:
    await execute_tool_calls(
        agent=_agent(tool),
        tool_calls=[_call()],
        ctx_wrapper=ctx,
        hooks=RunHooks(),
        config=RunConfig(),
        model="gpt-4o-mini",
    )


class TestToolGuardrailAudit:
    async def test_input_allow_records_tool_input_pass(self) -> None:
        ctx = RunContext(context=None)
        await _run(_tool(input_g=[_allow_in]), ctx)
        records = [r for r in ctx.collect_guardrail_audit() if r.level == "tool_input"]
        assert len(records) == 1
        assert records[0].action is GuardrailAction.PASS
        assert records[0].transformed_hash is None

    async def test_input_reject_content_is_transform_with_distinct_hashes(self) -> None:
        ctx = RunContext(context=None)
        await _run(_tool(input_g=[_reject_in]), ctx)
        record = next(r for r in ctx.collect_guardrail_audit() if r.level == "tool_input")
        assert record.action is GuardrailAction.TRANSFORM
        assert record.transformed_hash == hash_payload("blocked input")
        assert record.output_hash is not None
        assert record.output_hash != record.transformed_hash
        assert "blocked input" not in (record.transformed_hash or "")

    async def test_input_raise_records_tool_input_raise(self) -> None:
        ctx = RunContext(context=None)
        with contextlib.suppress(Exception):
            await _run(_tool(input_g=[_raise_in]), ctx)
        record = next(r for r in ctx.collect_guardrail_audit() if r.level == "tool_input")
        assert record.action is GuardrailAction.RAISE
        assert record.transformed_hash is None

    async def test_output_reject_content_records_tool_output_transform(self) -> None:
        ctx = RunContext(context=None)
        await _run(_tool(output_g=[_reject_out]), ctx)
        record = next(r for r in ctx.collect_guardrail_audit() if r.level == "tool_output")
        assert record.action is GuardrailAction.TRANSFORM
        assert record.transformed_hash == hash_payload("blocked output")
        assert record.output_hash != record.transformed_hash

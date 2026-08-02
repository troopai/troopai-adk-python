from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, override

import pytest

from troopai.adk.agents import Agent
from troopai.adk.exceptions import UsageLimitExceeded
from troopai.adk.llms.llm import LLM
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.llms.routing import LLMRouter, RoutedModel, RoutingContext
from troopai.adk.run import RunHooks, Runner
from troopai.adk.run.config import RunConfig
from troopai.adk.run.context import RunContext
from troopai.adk.run.llm_calls import call_llm_with_routing
from troopai.adk.run.tools_executor import execute_tool_calls
from troopai.adk.schemas import AgentOutputSchemaBase
from troopai.adk.tools import Tool
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.input import LLMInputContentItem
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
    LLMResponseText,
    LLMStreamEvent,
)
from troopai.adk.types.tokens.llm_usage import LLMUsage, LLMUsageLimits


class _CountingLLM(LLM):
    def __init__(self) -> None:
        self.calls = 0

    async def acomplete(  # type: ignore[override]
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
        self.calls += 1
        return LLMResponse(
            response_id="resp-1",
            model="counting",
            response=[LLMResponseText(text="done")],
        )


class _UsageLLM(LLM):
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    async def acomplete(  # type: ignore[override]
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
        self.calls += 1
        return LLMResponse(
            response_id=f"resp-{self.content}",
            model=self.content,
            response=[LLMResponseText(text=self.content)],
            usage=LLMUsage(requests=1, input_tokens=7, output_tokens=3, total_tokens=10),
        )


class _EscalatingRouter(LLMRouter):
    def __init__(self, candidates: Sequence[RoutedModel]) -> None:
        self._candidates = list(candidates)

    @override
    def candidates(self, ctx: RoutingContext) -> Sequence[RoutedModel]:
        del ctx
        return self._candidates

    @override
    def should_escalate(self, response: LLMResponse | None) -> bool:
        return response is not None and response.content == "bad"


def _tool_call(call_id: str, name: str) -> LLMResponseFunctionToolCall:
    return LLMResponseFunctionToolCall(call_id=call_id, name=name, arguments="{}")


def _tool(name: str, calls: list[str]) -> FunctionTool:
    async def invoke(_ctx: Any, _raw_args: str) -> str:
        calls.append(name)
        return "ok"

    return FunctionTool(name=name, schema={"type": "object", "properties": {}}, on_invoke=invoke)


def _failing_tool(name: str, calls: list[str]) -> FunctionTool:
    async def invoke(_ctx: Any, _raw_args: str) -> str:
        calls.append(name)
        raise RuntimeError("tool failed")

    return FunctionTool(name=name, schema={"type": "object", "properties": {}}, on_invoke=invoke)


async def test_request_limit_is_checked_before_model_call() -> None:
    llm = _CountingLLM()
    agent = Agent(name="limited", system_prompt="test", llm=llm)
    config = RunConfig(usage_limits=LLMUsageLimits(request_limit=0))

    with pytest.raises(UsageLimitExceeded, match="Request limit exceeded"):
        await Runner.arun(agent, "hello", run_config=config)

    assert llm.calls == 0


async def test_tool_calls_limit_is_checked_before_dispatch() -> None:
    calls: list[str] = []
    agent = Agent(name="limited", system_prompt="test", tools=[_tool("work", calls)])
    context = RunContext(context=None)
    context.usage.tool_calls = 1
    config = RunConfig(usage_limits=LLMUsageLimits(tool_calls_limit=1))

    with pytest.raises(UsageLimitExceeded, match="Tool call limit exceeded"):
        await execute_tool_calls(
            agent=agent,
            tool_calls=[_tool_call("call-1", "work")],
            ctx_wrapper=context,
            hooks=RunHooks(),
            config=config,
        )

    assert calls == []


async def test_max_tool_calls_per_turn_is_checked_before_parallel_dispatch() -> None:
    calls: list[str] = []
    agent = Agent(
        name="limited",
        system_prompt="test",
        tools=[_tool("one", calls), _tool("two", calls)],
    )
    config = RunConfig(max_tool_calls_per_turn=1)

    with pytest.raises(UsageLimitExceeded, match="Tool calls per turn exceeded"):
        await execute_tool_calls(
            agent=agent,
            tool_calls=[_tool_call("call-1", "one"), _tool_call("call-2", "two")],
            ctx_wrapper=RunContext(context=None),
            hooks=RunHooks(),
            config=config,
            parallel=True,
        )

    assert calls == []


async def test_tool_call_usage_tracks_executed_calls() -> None:
    calls: list[str] = []
    agent = Agent(
        name="limited",
        system_prompt="test",
        tools=[_tool("one", calls), _tool("two", calls)],
    )
    context = RunContext(context=None)

    results, deferred = await execute_tool_calls(
        agent=agent,
        tool_calls=[_tool_call("call-1", "one"), _tool_call("call-2", "two")],
        ctx_wrapper=context,
        hooks=RunHooks(),
        config=RunConfig(),
    )

    assert deferred is None
    assert [result.output for result in results] == ["ok", "ok"]
    assert calls == ["one", "two"]
    assert context.usage.tool_calls == 2


async def test_failed_tool_call_still_counts_as_dispatched() -> None:
    calls: list[str] = []
    agent = Agent(name="limited", system_prompt="test", tools=[_failing_tool("boom", calls)])
    context = RunContext(context=None)

    with pytest.raises(RuntimeError, match="tool failed"):
        await execute_tool_calls(
            agent=agent,
            tool_calls=[_tool_call("call-1", "boom")],
            ctx_wrapper=context,
            hooks=RunHooks(),
            config=RunConfig(fail_on_tool_error=True),
        )

    assert calls == ["boom"]
    assert context.usage.tool_calls == 1


async def test_rejected_routed_candidate_counts_usage_before_fallback() -> None:
    rejected = _UsageLLM("bad")
    fallback = _UsageLLM("good")
    router = _EscalatingRouter(
        [
            RoutedModel(llm=rejected, model="bad"),
            RoutedModel(llm=fallback, model="good"),
        ]
    )
    context = RunContext(context=None)
    config = RunConfig(router=router, usage_limits=LLMUsageLimits(request_limit=1))

    with pytest.raises(UsageLimitExceeded, match="Request limit exceeded"):
        await call_llm_with_routing(
            router,
            Agent(name="limited", system_prompt="test"),
            [],
            config,
            RunHooks(),
            context=context,
        )

    assert rejected.calls == 1
    assert fallback.calls == 0
    assert context.usage.requests == 1

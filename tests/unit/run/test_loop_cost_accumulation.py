"""Tests that the agent loop accumulates per-call cost onto ``RunContext.cost_usd``
and stamps it on ``GenerationSpanData.cost_usd``.

Uses :func:`run_agent_loop` directly (same pattern as
``test_reset_tool_choice.py``), patching ``call_llm`` to return a response
with usage and patching ``resolve_llm`` to return a fake ``LLM`` subclass
whose ``cost()`` returns a fixed value.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, override
from unittest.mock import patch

import pytest

from troopai.adk.llms.llm import LLM
from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText
from troopai.adk.types.tokens.llm_usage import LLMUsage


class _FixedCostLLM(LLM):
    """Minimal LLM stub that returns a fixed cost from ``cost()``."""

    def __init__(self, cost_per_call: float | None) -> None:
        self._cost_per_call = cost_per_call

    async def acomplete(  # type: ignore[override]  # stub: loop.call_llm is patched; acomplete never called
        self,
        messages: Any,
        llm_config: Any = None,
        tools: Any = None,
        output_schema: Any = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[Any]:
        # Not called in these tests — loop.call_llm is patched
        raise NotImplementedError("acomplete should not be called in this test")

    @override
    def cost(self, model: str, usage: LLMUsage) -> float | None:
        del model, usage
        return self._cost_per_call


def _make_response_with_usage() -> LLMResponse:
    """Single-turn text response carrying token usage."""
    return LLMResponse(
        response_id="resp-cost-test",
        model="fake",
        response=[LLMResponseText(text="done")],
        usage=LLMUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _make_agent() -> Any:
    """Minimal agent-like object for loop testing (mirrors test_reset_tool_choice.py)."""
    from types import SimpleNamespace

    from troopai.adk.agents.agent_guardrails import AgentGuardrails
    from troopai.adk.agents.middleware import Middleware
    from troopai.adk.skills.activation import SkillActivation

    return SimpleNamespace(
        name="cost-test-agent",
        tools=[],
        llm_config=None,
        tool_use_behavior="run_llm_again",
        handoffs=None,
        llm=None,
        output_schema=None,
        guardrails=AgentGuardrails(),
        system_prompt="You are a cost-test agent.",
        skills=[],
        skill_activation=SkillActivation.EAGER,
        hooks=None,
        middleware=Middleware(),
    )


@pytest.mark.asyncio
async def test_loop_accumulates_cost_onto_run_context() -> None:
    """A single-turn response with fixed cost → ``context.cost_usd`` equals that cost."""
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.run.config import DEFAULT_RUN_CONFIG
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.loop import run_agent_loop

    agent = _make_agent()
    fake_llm = _FixedCostLLM(cost_per_call=0.01)

    async def fake_call_llm(*args: Any, **kwargs: Any) -> LLMResponse:
        return _make_response_with_usage()

    ctx = RunContext(context=None)

    with (
        patch("troopai.adk.run.loop.call_llm", side_effect=fake_call_llm),
        patch("troopai.adk.run.loop.resolve_llm", return_value=fake_llm),
    ):
        result = await run_agent_loop(
            agent=agent,
            user_prompt="ping",
            context=ctx,
            ctx_wrapper=ctx,
            hooks=RunHooks(),
            max_turns=5,
            config=DEFAULT_RUN_CONFIG,
        )

    assert result.final_output == "done"
    assert result.context.cost_usd == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_loop_cost_none_does_not_accumulate() -> None:
    """When ``LLM.cost()`` returns ``None``, ``cost_usd`` stays ``0.0``."""
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.run.config import DEFAULT_RUN_CONFIG
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.loop import run_agent_loop

    agent = _make_agent()
    null_cost_llm = _FixedCostLLM(cost_per_call=None)

    async def fake_call_llm(*args: Any, **kwargs: Any) -> LLMResponse:
        return _make_response_with_usage()

    ctx = RunContext(context=None)

    with (
        patch("troopai.adk.run.loop.call_llm", side_effect=fake_call_llm),
        patch("troopai.adk.run.loop.resolve_llm", return_value=null_cost_llm),
    ):
        result = await run_agent_loop(
            agent=agent,
            user_prompt="ping",
            context=ctx,
            ctx_wrapper=ctx,
            hooks=RunHooks(),
            max_turns=5,
            config=DEFAULT_RUN_CONFIG,
        )

    assert result.final_output == "done"
    assert result.context.cost_usd == 0.0


@pytest.mark.asyncio
async def test_loop_accumulates_cost_across_turns() -> None:
    """Two turns each with cost 0.01 → total cost 0.02."""
    from unittest.mock import AsyncMock

    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.run.config import DEFAULT_RUN_CONFIG
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.loop import run_agent_loop
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

    tool = FunctionTool(
        name="echo",
        description="Echo",
        schema={"type": "object", "properties": {}},
        on_invoke=AsyncMock(return_value="echoed"),
    )

    from types import SimpleNamespace

    from troopai.adk.agents.agent_guardrails import AgentGuardrails
    from troopai.adk.agents.middleware import Middleware
    from troopai.adk.skills.activation import SkillActivation

    agent = SimpleNamespace(
        name="multi-turn-cost-agent",
        tools=[tool],
        llm_config=None,
        tool_use_behavior="run_llm_again",
        handoffs=None,
        llm=None,
        output_schema=None,
        guardrails=AgentGuardrails(),
        system_prompt="You are a test agent.",
        skills=[],
        skill_activation=SkillActivation.EAGER,
        hooks=None,
        middleware=Middleware(),
    )

    fake_llm = _FixedCostLLM(cost_per_call=0.01)
    call_count = 0

    async def fake_call_llm(*args: Any, **kwargs: Any) -> LLMResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                response_id="resp-1",
                model="fake",
                response=[
                    LLMResponseFunctionToolCall(
                        call_id="call_1",
                        name="echo",
                        arguments="{}",
                    )
                ],
                usage=LLMUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
            )
        return LLMResponse(
            response_id="resp-2",
            model="fake",
            response=[LLMResponseText(text="done")],
            usage=LLMUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
        )

    ctx = RunContext(context=None)

    with (
        patch("troopai.adk.run.loop.call_llm", side_effect=fake_call_llm),
        patch("troopai.adk.run.loop.resolve_llm", return_value=fake_llm),
    ):
        result = await run_agent_loop(
            agent=agent,  # type: ignore[arg-type]  # SimpleNamespace duck-types Agent in loop tests
            user_prompt="ping",
            context=ctx,
            ctx_wrapper=ctx,
            hooks=RunHooks(),
            max_turns=5,
            config=DEFAULT_RUN_CONFIG,
        )

    assert call_count == 2
    assert result.context.cost_usd == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_streaming_loop_accumulates_cost_onto_run_context() -> None:
    """Streaming path: single-turn response with fixed cost → ``context.cost_usd`` equals that cost.

    Drives ``run_agent_block_streamed`` via ``Runner.arun(stream=True)``,
    patching ``call_llm_streamed`` and ``resolve_llm`` in the same way as
    the non-streaming tests. The stream is drained inside the patch block
    so the background task's invocations hit the mocks.
    """
    from unittest.mock import AsyncMock, patch

    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.runner import Runner

    agent = Agent(
        name="streaming-cost-test-agent",
        system_prompt="You are a streaming cost-test agent.",
    )
    fake_llm = _FixedCostLLM(cost_per_call=0.01)

    async def fake_call_llm_streamed(*args: Any, **kwargs: Any) -> Any:
        return _make_response_with_usage()

    with (
        patch(
            "troopai.adk.run.loop.call_llm_streamed",
            new=AsyncMock(side_effect=fake_call_llm_streamed),
        ),
        patch("troopai.adk.run.loop.resolve_llm", return_value=fake_llm),
        patch(
            "troopai.adk.run.runner.run_blocking_input_guardrails",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "troopai.adk.run.runner.run_parallel_input_guardrails",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "troopai.adk.run.runner.run_output_guardrails",
            new=AsyncMock(return_value=[]),
        ),
    ):
        streaming = await Runner.arun(agent, "ping", max_turns=5, stream=True)
        async for _ in streaming.stream_events():
            pass

    assert streaming.final_output == "done"
    assert streaming.context is not None
    assert streaming.context.cost_usd == pytest.approx(0.01)


class _ModelCapturingLLM(LLM):
    """LLM stub that records which model name was passed to ``cost()``."""

    def __init__(self) -> None:
        self.cost_model_calls: list[str] = []

    async def acomplete(  # type: ignore[override]
        self,
        messages: Any,
        llm_config: Any = None,
        tools: Any = None,
        output_schema: Any = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[Any]:
        raise NotImplementedError("acomplete should not be called in this test")

    @override
    def cost(self, model: str, usage: LLMUsage) -> float | None:
        self.cost_model_calls.append(model)
        return 0.001


@pytest.mark.asyncio
async def test_non_streaming_loop_uses_response_model_for_cost_on_fallback() -> None:
    """Non-streaming path: when response.model differs from the configured model
    (litellm fallback scenario), ``llm.cost()`` must be called with the actual
    serving model, not the originally configured one.

    Regression test for the bug where the else-branch in ``run_agent_loop`` never
    updated ``llm_model_name`` from ``response.model`` after ``call_llm`` returned,
    causing cost to be computed against the wrong model's price table.
    """
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.run.config import DEFAULT_RUN_CONFIG
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.loop import run_agent_loop

    agent = _make_agent()
    capturing_llm = _ModelCapturingLLM()

    # The configured model is "gpt-4o" but the fallback served "gpt-4o-mini"
    configured_model = "gpt-4o"
    actual_serving_model = "gpt-4o-mini"

    async def fake_call_llm(*args: Any, **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            response_id="resp-fallback",
            model=actual_serving_model,  # differs from configured model
            response=[LLMResponseText(text="done")],
            usage=LLMUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
        )

    ctx = RunContext(context=None)

    with (
        patch("troopai.adk.run.loop.call_llm", side_effect=fake_call_llm),
        patch("troopai.adk.run.loop.resolve_llm", return_value=capturing_llm),
        patch(
            "troopai.adk.run.loop.resolve_model_name",
            return_value=configured_model,
        ),
    ):
        result = await run_agent_loop(
            agent=agent,
            user_prompt="ping",
            context=ctx,
            ctx_wrapper=ctx,
            hooks=RunHooks(),
            max_turns=5,
            config=DEFAULT_RUN_CONFIG,
        )

    assert result.final_output == "done"
    # cost() must be called with the actual serving model, not the configured one
    assert capturing_llm.cost_model_calls, "cost() was never called"
    assert capturing_llm.cost_model_calls[0] == actual_serving_model, (
        f"Expected cost() to be called with '{actual_serving_model}' "
        f"(actual serving model after fallback), got '{capturing_llm.cost_model_calls[0]}'"
    )


@pytest.mark.asyncio
async def test_streaming_loop_uses_response_model_for_cost_on_fallback() -> None:
    """Streaming path: when final_response.model differs from the configured model
    (litellm fallback scenario), ``llm.cost()`` must be called with the actual
    serving model, not the originally configured one.

    Regression test for the streaming else-branch in ``run_agent_block_streamed``
    which mirrored the same omission as the non-streaming path.
    """
    from unittest.mock import AsyncMock

    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.runner import Runner

    agent = Agent(
        name="streaming-fallback-cost-test-agent",
        system_prompt="You are a streaming cost-test agent.",
    )
    capturing_llm = _ModelCapturingLLM()

    configured_model = "gpt-4o"
    actual_serving_model = "gpt-4o-mini"

    async def fake_call_llm_streamed(*args: Any, **kwargs: Any) -> Any:
        return LLMResponse(
            response_id="resp-stream-fallback",
            model=actual_serving_model,  # differs from configured model
            response=[LLMResponseText(text="done")],
            usage=LLMUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
        )

    with (
        patch(
            "troopai.adk.run.loop.call_llm_streamed",
            new=AsyncMock(side_effect=fake_call_llm_streamed),
        ),
        patch("troopai.adk.run.loop.resolve_llm", return_value=capturing_llm),
        patch(
            "troopai.adk.run.loop.resolve_model_name",
            return_value=configured_model,
        ),
        patch(
            "troopai.adk.run.runner.run_blocking_input_guardrails",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "troopai.adk.run.runner.run_parallel_input_guardrails",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "troopai.adk.run.runner.run_output_guardrails",
            new=AsyncMock(return_value=[]),
        ),
    ):
        streaming = await Runner.arun(agent, "ping", max_turns=5, stream=True)
        async for _ in streaming.stream_events():
            pass

    assert streaming.final_output == "done"
    assert streaming.context is not None
    # cost() must be called with the actual serving model, not the configured one
    assert capturing_llm.cost_model_calls, "cost() was never called"
    assert capturing_llm.cost_model_calls[0] == actual_serving_model, (
        f"Expected cost() to be called with '{actual_serving_model}' "
        f"(actual serving model after fallback), got '{capturing_llm.cost_model_calls[0]}'"
    )

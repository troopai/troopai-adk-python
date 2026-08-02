"""Zero-span contract when ``RunConfig.tracing_enabled=False``.

Even if a user has installed a real recording tracer via
:func:`set_tracer`, a run with ``tracing_enabled=False`` MUST emit
exactly zero spans. The per-call-site
``disabled=not config.tracing_enabled`` kwarg threaded through every
instrumentation point is the mechanism.

The assertion matters because the default :class:`RunConfig` ships
with ``tracing_enabled=False``: any framework code path that stops
respecting the flag would silently start paying to build span-data
payloads on users' hot paths.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tracing import Span, set_tracer
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
    LLMResponseText,
)
from troopai.adk.types.tracing import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SpanData,
)


class _RecordingTracer:
    """Counts every factory call; failing this test means a hot path
    built a :class:`SpanData` and reached the tracer despite the flag."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, SpanData]] = []

    def _record(self, kind: str, data: SpanData) -> Span[Any]:
        self.spans.append((kind, data))
        return Span(data)

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        return self._record("agent", data)

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        return self._record("function", data)

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        return self._record("generation", data)

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        return self._record("response", data)

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        return self._record("handoff", data)

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        return self._record("guardrail", data)

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        return self._record("custom", data)


def _make_agent() -> Agent:
    async def _echo_invoker(_ctx: Any, _raw_args: str) -> str:
        return "echoed"

    echo = FunctionTool(
        name="echo",
        description="Echo back the input.",
        schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        on_invoke=_echo_invoker,
    )
    return Agent(
        name="disabled-trace-agent",
        system_prompt="You are a tracing-disabled-test agent.",
        tools=[echo],
    )


def _tool_call_response() -> LLMResponse:
    return LLMResponse(
        response_id="resp-1",
        model="fake",
        response=[
            LLMResponseFunctionToolCall(
                call_id="call_1",
                name="echo",
                arguments='{"value": "hello"}',
            )
        ],
    )


def _final_text_response() -> LLMResponse:
    return LLMResponse(
        response_id="resp-2",
        model="fake",
        response=[LLMResponseText(text="done")],
    )


@pytest.fixture(autouse=True)
def _reset_tracer() -> Any:
    yield
    set_tracer(None)


@pytest.mark.asyncio
async def test_disabled_tracing_emits_zero_spans_even_with_real_tracer() -> None:
    """A real tracer is installed; the flag is off; no spans must escape.

    If this fails, some framework call site is building a SpanData and
    dispatching to the tracer without threading through the
    ``disabled=not config.tracing_enabled`` kwarg. Treat it as a bug:
    framework code must not pay the span-construction cost for users
    who have not opted in.
    """
    tracer = _RecordingTracer()
    set_tracer(tracer)

    call_count = {"n": 0}

    async def fake_call_llm(*_args: Any, **_kwargs: Any) -> LLMResponse:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _tool_call_response()
        return _final_text_response()

    agent = _make_agent()
    config = RunConfig(tracing_enabled=False)

    with (
        patch(
            "troopai.adk.run.loop.call_llm",
            new=AsyncMock(side_effect=fake_call_llm),
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
        await Runner.arun(agent, "go", max_turns=5, run_config=config)

    assert len(tracer.spans) == 0, f"Expected zero spans when tracing_enabled=False; got {[k for k, _ in tracer.spans]}"


@pytest.mark.asyncio
async def test_default_runconfig_emits_zero_spans() -> None:
    """The *default* RunConfig ships with tracing off. Same contract —
    regression guard against a future default flip."""
    tracer = _RecordingTracer()
    set_tracer(tracer)

    async def fake_call_llm(*_args: Any, **_kwargs: Any) -> LLMResponse:
        return _final_text_response()

    agent = _make_agent()
    config = RunConfig()  # no explicit tracing_enabled

    with (
        patch(
            "troopai.adk.run.loop.call_llm",
            new=AsyncMock(side_effect=fake_call_llm),
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
        await Runner.arun(agent, "go", max_turns=3, run_config=config)

    assert len(tracer.spans) == 0

"""End-to-end tracing assertions for :meth:`Runner.arun`.

Opts in via ``RunConfig(tracing_enabled=True)`` with an in-memory
recording tracer, then asserts the emitted span tree shape:

* one outer :class:`AgentSpanData` per run,
* one :class:`GenerationSpanData` per LLM turn,
* one :class:`FunctionSpanData` per tool call.

The loop is driven by a patched :func:`call_llm` that returns a tool
call on turn 1 and a final text message on turn 2 — enough to hit
both the generation and tool span paths without touching a real LLM.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tracing import (
    Span,
    set_tracer,
)
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
    """Minimal in-memory tracer that logs every span created.

    Holds onto the :class:`SpanData` payload for shape assertions and
    returns real :class:`Span` instances so the
    :class:`~contextvars.ContextVar` parent chain works end-to-end.
    """

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
        name="tracing-agent",
        system_prompt="You are a tracing-test agent.",
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


async def _run_with_tracing(*, tracing_enabled: bool) -> _RecordingTracer:
    """Run ``Runner.arun`` with a mock LLM that returns tool-call then text.

    Returns the installed :class:`_RecordingTracer` so callers can
    inspect the emitted span list.
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
    config = RunConfig(tracing_enabled=tracing_enabled)

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

    return tracer


@pytest.fixture(autouse=True)
def _reset_tracer() -> Any:
    yield
    set_tracer(None)


@pytest.mark.asyncio
async def test_span_tree_shape_with_tracing_enabled() -> None:
    """A simple two-turn run must emit: 1 agent + 2 generation + 1 function."""
    tracer = await _run_with_tracing(tracing_enabled=True)

    kinds = [k for k, _ in tracer.spans]
    assert kinds.count("agent") == 1, f"expected 1 agent span, got kinds={kinds}"
    assert kinds.count("generation") == 2, f"expected 2 generation spans (one per turn), got kinds={kinds}"
    assert kinds.count("function") == 1, f"expected 1 function span (echo tool), got kinds={kinds}"


@pytest.mark.asyncio
async def test_agent_span_captures_agent_metadata() -> None:
    tracer = await _run_with_tracing(tracing_enabled=True)
    agent_spans = [d for k, d in tracer.spans if k == "agent"]
    assert len(agent_spans) == 1
    agent_data = agent_spans[0]
    assert isinstance(agent_data, AgentSpanData)
    assert agent_data.name == "tracing-agent"
    assert agent_data.tools == ["echo"]


@pytest.mark.asyncio
async def test_function_span_records_tool_name() -> None:
    tracer = await _run_with_tracing(tracing_enabled=True)
    function_spans = [d for k, d in tracer.spans if k == "function"]
    assert len(function_spans) == 1
    fn_data = function_spans[0]
    assert isinstance(fn_data, FunctionSpanData)
    assert fn_data.name == "echo"


@pytest.mark.asyncio
async def test_tracing_metadata_flows_to_agent_span() -> None:
    """``RunConfig.tracing_metadata`` must land on the root agent span.
    This is the only way user-supplied metadata travels to the tracer."""
    tracer = _RecordingTracer()
    set_tracer(tracer)

    agent = _make_agent()
    config = RunConfig(
        tracing_enabled=True,
        tracing_metadata={"tenant": "acme", "request_id": "req_42"},
    )

    async def fake_call_llm(*_args: Any, **_kwargs: Any) -> LLMResponse:
        return _final_text_response()

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

    agent_spans = [d for k, d in tracer.spans if k == "agent"]
    assert len(agent_spans) == 1
    assert isinstance(agent_spans[0], AgentSpanData)
    assert agent_spans[0].metadata == {"tenant": "acme", "request_id": "req_42"}

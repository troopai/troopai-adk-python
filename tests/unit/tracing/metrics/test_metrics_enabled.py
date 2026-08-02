"""Tests for RunConfig.metrics_enabled and the span-gate widening.

(A) Gate-expression + factory unit tests — verify that
    ``RunConfig(metrics_enabled=True, tracing_enabled=False)`` produces a
    live span that a :class:`MetricsTracer` records instruments for.

(B) Integration test — run a real :class:`Runner` with a mocked LLM and
    confirm the generation + agent-turn metric instruments fire when
    ``metrics_enabled=True`` and ``tracing_enabled=False``.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from troopai.adk.agents.agent import Agent
from troopai.adk.run.config import RunConfig
from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.metrics.instruments import Instruments
from troopai.adk.tracing.metrics.tracer import MetricsTracer
from troopai.adk.tracing.spans import generation_span
from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText
from troopai.adk.types.tokens.llm_usage import LLMUsage

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_reader_and_tracer() -> tuple[InMemoryMetricReader, MetricsTracer]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    return reader, MetricsTracer(Instruments(provider.get_meter("test")))


def _metric_names(reader: InMemoryMetricReader) -> set[str]:
    data = reader.get_metrics_data()
    if data is None:
        return set()
    return {metric.name for rm in data.resource_metrics for sm in rm.scope_metrics for metric in sm.metrics}


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_tracer():
    yield
    set_tracer(None)


# ── (A) Gate-expression + factory unit tests ──────────────────────────────────


def test_metrics_enabled_records_without_span_export():
    """generation_span with metrics_enabled=True records instruments even when
    tracing_enabled=False, because the gate evaluates to ``disabled=False``."""
    reader, tracer = _make_reader_and_tracer()
    set_tracer(tracer)

    config = RunConfig(metrics_enabled=True, tracing_enabled=False)
    with generation_span(
        model="gpt",
        disabled=not (config.tracing_enabled or config.metrics_enabled),
    ) as span:
        span.data = dataclasses.replace(span.data, usage={"input_tokens": 6, "output_tokens": 1})

    names = _metric_names(reader)
    assert "troopai.llm.tokens.prompt" in names, f"expected prompt instrument, got {names}"
    assert "troopai.llm.tokens.completion" in names, f"expected completion instrument, got {names}"


def test_both_disabled_is_noop():
    """When both flags are False (defaults) the gate evaluates to
    ``disabled=True`` and the span is a NoOpSpan — no tracer is consulted."""
    reader, tracer = _make_reader_and_tracer()
    set_tracer(tracer)

    config = RunConfig()
    assert not config.tracing_enabled
    assert not config.metrics_enabled

    with generation_span(
        model="gpt",
        disabled=not (config.tracing_enabled or config.metrics_enabled),
    ):
        pass

    # No instruments should have fired
    assert len(_metric_names(reader)) == 0


def test_tracing_enabled_alone_records():
    """Existing behaviour: tracing_enabled=True alone keeps the span live."""
    reader, tracer = _make_reader_and_tracer()
    set_tracer(tracer)

    config = RunConfig(tracing_enabled=True, metrics_enabled=False)
    with generation_span(
        model="gpt-4o",
        disabled=not (config.tracing_enabled or config.metrics_enabled),
    ) as span:
        span.data = dataclasses.replace(span.data, usage={"prompt_tokens": 3, "completion_tokens": 1})

    names = _metric_names(reader)
    assert "troopai.llm.tokens.prompt" in names


# ── (B) Integration test — Runner seams honour metrics_enabled ───────────────


async def test_runner_metrics_enabled_fires_generation_instrument():
    """Run a real Runner with a mocked LLM; assert generation instruments fire
    when ``metrics_enabled=True`` and ``tracing_enabled=False``."""

    reader, tracer = _make_reader_and_tracer()
    set_tracer(tracer)

    agent = Agent(name="metrics-test-agent", system_prompt="You are a test agent.")

    def _text_response() -> LLMResponse:
        return LLMResponse(
            response_id="resp-metrics-1",
            model="fake",
            response=[LLMResponseText(text="hello")],
            usage=LLMUsage(input_tokens=10, output_tokens=3),
        )

    async def fake_call_llm(*_args: Any, **_kwargs: Any) -> LLMResponse:
        return _text_response()

    config = RunConfig(metrics_enabled=True, tracing_enabled=False)

    from troopai.adk.run.runner import Runner

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
        result = await Runner.arun(agent, "hi", max_turns=2, run_config=config)

    assert result.final_output == "hello"

    names = _metric_names(reader)
    # The generation seam in loop.py was widened — tokens must be recorded.
    assert "troopai.llm.tokens.prompt" in names, f"expected prompt instrument, got {names}"
    # The agent-turn seam in runner.py was widened — agent duration must fire.
    assert "troopai.agent.turn.duration_ms" in names, f"expected agent-turn instrument, got {names}"

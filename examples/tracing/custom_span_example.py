"""
Example: Custom Spans with ``custom_span``

Demonstrates the application-side tracing API:

- Installing a custom tracer via ``set_tracer``
- Instrumenting application code with ``custom_span``
- Recording structured error context with ``Span.set_error``

The example defines a tiny in-memory tracer that collects finished spans
into a list, so the script is self-contained (no OpenTelemetry backend
required). In production you would install a real tracer (OpenTelemetry,
Datadog, Honeycomb, etc.) instead.

Run:
    python examples/tracing/custom_span_example.py
"""

from __future__ import annotations

import logging
from typing import Any, override

from troopai.adk.tracing import (
    NoOpSpan,
    Span,
    custom_span,
    set_tracer,
)
from troopai.adk.types.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SpanData,
)

logger = logging.getLogger(__name__)


class RecordingSpan[TData: SpanData](Span[TData]):
    """Span that logs start/finish and retains its data payload.

    Generic in ``TData`` so the ``RecordingTracer`` factory methods can
    instantiate ``RecordingSpan(data)`` and have pyright infer the concrete
    ``Span[AgentSpanData]`` / ``Span[FunctionSpanData]`` / ... return type
    without per-method escape hatches.
    """

    def __init__(self, data: TData, span_id: str | None = None) -> None:
        super().__init__(data, span_id=span_id)

    @override
    def start(self) -> None:
        logger.info("span start: %s", type(self.data).__name__)

    @override
    def finish(self) -> None:
        super().finish()
        logger.info(
            "span finish: %s data=%s error=%s",
            type(self.data).__name__,
            self.data,
            self.error,
        )


class RecordingTracer:
    """Tiny tracer that routes all factories to ``RecordingSpan``."""

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        return RecordingSpan(data)

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        return RecordingSpan(data)

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        return RecordingSpan(data)

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        return RecordingSpan(data)

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        return RecordingSpan(data)

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        return RecordingSpan(data)

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        return RecordingSpan(data)


def rank(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda it: it.get("score", 0), reverse=True)


def main() -> None:
    set_tracer(RecordingTracer())

    results = [
        {"id": "a", "score": 0.3},
        {"id": "b", "score": 0.9},
        {"id": "c", "score": 0.6},
    ]

    # Success path — span records input size, body runs, finish logs payload.
    with custom_span("rank_search_results", data={"n": len(results)}) as span:
        ranked = rank(results)
        span.data.data["top_id"] = ranked[0]["id"]
        logger.info("ranked: %s", ranked)

    # Error path — record context with set_error without re-raising.
    with custom_span("risky_step", data={"attempt": 1}) as span:
        try:
            raise RuntimeError("simulated failure")
        except RuntimeError as e:
            span.set_error(str(e), data={"type": type(e).__name__})
            logger.info("handled error; continuing")

    # NoOp short-circuit — disabled=True bypasses the installed tracer.
    with custom_span("disabled_span", disabled=True) as span:
        assert isinstance(span, NoOpSpan)


if __name__ == "__main__":
    main()

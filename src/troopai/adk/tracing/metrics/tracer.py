"""A Tracer that records OTel metric instruments from SpanData at finish()."""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from typing import override

from troopai.adk.tracing.metrics.instruments import Instruments
from troopai.adk.tracing.spans import Span
from troopai.adk.types.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GraphNodeSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SpanData,
    SwarmTurnSpanData,
)

logger = logging.getLogger(__name__)


class MetricSpan[TData: SpanData](Span[TData]):
    """Records instruments at finish(); never exports a span.

    Reads ``self.data`` at finish so it observes the post-call rebind
    (the runner does ``span.data = dataclasses.replace(...)``). Computes
    its own wall-clock duration. ``on_finish`` is ``None`` for span kinds
    with no instrument.
    """

    def __init__(self, data: TData, on_finish: Callable[[TData, float, bool], None] | None) -> None:
        """Construct a metric-recording span.

        Args:
            data: Typed span-data payload.
            on_finish: Callable invoked at :meth:`finish` with
                ``(data, duration_ms, error)``. Pass ``None`` for span
                kinds that have no associated instrument.
        """
        super().__init__(data)
        self._on_finish = on_finish
        self._start_monotonic: float | None = None

    @override
    def start(self) -> None:
        self._start_monotonic = time.monotonic()

    @override
    def finish(self) -> None:
        if self._finished:
            return
        if self._on_finish is not None and self._start_monotonic is not None:
            duration_ms = (time.monotonic() - self._start_monotonic) * 1000.0
            try:
                self._on_finish(self.data, duration_ms, self.error is not None)
            except Exception:
                logger.exception("MetricSpan record failed (kind=%s); run unaffected", type(self.data).__name__)
        self._finished = True


class MetricsTracer:
    """Tracer that records metric instruments. Compose via MultiTracer."""

    def __init__(self, instruments: Instruments) -> None:
        """Construct a metrics tracer backed by the given instruments.

        Args:
            instruments: The :class:`~troopai.adk.tracing.metrics.instruments.Instruments`
                instance that owns the OTel histogram and counter handles.
        """
        self._inst = instruments

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        """Return a span that records agent turn duration at finish.

        Args:
            data: Span data for the agent turn.

        Returns:
            A :class:`MetricSpan` that records ``troopai.agent.turn.duration_ms``
            at finish.
        """
        return MetricSpan(data, self._record_agent)

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        """Return a span that records a tool call counter at finish.

        Args:
            data: Span data for the function tool call.

        Returns:
            A :class:`MetricSpan` that increments ``troopai.agent.tool.calls``
            at finish.
        """
        return MetricSpan(data, self._record_function)

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        """Return a span that records LLM token counts and request counter at finish.

        Args:
            data: Span data for the LLM generation turn.

        Returns:
            A :class:`MetricSpan` that records token histograms and the
            ``troopai.llm.requests`` counter at finish.
        """
        return MetricSpan(data, self._record_generation)

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        """Return a no-record span (no instrument for response spans).

        Args:
            data: Span data for the provider response.

        Returns:
            A :class:`MetricSpan` with no ``on_finish`` callback.
        """
        return MetricSpan(data, None)

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        """Return a no-record span (no instrument for handoff spans).

        Args:
            data: Span data for the handoff.

        Returns:
            A :class:`MetricSpan` with no ``on_finish`` callback.
        """
        return MetricSpan(data, None)

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        """Return a no-record span (no instrument for guardrail spans).

        Args:
            data: Span data for the guardrail evaluation.

        Returns:
            A :class:`MetricSpan` with no ``on_finish`` callback.
        """
        return MetricSpan(data, None)

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        """Route graph_node and swarm_turn custom spans to their instruments.

        Inspects ``data.data["type"]``: ``"graph_node"`` records
        ``troopai.graph.node.duration_ms``; ``"swarm_turn"`` records
        ``troopai.swarm.turn.duration_ms``; all other types return a
        no-record span.

        Args:
            data: Span data for the custom span.

        Returns:
            A :class:`MetricSpan` bound to the appropriate instrument
            callback, or a no-record span when the type has no instrument.
        """
        inner = data.data.get("type")
        if inner == "graph_node":
            return MetricSpan(data, self._record_graph_node)
        if inner == "swarm_turn":
            return MetricSpan(data, self._record_swarm_turn)
        return MetricSpan(data, None)

    def _record_agent(self, data: AgentSpanData, duration_ms: float, _error: bool) -> None:
        self._inst.record_agent(data, duration_ms)

    def _record_function(self, data: FunctionSpanData, _dur: float, error: bool) -> None:
        self._inst.record_function(data, error)

    def _record_generation(self, data: GenerationSpanData, _dur: float, error: bool) -> None:
        self._inst.record_generation(data, error)

    def _record_graph_node(self, data: CustomSpanData, duration_ms: float, _error: bool) -> None:
        known = {f.name for f in dataclasses.fields(GraphNodeSpanData)}
        payload = {k: v for k, v in data.data.items() if k != "type" and k in known}
        self._inst.record_graph_node(GraphNodeSpanData(**payload), duration_ms)

    def _record_swarm_turn(self, data: CustomSpanData, duration_ms: float, _error: bool) -> None:
        known = {f.name for f in dataclasses.fields(SwarmTurnSpanData)}
        payload = {k: v for k, v in data.data.items() if k != "type" and k in known}
        self._inst.record_swarm_turn(SwarmTurnSpanData(**payload), duration_ms)

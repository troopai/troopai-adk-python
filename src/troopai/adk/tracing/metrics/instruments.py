"""OTel metric instruments recorded from typed SpanData."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from troopai.adk.types.tracing.span_data import (
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GraphNodeSpanData,
    SwarmTurnSpanData,
)

if TYPE_CHECKING:
    from opentelemetry.metrics import Meter

logger = logging.getLogger(__name__)


class Instruments:
    """Owns the OTel instruments and the SpanData -> instrument recording.

    Creates all framework histograms and counters against the supplied
    ``Meter`` on construction, then exposes typed ``record_*`` methods
    that :class:`~troopai.adk.tracing.metrics.tracer.MetricsTracer` calls
    at :meth:`~troopai.adk.tracing.spans.Span.finish` time.
    """

    def __init__(self, meter: Meter) -> None:
        """Create framework instruments against the given OTel meter.

        Args:
            meter: An ``opentelemetry.metrics.Meter`` obtained from the
                active ``MeterProvider``.
        """
        self._agent_turn_duration = meter.create_histogram(
            "troopai.agent.turn.duration_ms",
            unit="ms",
            description="Agent turn wall-clock duration.",
        )
        self._llm_tokens_prompt = meter.create_histogram(
            "troopai.llm.tokens.prompt",
            unit="{token}",
            description="Prompt tokens per LLM request.",
        )
        self._llm_tokens_completion = meter.create_histogram(
            "troopai.llm.tokens.completion",
            unit="{token}",
            description="Completion tokens per LLM request.",
        )
        self._llm_requests = meter.create_counter(
            "troopai.llm.requests",
            unit="1",
            description="LLM requests by model and status.",
        )
        self._llm_cost = meter.create_histogram(
            "troopai.llm.cost.usd",
            unit="{usd}",
            description="Actual USD cost per LLM request.",
        )
        self._tool_calls = meter.create_counter(
            "troopai.agent.tool.calls",
            unit="1",
            description="Tool calls by tool name and status.",
        )
        self._graph_node_duration = meter.create_histogram(
            "troopai.graph.node.duration_ms",
            unit="ms",
            description="Graph node wall-clock duration.",
        )
        self._swarm_turn_duration = meter.create_histogram(
            "troopai.swarm.turn.duration_ms",
            unit="ms",
            description="Swarm turn wall-clock duration.",
        )

    def record_agent(self, data: AgentSpanData, duration_ms: float) -> None:
        """Record a completed agent turn duration.

        Args:
            data: Span data for the agent turn.
            duration_ms: Wall-clock duration in milliseconds.
        """
        logger.debug("recording agent turn duration: agent=%s duration_ms=%s", data.name, duration_ms)
        self._agent_turn_duration.record(duration_ms, {"agent": data.name})

    def record_generation(self, data: GenerationSpanData, error: bool) -> None:
        """Record metrics for a completed LLM generation turn.

        Records prompt and completion token histograms (when usage is present),
        an LLM request counter, and — when ``data.cost_usd`` is not None —
        a ``troopai.llm.cost.usd`` histogram entry. Token key handling accepts
        both ``prompt_tokens``/``completion_tokens`` (OpenAI convention) and
        ``input_tokens``/``output_tokens`` (Anthropic convention).

        When ``data.tenant_id`` is not None a ``tenant`` dimension is added to
        all LLM instruments recorded by this call.

        Args:
            data: Span data for the LLM generation turn.
            error: Whether the request ended in an error.
        """
        model = data.model if data.model is not None else "unknown"
        logger.debug("recording generation: model=%s error=%s", model, error)
        dims: dict[str, str] = {"model": model}
        if data.tenant_id is not None:
            dims["tenant"] = data.tenant_id
        if data.usage is not None:
            prompt = data.usage.get("prompt_tokens")
            if prompt is None:
                prompt = data.usage.get("input_tokens")
            completion = data.usage.get("completion_tokens")
            if completion is None:
                completion = data.usage.get("output_tokens")
            if isinstance(prompt, int):
                self._llm_tokens_prompt.record(prompt, dims)
            if isinstance(completion, int):
                self._llm_tokens_completion.record(completion, dims)
        self._llm_requests.add(1, {**dims, "status": "error" if error else "success"})
        if data.cost_usd is not None:
            self._llm_cost.record(data.cost_usd, dims)

    def record_function(self, data: FunctionSpanData, error: bool) -> None:
        """Record a single tool-call invocation counter.

        Args:
            data: Span data for the function tool call.
            error: Whether the tool call ended in an error.
        """
        logger.debug("recording tool call: tool=%s error=%s", data.name, error)
        self._tool_calls.add(1, {"tool": data.name, "status": "error" if error else "success"})

    def record_graph_node(self, data: GraphNodeSpanData, duration_ms: float) -> None:
        """Record a graph node execution duration.

        Args:
            data: Span data for the graph node.
            duration_ms: Wall-clock duration in milliseconds.
        """
        status = data.status if data.status is not None else "unknown"
        logger.debug("recording graph node: node=%s status=%s duration_ms=%s", data.node_name, status, duration_ms)
        self._graph_node_duration.record(duration_ms, {"node": data.node_name, "status": status})

    def record_swarm_turn(self, data: SwarmTurnSpanData, duration_ms: float) -> None:
        """Record a swarm turn execution duration.

        Args:
            data: Span data for the swarm turn.
            duration_ms: Wall-clock duration in milliseconds.
        """
        status = data.status if data.status is not None else "unknown"
        logger.debug("recording swarm turn: member=%s status=%s duration_ms=%s", data.member, status, duration_ms)
        self._swarm_turn_duration.record(duration_ms, {"member": data.member, "status": status})

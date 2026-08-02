"""Tracer protocol and registry.

Defines the :class:`Tracer` protocol that observability backends
implement and a module-level registry with :func:`get_tracer` and
:func:`set_tracer`. The default tracer is :class:`NoOpTracer` — it
does nothing, costs nothing, and is active until an application
explicitly opts in.

Opt-in pattern::

    from troopai.adk.tracing import set_tracer

    set_tracer(MyBackendTracer())

Framework code (runner, LLM layer, tool executor) fetches the current
tracer with :func:`get_tracer` at call time and calls its ``span_*``
factory methods to create spans. When the tracer is a
:class:`NoOpTracer`, every factory returns a :class:`NoOpSpan`, so the
hot path is cost-free.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from troopai.adk.tracing.spans import Span
    from troopai.adk.types.tracing.span_data import (
        AgentSpanData,
        CustomSpanData,
        FunctionSpanData,
        GenerationSpanData,
        GuardrailSpanData,
        HandoffSpanData,
        ResponseSpanData,
    )

logger = logging.getLogger(__name__)


@runtime_checkable
class Flushable(Protocol):
    """Protocol for tracers that can synchronously drain pending spans.

    Implementing this protocol is optional.  Tracers that buffer spans
    (e.g. those backed by an OTel ``BatchSpanProcessor``) should implement
    :meth:`flush` to guarantee that all spans started before the call are
    delivered to the exporter before the method returns.
    """

    def flush(self) -> None:
        """Synchronously drain all pending spans to the exporter(s)."""
        ...


@runtime_checkable
class Tracer(Protocol):
    """Protocol for framework-level tracers.

    A tracer is a factory of :class:`Span` objects, one method per
    span kind. Implementations may record to any backend (OpenTelemetry,
    stdout, in-memory for tests, etc.).

    Every method returns a :class:`Span` that the caller uses as a
    context manager. The span's lifecycle is ``start → body → finish``,
    and its associated data dataclass is updated in-place during the
    body.
    """

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        """Start a span covering one agent turn."""
        ...

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        """Start a span covering one function-tool execution."""
        ...

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        """Start a span covering one LLM generation turn."""
        ...

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        """Start a span covering one provider-level LLM response."""
        ...

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        """Start a span covering an agent-to-agent handoff."""
        ...

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        """Start a span covering a guardrail evaluation."""
        ...

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        """Start a developer-authored custom span."""
        ...


class NoOpTracer:
    """Tracer that records nothing.

    The default tracer, active until an application opts in by calling
    :func:`set_tracer`. Every factory method returns a
    :class:`NoOpSpan`, so tracing adds zero overhead to the hot path.
    """

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        from troopai.adk.tracing.spans import NoOpSpan

        return NoOpSpan(data)

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        from troopai.adk.tracing.spans import NoOpSpan

        return NoOpSpan(data)

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        from troopai.adk.tracing.spans import NoOpSpan

        return NoOpSpan(data)

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        from troopai.adk.tracing.spans import NoOpSpan

        return NoOpSpan(data)

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        from troopai.adk.tracing.spans import NoOpSpan

        return NoOpSpan(data)

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        from troopai.adk.tracing.spans import NoOpSpan

        return NoOpSpan(data)

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        from troopai.adk.tracing.spans import NoOpSpan

        return NoOpSpan(data)


_current_tracer: Tracer = NoOpTracer()


def get_tracer() -> Tracer:
    """Return the current process-wide tracer.

    Returns:
        The tracer installed via :func:`set_tracer`, or a
        :class:`NoOpTracer` when tracing has not been opted into.
    """
    return _current_tracer


def set_tracer(tracer: Tracer | None) -> None:
    """Install a tracer process-wide.

    Args:
        tracer: The tracer implementation to install, or ``None`` to
            restore the default :class:`NoOpTracer`.
    """
    global _current_tracer
    if tracer is None:
        _current_tracer = NoOpTracer()
        logger.debug("Tracing reset to NoOpTracer")
        return
    _current_tracer = tracer
    logger.debug("Tracer installed: %s", type(tracer).__name__)


def flush_traces() -> None:
    """Synchronously drain pending spans on the active tracer.

    Short-lived scripts that use a tracer backed by a buffering processor
    (e.g. OTel's ``BatchSpanProcessor``) may exit before the background
    export thread delivers all queued spans.  Calling this function before
    process exit ensures every span started before the call is flushed to
    the exporter.

    The call is a no-op when the installed tracer does not implement the
    :class:`Flushable` protocol (e.g. :class:`NoOpTracer`). When the
    installed tracer is a ``MultiTracer``, failures on individual child
    tracers are logged and skipped — the call does not raise even if some
    backends fail to flush.

    Example::

        from troopai.adk.tracing import flush_traces, set_tracer
        from troopai.adk.tracing.otel import setup_otel

        set_tracer(setup_otel(service_name="my-agent"))
        try:
            run_agent()
        finally:
            flush_traces()
    """
    tracer = _current_tracer
    if isinstance(tracer, Flushable):
        tracer.flush()
        logger.debug("flush_traces: flushed %s", type(tracer).__name__)
    else:
        logger.debug("flush_traces: tracer %s is not Flushable; skipping", type(tracer).__name__)

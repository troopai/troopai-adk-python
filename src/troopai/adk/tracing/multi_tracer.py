"""Fan-out composite tracer.

:class:`MultiTracer` wraps a list of :class:`Tracer` implementations
and forwards every ``*_span`` factory call to all of them. The returned
:class:`CompositeSpan` delegates ``start`` / ``finish`` / ``set_error``
to every child span so a single ``with`` block lights up every backend
simultaneously.

Use when you need two unrelated backends to observe the same run (for
example, an OTel bridge exporting to a collector and an in-memory
recorder feeding integration tests). When both backends are
OTel-compatible, prefer configuring a second ``SpanProcessor`` on the
shared :class:`~opentelemetry.sdk.trace.TracerProvider` instead — OTel's
own pipeline already handles multi-exporter fan-out with proper batching
and sampling.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar, override

from troopai.adk.tracing.spans import NoOpSpan, Span
from troopai.adk.tracing.tracer import Flushable, Tracer
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

TData = TypeVar("TData", bound=SpanData)


class CompositeSpan[TData: SpanData](Span[TData]):
    """Span that fans lifecycle hooks out to a list of child spans.

    Intentionally does **not** call :meth:`Span.start` / :meth:`Span.finish`
    on itself (the base-class :class:`~contextvars.ContextVar` tracking).
    Each child manages its own parent chain — OTel children via OTel's
    own context, framework children via :data:`_current_span`. Layering a
    composite on top of them would create duplicate entries in either
    stack.

    Exceptions raised by one child span's lifecycle hook are logged and
    swallowed so a broken sub-tracer does not poison the other backends
    or crash the application.
    """

    def __init__(
        self,
        children: list[Span[TData]],
        data: TData,
        span_id: str | None = None,
    ) -> None:
        """Construct a composite span over a list of child spans.

        Args:
            children: The wrapped child spans, one per inner tracer. Each
                receives lifecycle calls (``start``, ``finish``,
                ``set_error``) in order.
            data: Typed span-data payload shared across all children.
            span_id: Optional caller-assigned framework span identifier.
        """
        # _children must exist before super().__init__ so that the data
        # setter (invoked by the base via ``self.data = data``) can fan out
        # immediately.
        self._children = children
        super().__init__(data, span_id=span_id)

    @property
    def data(  # pyright: ignore[reportImplicitOverride]  # @override cannot stack on @property; the override is intentional
        self,
    ) -> TData:
        return self._data

    @data.setter
    def data(self, value: TData) -> None:
        # Runner code rebinds the span after the LLM call
        # (``span.data = dataclasses.replace(span.data, usage=...)``); fan
        # the new value out to every child so composed tracers observe the
        # final payload at finish().
        self._data = value
        for child in self._children:
            child.data = value

    @override
    def start(self) -> None:
        for child in self._children:
            try:
                child.start()
            except Exception:
                logger.exception("Sub-tracer start raised; other children continue")

    @override
    def finish(self) -> None:
        # Finish children in REVERSE start order. Each child that tracks a
        # parent stack (the framework ContextVar, OTel's own context)
        # installs itself left-to-right on start(), so the last-started
        # child is the current one. Those stacks are LIFO: the newest token
        # must be reset first. Finishing in start order would reset an outer
        # token while an inner one is still active, corrupting the stack and
        # leaving a finished child as the current span.
        for child in reversed(self._children):
            try:
                child.finish()
            except Exception:
                logger.exception("Sub-tracer finish raised; other children continue")
        self._finished = True

    @override
    def set_error(self, message: str, *, data: dict[str, Any] | None = None) -> None:
        super().set_error(message, data=data)
        for child in self._children:
            try:
                child.set_error(message, data=data)
            except Exception:
                logger.exception("Sub-tracer set_error raised; other children continue")


class MultiTracer:
    """Composite :class:`Tracer` that fans calls to a list of tracers.

    Every ``*_span`` factory delegates to every wrapped tracer and
    returns a :class:`CompositeSpan` whose lifecycle hooks propagate to
    all children. Pass an empty list to disable fan-out (every span
    degrades to a :class:`NoOpSpan`).

    .. warning::

        Composing **two or more** :class:`~troopai.adk.tracing.otel.OTelTracer`
        instances in the same :class:`MultiTracer` causes the second
        OTel child to parent itself to the first OTel span (because
        ``child[0].start()`` pushes a new OTel context before
        ``child[1].start()`` runs). To fan out to multiple OTel
        exporters, configure multiple ``SpanExporter`` / ``SpanProcessor``
        objects on a shared ``TracerProvider`` instead — OTel's own
        multi-exporter pipeline handles this correctly. Pairing a single
        :class:`~troopai.adk.tracing.otel.OTelTracer` with a
        :class:`~troopai.adk.tracing.metrics.MetricsTracer` or an
        in-memory recorder is safe and the primary supported use-case.
    """

    def __init__(self, tracers: list[Tracer]) -> None:
        """Construct a fan-out tracer over the given list.

        Args:
            tracers: The wrapped tracers, called in order. Exceptions raised
                by any single tracer do not prevent the others from running.

        Raises:
            ValueError: When more than one of the supplied tracers is an
                :class:`~troopai.adk.tracing.otel.OTelTracer` instance.
                Use a single shared ``TracerProvider`` with multiple
                ``SpanExporter`` / ``SpanProcessor`` objects instead.
        """
        self._tracers = list(tracers)
        self._warn_multiple_otel_tracers()
        logger.debug("MultiTracer initialised with %d tracers", len(self._tracers))

    def _warn_multiple_otel_tracers(self) -> None:
        """Emit a warning when >1 OTelTracer is present."""
        try:
            from troopai.adk.tracing.otel import OTelTracer
        except ImportError:
            return
        if OTelTracer is None:
            return
        otel_count = sum(1 for t in self._tracers if isinstance(t, OTelTracer))
        if otel_count > 1:
            logger.warning(
                "MultiTracer received %d OTelTracer instances. "
                "Sequential OTel span starts cause child[1] to self-parent under child[0]. "
                "Configure multiple SpanExporter/SpanProcessor objects on a shared "
                "TracerProvider instead.",
                otel_count,
            )

    def _collect[TData: SpanData](
        self,
        data: TData,
        factory_name: str,
    ) -> Span[TData]:
        """Fan out a ``*_span`` factory call to all wrapped tracers.

        Each tracer's factory call is isolated in a try/except so a
        broken sub-tracer cannot abort the remaining ones.  When *no*
        tracer succeeds a :class:`NoOpSpan` is returned so callers
        always get a valid span.

        Args:
            data: Typed span-data payload forwarded to every tracer.
            factory_name: Name of the ``Tracer`` method to call
                (e.g. ``"agent_span"``).

        Returns:
            A :class:`CompositeSpan` over all successfully-created
            children, or a :class:`NoOpSpan` when all factories failed
            or ``self._tracers`` is empty.
        """
        if len(self._tracers) == 0:
            return NoOpSpan(data)
        children: list[Span[TData]] = []
        for t in self._tracers:
            try:
                span: Span[TData] = getattr(t, factory_name)(data)
                children.append(span)
            except Exception:
                logger.exception(
                    "MultiTracer tracer factory raised; skipping (factory=%s, tracer=%r)",
                    factory_name,
                    t,
                )
        if len(children) == 0:
            return NoOpSpan(data)
        return CompositeSpan(children, data)

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        return self._collect(data, "agent_span")

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        return self._collect(data, "function_span")

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        return self._collect(data, "generation_span")

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        return self._collect(data, "response_span")

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        return self._collect(data, "handoff_span")

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        return self._collect(data, "guardrail_span")

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        return self._collect(data, "custom_span")

    def flush(self) -> None:
        """Synchronously drain pending spans on every :class:`Flushable` inner tracer.

        Tracers that do not implement :class:`~troopai.adk.tracing.tracer.Flushable`
        are silently skipped.  Exceptions raised by individual flushes are logged
        and swallowed so one broken sub-tracer does not prevent the others from
        flushing.
        """
        for tracer in self._tracers:
            if isinstance(tracer, Flushable):
                try:
                    tracer.flush()
                except Exception:
                    logger.exception("MultiTracer flush raised on %r; continuing", tracer)

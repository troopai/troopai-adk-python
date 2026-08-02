"""OpenTelemetry span adapter.

:class:`OTelSpan` wraps an ``opentelemetry.trace.Span`` inside the
framework :class:`~troopai.adk.tracing.spans.Span` abstraction. It
manages the OTel span lifecycle (start, activate in the OTel context,
flush attributes on finish, propagate errors) without exposing any OTel
types on the framework surface.

OTelSpan intentionally does **not** participate in the framework
:class:`~contextvars.ContextVar` chain — OTel has its own
``opentelemetry.context`` propagation that auto-parents children when
the span is set as current. Double-tracking would just waste cycles.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, override

from troopai.adk.tracing.spans import Span
from troopai.adk.types.tracing.span_data import SpanData

if TYPE_CHECKING:
    from contextvars import Token as _ContextVarsToken

    from opentelemetry.context import Context as OTelContext
    from opentelemetry.trace import Span as OTelBaseSpan, Tracer as OTelBaseTracer

    OTelContextToken = _ContextVarsToken[OTelContext]

logger = logging.getLogger(__name__)

TData = TypeVar("TData", bound=SpanData)


class OTelSpan[TData: SpanData](Span[TData]):
    """Framework span backed by an ``opentelemetry.trace.Span``.

    The OTel span is started lazily in :meth:`start` so the parent
    context is captured at the call site (not at construction time).
    Attributes from ``self.data`` are flushed into the OTel span at
    :meth:`finish` — this lets the runner mutate ``self.data`` in place
    during the span body (e.g. updating ``usage`` after the LLM
    response) and have the final state recorded.
    """

    def __init__(
        self,
        data: TData,
        otel_tracer: OTelBaseTracer,
        name: str,
        attribute_flattener: Callable[[dict[str, Any]], dict[str, Any]],
        span_id: str | None = None,
    ) -> None:
        """Construct an OTel-backed span.

        Args:
            data: Typed span-data payload.
            otel_tracer: The underlying ``opentelemetry.trace.Tracer``.
            name: Span name (OTel-conventional, e.g. ``agent.triage``).
            attribute_flattener: Callable that converts ``data.export()``
                to a flat ``dict[str, str | int | float | bool]`` suitable
                for :meth:`Span.set_attributes`.
            span_id: Caller-assigned framework span identifier (does not
                control the OTel span ID).
        """
        super().__init__(data, span_id=span_id)
        self._otel_tracer = otel_tracer
        self._otel_name = name
        self._flatten = attribute_flattener
        self._otel_span: OTelBaseSpan | None = None
        self._otel_ctx_token: OTelContextToken | None = None

    @override
    def start(self) -> None:
        """Start and activate the underlying OTel span.

        OTel's own context (``opentelemetry.context``) propagates the
        span so child spans started during the body auto-attach as
        children. No framework :class:`~contextvars.ContextVar`
        bookkeeping is done here.
        """
        from opentelemetry import context as otel_context, trace as otel_trace

        self._otel_span = self._otel_tracer.start_span(self._otel_name)
        self._otel_ctx_token = otel_context.attach(otel_trace.set_span_in_context(self._otel_span))

    @override
    def finish(self) -> None:
        """Flush attributes, record errors, end the OTel span."""
        from opentelemetry import context as otel_context
        from opentelemetry.trace import Status, StatusCode

        if self._otel_span is not None:
            try:
                attrs = self._flatten(self.data.export())
                if len(attrs) > 0:
                    self._otel_span.set_attributes(attrs)
                if self.error is not None:
                    self._otel_span.set_status(Status(StatusCode.ERROR, self.error.get("message", "")))
            except Exception:
                logger.exception("OTelSpan attribute/status flush failed")
            finally:
                # end() MUST run even if attribute flattening or status
                # setting raised above — otherwise the span is
                # started-but-never-ended and leaks as a dangling open span
                # in the exporter's buffer (the context is already detached
                # below). Guard end() itself so a rare end() failure does
                # not propagate out of finish().
                try:
                    self._otel_span.end()
                except Exception:
                    logger.exception("OTelSpan end failed")

        if self._otel_ctx_token is not None:
            try:
                otel_context.detach(self._otel_ctx_token)
            except Exception:
                logger.exception("OTel context detach failed")
            self._otel_ctx_token = None

        self._finished = True

    @override
    def set_error(self, message: str, *, data: dict[str, Any] | None = None) -> None:
        """Record the error on both the framework and OTel spans.

        Args:
            message: Human-readable error message, recorded as the
                ``exception.message`` attribute on the emitted exception
                event. The span status (``StatusCode.ERROR``) is applied
                at ``finish()`` time.
            data: Optional metadata dict. When present, ``data["type"]``
                is used as the ``exception.type`` attribute (defaults to
                ``"Error"``).
        """
        super().set_error(message, data=data)
        if self._otel_span is not None:
            try:
                self._otel_span.add_event(
                    "exception",
                    attributes={
                        "exception.message": message,
                        "exception.type": (data or {}).get("type", "Error"),
                    },
                )
            except Exception:
                logger.exception("OTelSpan set_error failed")

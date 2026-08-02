"""Fluent helpers for wiring an OpenTelemetry pipeline.

:func:`setup_otel` installs a ``TracerProvider`` with a
``BatchSpanProcessor(OTLPSpanExporter)`` (and an optional
``ConsoleSpanExporter``) and returns an :class:`OTelTracer` ready to be
handed to :func:`troopai.adk.tracing.set_tracer`. It reads
``OTEL_EXPORTER_OTLP_ENDPOINT`` when ``endpoint`` is ``None`` so users
following the upstream conventions need zero keyword arguments.

:func:`setup_otel_from_env` is a thin convenience wrapper driven by the
standard ``OTEL_*`` environment variables
(``OTEL_EXPORTER_OTLP_ENDPOINT``, ``OTEL_EXPORTER_OTLP_HEADERS``,
``OTEL_SERVICE_NAME``).  Endpoint and headers are read natively by the
OpenTelemetry SDK; ``OTEL_SERVICE_NAME`` is resolved explicitly and
threaded into the ``service.name`` resource attribute.  It MUST be called
explicitly — the framework never invokes it automatically.

The helpers are intentionally thin: advanced configurations (multiple
processors, custom resource attributes, sampler tuning) should
construct the ``TracerProvider`` directly and pass it to
:class:`OTelTracer`.

Example::

    from troopai.adk.tracing import set_tracer
    from troopai.adk.tracing.otel import setup_otel

    tracer = setup_otel(service_name="my-agent", console=True)
    set_tracer(tracer)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from troopai.adk.exceptions import TracingDependencyError
from troopai.adk.tracing.otel.otel_tracer import OTelTracer
from troopai.adk.types.tracing.convention import TracingConvention

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import SpanProcessor

logger = logging.getLogger(__name__)


def setup_otel(
    *,
    endpoint: str | None = None,
    service_name: str = "troopai-adk-python",
    console: bool = False,
    additional_processors: list[SpanProcessor] | None = None,
    headers: dict[str, str] | None = None,
    convention: TracingConvention = TracingConvention.DEFAULT,
) -> OTelTracer:
    """Install an OTel :class:`TracerProvider` and return an :class:`OTelTracer`.

    Args:
        endpoint: OTLP collector endpoint (e.g. ``http://localhost:4317``).
            When ``None``, OTel reads ``OTEL_EXPORTER_OTLP_ENDPOINT``
            from the environment; when that is unset the OTLP exporter
            falls back to its own default (``http://localhost:4317``).
        service_name: Value for the ``service.name`` resource attribute.
            Shows up as the service in Jaeger/Datadog/Honeycomb UIs.
        console: When ``True``, also install a
            ``ConsoleSpanExporter`` so spans are printed to stdout in
            addition to being shipped to the OTLP collector — handy for
            development.
        additional_processors: Extra ``SpanProcessor`` instances to
            attach after the default batch processor. Use this for
            custom exporters (e.g. an in-memory recorder for tests,
            OpenInference processors).
        headers: Optional headers dict for the OTLP exporter — commonly
            used to pass a vendor API key. **Never hard-code the key in
            source** — load it from the environment, e.g.
            ``{"x-honeycomb-team": os.environ["HONEYCOMB_API_KEY"]}``.
        convention: Span-attribute vocabulary the returned tracer emits.
            Defaults to ``TracingConvention.DEFAULT`` (``gen_ai.*`` plus
            ``troopai.*``). Pass ``TracingConvention.OPENINFERENCE`` to
            emit OpenInference attributes (read natively by
            Phoenix/Arize).

    Returns:
        An :class:`OTelTracer` bound to the new provider.

    Raises:
        TracingDependencyError: When the ``opentelemetry`` packages are
            not installed. Install the optional extra via
            ``pip install 'troopai-adk-python[otel]'``.
    """
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
    except ImportError as exc:
        raise TracingDependencyError(missing="opentelemetry") from exc

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    otlp_kwargs: dict[str, Any] = {}
    if endpoint is not None:
        otlp_kwargs["endpoint"] = endpoint
    if headers is not None and len(headers) > 0:
        otlp_kwargs["headers"] = headers

    otlp_exporter = OTLPSpanExporter(**otlp_kwargs)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    if console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    if additional_processors is not None:
        for processor in additional_processors:
            provider.add_span_processor(processor)

    otel_trace.set_tracer_provider(provider)

    logger.info(
        "OTel tracer provider installed (service_name=%s, endpoint=%s, console=%s)",
        service_name,
        endpoint if endpoint is not None else "<env>",
        console,
    )
    return OTelTracer(provider=provider, service_name=service_name, convention=convention)


def setup_otel_from_env(
    *,
    console: bool = False,
    additional_processors: list[SpanProcessor] | None = None,
    convention: TracingConvention = TracingConvention.DEFAULT,
) -> OTelTracer:
    """Set up an OTel pipeline driven entirely by standard ``OTEL_*`` env vars.

    This is a convenience wrapper around :func:`setup_otel` driven by the
    standard ``OTEL_*`` environment variables.  Endpoint and headers are read
    natively by the OpenTelemetry SDK; ``OTEL_SERVICE_NAME`` is resolved
    explicitly and threaded into the ``service.name`` resource attribute.
    The relevant variables are:

    * ``OTEL_EXPORTER_OTLP_ENDPOINT`` — collector endpoint (e.g.
      ``http://localhost:4317``).  **Required** (or the signal-specific ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``) — raises :class:`ValueError`
      when unset or empty so the misconfiguration is immediately visible
      rather than silently shipping spans to a default address.
    * ``OTEL_EXPORTER_OTLP_HEADERS`` — comma-separated ``key=value`` pairs
      forwarded as gRPC metadata (e.g. ``x-honeycomb-team=abc123``).
      Optional; the SDK parses the format natively.
    * ``OTEL_SERVICE_NAME`` — value for the ``service.name`` resource
      attribute.  When absent the framework defaults to ``"troopai-adk-python"``.

    This function MUST be called explicitly.  The framework never invokes
    it automatically — that would violate the no-implicit-behavior contract.

    Args:
        console: When ``True``, also install a ``ConsoleSpanExporter`` so
            spans are printed to stdout in addition to being exported via
            OTLP — handy for local development alongside a collector.
        additional_processors: Extra ``SpanProcessor`` instances to attach
            after the default batch processor.
        convention: Span-attribute vocabulary the returned tracer emits.
            Defaults to ``TracingConvention.DEFAULT``.

    Returns:
        An :class:`OTelTracer` bound to the new provider.

    Raises:
        ValueError: When ``OTEL_EXPORTER_OTLP_ENDPOINT`` is not set or is
            empty.  A missing endpoint is an explicit misconfiguration, not
            a silent no-op.
        TracingDependencyError: When the ``opentelemetry`` packages are not
            installed.  Install the optional extra via
            ``pip install 'troopai-adk-python[otel]'``.
    """
    endpoint_env = (
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
    )
    if not endpoint_env:
        raise ValueError(
            "Neither OTEL_EXPORTER_OTLP_ENDPOINT nor OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is set. "
            "Set one to your OTLP collector endpoint (e.g. http://localhost:4317) "
            "before calling setup_otel_from_env()."
        )

    # Delegate to setup_otel with endpoint=None so the SDK reads the env var
    # directly (avoiding double-parsing), and let it also pick up
    # OTEL_EXPORTER_OTLP_HEADERS natively. OTEL_SERVICE_NAME is resolved here
    # explicitly: setup_otel always sets the service.name resource attribute,
    # which shadows the SDK's env-driven OTELResourceDetector, so the env var
    # would otherwise be silently ignored.
    service_name = os.environ.get("OTEL_SERVICE_NAME", "troopai-adk-python")
    logger.debug(
        "setup_otel_from_env: delegating to setup_otel with env-driven endpoint=%s, service_name=%s",
        endpoint_env,
        service_name,
    )
    return setup_otel(
        service_name=service_name,
        console=console,
        additional_processors=additional_processors,
        convention=convention,
    )

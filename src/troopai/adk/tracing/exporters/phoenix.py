"""Arize Phoenix exporter — OTLP + OpenInference convention.

Phoenix ingests OTLP and reads OpenInference attributes natively, so this
is ``setup_otel`` parameterized with the OpenInference convention.
Docs: https://docs.arize.com/phoenix/tracing/how-to-tracing/setup-tracing
"""

from __future__ import annotations

from troopai.adk.tracing.otel.otel_tracer import OTelTracer
from troopai.adk.tracing.otel.setup import setup_otel
from troopai.adk.types.tracing.convention import TracingConvention


def setup_phoenix(*, endpoint: str | None = None, service_name: str = "troopai-adk-python") -> OTelTracer:
    """Return an :class:`OTelTracer` exporting OpenInference spans to Phoenix.

    Args:
        endpoint: Phoenix OTLP collector endpoint; ``None`` uses the OTel
            environment default.
        service_name: Value for the ``service.name`` resource attribute.

    Returns:
        A configured :class:`OTelTracer` using the
        ``TracingConvention.OPENINFERENCE`` attribute vocabulary, pointing
        at the Phoenix collector.

    Raises:
        TracingDependencyError: When the ``opentelemetry`` packages are
            not installed (``pip install 'troopai-adk-python[otel]'``).
    """
    return setup_otel(
        endpoint=endpoint,
        service_name=service_name,
        convention=TracingConvention.OPENINFERENCE,
    )

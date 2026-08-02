"""Wire an OTel MeterProvider and return a MetricsTracer."""

from __future__ import annotations

import logging

from troopai.adk.exceptions import TracingDependencyError
from troopai.adk.tracing.metrics.instruments import Instruments
from troopai.adk.tracing.metrics.tracer import MetricsTracer

logger = logging.getLogger(__name__)

_DEFAULT_EXPORT_INTERVAL_MS = 60_000


def setup_metrics(
    *,
    endpoint: str | None = None,
    service_name: str = "troopai-adk-python",
    export_interval_ms: int = _DEFAULT_EXPORT_INTERVAL_MS,
) -> MetricsTracer:
    """Install an OTel ``MeterProvider`` and return a :class:`MetricsTracer`.

    Args:
        endpoint: OTLP metrics endpoint; ``None`` falls back to OTel's
            ``OTEL_EXPORTER_OTLP_ENDPOINT`` / built-in default.
        service_name: ``service.name`` resource attribute.
        export_interval_ms: Periodic export interval in milliseconds.

    Raises:
        TracingDependencyError: when the ``opentelemetry`` packages are
            not installed (``pip install 'troopai-adk-python[otel]'``).
    """
    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
    except ImportError as exc:
        raise TracingDependencyError(missing="opentelemetry") from exc

    exporter = OTLPMetricExporter(endpoint=endpoint) if endpoint is not None else OTLPMetricExporter()
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=export_interval_ms)
    provider = MeterProvider(resource=Resource.create({"service.name": service_name}), metric_readers=[reader])
    otel_metrics.set_meter_provider(provider)
    logger.info(
        "OTel meter provider installed (service_name=%s, endpoint=%s, export_interval_ms=%d)",
        service_name,
        endpoint if endpoint is not None else "<env>",
        export_interval_ms,
    )
    return MetricsTracer(Instruments(provider.get_meter(service_name)))

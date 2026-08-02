import pytest

from troopai.adk.tracing.metrics import MetricsTracer
from troopai.adk.tracing.metrics.setup import setup_metrics


def test_setup_metrics_returns_tracer():
    pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc.metric_exporter")

    from unittest.mock import MagicMock, patch

    with (
        patch("opentelemetry.metrics.set_meter_provider") as mock_set,
        patch(
            "opentelemetry.exporter.otlp.proto.grpc.metric_exporter.OTLPMetricExporter",
            return_value=MagicMock(),
        ),
        patch(
            "opentelemetry.sdk.metrics.export.PeriodicExportingMetricReader",
            return_value=MagicMock(),
        ),
    ):
        tracer = setup_metrics(service_name="test-agent", endpoint="http://localhost:4317")

    assert isinstance(tracer, MetricsTracer)
    mock_set.assert_called_once()

"""Smoke tests for the tracing public API exports."""

from __future__ import annotations

import pytest


def test_public_symbols_importable() -> None:
    pytest.importorskip("opentelemetry")
    from troopai.adk.tracing import MetricsTracer, OTelTracer, TracingConvention, log_event, setup_metrics, setup_otel
    from troopai.adk.tracing.exporters import (
        setup_helicone,
        setup_langsmith,
        setup_logfire,
        setup_phoenix,
    )

    assert MetricsTracer is not None
    assert OTelTracer is not None
    assert TracingConvention is not None
    assert log_event is not None
    assert setup_metrics is not None
    assert setup_otel is not None
    assert setup_helicone is not None
    assert setup_langsmith is not None
    assert setup_logfire is not None
    assert setup_phoenix is not None

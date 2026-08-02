"""OTel attribute mapping for swarm spans.

Verifies that opening + closing swarm_span / swarm_turn_span via the
OTelTracer surfaces the expected troopai.swarm.* attributes through the
in-memory test exporter. None-valued fields are absent.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from troopai.adk.tracing.otel.otel_tracer import OTelTracer
from troopai.adk.tracing.spans import swarm_span, swarm_turn_span
from troopai.adk.tracing.tracer import set_tracer


@pytest.fixture
def otel_exporter() -> Iterator[InMemorySpanExporter]:
    """Wire an OTel exporter + matching OTelTracer for the test."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    set_tracer(OTelTracer(provider=provider, service_name="troopai-adk-python-test"))
    yield exporter
    exporter.clear()
    set_tracer(None)


class TestSwarmSpanAttributes:
    def test_root_span_emits_troopai_swarm_namespace(self, otel_exporter: InMemorySpanExporter) -> None:
        span = swarm_span(
            swarm_id="abc-123",
            entry="approver",
            status="completed",
            turns_total=4,
        )
        span.start()
        span.finish()

        finished = otel_exporter.get_finished_spans()
        assert len(finished) == 1
        attrs = finished[0].attributes or {}
        assert attrs.get("troopai.swarm.id") == "abc-123"
        assert attrs.get("troopai.swarm.entry") == "approver"
        assert attrs.get("troopai.swarm.status") == "completed"
        assert attrs.get("troopai.swarm.turns_total") == 4

    def test_root_span_omits_none_valued_fields(self, otel_exporter: InMemorySpanExporter) -> None:
        span = swarm_span(swarm_id="abc-123")
        span.start()
        span.finish()

        attrs = otel_exporter.get_finished_spans()[0].attributes or {}
        assert attrs.get("troopai.swarm.id") == "abc-123"
        assert "troopai.swarm.entry" not in attrs
        assert "troopai.swarm.status" not in attrs
        assert "troopai.swarm.turns_total" not in attrs


class TestSwarmTurnSpanAttributes:
    def test_turn_span_emits_troopai_swarm_turn_namespace(self, otel_exporter: InMemorySpanExporter) -> None:
        span = swarm_turn_span(
            swarm_id="abc-123",
            index=3,
            member="approver",
            status="success",
            duration_ms=147,
            resume_attempt=2,
        )
        span.start()
        span.finish()

        attrs = otel_exporter.get_finished_spans()[0].attributes or {}
        assert attrs.get("troopai.swarm.id") == "abc-123"
        assert attrs.get("troopai.swarm.turn.index") == 3
        assert attrs.get("troopai.swarm.turn.member") == "approver"
        assert attrs.get("troopai.swarm.turn.status") == "success"
        assert attrs.get("troopai.swarm.turn.duration_ms") == 147
        assert attrs.get("troopai.swarm.turn.resume_attempt") == 2

    def test_turn_span_omits_none_valued_fields(self, otel_exporter: InMemorySpanExporter) -> None:
        span = swarm_turn_span(swarm_id="abc-123", index=1, member="approver")
        span.start()
        span.finish()

        attrs = otel_exporter.get_finished_spans()[0].attributes or {}
        assert attrs.get("troopai.swarm.id") == "abc-123"
        assert attrs.get("troopai.swarm.turn.index") == 1
        assert attrs.get("troopai.swarm.turn.member") == "approver"
        assert "troopai.swarm.turn.status" not in attrs
        assert "troopai.swarm.turn.duration_ms" not in attrs
        assert "troopai.swarm.turn.resume_attempt" not in attrs

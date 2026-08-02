"""OTel-bridge attribute mapping for graph / graph-superstep / graph-node spans.

The graph-tracing factories route through ``custom_span``, so the OTel
bridge sees :class:`CustomSpanData` and dispatches to graph-specific
attribute helpers based on the inner ``data["type"]`` discriminator.

Tests use OTel's own :class:`InMemorySpanExporter` so assertions run
against true-to-runtime span shapes — names, attributes, parent chains.
"""

from __future__ import annotations

from typing import Any

import pytest

from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.spans import graph_node_span, graph_span, graph_superstep_span

otel_sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
otel_sdk_export = pytest.importorskip("opentelemetry.sdk.trace.export")
otel_in_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from troopai.adk.tracing.otel import OTelTracer


@pytest.fixture
def exporter_and_tracer() -> Any:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OTelTracer(provider=provider, service_name="test-svc")
    set_tracer(tracer)
    yield exporter, tracer
    exporter.clear()
    set_tracer(None)


def _finished_span_by_name(exporter: InMemorySpanExporter, name: str) -> Any:
    matches = [s for s in exporter.get_finished_spans() if s.name == name]
    assert len(matches) == 1, (
        f"expected exactly one span named {name!r}; got {[s.name for s in exporter.get_finished_spans()]}"
    )
    return matches[0]


class TestGraphSpanOTelAttributes:
    def test_graph_span_emits_troopai_graph_attributes(self, exporter_and_tracer: Any) -> None:
        exporter, _ = exporter_and_tracer
        with graph_span(
            graph_id="my-graph",
            entry="start_node",
            status="completed",
            supersteps_total=4,
        ):
            pass

        otel_span = _finished_span_by_name(exporter, "graph.my-graph")
        attrs = otel_span.attributes
        assert attrs is not None
        assert attrs["troopai.graph.id"] == "my-graph"
        assert attrs["troopai.graph.entry"] == "start_node"
        assert attrs["troopai.graph.status"] == "completed"
        assert attrs["troopai.graph.supersteps_total"] == 4

    def test_graph_span_omits_optional_none_attributes(self, exporter_and_tracer: Any) -> None:
        exporter, _ = exporter_and_tracer
        with graph_span(graph_id="g1"):
            pass

        otel_span = _finished_span_by_name(exporter, "graph.g1")
        attrs = otel_span.attributes
        assert attrs is not None
        assert attrs["troopai.graph.id"] == "g1"
        # None-valued fields MUST be omitted (OTel rejects None values).
        assert "troopai.graph.entry" not in attrs
        assert "troopai.graph.status" not in attrs
        assert "troopai.graph.supersteps_total" not in attrs


class TestGraphSuperstepSpanOTelAttributes:
    def test_graph_superstep_span_emits_indexed_attributes(self, exporter_and_tracer: Any) -> None:
        exporter, _ = exporter_and_tracer
        with graph_superstep_span(
            graph_id="g1",
            index=2,
            ready_nodes=("a", "b"),
            fired_nodes=("a", "b"),
        ):
            pass

        otel_span = _finished_span_by_name(exporter, "graph.superstep.2")
        attrs = otel_span.attributes
        assert attrs is not None
        assert attrs["troopai.graph.id"] == "g1"
        assert attrs["troopai.graph.superstep.index"] == 2
        assert list(attrs["troopai.graph.superstep.ready_nodes"]) == ["a", "b"]
        assert list(attrs["troopai.graph.superstep.fired_nodes"]) == ["a", "b"]

    def test_graph_superstep_span_omits_unset_node_lists(self, exporter_and_tracer: Any) -> None:
        exporter, _ = exporter_and_tracer
        with graph_superstep_span(graph_id="g1", index=0):
            pass

        otel_span = _finished_span_by_name(exporter, "graph.superstep.0")
        attrs = otel_span.attributes
        assert attrs is not None
        assert attrs["troopai.graph.superstep.index"] == 0
        assert "troopai.graph.superstep.ready_nodes" not in attrs
        assert "troopai.graph.superstep.fired_nodes" not in attrs


class TestGraphNodeSpanOTelAttributes:
    def test_graph_node_span_emits_node_attributes(self, exporter_and_tracer: Any) -> None:
        exporter, _ = exporter_and_tracer
        with graph_node_span(
            graph_id="g1",
            node_name="enrich",
            attempts=2,
            status="success",
            duration_ms=512,
        ):
            pass

        otel_span = _finished_span_by_name(exporter, "graph.node.enrich")
        attrs = otel_span.attributes
        assert attrs is not None
        assert attrs["troopai.graph.id"] == "g1"
        assert attrs["troopai.graph.node.name"] == "enrich"
        assert attrs["troopai.graph.node.attempts"] == 2
        assert attrs["troopai.graph.node.status"] == "success"
        assert attrs["troopai.graph.node.duration_ms"] == 512

    def test_graph_node_span_includes_resume_attempt_when_set(self, exporter_and_tracer: Any) -> None:
        exporter, _ = exporter_and_tracer
        with graph_node_span(
            graph_id="g1",
            node_name="enrich",
            status="interrupted",
            resume_attempt=1,
        ):
            pass

        otel_span = _finished_span_by_name(exporter, "graph.node.enrich")
        attrs = otel_span.attributes
        assert attrs is not None
        assert attrs["troopai.graph.node.resume_attempt"] == 1

    def test_graph_node_span_omits_unset_optional_fields(self, exporter_and_tracer: Any) -> None:
        exporter, _ = exporter_and_tracer
        with graph_node_span(graph_id="g1", node_name="enrich"):
            pass

        otel_span = _finished_span_by_name(exporter, "graph.node.enrich")
        attrs = otel_span.attributes
        assert attrs is not None
        assert "troopai.graph.node.attempts" not in attrs
        assert "troopai.graph.node.status" not in attrs
        assert "troopai.graph.node.duration_ms" not in attrs
        assert "troopai.graph.node.resume_attempt" not in attrs


class TestNestedGraphSpanHierarchy:
    def test_node_span_nests_under_superstep_span_under_graph_span(self, exporter_and_tracer: Any) -> None:
        exporter, _ = exporter_and_tracer
        with (
            graph_span(graph_id="parent"),
            graph_superstep_span(graph_id="parent", index=0),
            graph_node_span(graph_id="parent", node_name="leaf"),
        ):
            pass

        spans = {s.name: s for s in exporter.get_finished_spans()}
        graph = spans["graph.parent"]
        superstep = spans["graph.superstep.0"]
        node = spans["graph.node.leaf"]

        # Parent chain: graph (root) ← superstep ← node.
        assert graph.parent is None
        assert superstep.parent is not None
        assert superstep.parent.span_id == graph.context.span_id
        assert node.parent is not None
        assert node.parent.span_id == superstep.context.span_id

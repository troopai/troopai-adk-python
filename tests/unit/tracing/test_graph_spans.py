"""Graph-tracing span factories and data classes.

Three new span kinds for the graph orchestration layer:

- :class:`GraphSpanData` — root span for a whole graph run.
- :class:`GraphSuperstepSpanData` — one BSP superstep boundary.
- :class:`GraphNodeSpanData` — one node attempt inside a superstep.

Factories route through ``custom_span``: the inner :class:`CustomSpanData`
carries the graph-typed payload as ``data["type"]`` + the exported fields.
OTel-bridge attribute mapping (follow-up phase) inspects the discriminator
to route to graph-specific attribute conventions.
"""

from __future__ import annotations

from troopai.adk.tracing import Span, set_tracer
from troopai.adk.tracing.spans import graph_node_span, graph_span, graph_superstep_span
from troopai.adk.types.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GraphNodeSpanData,
    GraphSpanData,
    GraphSuperstepSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SpanData,
)


class _Recorder:
    """In-memory tracer recording every span created — for assertion in tests."""

    def __init__(self) -> None:
        self.spans: list[SpanData] = []

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        self.spans.append(data)
        return Span(data)

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        self.spans.append(data)
        return Span(data)

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        self.spans.append(data)
        return Span(data)

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        self.spans.append(data)
        return Span(data)

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        self.spans.append(data)
        return Span(data)

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        self.spans.append(data)
        return Span(data)

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        self.spans.append(data)
        return Span(data)


class TestGraphSpanFactory:
    def teardown_method(self) -> None:
        set_tracer(None)

    def test_graph_span_records_graph_id_under_custom_span(self) -> None:
        recorder = _Recorder()
        set_tracer(recorder)
        with graph_span(graph_id="my-graph"):
            pass
        assert len(recorder.spans) == 1
        recorded = recorder.spans[0]
        assert isinstance(recorded, CustomSpanData)
        assert recorded.name == "graph.my-graph"
        assert recorded.data["type"] == "graph"
        assert recorded.data["graph_id"] == "my-graph"

    def test_graph_span_propagates_entry_and_status_fields(self) -> None:
        recorder = _Recorder()
        set_tracer(recorder)
        with graph_span(graph_id="g1", entry="start_node", status="completed", supersteps_total=4):
            pass
        recorded = recorder.spans[0]
        assert isinstance(recorded, CustomSpanData)
        assert recorded.data["entry"] == "start_node"
        assert recorded.data["status"] == "completed"
        assert recorded.data["supersteps_total"] == 4

    def test_graph_span_disabled_bypasses_tracer(self) -> None:
        recorder = _Recorder()
        set_tracer(recorder)
        with graph_span(graph_id="g1", disabled=True):
            pass
        assert len(recorder.spans) == 0


class TestGraphSuperstepSpanFactory:
    def teardown_method(self) -> None:
        set_tracer(None)

    def test_graph_superstep_span_records_index_under_custom_span(self) -> None:
        recorder = _Recorder()
        set_tracer(recorder)
        with graph_superstep_span(graph_id="g1", index=2):
            pass
        recorded = recorder.spans[0]
        assert isinstance(recorded, CustomSpanData)
        assert recorded.name == "graph.superstep.2"
        assert recorded.data["type"] == "graph_superstep"
        assert recorded.data["graph_id"] == "g1"
        assert recorded.data["index"] == 2

    def test_graph_superstep_span_carries_ready_and_fired_nodes(self) -> None:
        recorder = _Recorder()
        set_tracer(recorder)
        with graph_superstep_span(
            graph_id="g1",
            index=0,
            ready_nodes=("a", "b"),
            fired_nodes=("a", "b"),
        ):
            pass
        recorded = recorder.spans[0]
        assert isinstance(recorded, CustomSpanData)
        # tuples serialize as lists via JSON-safe export
        assert recorded.data["ready_nodes"] == ["a", "b"]
        assert recorded.data["fired_nodes"] == ["a", "b"]


class TestGraphNodeSpanFactory:
    def teardown_method(self) -> None:
        set_tracer(None)

    def test_graph_node_span_records_node_name(self) -> None:
        recorder = _Recorder()
        set_tracer(recorder)
        with graph_node_span(graph_id="g1", node_name="fetch_data"):
            pass
        recorded = recorder.spans[0]
        assert isinstance(recorded, CustomSpanData)
        assert recorded.name == "graph.node.fetch_data"
        assert recorded.data["type"] == "graph_node"
        assert recorded.data["graph_id"] == "g1"
        assert recorded.data["node_name"] == "fetch_data"

    def test_graph_node_span_carries_attempts_status_and_resume_attempt(self) -> None:
        recorder = _Recorder()
        set_tracer(recorder)
        with graph_node_span(
            graph_id="g1",
            node_name="enrich",
            attempts=3,
            status="interrupted",
            duration_ms=1234,
            resume_attempt=2,
        ):
            pass
        recorded = recorder.spans[0]
        assert isinstance(recorded, CustomSpanData)
        assert recorded.data["attempts"] == 3
        assert recorded.data["status"] == "interrupted"
        assert recorded.data["duration_ms"] == 1234
        assert recorded.data["resume_attempt"] == 2


class TestGraphSpanDataExport:
    def test_graph_span_data_export_includes_type_discriminator(self) -> None:
        data = GraphSpanData(graph_id="g1")
        exported = data.export()
        assert exported["type"] == "graph"
        assert exported["graph_id"] == "g1"

    def test_graph_superstep_span_data_export_includes_type_discriminator(self) -> None:
        data = GraphSuperstepSpanData(graph_id="g1", index=0)
        exported = data.export()
        assert exported["type"] == "graph_superstep"
        assert exported["graph_id"] == "g1"
        assert exported["index"] == 0

    def test_graph_node_span_data_export_includes_type_discriminator(self) -> None:
        data = GraphNodeSpanData(graph_id="g1", node_name="ask")
        exported = data.export()
        assert exported["type"] == "graph_node"
        assert exported["graph_id"] == "g1"
        assert exported["node_name"] == "ask"

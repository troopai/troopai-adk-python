from typing import Any

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from troopai.adk.tracing.metrics.instruments import Instruments
from troopai.adk.types.tracing.span_data import (
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
)


def _meter_and_reader():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    return provider.get_meter("test"), reader


def _points(reader: InMemoryMetricReader, name: str) -> list[Any]:
    data = reader.get_metrics_data()
    if data is None:
        return []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    return list(metric.data.data_points)
    return []


def test_generation_records_token_histograms_by_model():
    meter, reader = _meter_and_reader()
    inst = Instruments(meter)
    inst.record_generation(
        GenerationSpanData(model="claude-3", usage={"input_tokens": 10, "output_tokens": 4}),
        error=False,
    )
    pts = _points(reader, "troopai.llm.tokens.prompt")
    assert len(pts) == 1
    assert pts[0].attributes["model"] == "claude-3"
    assert pts[0].sum == 10
    completion_pts = _points(reader, "troopai.llm.tokens.completion")
    assert completion_pts[0].sum == 4
    req = _points(reader, "troopai.llm.requests")
    assert req[0].attributes == {"model": "claude-3", "status": "success"}


def test_generation_records_error_request_label():
    meter, reader = _meter_and_reader()
    inst = Instruments(meter)
    inst.record_generation(GenerationSpanData(model="m", usage=None), error=True)
    req = _points(reader, "troopai.llm.requests")
    assert req[0].attributes == {"model": "m", "status": "error"}


def test_tool_calls_counter_labels_status():
    meter, reader = _meter_and_reader()
    inst = Instruments(meter)
    inst.record_function(FunctionSpanData(name="lookup"), error=True)
    pts = _points(reader, "troopai.agent.tool.calls")
    assert pts[0].attributes == {"tool": "lookup", "status": "error"}


def test_agent_turn_duration_histogram():
    meter, reader = _meter_and_reader()
    inst = Instruments(meter)
    inst.record_agent(AgentSpanData(name="triage"), duration_ms=12.5)
    pts = _points(reader, "troopai.agent.turn.duration_ms")
    assert pts[0].attributes == {"agent": "triage"}
    assert pts[0].sum == 12.5


def test_graph_node_duration_records_node_and_status():
    meter, reader = _meter_and_reader()
    inst = Instruments(meter)
    from troopai.adk.types.tracing.span_data import GraphNodeSpanData

    inst.record_graph_node(GraphNodeSpanData(graph_id="g", node_name="planner", status="success"), duration_ms=7.0)
    pts = _points(reader, "troopai.graph.node.duration_ms")
    assert pts[0].attributes == {"node": "planner", "status": "success"}
    assert pts[0].sum == 7.0


def test_swarm_turn_duration_defaults_status_when_none():
    meter, reader = _meter_and_reader()
    inst = Instruments(meter)
    from troopai.adk.types.tracing.span_data import SwarmTurnSpanData

    inst.record_swarm_turn(SwarmTurnSpanData(swarm_id="s", index=1, member="alice"), duration_ms=3.0)
    pts = _points(reader, "troopai.swarm.turn.duration_ms")
    assert pts[0].attributes == {"member": "alice", "status": "unknown"}

import dataclasses
from typing import Any

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from troopai.adk.tracing.metrics.instruments import Instruments
from troopai.adk.tracing.metrics.tracer import MetricsTracer
from troopai.adk.types.tracing.span_data import (
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GraphNodeSpanData,
    HandoffSpanData,
    SwarmTurnSpanData,
)


def _tracer_and_reader() -> tuple[MetricsTracer, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    return MetricsTracer(Instruments(provider.get_meter("t"))), reader


def _points(reader: InMemoryMetricReader, name: str) -> list[Any]:
    metrics_data = reader.get_metrics_data()
    if metrics_data is None:
        return []
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    return list(metric.data.data_points)
    return []


def test_generation_span_records_tokens_after_data_rebind():
    tracer, reader = _tracer_and_reader()
    with tracer.generation_span(GenerationSpanData(model="gpt")) as span:
        span.data = dataclasses.replace(span.data, usage={"input_tokens": 8, "output_tokens": 2})
    pts = _points(reader, "troopai.llm.tokens.prompt")
    assert pts[0].sum == 8
    assert pts[0].attributes["model"] == "gpt"
    completion_pts = _points(reader, "troopai.llm.tokens.completion")
    assert completion_pts[0].sum == 2


def test_custom_swarm_turn_records_duration():
    tracer, reader = _tracer_and_reader()
    payload = SwarmTurnSpanData(swarm_id="s", index=1, member="alice", status="success").export()
    with tracer.custom_span(CustomSpanData(name="swarm.turn.1", data=payload)):
        pass
    pts = _points(reader, "troopai.swarm.turn.duration_ms")
    assert pts[0].attributes == {"member": "alice", "status": "success"}


def test_custom_graph_node_records_duration():
    tracer, reader = _tracer_and_reader()
    payload = GraphNodeSpanData(graph_id="g", node_name="planner", status="success").export()
    with tracer.custom_span(CustomSpanData(name="graph.node.planner", data=payload)):
        pass
    pts = _points(reader, "troopai.graph.node.duration_ms")
    assert pts[0].attributes == {"node": "planner", "status": "success"}


def test_agent_and_function_spans_record_through_tracer():
    from troopai.adk.types.tracing.span_data import AgentSpanData

    tracer, reader = _tracer_and_reader()
    with tracer.agent_span(AgentSpanData(name="triage")):
        pass
    with tracer.function_span(FunctionSpanData(name="lookup")):
        pass
    agent_pts = _points(reader, "troopai.agent.turn.duration_ms")
    assert agent_pts[0].attributes == {"agent": "triage"}
    tool_pts = _points(reader, "troopai.agent.tool.calls")
    assert tool_pts[0].attributes == {"tool": "lookup", "status": "success"}


def test_metric_span_does_not_touch_contextvar():
    from troopai.adk.tracing.spans import current_span
    from troopai.adk.types.tracing.span_data import AgentSpanData

    tracer, _reader = _tracer_and_reader()
    assert current_span() is None
    with tracer.agent_span(AgentSpanData(name="a")):
        assert current_span() is None  # MetricSpan.start() does not push
    assert current_span() is None


def test_finish_is_idempotent_does_not_double_record():
    tracer, reader = _tracer_and_reader()
    span = tracer.function_span(FunctionSpanData(name="lookup"))
    span.start()
    span.finish()
    span.finish()  # second finish must be a no-op
    tool_pts = _points(reader, "troopai.agent.tool.calls")
    assert len(tool_pts) == 1
    assert tool_pts[0].value == 1  # counted once, not twice


def test_explicit_finish_inside_with_block_records_once():
    tracer, reader = _tracer_and_reader()
    with tracer.generation_span(GenerationSpanData(model="gpt")) as span:
        span.data = dataclasses.replace(span.data, usage={"input_tokens": 8, "output_tokens": 2})
        span.finish()  # explicit close; __exit__ then calls finish() again
    prompt_pts = _points(reader, "troopai.llm.tokens.prompt")
    assert len(prompt_pts) == 1
    assert prompt_pts[0].sum == 8  # not 16
    request_pts = _points(reader, "troopai.llm.requests")
    assert request_pts[0].value == 1  # one request, not two


def test_unknown_kind_records_nothing():
    tracer, reader = _tracer_and_reader()
    with tracer.handoff_span(HandoffSpanData(from_agent="a", to_agent="b")):
        pass
    # no handoff instrument exists; verify every instrument is silent
    for name in (
        "troopai.agent.turn.duration_ms",
        "troopai.llm.tokens.prompt",
        "troopai.llm.tokens.completion",
        "troopai.llm.requests",
        "troopai.agent.tool.calls",
        "troopai.graph.node.duration_ms",
        "troopai.swarm.turn.duration_ms",
    ):
        assert _points(reader, name) == []

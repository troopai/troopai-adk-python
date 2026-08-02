from typing import Any

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from troopai.adk.tracing.metrics.instruments import Instruments
from troopai.adk.types.tracing.span_data import GenerationSpanData


def _meter_and_reader():
    r = InMemoryMetricReader()
    return MeterProvider(metric_readers=[r]).get_meter("t"), r


def _points(reader: InMemoryMetricReader, name: str) -> list[Any]:
    data = reader.get_metrics_data()
    if data is None:
        return []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name == name:
                    return list(m.data.data_points)
    return []


def test_cost_metric_recorded_with_tenant_dimension():
    meter, reader = _meter_and_reader()
    inst = Instruments(meter)
    inst.record_generation(
        GenerationSpanData(
            model="gpt", usage={"input_tokens": 10, "output_tokens": 4}, cost_usd=0.03, tenant_id="acme"
        ),
        error=False,
    )
    cost_pts = _points(reader, "troopai.llm.cost.usd")
    assert cost_pts[0].sum == 0.03
    assert cost_pts[0].attributes == {"model": "gpt", "tenant": "acme"}
    tok_pts = _points(reader, "troopai.llm.tokens.prompt")
    assert tok_pts[0].attributes == {"model": "gpt", "tenant": "acme"}


def test_cost_metric_not_recorded_when_none():
    meter, reader = _meter_and_reader()
    inst = Instruments(meter)
    inst.record_generation(
        GenerationSpanData(model="gpt", usage={"input_tokens": 5, "output_tokens": 2}, cost_usd=None, tenant_id=None),
        error=False,
    )
    cost_pts = _points(reader, "troopai.llm.cost.usd")
    assert len(cost_pts) == 0


def test_untenanted_span_has_no_tenant_dimension():
    meter, reader = _meter_and_reader()
    inst = Instruments(meter)
    inst.record_generation(
        GenerationSpanData(model="m", usage={"input_tokens": 1, "output_tokens": 1}, cost_usd=None, tenant_id=None),
        error=False,
    )
    tok_pts = _points(reader, "troopai.llm.tokens.prompt")
    assert tok_pts[0].attributes == {"model": "m"}
    req_pts = _points(reader, "troopai.llm.requests")
    assert req_pts[0].attributes == {"model": "m", "status": "success"}


def test_request_counter_includes_tenant_when_present():
    meter, reader = _meter_and_reader()
    inst = Instruments(meter)
    inst.record_generation(
        GenerationSpanData(model="gpt4", usage=None, cost_usd=None, tenant_id="tenant-x"),
        error=True,
    )
    req_pts = _points(reader, "troopai.llm.requests")
    assert req_pts[0].attributes == {"model": "gpt4", "tenant": "tenant-x", "status": "error"}


def test_cost_zero_is_recorded():
    meter, reader = _meter_and_reader()
    inst = Instruments(meter)
    inst.record_generation(
        GenerationSpanData(model="m", usage=None, cost_usd=0.0, tenant_id=None),
        error=False,
    )
    cost_pts = _points(reader, "troopai.llm.cost.usd")
    assert len(cost_pts) == 1
    assert cost_pts[0].sum == 0.0

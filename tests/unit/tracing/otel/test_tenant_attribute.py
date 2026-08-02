"""Tests that troopai.tenant.id is emitted on agent and generation spans."""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from troopai.adk.tracing.otel.otel_tracer import OTelTracer
from troopai.adk.types.tracing.convention import TracingConvention
from troopai.adk.types.tracing.span_data import AgentSpanData, GenerationSpanData


def _exp() -> tuple[TracerProvider, InMemorySpanExporter]:
    e = InMemorySpanExporter()
    p = TracerProvider()
    p.add_span_processor(SimpleSpanProcessor(e))
    return p, e


def test_tenant_id_emitted_default_convention() -> None:
    p, e = _exp()
    with OTelTracer(provider=p).agent_span(AgentSpanData(name="a", tenant_id="acme")):
        pass
    (s,) = e.get_finished_spans()
    assert s.attributes is not None
    assert s.attributes["troopai.tenant.id"] == "acme"


def test_tenant_id_emitted_openinference_convention() -> None:
    p, e = _exp()
    with OTelTracer(provider=p, convention=TracingConvention.OPENINFERENCE).agent_span(
        AgentSpanData(name="a", tenant_id="acme")
    ):
        pass
    (s,) = e.get_finished_spans()
    assert s.attributes is not None
    assert s.attributes["troopai.tenant.id"] == "acme"


def test_generation_tenant_id_emitted_default_convention() -> None:
    p, e = _exp()
    with OTelTracer(provider=p).generation_span(GenerationSpanData(model="m", tenant_id="acme")):
        pass
    (s,) = e.get_finished_spans()
    assert s.attributes is not None
    assert s.attributes["troopai.tenant.id"] == "acme"


def test_generation_tenant_id_emitted_openinference_convention() -> None:
    p, e = _exp()
    with OTelTracer(provider=p, convention=TracingConvention.OPENINFERENCE).generation_span(
        GenerationSpanData(model="m", tenant_id="acme")
    ):
        pass
    (s,) = e.get_finished_spans()
    assert s.attributes is not None
    assert s.attributes["troopai.tenant.id"] == "acme"


def test_tenant_id_absent_when_unset_default_convention() -> None:
    p, e = _exp()
    with OTelTracer(provider=p).agent_span(AgentSpanData(name="a")):
        pass
    (s,) = e.get_finished_spans()
    assert "troopai.tenant.id" not in (s.attributes or {})


def test_tenant_id_absent_when_unset_openinference_convention() -> None:
    p, e = _exp()
    with OTelTracer(provider=p, convention=TracingConvention.OPENINFERENCE).agent_span(AgentSpanData(name="a")):
        pass
    (s,) = e.get_finished_spans()
    assert "troopai.tenant.id" not in (s.attributes or {})

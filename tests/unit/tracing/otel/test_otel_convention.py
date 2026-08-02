"""Convention-aware OTel attribute flattening.

Verifies that ``OTelTracer(convention=TracingConvention.OPENINFERENCE)``
emits OpenInference attributes, and that the default convention is
unchanged from the GenAI semconv baseline. Also confirms that token-count
attributes reflect post-construction data rebinds (the runner rebinds
``span.data`` after the LLM call).
"""

from __future__ import annotations

import dataclasses

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from troopai.adk.tracing.otel.otel_tracer import OTelTracer
from troopai.adk.types.tracing.convention import TracingConvention
from troopai.adk.types.tracing.span_data import GenerationSpanData


def _provider_and_exporter() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_openinference_convention_emits_span_kind_and_token_counts() -> None:
    provider, exporter = _provider_and_exporter()
    tracer = OTelTracer(provider=provider, convention=TracingConvention.OPENINFERENCE)
    with tracer.generation_span(GenerationSpanData(model="claude-3")) as span:
        span.data = dataclasses.replace(span.data, usage={"input_tokens": 9, "output_tokens": 4})
    (finished,) = exporter.get_finished_spans()
    attrs = finished.attributes
    assert attrs is not None
    assert attrs["openinference.span.kind"] == "LLM"
    assert attrs["llm.model_name"] == "claude-3"
    assert attrs["llm.token_count.prompt"] == 9


def test_default_convention_unchanged() -> None:
    provider, exporter = _provider_and_exporter()
    tracer = OTelTracer(provider=provider)  # default convention
    with tracer.generation_span(GenerationSpanData(model="gpt")):
        pass
    (finished,) = exporter.get_finished_spans()
    attrs = finished.attributes
    assert attrs is not None
    assert attrs["gen_ai.system"] == "troopai"
    assert "openinference.span.kind" not in attrs


def test_default_convention_reflects_post_rebind_usage() -> None:
    """Default flattener must read the rebound data, not the construction-time data."""
    provider, exporter = _provider_and_exporter()
    tracer = OTelTracer(provider=provider)
    with tracer.generation_span(GenerationSpanData(model="gpt-4")) as span:
        span.data = dataclasses.replace(span.data, usage={"input_tokens": 5, "output_tokens": 3})
    (finished,) = exporter.get_finished_spans()
    attrs = finished.attributes
    assert attrs is not None
    assert attrs["gen_ai.usage.input_tokens"] == 5
    assert attrs["gen_ai.usage.output_tokens"] == 3


def test_openinference_agent_span() -> None:
    from troopai.adk.types.tracing.span_data import AgentSpanData

    provider, exporter = _provider_and_exporter()
    tracer = OTelTracer(provider=provider, convention=TracingConvention.OPENINFERENCE)
    with tracer.agent_span(AgentSpanData(name="my-agent")):
        pass
    (finished,) = exporter.get_finished_spans()
    attrs = finished.attributes
    assert attrs is not None
    assert attrs["openinference.span.kind"] == "AGENT"
    assert attrs["troopai.agent.name"] == "my-agent"


def test_openinference_guardrail_span() -> None:
    from troopai.adk.types.tracing.span_data import GuardrailSpanData

    provider, exporter = _provider_and_exporter()
    tracer = OTelTracer(provider=provider, convention=TracingConvention.OPENINFERENCE)
    with tracer.guardrail_span(GuardrailSpanData(name="pii-check", triggered=True)):
        pass
    (finished,) = exporter.get_finished_spans()
    attrs = finished.attributes
    assert attrs is not None
    assert attrs["openinference.span.kind"] == "GUARDRAIL"
    assert attrs["troopai.guardrail.triggered"] is True


def test_openinference_function_span_redacts_tool_io() -> None:
    """OPENINFERENCE tool I/O must be credential-redacted when record_tool_io_full is False."""
    from troopai.adk.types.tracing.span_data import FunctionSpanData

    provider, exporter = _provider_and_exporter()
    tracer = OTelTracer(provider=provider, convention=TracingConvention.OPENINFERENCE)
    secret_input = '{"api_key": "sk-abcdefghijklmnopqrstuvwxyz123456"}'
    with tracer.function_span(FunctionSpanData(name="db", input=secret_input)):
        pass
    (finished,) = exporter.get_finished_spans()
    attrs = finished.attributes
    assert attrs is not None
    assert attrs["openinference.span.kind"] == "TOOL"
    # The raw secret must NOT appear; the sk- pattern replaces the credential body with sk-***
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in attrs["input.value"]
    assert "sk-***" in attrs["input.value"]


def test_openinference_function_span_record_full_emits_raw() -> None:
    """With record_tool_io_full=True the raw tool I/O must be emitted verbatim."""
    from troopai.adk.types.tracing.span_data import FunctionSpanData

    provider, exporter = _provider_and_exporter()
    tracer = OTelTracer(
        provider=provider,
        convention=TracingConvention.OPENINFERENCE,
        record_tool_io_full=True,
    )
    secret_input = '{"api_key": "sk-abcdefghijklmnopqrstuvwxyz123456"}'
    with tracer.function_span(FunctionSpanData(name="db", input=secret_input)):
        pass
    (finished,) = exporter.get_finished_spans()
    attrs = finished.attributes
    assert attrs is not None
    assert attrs["openinference.span.kind"] == "TOOL"
    # opt-in to full I/O — raw value must be present
    assert attrs["input.value"] == secret_input


def test_setup_otel_forwards_convention() -> None:
    """setup_otel must accept and forward the convention kwarg."""
    from unittest.mock import patch

    import pytest

    pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")

    from troopai.adk.tracing.otel.setup import setup_otel

    with patch("troopai.adk.tracing.otel.setup.OTelTracer") as mock_cls:
        mock_cls.return_value = object()
        setup_otel(
            convention=TracingConvention.OPENINFERENCE,
            service_name="test-svc",
        )
        assert mock_cls.call_count == 1
        _, kwargs = mock_cls.call_args
        assert kwargs.get("convention") is TracingConvention.OPENINFERENCE

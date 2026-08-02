"""Tests for the Tracer protocol, NoOpTracer, and span factories."""

from typing import Any

from troopai.adk.tracing import (
    NoOpSpan,
    NoOpTracer,
    Span,
    Tracer,
    agent_span,
    custom_span,
    function_span,
    generation_span,
    get_tracer,
    guardrail_span,
    handoff_span,
    response_span,
    set_tracer,
)
from troopai.adk.types.tracing import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SpanData,
)


class RecordingTracer:
    """Tracer double that records every span it creates."""

    def __init__(self) -> None:
        self.spans: list[SpanData] = []

    def _record(self, data: SpanData) -> Span[Any]:
        self.spans.append(data)
        return Span(data)

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


class TestTracerRegistry:
    def teardown_method(self) -> None:
        set_tracer(None)

    def test_default_tracer_is_noop(self) -> None:
        assert isinstance(get_tracer(), NoOpTracer)

    def test_set_tracer_installs_backend(self) -> None:
        backend = RecordingTracer()
        set_tracer(backend)
        assert get_tracer() is backend

    def test_set_tracer_none_restores_noop(self) -> None:
        set_tracer(RecordingTracer())
        set_tracer(None)
        assert isinstance(get_tracer(), NoOpTracer)

    def test_noop_tracer_satisfies_protocol(self) -> None:
        assert isinstance(NoOpTracer(), Tracer)

    def test_recording_tracer_satisfies_protocol(self) -> None:
        assert isinstance(RecordingTracer(), Tracer)


class TestSpanContextManager:
    def teardown_method(self) -> None:
        set_tracer(None)

    def test_span_enter_exit_normal(self) -> None:
        with custom_span("test", data={"x": 1}) as span:
            assert span.data.name == "test"
            assert span.error is None
        assert span._finished is True

    def test_span_captures_exception(self) -> None:
        span_ref = None
        try:
            with custom_span("boom") as span:
                span_ref = span
                raise ValueError("oops")
        except ValueError:
            pass
        assert span_ref is not None
        assert span_ref.error is not None
        assert span_ref.error["message"] == "oops"
        assert span_ref.error["data"]["type"] == "ValueError"

    def test_noop_span_records_nothing_persistently(self) -> None:
        with custom_span("noop") as span:
            assert isinstance(span, NoOpSpan)
        # NoOpSpan still tracks finished flag + error for introspection.
        assert span._finished is True


class TestSpanFactories:
    def teardown_method(self) -> None:
        set_tracer(None)

    def test_agent_span_routes_to_tracer(self) -> None:
        backend = RecordingTracer()
        set_tracer(backend)
        with agent_span(name="a", handoffs=["b"], tools=["t"], output_type="str"):
            pass
        assert len(backend.spans) == 1
        recorded = backend.spans[0]
        assert isinstance(recorded, AgentSpanData)
        assert recorded.name == "a"

    def test_function_span_routes_to_tracer(self) -> None:
        backend = RecordingTracer()
        set_tracer(backend)
        with function_span(name="lookup", input='{"id": 1}'):
            pass
        assert len(backend.spans) == 1
        assert isinstance(backend.spans[0], FunctionSpanData)

    def test_generation_span_routes_to_tracer(self) -> None:
        backend = RecordingTracer()
        set_tracer(backend)
        with generation_span(model="gpt-4o-mini", usage={"input_tokens": 5}):
            pass
        assert len(backend.spans) == 1
        assert isinstance(backend.spans[0], GenerationSpanData)

    def test_response_span_routes_to_tracer(self) -> None:
        backend = RecordingTracer()
        set_tracer(backend)
        with response_span(response_id="chatcmpl-x"):
            pass
        assert len(backend.spans) == 1
        assert isinstance(backend.spans[0], ResponseSpanData)

    def test_handoff_span_routes_to_tracer(self) -> None:
        backend = RecordingTracer()
        set_tracer(backend)
        with handoff_span(from_agent="router", to_agent="billing"):
            pass
        assert len(backend.spans) == 1
        assert isinstance(backend.spans[0], HandoffSpanData)

    def test_guardrail_span_routes_to_tracer(self) -> None:
        backend = RecordingTracer()
        set_tracer(backend)
        with guardrail_span(name="pii", triggered=True):
            pass
        assert len(backend.spans) == 1
        data = backend.spans[0]
        assert isinstance(data, GuardrailSpanData)
        assert data.triggered is True

    def test_custom_span_routes_to_tracer(self) -> None:
        backend = RecordingTracer()
        set_tracer(backend)
        with custom_span("checkout", data={"sku": 42}):
            pass
        assert len(backend.spans) == 1
        assert isinstance(backend.spans[0], CustomSpanData)

    def test_disabled_flag_bypasses_tracer(self) -> None:
        backend = RecordingTracer()
        set_tracer(backend)
        with custom_span("skip", disabled=True) as span:
            assert isinstance(span, NoOpSpan)
        # Backend was not consulted.
        assert len(backend.spans) == 0

    def test_custom_span_id_preserved(self) -> None:
        with custom_span("x", span_id="custom-id") as span:
            assert span.span_id == "custom-id"

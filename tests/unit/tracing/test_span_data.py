"""Tests for typed span-data dataclasses."""

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


class TestSpanDataTypes:
    def test_agent_span_data_export(self) -> None:
        data = AgentSpanData(
            name="triage",
            handoffs=["billing"],
            tools=["lookup_order"],
            output_type="str",
        )
        exported = data.export()
        assert exported["type"] == "agent"
        assert exported["name"] == "triage"
        assert exported["handoffs"] == ["billing"]
        assert exported["tools"] == ["lookup_order"]
        assert exported["output_type"] == "str"

    def test_agent_span_data_defaults(self) -> None:
        data = AgentSpanData(name="x")
        assert data.handoffs is None
        assert data.tools is None
        assert data.output_type is None
        assert data.type == "agent"

    def test_function_span_data_export(self) -> None:
        data = FunctionSpanData(
            name="lookup_order",
            input='{"id": 42}',
            output={"status": "shipped"},
        )
        exported = data.export()
        assert exported["type"] == "function"
        assert exported["name"] == "lookup_order"
        assert exported["input"] == '{"id": 42}'
        # Output is stringified for JSON-safety.
        assert exported["output"] == "{'status': 'shipped'}"

    def test_function_span_data_none_output(self) -> None:
        data = FunctionSpanData(name="noop")
        exported = data.export()
        assert exported["output"] is None

    def test_generation_span_data_export(self) -> None:
        data = GenerationSpanData(
            input=[{"role": "user", "content": "hi"}],
            output=[{"role": "assistant", "content": "hello"}],
            model="gpt-4o-mini",
            model_config={"temperature": 0.2},
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        exported = data.export()
        assert exported["type"] == "generation"
        assert exported["model"] == "gpt-4o-mini"
        assert exported["model_config"]["temperature"] == 0.2
        assert exported["usage"]["input_tokens"] == 10

    def test_response_span_data_export(self) -> None:
        data = ResponseSpanData(
            response_id="chatcmpl-abc",
            input=[{"role": "user", "content": "hi"}],
        )
        exported = data.export()
        assert exported["type"] == "response"
        assert exported["response_id"] == "chatcmpl-abc"
        assert len(exported["input"]) == 1

    def test_handoff_span_data_export(self) -> None:
        data = HandoffSpanData(from_agent="router", to_agent="billing")
        exported = data.export()
        assert exported["type"] == "handoff"
        assert exported["from_agent"] == "router"
        assert exported["to_agent"] == "billing"

    def test_guardrail_span_data_export(self) -> None:
        data = GuardrailSpanData(name="pii", triggered=True)
        exported = data.export()
        assert exported["type"] == "guardrail"
        assert exported["name"] == "pii"
        assert exported["triggered"] is True

    def test_custom_span_data_default_data(self) -> None:
        data = CustomSpanData(name="checkout")
        assert data.data == {}
        exported = data.export()
        assert exported["type"] == "custom"
        assert exported["data"] == {}

    def test_custom_span_data_with_payload(self) -> None:
        data = CustomSpanData(name="rank", data={"n": 10})
        exported = data.export()
        assert exported["data"]["n"] == 10

    def test_span_data_is_frozen(self) -> None:
        data = AgentSpanData(name="x")
        try:
            data.name = "y"  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("Frozen dataclass should reject attribute assignment")

    def test_span_data_subclasses_of_base(self) -> None:
        assert issubclass(AgentSpanData, SpanData)
        assert issubclass(CustomSpanData, SpanData)
        assert issubclass(FunctionSpanData, SpanData)
        assert issubclass(GenerationSpanData, SpanData)
        assert issubclass(GuardrailSpanData, SpanData)
        assert issubclass(HandoffSpanData, SpanData)
        assert issubclass(ResponseSpanData, SpanData)

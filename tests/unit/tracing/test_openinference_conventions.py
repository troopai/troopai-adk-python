import json

from troopai.adk.tracing.openinference.conventions import (
    agent_attrs,
    custom_attrs_by_type,
    function_attrs,
    generation_attrs,
    guardrail_attrs,
    handoff_attrs,
    response_attrs,
)
from troopai.adk.types.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GraphNodeSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
)


def test_generation_maps_llm_kind_and_token_counts():
    data = GenerationSpanData(
        model="claude-3",
        model_config={"temperature": 0.4},
        input=[{"role": "user", "content": "hi"}],
        output=[{"role": "assistant", "content": "yo"}],
        usage={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
    )
    attrs = generation_attrs(data)
    assert attrs["openinference.span.kind"] == "LLM"
    assert attrs["llm.system"] == "troopai"
    assert attrs["llm.model_name"] == "claude-3"
    assert attrs["llm.token_count.prompt"] == 11
    assert attrs["llm.token_count.completion"] == 7
    assert attrs["llm.token_count.total"] == 18
    assert json.loads(attrs["llm.invocation_parameters"]) == {"temperature": 0.4}
    assert attrs["input.mime_type"] == "application/json"
    assert json.loads(attrs["input.value"]) == [{"role": "user", "content": "hi"}]
    assert attrs["output.mime_type"] == "application/json"
    assert json.loads(attrs["output.value"]) == [{"role": "assistant", "content": "yo"}]


def test_generation_handles_chat_completions_token_keys():
    data = GenerationSpanData(model="gpt", usage={"prompt_tokens": 3, "completion_tokens": 2})
    attrs = generation_attrs(data)
    assert attrs["llm.token_count.prompt"] == 3
    assert attrs["llm.token_count.completion"] == 2


def test_agent_function_guardrail_kinds():
    assert agent_attrs(AgentSpanData(name="triage"))["openinference.span.kind"] == "AGENT"
    fn = function_attrs(FunctionSpanData(name="lookup", input="{}", output="ok"))
    assert fn["openinference.span.kind"] == "TOOL"
    assert fn["tool.name"] == "lookup"
    gr = guardrail_attrs(GuardrailSpanData(name="pii", triggered=True))
    assert gr["openinference.span.kind"] == "GUARDRAIL"


def test_custom_graph_node_maps_to_chain():
    data = CustomSpanData(
        name="graph.node.x",
        data=GraphNodeSpanData(graph_id="g", node_name="x").export(),
    )
    attrs = custom_attrs_by_type(data)
    assert attrs["openinference.span.kind"] == "CHAIN"


def test_response_attrs_maps_llm_kind():
    attrs = response_attrs(ResponseSpanData())
    assert attrs["openinference.span.kind"] == "LLM"


def test_handoff_attrs_maps_agent_kind_and_agent_names():
    attrs = handoff_attrs(HandoffSpanData(from_agent="a", to_agent="b"))
    assert attrs["openinference.span.kind"] == "AGENT"
    assert "troopai.handoff.from" in attrs
    assert attrs["troopai.handoff.from"] == "a"
    assert "troopai.handoff.to" in attrs
    assert attrs["troopai.handoff.to"] == "b"


def test_custom_attrs_by_type_dispatch_branches():
    assert custom_attrs_by_type(CustomSpanData(name="s", data={"type": "swarm"}))["openinference.span.kind"] == "AGENT"
    assert (
        custom_attrs_by_type(CustomSpanData(name="t", data={"type": "swarm_turn"}))["openinference.span.kind"]
        == "AGENT"
    )
    assert (
        custom_attrs_by_type(CustomSpanData(name="sb", data={"type": "sandbox"}))["openinference.span.kind"] == "TOOL"
    )
    assert (
        custom_attrs_by_type(CustomSpanData(name="u", data={"type": "totally_unknown"}))["openinference.span.kind"]
        == "CHAIN"
    )
    assert custom_attrs_by_type(CustomSpanData(name="n", data={}))["openinference.span.kind"] == "CHAIN"
    assert custom_attrs_by_type(CustomSpanData(name="g", data={"type": "graph"}))["openinference.span.kind"] == "CHAIN"
    assert (
        custom_attrs_by_type(CustomSpanData(name="gs", data={"type": "graph_superstep"}))["openinference.span.kind"]
        == "CHAIN"
    )

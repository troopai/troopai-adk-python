"""Map framework SpanData onto OpenInference semantic-convention attributes.

OpenInference (https://github.com/Arize-ai/openinference) is the de-facto
LLM span convention read natively by Phoenix/Arize. Attribute-name
constants are vendored here (plain strings) to avoid a dependency on the
``openinference-semantic-conventions`` package.
"""

from __future__ import annotations

import json
from typing import Any

from troopai.adk.types.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
)

SPAN_KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
INPUT_MIME = "input.mime_type"
OUTPUT_VALUE = "output.value"
OUTPUT_MIME = "output.mime_type"
JSON_MIME = "application/json"


def _json(value: Any) -> str:
    return json.dumps(value, default=str)


def agent_attrs(data: AgentSpanData) -> dict[str, Any]:
    """Map an agent span to OpenInference attributes.

    Args:
        data: Span data for the agent turn.

    Returns:
        Flat attribute dict with ``openinference.span.kind = "AGENT"`` and
        any populated agent fields.
    """
    attrs: dict[str, Any] = {SPAN_KIND: "AGENT", "troopai.agent.name": data.name}
    if data.output_type is not None:
        attrs["troopai.agent.output_type"] = data.output_type
    if data.tenant_id is not None:
        attrs["troopai.tenant.id"] = data.tenant_id
    return attrs


def function_attrs(data: FunctionSpanData) -> dict[str, Any]:
    """Map a function/tool span to OpenInference attributes.

    Args:
        data: Span data for the function tool call.

    Returns:
        Flat attribute dict with ``openinference.span.kind = "TOOL"`` and
        any populated input/output fields.
    """
    attrs: dict[str, Any] = {SPAN_KIND: "TOOL", "tool.name": data.name}
    if data.input is not None:
        attrs[INPUT_VALUE] = data.input
        attrs[INPUT_MIME] = JSON_MIME
    if data.output is not None:
        attrs[OUTPUT_VALUE] = str(data.output)
    return attrs


def generation_attrs(data: GenerationSpanData) -> dict[str, Any]:
    """Map an LLM generation span to OpenInference attributes.

    Accepts both OpenAI-convention token keys (``prompt_tokens`` /
    ``completion_tokens``) and Anthropic-convention keys
    (``input_tokens`` / ``output_tokens``).

    Args:
        data: Span data for the LLM generation turn.

    Returns:
        Flat attribute dict with ``openinference.span.kind = "LLM"``
        and populated ``llm.*`` fields.
    """
    attrs: dict[str, Any] = {SPAN_KIND: "LLM", "llm.system": "troopai"}
    if data.model is not None:
        attrs["llm.model_name"] = data.model
    if data.model_config is not None:
        attrs["llm.invocation_parameters"] = _json(data.model_config)
    if data.input is not None:
        attrs[INPUT_VALUE] = _json(data.input)
        attrs[INPUT_MIME] = JSON_MIME
    if data.output is not None:
        attrs[OUTPUT_VALUE] = _json(data.output)
        attrs[OUTPUT_MIME] = JSON_MIME
    if data.usage is not None:
        prompt = data.usage.get("prompt_tokens")
        if prompt is None:
            prompt = data.usage.get("input_tokens")
        completion = data.usage.get("completion_tokens")
        if completion is None:
            completion = data.usage.get("output_tokens")
        total = data.usage.get("total_tokens")
        if isinstance(prompt, int):
            attrs["llm.token_count.prompt"] = prompt
        if isinstance(completion, int):
            attrs["llm.token_count.completion"] = completion
        if isinstance(total, int):
            attrs["llm.token_count.total"] = total
    if data.tenant_id is not None:
        attrs["troopai.tenant.id"] = data.tenant_id
    return attrs


def response_attrs(data: ResponseSpanData) -> dict[str, Any]:
    """Map a provider-level response span to OpenInference attributes.

    Args:
        data: Span data for the provider response.

    Returns:
        Flat attribute dict with ``openinference.span.kind = "LLM"`` and
        any populated response fields.
    """
    attrs: dict[str, Any] = {SPAN_KIND: "LLM", "llm.system": "troopai"}
    if data.response_id is not None:
        attrs["troopai.response.id"] = data.response_id
    return attrs


def handoff_attrs(data: HandoffSpanData) -> dict[str, Any]:
    """Map an agent-handoff span to OpenInference attributes.

    Args:
        data: Span data for the handoff.

    Returns:
        Flat attribute dict with ``openinference.span.kind = "AGENT"`` and
        any populated handoff-direction fields.
    """
    attrs: dict[str, Any] = {SPAN_KIND: "AGENT"}
    if data.from_agent is not None:
        attrs["troopai.handoff.from"] = data.from_agent
    if data.to_agent is not None:
        attrs["troopai.handoff.to"] = data.to_agent
    return attrs


def guardrail_attrs(data: GuardrailSpanData) -> dict[str, Any]:
    """Map a guardrail-evaluation span to OpenInference attributes.

    Args:
        data: Span data for the guardrail evaluation.

    Returns:
        Flat attribute dict with ``openinference.span.kind = "GUARDRAIL"``,
        the guardrail name, and the triggered flag.
    """
    return {
        SPAN_KIND: "GUARDRAIL",
        "tool.name": data.name,
        "troopai.guardrail.triggered": data.triggered,
    }


_CUSTOM_TYPE_TO_KIND: dict[str, str] = {
    "graph": "CHAIN",
    "graph_superstep": "CHAIN",
    "graph_node": "CHAIN",
    "swarm": "AGENT",
    "swarm_turn": "AGENT",
    "sandbox": "TOOL",
}


def custom_attrs_by_type(data: CustomSpanData) -> dict[str, Any]:
    """Map a custom span to OpenInference attributes, routing by inner type.

    The ``data["type"]`` discriminator maps framework-internal custom span
    kinds (``graph``, ``graph_node``, ``swarm``, ``swarm_turn``,
    ``sandbox``) to an appropriate ``openinference.span.kind`` value.
    Unknown or absent types default to ``"CHAIN"``.

    Args:
        data: Span data for the custom span.

    Returns:
        Flat attribute dict with a ``openinference.span.kind`` determined
        from the inner type and the span name.
    """
    inner = data.data.get("type")
    kind = _CUSTOM_TYPE_TO_KIND.get(inner, "CHAIN") if isinstance(inner, str) else "CHAIN"
    attrs: dict[str, Any] = {SPAN_KIND: kind, "troopai.span.name": data.name}
    return attrs

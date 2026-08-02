"""Tests for :class:`OpenAIResponsesConverter`.

Covers the Layer 1 ↔ ``openai.types.responses.*`` wire-type bridge:

- ``items_to_input`` across every ``LLMInputContentItem`` variant
  (including the replay shapes that flow back as history).
- ``convert_tools`` for ``FunctionTool`` and ``ExecutableBuiltinTool``.
- ``convert_tool_choice`` — MUST emit the FLAT Responses-API shape
  ``{"type": "function", "name": ...}`` (not the Chat-Completions
  nested ``function: {name: ...}``).
- ``resolve_response_format`` — structured output via
  ``ResponseTextConfigParam``.
- ``response_to_llm_response`` — the four core item types +
  hosted-tool items landing in :class:`LLMResponseProviderItem`.
- ``parse_usage`` — cached- / reasoning-token mapping.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_reasoning_item import Content as ReasoningContent, Summary as ReasoningSummary
from openai.types.responses.response_usage import (
    InputTokensDetails as SDKInputTokensDetails,
    OutputTokensDetails as SDKOutputTokensDetails,
    ResponseUsage,
)
from pydantic import BaseModel

from troopai.adk.llms.openai.openai_responses_converter import OpenAIResponsesConverter
from troopai.adk.schemas.utils import SchemaEnforcement
from troopai.adk.tools.builtin.builtin_tool import ExecutableBuiltinTool
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.responses.llm_response import (
    LLMResponseFunctionToolCall,
    LLMResponseProviderItem,
    LLMResponseReasoning,
    LLMResponseRefusal,
    LLMResponseText,
)

ResponseStatus = Literal["completed", "failed", "in_progress", "cancelled", "queued", "incomplete"]


# ==================================================================
# Helpers
# ==================================================================


def _make_response(
    output: list[Any],
    *,
    status: ResponseStatus = "completed",
    incomplete_details: Any = None,
    usage: ResponseUsage | None = None,
    response_id: str = "resp_abc",
    model: str = "gpt-5.1",
) -> Response:
    """Build a minimal SDK ``Response`` object for converter tests.

    The SDK's Response model has ~30 required fields; most are irrelevant
    to the converter. We construct one with ``model_construct`` to skip
    validation, but mypy still demands the full required-field set — we
    fill the unused ones with placeholder defaults.
    """
    return Response.model_construct(
        id=response_id,
        model=model,
        output=output,
        status=status,
        incomplete_details=incomplete_details,
        usage=usage,
        created_at=0.0,
        object="response",
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _dummy_tool(
    name: str = "lookup",
    *,
    description: str | None = "Look something up",
    enforcement: SchemaEnforcement = SchemaEnforcement.NORMALIZED,
) -> FunctionTool:
    """Minimal ``FunctionTool`` for convert_tools tests."""

    async def _invoke(_ctx: Any, raw_args: str) -> str:  # pragma: no cover
        return raw_args

    return FunctionTool(
        name=name,
        schema={"type": "object", "properties": {"q": {"type": "string"}}},
        description=description,
        schema_enforcement=enforcement,
        on_invoke=_invoke,
    )


async def _noop_invoke(_ctx: Any, raw_args: str) -> str:  # pragma: no cover
    return raw_args


def _echo_executable_tool() -> ExecutableBuiltinTool:
    """Build an ``ExecutableBuiltinTool`` with a dict schema."""
    return ExecutableBuiltinTool(
        name="echo",
        description="Echo the input back.",
        schema={"type": "object", "properties": {"msg": {"type": "string"}}},
        on_invoke=_noop_invoke,
    )


def _as_dict(obj: Any) -> dict[str, Any]:
    """Cast a TypedDict union element to ``dict[str, Any]`` for test assertions.

    The converter returns ``list[ResponseInputItemParam]`` / ``list[ToolParam]``
    — both large TypedDict unions. Pyright cannot narrow field access on the
    union without a runtime discriminator, and test bodies repeatedly
    access fields that are valid at runtime but absent from some union
    members. Casting to ``dict[str, Any]`` at the test boundary is
    semantically free (TypedDicts are dicts at runtime) and keeps the
    test file free of scattered ``# type: ignore`` markers.
    """
    assert isinstance(obj, dict)
    return dict(obj)


# ==================================================================
# items_to_input
# ==================================================================


class TestItemsToInput:
    def test_empty_list_returns_empty_list(self) -> None:
        assert OpenAIResponsesConverter.items_to_input([]) == []

    def test_easy_message_string_content_pass_through(self) -> None:
        msg: Any = {"role": "user", "content": "hello"}
        out = OpenAIResponsesConverter.items_to_input([msg])
        assert [_as_dict(x) for x in out] == [{"role": "user", "content": "hello"}]

    def test_easy_message_list_content_pass_through(self) -> None:
        msg: Any = {
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}],
        }
        out = OpenAIResponsesConverter.items_to_input([msg])
        assert [_as_dict(x) for x in out] == [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]

    def test_strict_message_with_role_user(self) -> None:
        msg: Any = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "yo"}],
        }
        out = OpenAIResponsesConverter.items_to_input([msg])
        # "message" + user role → EasyInputMessageParam (discriminator dropped).
        assert [_as_dict(x) for x in out] == [{"role": "user", "content": [{"type": "input_text", "text": "yo"}]}]

    def test_assistant_message_replay_passes_through(self) -> None:
        msg: dict[str, Any] = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            "id": "msg_1",
            "status": "completed",
        }
        out = OpenAIResponsesConverter.items_to_input([msg])  # type: ignore[list-item]
        assert [_as_dict(x) for x in out] == [msg]

    def test_assistant_output_text_missing_annotations_gets_empty_list(self) -> None:
        # A plain assistant turn (no web-search citations) round-trips
        # through ``LLMResponseText.to_param()`` which omits ``annotations``.
        # The Responses wire shape types ``annotations`` as required on
        # ``output_text``; the converter must default it to ``[]`` so the
        # verbatim replay does not 400.
        msg: dict[str, Any] = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi"}],
            "id": "msg_1",
            "status": "completed",
        }
        out = OpenAIResponsesConverter.items_to_input([msg])  # type: ignore[list-item]
        result = _as_dict(out[0])
        assert result["content"][0] == {"type": "output_text", "text": "hi", "annotations": []}

    def test_assistant_output_text_preserves_existing_annotations(self) -> None:
        anns = [{"type": "url_citation", "url": "https://x", "title": "X", "start_index": 0, "end_index": 1}]
        msg: dict[str, Any] = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi", "annotations": anns}],
            "id": "msg_1",
            "status": "completed",
        }
        out = OpenAIResponsesConverter.items_to_input([msg])  # type: ignore[list-item]
        assert _as_dict(out[0])["content"][0]["annotations"] == anns

    def test_assistant_message_missing_status_defaults_completed(self) -> None:
        msg: dict[str, Any] = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            "id": "msg_1",
        }
        out = OpenAIResponsesConverter.items_to_input([msg])  # type: ignore[list-item]
        assert _as_dict(out[0])["status"] == "completed"

    def test_assistant_message_missing_id_falls_back_to_easy_message(self) -> None:
        # Cross-provider replay (e.g. Chat-Completions history) can yield an
        # assistant message with no ``id``. ``ResponseOutputMessageParam``
        # requires ``id``; fall back to an ``EasyInputMessageParam`` (assistant
        # role, joined text) the Responses API accepts without ``id``.
        msg: dict[str, Any] = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello there"}],
            "status": "completed",
        }
        out = OpenAIResponsesConverter.items_to_input([msg])  # type: ignore[list-item]
        result = _as_dict(out[0])
        assert "id" not in result
        assert result == {"role": "assistant", "content": "hello there"}

    def test_function_call_replay_passes_through(self) -> None:
        call: dict[str, Any] = {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q": "x"}',
        }
        out = OpenAIResponsesConverter.items_to_input([call])  # type: ignore[list-item]
        assert [_as_dict(x) for x in out] == [call]

    def test_function_call_output_passes_through(self) -> None:
        out_item: dict[str, Any] = {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "42",
        }
        out = OpenAIResponsesConverter.items_to_input([out_item])  # type: ignore[list-item]
        assert [_as_dict(x) for x in out] == [out_item]

    def test_reasoning_replay_passes_through(self) -> None:
        reasoning: dict[str, Any] = {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "thought about it"}],
            "id": "rs_1",
            "encrypted_content": "opaque",
        }
        out = OpenAIResponsesConverter.items_to_input([reasoning])  # type: ignore[list-item]
        assert [_as_dict(x) for x in out] == [reasoning]

    def test_provider_item_unwraps_raw(self) -> None:
        raw: dict[str, Any] = {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "claude"},
        }
        provider: dict[str, Any] = {
            "type": "provider_item",
            "item_type": "web_search_call",
            "raw": raw,
        }
        out = OpenAIResponsesConverter.items_to_input([provider])  # type: ignore[list-item]
        # The ``raw`` payload is the verbatim Responses-API item — echo it.
        assert [_as_dict(x) for x in out] == [raw]

    def test_provider_item_with_non_dict_raw_dropped(self, caplog: Any) -> None:
        bogus: dict[str, Any] = {
            "type": "provider_item",
            "item_type": "x",
            "raw": "not-a-dict",
        }
        with caplog.at_level("WARNING"):
            out = OpenAIResponsesConverter.items_to_input([bogus])  # type: ignore[list-item]
        assert out == []

    def test_bare_input_text_wrapped_in_user_message(self) -> None:
        part: Any = {"type": "input_text", "text": "bare"}
        out = OpenAIResponsesConverter.items_to_input([part])
        assert len(out) == 1
        msg = _as_dict(out[0])
        assert msg["role"] == "user"
        content = msg["content"]
        assert isinstance(content, list)
        assert content[0] == part

    def test_bare_input_image_wrapped_in_user_message(self) -> None:
        part: Any = {
            "type": "input_image",
            "image_url": "https://example.com/x.png",
            "detail": "auto",
        }
        out = OpenAIResponsesConverter.items_to_input([part])
        assert _as_dict(out[0])["role"] == "user"

    def test_unknown_role_coerced_to_user(self) -> None:
        msg: Any = {"role": "moderator", "content": "hello"}
        out = OpenAIResponsesConverter.items_to_input([msg])
        assert _as_dict(out[0])["role"] == "user"

    def test_unrecognised_item_dropped(self, caplog: Any) -> None:
        with caplog.at_level("WARNING"):
            out = OpenAIResponsesConverter.items_to_input([{"type": "mystery"}])  # type: ignore[list-item]
        assert out == []


# ==================================================================
# convert_tools
# ==================================================================


class TestConvertTools:
    def test_function_tool_emits_function_param_shape(self) -> None:
        tool = _dummy_tool()
        out = OpenAIResponsesConverter.convert_tools([tool])
        assert len(out) == 1
        param = _as_dict(out[0])
        assert param["type"] == "function"
        assert param["name"] == "lookup"
        assert param["description"] == "Look something up"
        assert param["strict"] is False
        params_schema = param["parameters"]
        assert isinstance(params_schema, dict)
        assert params_schema.get("type") == "object"
        # The schema-normalizer may add description/required — assert shape,
        # not exact content.
        props = params_schema.get("properties", {})
        assert "q" in props

    def test_function_tool_strict_mode_on(self) -> None:
        tool = _dummy_tool(enforcement=SchemaEnforcement.STRICT)
        out = OpenAIResponsesConverter.convert_tools([tool])
        assert _as_dict(out[0])["strict"] is True

    def test_function_tool_without_description_omits_key(self) -> None:
        tool = _dummy_tool(description=None)
        out = OpenAIResponsesConverter.convert_tools([tool])
        assert "description" not in _as_dict(out[0])

    def test_executable_builtin_tool_is_flattened_to_function(self) -> None:
        tool = _echo_executable_tool()
        out = OpenAIResponsesConverter.convert_tools([tool])
        assert len(out) == 1
        param = _as_dict(out[0])
        assert param["type"] == "function"
        assert param["name"] == "echo"
        assert param["strict"] is True
        assert param["description"] == "Echo the input back."

    def test_pydantic_schema_model_converted_to_json_schema(self) -> None:
        class _Input(BaseModel):
            x: int

        tool = ExecutableBuiltinTool(
            name="typed",
            description="",
            schema=_Input,
            on_invoke=_noop_invoke,
        )
        out = OpenAIResponsesConverter.convert_tools([tool])
        param = _as_dict(out[0])
        assert param["name"] == "typed"
        params = param["parameters"]
        assert isinstance(params, dict)
        assert params.get("type") == "object"
        # Pydantic-derived schemas expose ``x`` under ``properties``.
        assert "x" in params.get("properties", {})

    def test_unknown_tool_type_is_skipped_with_warning(self, caplog: Any) -> None:
        class _NotATool:
            pass

        with caplog.at_level("WARNING"):
            out = OpenAIResponsesConverter.convert_tools([_NotATool()])  # type: ignore[list-item]
        assert out == []


# ==================================================================
# convert_tool_choice
# ==================================================================


class TestConvertToolChoice:
    def test_none_when_tool_choice_unset(self) -> None:
        assert OpenAIResponsesConverter.convert_tool_choice(None, tools_present=True) is None

    def test_none_when_tools_absent(self) -> None:
        assert OpenAIResponsesConverter.convert_tool_choice("auto", tools_present=False) is None

    @pytest.mark.parametrize("mode", ["auto", "required", "none"])
    def test_string_literals_returned_as_is(self, mode: str) -> None:
        assert OpenAIResponsesConverter.convert_tool_choice(mode, tools_present=True) == mode

    def test_named_tool_uses_flat_shape(self) -> None:
        """Responses API named-tool shape MUST be flat, not nested."""
        result = OpenAIResponsesConverter.convert_tool_choice("lookup", tools_present=True)
        assert isinstance(result, dict)
        payload = _as_dict(result)
        assert payload == {"type": "function", "name": "lookup"}
        # Critical contrast against Chat Completions' nested shape:
        assert "function" not in payload


# ==================================================================
# resolve_response_format
# ==================================================================


class _FakeSchema:
    """Minimal stand-in for ``AgentOutputSchemaBase`` for converter tests."""

    def name(self) -> str:
        return "ThingOutput"

    def json_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"x": {"type": "integer"}}}

    def is_strict_json_schema(self) -> bool:
        return True


class TestResolveResponseFormat:
    def test_none_when_no_schema(self) -> None:
        assert OpenAIResponsesConverter.resolve_response_format(None) is None

    def test_builds_response_text_config(self) -> None:
        cfg = OpenAIResponsesConverter.resolve_response_format(_FakeSchema())  # type: ignore[arg-type]
        assert cfg is not None
        # ``format`` is a NotRequired field on ``ResponseTextConfigParam``;
        # go through ``dict.get`` so the narrowing is explicit, then assert
        # non-None to keep the rest of the test readable.
        fmt_raw = dict(cfg).get("format")
        assert fmt_raw is not None
        fmt = _as_dict(fmt_raw)
        assert fmt["type"] == "json_schema"
        assert fmt["name"] == "ThingOutput"
        assert fmt["strict"] is True
        assert fmt["schema"] == {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }


# ==================================================================
# response_to_llm_response
# ==================================================================


class TestResponseToLLMResponse:
    def test_text_message_maps_to_llm_response_text(self) -> None:
        text_block = ResponseOutputText.model_construct(type="output_text", text="Hello!", annotations=[])
        msg = ResponseOutputMessage.model_construct(
            id="msg_1",
            type="message",
            role="assistant",
            content=[text_block],
            status="completed",
        )
        response = _make_response([msg])
        out = OpenAIResponsesConverter.response_to_llm_response(response)
        assert len(out.response) == 1
        part = out.response[0]
        assert isinstance(part, LLMResponseText)
        assert part.text == "Hello!"
        assert out.finish_reason == "stop"

    def test_refusal_block_maps_to_refusal_part(self) -> None:
        refusal_block = ResponseOutputRefusal.model_construct(type="refusal", refusal="I can't help with that.")
        msg = ResponseOutputMessage.model_construct(
            id="msg_1",
            type="message",
            role="assistant",
            content=[refusal_block],
            status="completed",
        )
        response = _make_response([msg])
        out = OpenAIResponsesConverter.response_to_llm_response(response)
        parts = [p for p in out.response if isinstance(p, LLMResponseRefusal)]
        assert len(parts) == 1
        assert parts[0].refusal == "I can't help with that."
        assert out.finish_reason == "refusal"

    def test_function_call_maps_to_function_call_part(self) -> None:
        fn = ResponseFunctionToolCall.model_construct(
            id="fc_1",
            type="function_call",
            call_id="call_1",
            name="lookup",
            arguments='{"q":"x"}',
            status="completed",
        )
        response = _make_response([fn])
        out = OpenAIResponsesConverter.response_to_llm_response(response)
        assert len(out.response) == 1
        part = out.response[0]
        assert isinstance(part, LLMResponseFunctionToolCall)
        assert part.call_id == "call_1"
        assert part.name == "lookup"
        assert part.arguments == '{"q":"x"}'
        assert out.finish_reason == "tool_calls"

    def test_reasoning_item_maps_with_summary_and_content(self) -> None:
        reasoning = ResponseReasoningItem.model_construct(
            id="rs_1",
            type="reasoning",
            summary=[ReasoningSummary.model_construct(type="summary_text", text="quick thought")],
            content=[ReasoningContent.model_construct(type="reasoning_text", text="full chain")],
            encrypted_content="opaque",
            status="completed",
        )
        response = _make_response([reasoning])
        out = OpenAIResponsesConverter.response_to_llm_response(response)
        assert len(out.response) == 1
        part = out.response[0]
        assert isinstance(part, LLMResponseReasoning)
        assert part.thinking == "full chain"
        assert part.summary == "quick thought"
        assert part.encrypted_content == "opaque"
        assert part.id == "rs_1"

    def test_reasoning_item_with_only_summary(self) -> None:
        reasoning = ResponseReasoningItem.model_construct(
            id="rs_1",
            type="reasoning",
            summary=[ReasoningSummary.model_construct(type="summary_text", text="s")],
            content=None,
            encrypted_content=None,
            status="completed",
        )
        response = _make_response([reasoning])
        out = OpenAIResponsesConverter.response_to_llm_response(response)
        part = out.response[0]
        assert isinstance(part, LLMResponseReasoning)
        assert part.summary == "s"
        assert part.thinking == ""

    def test_hosted_tool_item_routes_to_provider_item(self) -> None:
        """A ``file_search_call`` (or any non-core item) must land in
        :class:`LLMResponseProviderItem` with the verbatim payload."""

        class _FileSearchCall(BaseModel):
            type: str = "file_search_call"
            id: str = "fs_1"
            status: str = "completed"
            queries: list[str] = ["claude"]

        response = _make_response([_FileSearchCall()])
        out = OpenAIResponsesConverter.response_to_llm_response(response)
        assert len(out.response) == 1
        part = out.response[0]
        assert isinstance(part, LLMResponseProviderItem)
        assert part.item_type == "file_search_call"
        assert part.raw["id"] == "fs_1"
        assert part.raw["status"] == "completed"
        assert part.raw["queries"] == ["claude"]

    def test_finish_reason_length_on_incomplete_max_tokens(self) -> None:
        class _IncompleteDetails(BaseModel):
            reason: str = "max_output_tokens"

        response = _make_response(
            [],
            status="incomplete",
            incomplete_details=_IncompleteDetails(),
        )
        out = OpenAIResponsesConverter.response_to_llm_response(response)
        assert out.finish_reason == "length"

    def test_finish_reason_incomplete_on_unknown_reason(self) -> None:
        class _IncompleteDetails(BaseModel):
            reason: str = "content_filter"

        response = _make_response(
            [],
            status="incomplete",
            incomplete_details=_IncompleteDetails(),
        )
        out = OpenAIResponsesConverter.response_to_llm_response(response)
        assert out.finish_reason == "incomplete"

    def test_incomplete_wins_over_truncated_tool_call(self) -> None:
        # A truncated response can still carry a (partial) function call whose
        # JSON arguments were cut off. finish_reason must be length/incomplete
        # so the Runner does NOT execute the tool with malformed arguments.
        class _IncompleteDetails(BaseModel):
            reason: str = "max_output_tokens"

        fn = ResponseFunctionToolCall.model_construct(
            id="fc_1",
            type="function_call",
            call_id="call_1",
            name="lookup",
            arguments='{"q":"unterm',  # truncated JSON
            status="incomplete",
        )
        response = _make_response(
            [fn],
            status="incomplete",
            incomplete_details=_IncompleteDetails(),
        )
        out = OpenAIResponsesConverter.response_to_llm_response(response)
        assert out.finish_reason == "length"
        # The call is still surfaced (for tracing) but not signalled as a
        # ready-to-run tool_calls turn.
        assert any(isinstance(p, LLMResponseFunctionToolCall) for p in out.response)

    def test_failed_status_maps_to_error(self) -> None:
        response = _make_response([], status="failed")
        out = OpenAIResponsesConverter.response_to_llm_response(response)
        assert out.finish_reason == "error"

    def test_response_metadata_carried_through(self) -> None:
        response = _make_response(
            [],
            response_id="resp_xyz",
            model="gpt-5.1-mini",
            usage=ResponseUsage.model_construct(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                input_tokens_details=SDKInputTokensDetails.model_construct(cached_tokens=3),
                output_tokens_details=SDKOutputTokensDetails.model_construct(reasoning_tokens=2),
            ),
        )
        out = OpenAIResponsesConverter.response_to_llm_response(response)
        assert out.response_id == "resp_xyz"
        assert out.model == "gpt-5.1-mini"
        assert out.usage is not None
        assert out.usage.total_tokens == 15


# ==================================================================
# parse_usage
# ==================================================================


class TestParseUsage:
    def test_maps_tokens_and_details(self) -> None:
        usage = ResponseUsage.model_construct(
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            input_tokens_details=SDKInputTokensDetails.model_construct(cached_tokens=25),
            output_tokens_details=SDKOutputTokensDetails.model_construct(reasoning_tokens=10),
        )
        out = OpenAIResponsesConverter.parse_usage(usage)
        assert out.requests == 1
        assert out.input_tokens == 100
        assert out.output_tokens == 40
        assert out.total_tokens == 140
        assert out.input_tokens_details.cached_tokens == 25
        assert out.input_tokens_details.cache_creation_input_tokens == 0
        assert out.output_tokens_details.reasoning_tokens == 10

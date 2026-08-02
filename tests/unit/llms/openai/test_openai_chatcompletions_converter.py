"""Tests for :class:`OpenAIChatCompletionsConverter`.

Covers the Layer 1 ↔ ``openai.types.chat.*`` wire-type bridge:

- ``items_to_messages`` — every ``LLMInputContentItem`` variant,
  including ``function_call`` folding into one assistant message and
  ``function_call_output`` → ``role="tool"`` messages.
- ``convert_tools`` — ``FunctionTool`` and ``ExecutableBuiltinTool``
  routed into the ``{type: "function", function: {...}}`` shape.
- ``convert_tool_choice`` — MUST emit the NESTED Chat-Completions shape
  ``{"type": "function", "function": {"name": ...}}``. This is the
  load-bearing contrast with the Responses-API FLAT shape.
- ``resolve_response_format`` — structured output via the
  ``json_schema`` dict.
- ``chat_completion_to_llm_response`` — text / tool_call / refusal
  branches.
- ``parse_usage`` — cached- / reasoning-token mapping.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from pydantic import BaseModel

from troopai.adk.llms.openai.openai_chatcompletions_converter import (
    OpenAIChatCompletionsConverter,
)
from troopai.adk.schemas.utils import SchemaEnforcement
from troopai.adk.tools.builtin.builtin_tool import ExecutableBuiltinTool
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.responses.llm_response import (
    LLMResponseFunctionToolCall,
    LLMResponseRefusal,
    LLMResponseText,
)

FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


# ==================================================================
# Helpers
# ==================================================================


async def _noop_invoke(_ctx: Any, raw_args: str) -> str:  # pragma: no cover
    return raw_args


def _dummy_tool(
    name: str = "lookup",
    *,
    description: str | None = "Look something up",
    enforcement: SchemaEnforcement = SchemaEnforcement.NORMALIZED,
) -> FunctionTool:
    """Minimal ``FunctionTool`` for ``convert_tools`` tests."""
    return FunctionTool(
        name=name,
        schema={"type": "object", "properties": {"q": {"type": "string"}}},
        description=description,
        schema_enforcement=enforcement,
        on_invoke=_noop_invoke,
    )


def _echo_executable_tool() -> ExecutableBuiltinTool:
    """Build an ``ExecutableBuiltinTool`` with a dict schema."""
    return ExecutableBuiltinTool(
        name="echo",
        description="Echo the input back.",
        schema={"type": "object", "properties": {"msg": {"type": "string"}}},
        on_invoke=_noop_invoke,
    )


def _as_dict(obj: Any) -> dict[str, Any]:
    """Cast a TypedDict-union element to ``dict`` for test assertions.

    The converter returns ``list[ChatCompletionMessageParam]`` — a TypedDict
    union that pyright cannot narrow without a runtime discriminator. Casting
    to a plain dict at the test boundary is semantically free (TypedDicts
    ARE dicts at runtime) and keeps the test file free of scattered
    ``# type: ignore`` markers.
    """
    assert isinstance(obj, dict)
    return dict(obj)


def _make_completion(
    *,
    message: ChatCompletionMessage,
    finish_reason: FinishReason = "stop",
    usage: CompletionUsage | None = None,
    completion_id: str = "chatcmpl_abc",
    model: str = "gpt-4o",
) -> ChatCompletion:
    """Build a minimal ``ChatCompletion`` object for converter tests."""
    return ChatCompletion.model_construct(
        id=completion_id,
        model=model,
        object="chat.completion",
        created=0,
        choices=[
            Choice.model_construct(
                finish_reason=finish_reason,
                index=0,
                logprobs=None,
                message=message,
            )
        ],
        usage=usage,
    )


# ==================================================================
# items_to_messages
# ==================================================================


class TestItemsToMessages:
    def test_empty_list_returns_empty_list(self) -> None:
        assert OpenAIChatCompletionsConverter.items_to_messages([]) == []

    def test_easy_message_string_content_becomes_user_message(self) -> None:
        msg: Any = {"role": "user", "content": "hello"}
        out = OpenAIChatCompletionsConverter.items_to_messages([msg])
        assert len(out) == 1
        assert _as_dict(out[0]) == {"role": "user", "content": "hello"}

    def test_system_message_routed_to_system_role(self) -> None:
        msg: Any = {"role": "system", "content": "You are helpful."}
        out = OpenAIChatCompletionsConverter.items_to_messages([msg])
        assert _as_dict(out[0]) == {"role": "system", "content": "You are helpful."}

    def test_developer_message_mapped_to_system_role(self) -> None:
        """Chat Completions treats developer/system as equivalent — we normalise to system."""
        msg: Any = {"role": "developer", "content": "You are helpful."}
        out = OpenAIChatCompletionsConverter.items_to_messages([msg])
        assert _as_dict(out[0])["role"] == "system"

    def test_strict_message_with_user_role_becomes_user_message(self) -> None:
        msg: Any = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "yo"}],
        }
        out = OpenAIChatCompletionsConverter.items_to_messages([msg])
        wire = _as_dict(out[0])
        assert wire["role"] == "user"
        content = wire["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "yo"}

    def test_user_message_image_part_becomes_image_url(self) -> None:
        # A multimodal user turn previously dropped its image, silently
        # reducing to text-only.
        msg: Any = {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what is this?"},
                {"type": "input_image", "image_url": "https://ex.com/cat.png", "detail": "high"},
            ],
        }
        out = OpenAIChatCompletionsConverter.items_to_messages([msg])
        content = _as_dict(out[0])["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "what is this?"}
        assert content[1] == {
            "type": "image_url",
            "image_url": {"url": "https://ex.com/cat.png", "detail": "high"},
        }

    def test_user_message_image_without_detail_omits_detail(self) -> None:
        msg: Any = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}],
        }
        out = OpenAIChatCompletionsConverter.items_to_messages([msg])
        content = _as_dict(out[0])["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}

    def test_user_message_audio_part_becomes_input_audio(self) -> None:
        msg: Any = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_audio", "data": "Zm9v", "format": "wav"}],
        }
        out = OpenAIChatCompletionsConverter.items_to_messages([msg])
        content = _as_dict(out[0])["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "input_audio", "input_audio": {"data": "Zm9v", "format": "wav"}}

    def test_user_message_malformed_image_dropped(self) -> None:
        # Missing image_url → dropped, not sent as an empty reference.
        msg: Any = {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "hi"},
                {"type": "input_image"},
            ],
        }
        out = OpenAIChatCompletionsConverter.items_to_messages([msg])
        content = _as_dict(out[0])["content"]
        assert isinstance(content, list)
        assert content == [{"type": "text", "text": "hi"}]

    def test_assistant_message_with_text_flattens_content(self) -> None:
        msg: Any = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi there"}],
        }
        out = OpenAIChatCompletionsConverter.items_to_messages([msg])
        wire = _as_dict(out[0])
        assert wire["role"] == "assistant"
        # Single text chunk collapses to a string.
        assert wire["content"] == "hi there"

    def test_assistant_message_with_refusal_sets_refusal_field(self) -> None:
        msg: Any = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "refusal", "refusal": "I can't help."}],
        }
        out = OpenAIChatCompletionsConverter.items_to_messages([msg])
        wire = _as_dict(out[0])
        assert wire["role"] == "assistant"
        assert wire["refusal"] == "I can't help."

    def test_function_call_output_becomes_tool_role_message(self) -> None:
        """`function_call_output` MUST emit a ``role="tool"`` message keyed by ``tool_call_id``."""
        out_item: Any = {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "42",
        }
        out = OpenAIChatCompletionsConverter.items_to_messages([out_item])
        wire = _as_dict(out[0])
        assert wire == {"role": "tool", "tool_call_id": "call_1", "content": "42"}

    def test_function_call_output_non_string_output_stringified(self) -> None:
        out_item: Any = {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": {"a": 1},
        }
        out = OpenAIChatCompletionsConverter.items_to_messages([out_item])
        assert _as_dict(out[0])["content"] == "{'a': 1}"

    def test_orphan_function_call_output_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """`function_call_output` with no preceding `function_call` MUST warn.

        The Chat Completions API rejects tool messages that don't
        follow an assistant message carrying the matching tool_call
        id. Surfacing the issue at conversion time makes the source
        of the malformed history visible before the upstream 400.
        """
        import logging

        out_item: Any = {
            "type": "function_call_output",
            "call_id": "orphan_1",
            "output": "oops",
        }
        with caplog.at_level(logging.WARNING, logger="troopai.adk.llms.openai.openai_chatcompletions_converter"):
            out = OpenAIChatCompletionsConverter.items_to_messages([out_item])

        # Conversion still emits the tool message — the warning is
        # diagnostic, not a short-circuit. Downstream tests on this
        # codepath still exercise the happy shape.
        assert len(out) == 1
        assert _as_dict(out[0])["role"] == "tool"
        assert any(
            "has no preceding function_call" in record.message and "orphan_1" in record.message
            for record in caplog.records
        )

    def test_function_call_output_after_assistant_tool_calls_no_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A `function_call` + `function_call_output` pair MUST NOT warn."""
        import logging

        items: Any = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
            },
        ]
        with caplog.at_level(logging.WARNING, logger="troopai.adk.llms.openai.openai_chatcompletions_converter"):
            OpenAIChatCompletionsConverter.items_to_messages(items)

        assert not any("has no preceding function_call" in record.message for record in caplog.records)

    def test_single_function_call_folds_into_assistant_tool_calls(self) -> None:
        """One `function_call` item → one assistant message with one tool_call entry."""
        call: Any = {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q": "x"}',
        }
        out = OpenAIChatCompletionsConverter.items_to_messages([call])
        assert len(out) == 1
        wire = _as_dict(out[0])
        assert wire["role"] == "assistant"
        assert wire["content"] is None
        tool_calls = wire["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0] == {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q": "x"}'},
        }

    def test_multiple_function_calls_fold_into_one_assistant_message(self) -> None:
        """Consecutive `function_call` items MUST fold into ONE assistant message
        with the full ``tool_calls`` list. This is the core Chat-Completions
        shape requirement the fold-on-the-way-in invariant is meant to uphold.
        """
        calls: list[Any] = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"q": "x"}',
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "fetch",
                "arguments": '{"url": "y"}',
            },
        ]
        out = OpenAIChatCompletionsConverter.items_to_messages(calls)
        assert len(out) == 1, "Multiple function_calls MUST fold into one assistant message"
        wire = _as_dict(out[0])
        assert wire["role"] == "assistant"
        assert wire["content"] is None
        tool_calls = wire["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["id"] == "call_1"
        assert tool_calls[1]["id"] == "call_2"

    def test_function_call_then_output_pair(self) -> None:
        """Realistic tool turn: assistant issues one call, then the result flows back."""
        stream: list[Any] = [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"q": "x"}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "42"},
        ]
        out = OpenAIChatCompletionsConverter.items_to_messages(stream)
        assert len(out) == 2
        assistant = _as_dict(out[0])
        assert assistant["role"] == "assistant"
        assert len(assistant["tool_calls"]) == 1
        tool_msg = _as_dict(out[1])
        assert tool_msg == {"role": "tool", "tool_call_id": "call_1", "content": "42"}

    def test_two_function_calls_then_two_outputs(self) -> None:
        """Two calls + two outputs: ONE assistant message with TWO tool_calls,
        followed by TWO tool messages — the Chat-Completions wire canonical shape."""
        stream: list[Any] = [
            {"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{}"},
            {"type": "function_call", "call_id": "c2", "name": "b", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "r1"},
            {"type": "function_call_output", "call_id": "c2", "output": "r2"},
        ]
        out = OpenAIChatCompletionsConverter.items_to_messages(stream)
        assert len(out) == 3
        assert _as_dict(out[0])["role"] == "assistant"
        assert len(_as_dict(out[0])["tool_calls"]) == 2
        assert _as_dict(out[1]) == {"role": "tool", "tool_call_id": "c1", "content": "r1"}
        assert _as_dict(out[2]) == {"role": "tool", "tool_call_id": "c2", "content": "r2"}

    def test_multi_output_replay_does_not_warn_on_later_outputs(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Multiple `function_call_output` items after ONE multi-call assistant
        message are valid history. Only the first output flushes the pending
        calls into the assistant message; the 2nd..Nth outputs sit behind
        sibling `role="tool"` messages but are still preceded by that assistant
        message. The converter MUST scan past those sibling tool messages and
        NOT emit a spurious "no preceding function_call" warning for them.
        """
        import logging

        stream: list[Any] = [
            {"type": "function_call", "call_id": "c1", "name": "a", "arguments": "{}"},
            {"type": "function_call", "call_id": "c2", "name": "b", "arguments": "{}"},
            {"type": "function_call", "call_id": "c3", "name": "c", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "r1"},
            {"type": "function_call_output", "call_id": "c2", "output": "r2"},
            {"type": "function_call_output", "call_id": "c3", "output": "r3"},
        ]
        with caplog.at_level(logging.WARNING, logger="troopai.adk.llms.openai.openai_chatcompletions_converter"):
            out = OpenAIChatCompletionsConverter.items_to_messages(stream)

        # Shape stays correct: one assistant message + three tool messages.
        assert len(out) == 4
        # No spurious malformed-history warning for any of the outputs.
        assert not any("has no preceding function_call" in record.message for record in caplog.records)

    def test_reasoning_replay_dropped_with_warning(self, caplog: Any) -> None:
        """Chat Completions cannot replay reasoning — the converter MUST drop
        reasoning items with a warning. Contrast with the Responses API where
        reasoning IS replayable."""
        reasoning: Any = {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "thought"}],
            "id": "rs_1",
            "encrypted_content": "opaque",
        }
        with caplog.at_level("WARNING"):
            out = OpenAIChatCompletionsConverter.items_to_messages([reasoning])
        assert out == []
        assert any("reasoning" in rec.message.lower() for rec in caplog.records)

    def test_provider_item_dropped_with_warning(self, caplog: Any) -> None:
        """Hosted-tool items (file_search, web_search, etc.) cannot be replayed
        through Chat Completions; the converter drops them with a warning."""
        provider: Any = {
            "type": "provider_item",
            "item_type": "web_search_call",
            "raw": {"type": "web_search_call", "id": "ws_1"},
        }
        with caplog.at_level("WARNING"):
            out = OpenAIChatCompletionsConverter.items_to_messages([provider])
        assert out == []

    def test_bare_input_text_wrapped_in_user_message(self) -> None:
        part: Any = {"type": "input_text", "text": "bare"}
        out = OpenAIChatCompletionsConverter.items_to_messages([part])
        wire = _as_dict(out[0])
        assert wire["role"] == "user"
        content = wire["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "bare"}

    def test_unrecognised_item_dropped_with_warning(self, caplog: Any) -> None:
        mystery: Any = {"type": "mystery"}
        with caplog.at_level("WARNING"):
            out = OpenAIChatCompletionsConverter.items_to_messages([mystery])
        assert out == []


# ==================================================================
# convert_tools
# ==================================================================


class TestConvertTools:
    def test_function_tool_emits_nested_function_shape(self) -> None:
        """Chat Completions tool shape MUST nest the function definition under a
        ``function`` key — this is the load-bearing contrast with the flat
        Responses-API shape."""
        tool = _dummy_tool()
        out = OpenAIChatCompletionsConverter.convert_tools([tool])
        assert len(out) == 1
        param = _as_dict(out[0])
        assert param["type"] == "function"
        fn = param["function"]
        assert isinstance(fn, dict)
        assert fn["name"] == "lookup"
        assert fn["description"] == "Look something up"
        assert fn["strict"] is False
        params_schema = fn["parameters"]
        assert isinstance(params_schema, dict)
        assert params_schema.get("type") == "object"
        assert "q" in params_schema.get("properties", {})

    def test_function_tool_strict_mode_on(self) -> None:
        tool = _dummy_tool(enforcement=SchemaEnforcement.STRICT)
        out = OpenAIChatCompletionsConverter.convert_tools([tool])
        fn = _as_dict(out[0])["function"]
        assert fn["strict"] is True

    def test_function_tool_without_description_omits_key(self) -> None:
        tool = _dummy_tool(description=None)
        out = OpenAIChatCompletionsConverter.convert_tools([tool])
        fn = _as_dict(out[0])["function"]
        assert "description" not in fn

    def test_executable_builtin_tool_flattened_to_function(self) -> None:
        tool = _echo_executable_tool()
        out = OpenAIChatCompletionsConverter.convert_tools([tool])
        assert len(out) == 1
        param = _as_dict(out[0])
        assert param["type"] == "function"
        fn = param["function"]
        assert fn["name"] == "echo"
        assert fn["strict"] is True
        assert fn["description"] == "Echo the input back."

    def test_pydantic_schema_model_converted_to_json_schema(self) -> None:
        class _Input(BaseModel):
            x: int

        tool = ExecutableBuiltinTool(
            name="typed",
            description="",
            schema=_Input,
            on_invoke=_noop_invoke,
        )
        out = OpenAIChatCompletionsConverter.convert_tools([tool])
        fn = _as_dict(out[0])["function"]
        assert fn["name"] == "typed"
        params = fn["parameters"]
        assert isinstance(params, dict)
        assert params.get("type") == "object"
        assert "x" in params.get("properties", {})

    def test_unknown_tool_type_skipped_with_warning(self, caplog: Any) -> None:
        class _NotATool:
            pass

        with caplog.at_level("WARNING"):
            out = OpenAIChatCompletionsConverter.convert_tools([_NotATool()])  # type: ignore[list-item]
        assert out == []


# ==================================================================
# convert_tool_choice  (NESTED shape — contrast with Responses FLAT)
# ==================================================================


class TestConvertToolChoice:
    def test_none_when_tool_choice_unset(self) -> None:
        assert OpenAIChatCompletionsConverter.convert_tool_choice(None, tools_present=True) is None

    def test_none_when_tools_absent(self) -> None:
        assert OpenAIChatCompletionsConverter.convert_tool_choice("auto", tools_present=False) is None

    @pytest.mark.parametrize("mode", ["auto", "required", "none"])
    def test_string_literals_returned_as_is(self, mode: str) -> None:
        assert OpenAIChatCompletionsConverter.convert_tool_choice(mode, tools_present=True) == mode

    def test_named_tool_uses_nested_shape(self) -> None:
        """Chat Completions named-tool shape MUST be NESTED:
        ``{type: function, function: {name: ...}}``. This is the load-bearing
        contrast with the Responses API's flat ``{type: function, name: ...}``.
        """
        result = OpenAIChatCompletionsConverter.convert_tool_choice("lookup", tools_present=True)
        assert isinstance(result, dict)
        payload = _as_dict(result)
        assert payload == {"type": "function", "function": {"name": "lookup"}}
        # Explicit contrast assertion: the nested ``function`` key MUST be present.
        assert "function" in payload
        # And the FLAT-shape ``name`` key MUST NOT be at the top level.
        assert "name" not in payload


# ==================================================================
# resolve_response_format
# ==================================================================


class _FakeSchema:
    def name(self) -> str:
        return "ThingOutput"

    def json_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"x": {"type": "integer"}}}

    def is_strict_json_schema(self) -> bool:
        return True


class TestResolveResponseFormat:
    def test_none_when_no_schema(self) -> None:
        assert OpenAIChatCompletionsConverter.resolve_response_format(None) is None

    def test_builds_json_schema_dict(self) -> None:
        cfg = OpenAIChatCompletionsConverter.resolve_response_format(
            _FakeSchema()  # type: ignore[arg-type]
        )
        assert cfg is not None
        assert cfg["type"] == "json_schema"
        inner = cfg["json_schema"]
        assert isinstance(inner, dict)
        assert inner["name"] == "ThingOutput"
        assert inner.get("strict") is True
        assert inner.get("schema") == {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }


# ==================================================================
# chat_completion_to_llm_response
# ==================================================================


class TestChatCompletionToLLMResponse:
    def test_text_message_maps_to_llm_response_text(self) -> None:
        msg = ChatCompletionMessage.model_construct(
            role="assistant",
            content="Hello!",
            refusal=None,
            tool_calls=None,
        )
        completion = _make_completion(message=msg, finish_reason="stop")
        out = OpenAIChatCompletionsConverter.chat_completion_to_llm_response(completion)
        assert len(out.response) == 1
        part = out.response[0]
        assert isinstance(part, LLMResponseText)
        assert part.text == "Hello!"
        assert out.finish_reason == "stop"

    def test_refusal_takes_precedence_over_content(self) -> None:
        msg = ChatCompletionMessage.model_construct(
            role="assistant",
            content=None,
            refusal="I can't help with that.",
            tool_calls=None,
        )
        completion = _make_completion(message=msg, finish_reason="stop")
        out = OpenAIChatCompletionsConverter.chat_completion_to_llm_response(completion)
        parts = [p for p in out.response if isinstance(p, LLMResponseRefusal)]
        assert len(parts) == 1
        assert parts[0].refusal == "I can't help with that."
        # The converter overrides finish_reason to "refusal" when a refusal is present.
        assert out.finish_reason == "refusal"

    def test_function_tool_call_maps_to_function_call_part(self) -> None:
        call = ChatCompletionMessageToolCall.model_construct(
            id="call_1",
            type="function",
            function=Function.model_construct(name="lookup", arguments='{"q":"x"}'),
        )
        msg = ChatCompletionMessage.model_construct(
            role="assistant",
            content=None,
            refusal=None,
            tool_calls=[call],
        )
        completion = _make_completion(message=msg, finish_reason="tool_calls")
        out = OpenAIChatCompletionsConverter.chat_completion_to_llm_response(completion)
        fn_parts = [p for p in out.response if isinstance(p, LLMResponseFunctionToolCall)]
        assert len(fn_parts) == 1
        part = fn_parts[0]
        assert part.call_id == "call_1"
        assert part.name == "lookup"
        assert part.arguments == '{"q":"x"}'
        assert out.finish_reason == "tool_calls"

    def test_content_and_tool_calls_coexist(self) -> None:
        """OpenAI may emit both free-text and a tool_call in the same turn.
        The converter MUST surface both parts."""
        call = ChatCompletionMessageToolCall.model_construct(
            id="call_1",
            type="function",
            function=Function.model_construct(name="lookup", arguments="{}"),
        )
        msg = ChatCompletionMessage.model_construct(
            role="assistant",
            content="Let me check that for you.",
            refusal=None,
            tool_calls=[call],
        )
        completion = _make_completion(message=msg, finish_reason="tool_calls")
        out = OpenAIChatCompletionsConverter.chat_completion_to_llm_response(completion)
        text_parts = [p for p in out.response if isinstance(p, LLMResponseText)]
        fn_parts = [p for p in out.response if isinstance(p, LLMResponseFunctionToolCall)]
        assert len(text_parts) == 1
        assert text_parts[0].text == "Let me check that for you."
        assert len(fn_parts) == 1

    def test_empty_choices_returns_empty_response(self) -> None:
        completion = ChatCompletion.model_construct(
            id="chatcmpl_empty",
            model="gpt-4o",
            object="chat.completion",
            created=0,
            choices=[],
            usage=None,
        )
        out = OpenAIChatCompletionsConverter.chat_completion_to_llm_response(completion)
        assert out.response == []
        assert out.finish_reason == "stop"

    def test_metadata_carried_through(self) -> None:
        msg = ChatCompletionMessage.model_construct(
            role="assistant",
            content="ok",
            refusal=None,
            tool_calls=None,
        )
        completion = _make_completion(
            message=msg,
            finish_reason="stop",
            completion_id="chatcmpl_xyz",
            model="gpt-4o-mini",
            usage=CompletionUsage.model_construct(
                prompt_tokens=10,
                completion_tokens=3,
                total_tokens=13,
                prompt_tokens_details=PromptTokensDetails.model_construct(cached_tokens=2),
                completion_tokens_details=CompletionTokensDetails.model_construct(reasoning_tokens=1),
            ),
        )
        out = OpenAIChatCompletionsConverter.chat_completion_to_llm_response(completion)
        assert out.response_id == "chatcmpl_xyz"
        assert out.model == "gpt-4o-mini"
        assert out.usage is not None
        assert out.usage.total_tokens == 13


# ==================================================================
# parse_usage
# ==================================================================


class TestParseUsage:
    def test_maps_prompt_and_completion_tokens(self) -> None:
        usage = CompletionUsage.model_construct(
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
            prompt_tokens_details=PromptTokensDetails.model_construct(cached_tokens=25),
            completion_tokens_details=CompletionTokensDetails.model_construct(reasoning_tokens=10),
        )
        out = OpenAIChatCompletionsConverter.parse_usage(usage)
        assert out.requests == 1
        assert out.input_tokens == 100
        assert out.output_tokens == 40
        assert out.total_tokens == 140
        assert out.input_tokens_details.cached_tokens == 25
        assert out.input_tokens_details.cache_creation_input_tokens == 0
        assert out.output_tokens_details.reasoning_tokens == 10

    def test_missing_details_default_to_zero(self) -> None:
        usage = CompletionUsage.model_construct(
            prompt_tokens=5,
            completion_tokens=2,
            total_tokens=7,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        )
        out = OpenAIChatCompletionsConverter.parse_usage(usage)
        assert out.input_tokens_details.cached_tokens == 0
        assert out.output_tokens_details.reasoning_tokens == 0

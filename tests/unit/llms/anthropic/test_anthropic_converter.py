"""Tests for ``AnthropicConverter``.

Covers Layer 1 → Anthropic message conversion, tool / tool_choice
conversion, response parsing, structured-output synthetic-tool
helpers, and usage extraction.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from anthropic.types import (
    Message,
    TextBlock,
    ToolUseBlock,
    Usage,
)

from troopai.adk.llms.anthropic.anthropic_converter import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    AnthropicConverter,
)
from troopai.adk.schemas import AgentOutputSchema
from troopai.adk.tools import function_tool


class TestItemsToMessages:
    def test_string_input_becomes_user_message(self) -> None:
        system, msgs = AnthropicConverter.items_to_messages("hello")
        assert system is None
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"

    def test_system_message_extracted_to_system_prompt(self) -> None:
        items: list[Any] = [
            {"type": "message", "role": "system", "content": "You are helpful."},
            {"type": "message", "role": "user", "content": "hi"},
        ]
        system, msgs = AnthropicConverter.items_to_messages(items)
        assert system == "You are helpful."
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_developer_message_also_extracted_to_system(self) -> None:
        items: list[Any] = [
            {"type": "message", "role": "developer", "content": "Be concise."},
            {"type": "message", "role": "user", "content": "ping"},
        ]
        system, _msgs = AnthropicConverter.items_to_messages(items)
        assert system == "Be concise."

    def test_multiple_system_messages_joined(self) -> None:
        items: list[Any] = [
            {"type": "message", "role": "system", "content": "A"},
            {"type": "message", "role": "system", "content": "B"},
            {"type": "message", "role": "user", "content": "q"},
        ]
        system, _msgs = AnthropicConverter.items_to_messages(items)
        assert system == "A\n\nB"

    def test_function_call_replay_produces_tool_use_block(self) -> None:
        items: list[Any] = [
            {"type": "message", "role": "user", "content": "find x"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"x": 1}',
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "found"},
        ]
        _system, msgs = AnthropicConverter.items_to_messages(items)
        # First user message
        assert msgs[0]["role"] == "user"
        # Assistant tool_use
        assert msgs[1]["role"] == "assistant"
        assistant_content = msgs[1]["content"]
        assert isinstance(assistant_content, list)
        block = cast("dict[str, Any]", assistant_content[0])
        assert block["type"] == "tool_use"
        assert block["id"] == "call_1"
        assert block["name"] == "lookup"
        assert block["input"] == {"x": 1}
        # User tool_result
        assert msgs[2]["role"] == "user"
        result_content = msgs[2]["content"]
        assert isinstance(result_content, list)
        result = cast("dict[str, Any]", result_content[0])
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "call_1"

    def test_function_call_output_with_incomplete_status_marks_is_error(self) -> None:
        items: list[Any] = [
            {"type": "message", "role": "user", "content": "x"},
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": "boom",
                "status": "incomplete",
            },
        ]
        _system, msgs = AnthropicConverter.items_to_messages(items)
        result_msg_content = msgs[-1]["content"]
        assert isinstance(result_msg_content, list)
        result = cast("dict[str, Any]", result_msg_content[0])
        assert result.get("is_error") is True

    def test_function_call_output_completed_status_no_is_error(self) -> None:
        items: list[Any] = [
            {"type": "message", "role": "user", "content": "x"},
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ]
        _system, msgs = AnthropicConverter.items_to_messages(items)
        result_content = msgs[-1]["content"]
        assert isinstance(result_content, list)
        result = cast("dict[str, Any]", result_content[0])
        assert result.get("is_error") is None

    def test_assistant_first_history_raises_instead_of_injecting(self) -> None:
        # Anthropic requires the first message to be a user turn. The
        # framework's normal flow always builds a user prompt first, so
        # only a developer-supplied assistant-first history reaches this
        # branch. Fabricating a filler "Continue." user message would
        # inject a token the developer never wrote; the converter raises
        # a clear ValueError instead.
        items: list[Any] = [
            {
                "type": "message",
                "role": "assistant",
                "content": "How can I help?",
            },
        ]
        with pytest.raises(ValueError, match="first message"):
            AnthropicConverter.items_to_messages(items)

    def test_easy_message_system_with_list_content_extracted(self) -> None:
        # Regression: an easy-message (no ``type`` key) with role system and
        # list-form content must extract its ``input_text`` parts into the
        # system prompt — not silently drop the entire instruction.
        items: list[Any] = [
            {"role": "system", "content": [{"type": "input_text", "text": "You are helpful."}]},
            {"role": "user", "content": "hi"},
        ]
        system, msgs = AnthropicConverter.items_to_messages(items)
        assert system == "You are helpful."
        assert [m["role"] for m in msgs] == ["user"]

    def test_reasoning_replay_preserves_thinking_text(self) -> None:
        # An LLMResponseReasoning replayed on a later turn must reconstruct the
        # thinking text. ``LLMResponseReasoning.to_param()`` emits the text as a
        # ``reasoning_text`` content part with the signature in ``encrypted_content``;
        # the converter must read that — NOT replay an empty thinking block (which
        # Anthropic rejects because the signature no longer matches empty content).
        from troopai.adk.types.responses.llm_response import LLMResponseReasoning

        param = LLMResponseReasoning(thinking="Let me think step by step.", signature="sig-abc", id="r1").to_param()
        # A valid Anthropic history opens with a user turn; the reasoning
        # replays as the following assistant turn.
        items: list[Any] = [{"type": "message", "role": "user", "content": "hi"}, param]
        _system, msgs = AnthropicConverter.items_to_messages(items)

        thinking_blocks = [
            block
            for m in msgs
            for block in (m["content"] if isinstance(m["content"], list) else [])
            if isinstance(block, dict) and block.get("type") == "thinking"
        ]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["thinking"] == "Let me think step by step."
        assert thinking_blocks[0]["signature"] == "sig-abc"

    def test_redacted_reasoning_replay_emits_redacted_block(self) -> None:
        # A redacted-thinking block must round-trip as a ``redacted_thinking``
        # block, NOT a plain thinking block: its opaque data is not a valid
        # signature for empty thinking content, so Anthropic rejects the replay
        # on multi-turn extended-thinking tool use if it is mis-typed.
        from troopai.adk.types.responses.llm_response import LLMResponseReasoning

        param = LLMResponseReasoning(thinking="", encrypted_content="REDACTED_BLOB", is_redacted=True).to_param()
        assert param["content"] == [{"type": "redacted_thinking", "data": "REDACTED_BLOB"}]

        # A valid Anthropic history opens with a user turn; the reasoning
        # replays as the following assistant turn.
        items: list[Any] = [{"type": "message", "role": "user", "content": "hi"}, param]
        _system, msgs = AnthropicConverter.items_to_messages(items)
        blocks = [
            block
            for m in msgs
            for block in (m["content"] if isinstance(m["content"], list) else [])
            if isinstance(block, dict)
        ]
        redacted = [b for b in blocks if b.get("type") == "redacted_thinking"]
        thinking = [b for b in blocks if b.get("type") == "thinking"]
        assert len(redacted) == 1
        assert redacted[0]["data"] == "REDACTED_BLOB"
        # No empty plain-thinking block must be emitted for the redacted block.
        assert len(thinking) == 0


class TestConvertToolChoice:
    def test_auto(self) -> None:
        result = AnthropicConverter.convert_tool_choice("auto", tools_present=True)
        assert result == {"type": "auto"}

    def test_required_maps_to_any(self) -> None:
        result = AnthropicConverter.convert_tool_choice("required", tools_present=True)
        assert result == {"type": "any"}

    def test_none(self) -> None:
        result = AnthropicConverter.convert_tool_choice("none", tools_present=True)
        assert result == {"type": "none"}

    def test_named_tool(self) -> None:
        result = AnthropicConverter.convert_tool_choice("lookup", tools_present=True)
        assert result == {"type": "tool", "name": "lookup"}

    def test_returns_none_when_no_tools(self) -> None:
        result = AnthropicConverter.convert_tool_choice("auto", tools_present=False)
        assert result is None

    def test_returns_none_when_choice_is_none(self) -> None:
        result = AnthropicConverter.convert_tool_choice(None, tools_present=True)
        assert result is None


class TestConvertTools:
    def test_function_tool_becomes_tool_param(self) -> None:
        @function_tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        wire = AnthropicConverter.convert_tools([add])
        assert len(wire) == 1
        tool = cast("dict[str, Any]", wire[0])
        assert tool["name"] == "add"
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"


class TestStructuredOutputHelpers:
    def test_build_structured_output_tool_uses_canonical_name(self) -> None:
        from pydantic import BaseModel

        class Out(BaseModel):
            answer: str

        schema = AgentOutputSchema(Out)
        tool = AnthropicConverter.build_structured_output_tool(schema)
        assert tool["name"] == STRUCTURED_OUTPUT_TOOL_NAME
        # input_schema is typed as ``object`` upstream so the SDK can
        # accept any JSON-schema-shaped dict; narrow for assertion.
        input_schema = cast("dict[str, Any]", tool["input_schema"])
        assert input_schema["type"] == "object"
        assert "answer" in input_schema.get("properties", {})

    def test_build_structured_output_tool_choice_pins_synthetic_tool(self) -> None:
        choice = AnthropicConverter.build_structured_output_tool_choice()
        assert choice == {"type": "tool", "name": STRUCTURED_OUTPUT_TOOL_NAME}

    def test_parse_structured_output_validates_tool_use_input(self) -> None:
        from pydantic import BaseModel

        class Out(BaseModel):
            answer: str

        schema = AgentOutputSchema(Out)
        message = Message(
            id="msg_1",
            type="message",
            role="assistant",
            model="claude-sonnet-4",
            content=[
                ToolUseBlock(
                    type="tool_use",
                    id="tu_1",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    input={"answer": "42"},
                )
            ],
            stop_reason="tool_use",
            stop_sequence=None,
            usage=Usage(input_tokens=5, output_tokens=10),
        )
        result = AnthropicConverter.parse_structured_output(message, schema)
        assert isinstance(result, Out)
        assert result.answer == "42"

    def test_parse_structured_output_raises_on_missing_tool_use(self) -> None:
        from pydantic import BaseModel

        class Out(BaseModel):
            answer: str

        schema = AgentOutputSchema(Out)
        message = Message(
            id="msg_1",
            type="message",
            role="assistant",
            model="claude-sonnet-4",
            content=[TextBlock(type="text", text="I refuse", citations=None)],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=5, output_tokens=10),
        )
        with pytest.raises(ValueError, match=STRUCTURED_OUTPUT_TOOL_NAME):
            AnthropicConverter.parse_structured_output(message, schema)


class TestResponseToLLMResponse:
    def test_text_only_response(self) -> None:
        message = Message(
            id="msg_1",
            type="message",
            role="assistant",
            model="claude-sonnet-4",
            content=[TextBlock(type="text", text="Hello!", citations=None)],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        response = AnthropicConverter.response_to_llm_response(message)
        assert response.response_id == "msg_1"
        assert response.content == "Hello!"
        assert response.finish_reason == "end_turn"
        assert response.usage is not None
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5

    def test_tool_use_response(self) -> None:
        message = Message(
            id="msg_2",
            type="message",
            role="assistant",
            model="claude-sonnet-4",
            content=[
                ToolUseBlock(
                    type="tool_use",
                    id="call_a",
                    name="weather",
                    input={"city": "London"},
                )
            ],
            stop_reason="tool_use",
            stop_sequence=None,
            usage=Usage(input_tokens=20, output_tokens=15),
        )
        response = AnthropicConverter.response_to_llm_response(message)
        assert len(response.tool_calls) == 1
        call = response.tool_calls[0]
        assert call.call_id == "call_a"
        assert call.name == "weather"
        assert json.loads(call.arguments) == {"city": "London"}


class TestParseUsage:
    def test_basic_usage(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50)
        parsed = AnthropicConverter._parse_usage(usage)
        assert parsed.input_tokens == 100
        assert parsed.output_tokens == 50
        assert parsed.total_tokens == 150
        assert parsed.input_tokens_details.cached_tokens == 0

    def test_cache_fields_propagated(self) -> None:
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=20,
        )
        parsed = AnthropicConverter._parse_usage(usage)
        assert parsed.input_tokens_details.cached_tokens == 80
        assert parsed.input_tokens_details.cache_creation_input_tokens == 20

    def test_input_tokens_inclusive_of_cache_tokens(self) -> None:
        # Anthropic reports ``input_tokens`` EXCLUSIVE of cache-read and
        # cache-creation tokens. Token-limit checks and cost tracking must
        # see the inclusive total (matching the litellm path, which sums
        # the same three counts into prompt_tokens) — otherwise limits
        # never trip when prompt caching is active.
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=80,
            cache_creation_input_tokens=20,
        )
        parsed = AnthropicConverter._parse_usage(usage)
        assert parsed.input_tokens == 200  # 100 + 80 + 20
        assert parsed.total_tokens == 250  # 200 + 50
        assert parsed.usage[0].input_tokens == 200
        assert parsed.usage[0].total_tokens == 250

    def test_input_tokens_no_cache_unchanged(self) -> None:
        # With no cache activity the inclusive total equals the raw count.
        usage = Usage(input_tokens=100, output_tokens=50)
        parsed = AnthropicConverter._parse_usage(usage)
        assert parsed.input_tokens == 100
        assert parsed.total_tokens == 150


class TestMidSystemPreservation:
    """Opt-in in-place ``role:"system"`` messages (mid-conversation beta)."""

    _ITEMS: list[Any] = [
        {"type": "message", "role": "system", "content": "You are helpful."},
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "message", "role": "system", "content": "Terse mode enabled."},
        {"type": "message", "role": "user", "content": "go on"},
    ]

    def test_default_hoists_every_system_item(self) -> None:
        system, msgs = AnthropicConverter.items_to_messages(self._ITEMS)
        assert system is not None
        assert "You are helpful." in system
        assert "Terse mode enabled." in system
        assert all(m["role"] != "system" for m in msgs)

    def test_preserve_keeps_mid_system_in_place(self) -> None:
        system, msgs = AnthropicConverter.items_to_messages(self._ITEMS, preserve_mid_system=True)
        # Leading system item still hoists — role:"system" cannot be messages[0].
        assert system is not None
        assert "You are helpful." in system
        assert "Terse mode enabled." not in system
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "system", "user"]
        system_msg = msgs[1]
        content = system_msg["content"]
        assert isinstance(content, list)
        assert content[0]["text"] == "Terse mode enabled."

    def test_preserve_with_only_leading_system_hoists(self) -> None:
        items: list[Any] = [
            {"type": "message", "role": "system", "content": "Lead one."},
            {"type": "message", "role": "system", "content": "Lead two."},
            {"type": "message", "role": "user", "content": "hi"},
        ]
        system, msgs = AnthropicConverter.items_to_messages(items, preserve_mid_system=True)
        assert system is not None
        assert "Lead one." in system
        assert "Lead two." in system
        assert [m["role"] for m in msgs] == ["user"]

    def test_preserve_easy_message_form(self) -> None:
        items: list[Any] = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "Switch to French."},
        ]
        system, msgs = AnthropicConverter.items_to_messages(items, preserve_mid_system=True)
        assert system is None
        assert [m["role"] for m in msgs] == ["user", "system"]


class TestConvertImage:
    """Image content-part conversion to Anthropic ``ImageBlockParam``."""

    def test_http_url_becomes_url_source(self) -> None:
        block = AnthropicConverter._convert_image({"type": "input_image", "image_url": "https://x.test/a.png"})
        source = cast("dict[str, Any]", block["source"])
        assert source["type"] == "url"
        assert source["url"] == "https://x.test/a.png"

    def test_base64_data_uri_becomes_base64_source(self) -> None:
        # Regression: a ``data:`` URI string must split into a base64 source
        # with separate media_type/data — NOT a url source (Anthropic rejects
        # a data URI in the URL field).
        data_uri = "data:image/png;base64,aGVsbG8="
        block = AnthropicConverter._convert_image({"type": "input_image", "image_url": data_uri})
        source = cast("dict[str, Any]", block["source"])
        assert source["type"] == "base64"
        assert source["media_type"] == "image/png"
        assert source["data"] == "aGVsbG8="

    def test_base64_data_uri_unknown_media_type_falls_back(self) -> None:
        data_uri = "data:image/svg+xml;base64,YWJj"
        block = AnthropicConverter._convert_image({"type": "input_image", "image_url": data_uri})
        source = cast("dict[str, Any]", block["source"])
        assert source["type"] == "base64"
        assert source["media_type"] == "image/jpeg"

    def test_dict_base64_source_missing_media_type_raises_value_error(self) -> None:
        # Regression: a base64 source dict without ``media_type`` must raise a
        # descriptive ValueError, not a bare KeyError.
        with pytest.raises(ValueError, match="media_type"):
            AnthropicConverter._convert_image({"type": "input_image", "source": {"type": "base64", "data": "YWJj"}})

    def test_dict_base64_source_with_media_type(self) -> None:
        block = AnthropicConverter._convert_image(
            {"type": "input_image", "source": {"type": "base64", "media_type": "image/webp", "data": "YWJj"}}
        )
        source = cast("dict[str, Any]", block["source"])
        assert source["type"] == "base64"
        assert source["media_type"] == "image/webp"
        assert source["data"] == "YWJj"


class TestMultimodalToolResult:
    def test_string_output_stays_string(self) -> None:
        items: list[Any] = [
            {"type": "message", "role": "user", "content": "x"},
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "plain text"},
        ]
        _system, msgs = AnthropicConverter.items_to_messages(items)
        content = cast("list[Any]", msgs[-1]["content"])
        result = cast("dict[str, Any]", content[0])
        assert result["content"] == "plain text"

    def test_multimodal_list_output_becomes_content_blocks(self) -> None:
        # A tool that returns text + an image must NOT be collapsed to a
        # Python ``repr`` via ``str(list)``. Each part maps to a typed
        # Anthropic tool_result content block.
        items: list[Any] = [
            {"type": "message", "role": "user", "content": "x"},
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [
                    {"type": "input_text", "text": "here is the chart"},
                    {"type": "input_image", "image_url": "https://example.com/c.png"},
                ],
            },
        ]
        _system, msgs = AnthropicConverter.items_to_messages(items)
        result = cast("dict[str, Any]", cast("list[Any]", msgs[-1]["content"])[0])
        blocks = result["content"]
        assert isinstance(blocks, list)
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "here is the chart"
        assert blocks[1]["type"] == "image"
        assert blocks[1]["source"]["url"] == "https://example.com/c.png"

    def test_multimodal_output_preserves_is_error(self) -> None:
        items: list[Any] = [
            {"type": "message", "role": "user", "content": "x"},
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_text", "text": "failed"}],
                "status": "incomplete",
            },
        ]
        _system, msgs = AnthropicConverter.items_to_messages(items)
        result = cast("dict[str, Any]", cast("list[Any]", msgs[-1]["content"])[0])
        assert result.get("is_error") is True
        assert isinstance(result["content"], list)


class TestServerToolBlockSurfacing:
    def test_server_tool_use_block_surfaced_as_provider_item(self) -> None:
        # Server-executed tool blocks (web search etc.) must be surfaced
        # via the provider-item channel, not silently dropped.
        from anthropic.types import ServerToolUseBlock

        from troopai.adk.types.responses.llm_response import LLMResponseProviderItem

        message = Message(
            id="msg_srv",
            type="message",
            role="assistant",
            model="claude-sonnet-4",
            content=[
                ServerToolUseBlock(
                    type="server_tool_use",
                    id="srvtoolu_1",
                    name="web_search",
                    input={"query": "weather"},
                ),
                TextBlock(type="text", text="It is sunny.", citations=None),
            ],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=5, output_tokens=10),
        )
        response = AnthropicConverter.response_to_llm_response(message)
        provider_items = [p for p in response.response if isinstance(p, LLMResponseProviderItem)]
        assert len(provider_items) == 1
        assert provider_items[0].item_type == "server_tool_use"
        assert provider_items[0].raw["id"] == "srvtoolu_1"
        # The plain text part still comes through.
        assert response.content == "It is sunny."

    def test_pause_turn_stop_reason_surfaced_on_finish_reason(self) -> None:
        message = Message(
            id="msg_pause",
            type="message",
            role="assistant",
            model="claude-sonnet-4",
            content=[TextBlock(type="text", text="searching...", citations=None)],
            stop_reason="pause_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=5, output_tokens=10),
        )
        response = AnthropicConverter.response_to_llm_response(message)
        assert response.finish_reason == "pause_turn"


class TestSignatureOnlyReasoningReplay:
    def test_signature_only_reasoning_preserved_after_preceding_content(self) -> None:
        # A signature-only reasoning item (encrypted_content, empty content
        # list) must replay its signature even when the assistant turn
        # already carries preceding content — the fallback keys on THIS
        # item being empty, not on the whole message being empty.
        items: list[Any] = [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            {"type": "reasoning", "content": [], "encrypted_content": "sig-standalone"},
        ]
        _system, msgs = AnthropicConverter.items_to_messages(items)
        assistant_msg = msgs[-1]
        blocks = cast("list[Any]", assistant_msg["content"])
        thinking_blocks = [b for b in blocks if isinstance(b, dict) and b.get("type") == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["signature"] == "sig-standalone"
        # The preceding tool_use block is still present.
        assert any(isinstance(b, dict) and b.get("type") == "tool_use" for b in blocks)

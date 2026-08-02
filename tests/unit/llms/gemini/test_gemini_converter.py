"""Tests for ``GeminiConverter``."""

from __future__ import annotations

import base64
import json
from typing import Any, cast

from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    FunctionCall,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    Part,
)

from troopai.adk.llms.gemini.gemini_converter import GeminiConverter
from troopai.adk.tools import function_tool
from troopai.adk.types.items import ItemHelpers


class TestItemsToContents:
    def test_string_input(self) -> None:
        system, contents = GeminiConverter.items_to_contents("hello")
        assert system is None
        assert len(contents) == 1
        assert contents[0].role == "user"
        first_part = contents[0].parts[0] if contents[0].parts is not None else None
        assert first_part is not None
        assert first_part.text == "hello"

    def test_system_extracted(self) -> None:
        items: list[Any] = [
            {"type": "message", "role": "system", "content": "Be helpful."},
            {"type": "message", "role": "user", "content": "ping"},
        ]
        system, contents = GeminiConverter.items_to_contents(items)
        assert system == "Be helpful."
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_assistant_role_mapped_to_model(self) -> None:
        items: list[Any] = [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "message", "role": "assistant", "content": "hello"},
        ]
        _system, contents = GeminiConverter.items_to_contents(items)
        roles = [c.role for c in contents]
        assert roles == ["user", "model"]

    def test_function_call_replay(self) -> None:
        items: list[Any] = [
            {"type": "message", "role": "user", "content": "do x"},
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "do_x",
                "arguments": '{"y": 42}',
            },
            {"type": "function_call_output", "call_id": "c1", "name": "do_x", "output": '{"ok": true}'},
        ]
        _system, contents = GeminiConverter.items_to_contents(items)
        assert len(contents) == 3
        # Model turn carries the function_call part.
        assert contents[1].role == "model"
        model_parts = contents[1].parts or []
        assert len(model_parts) == 1
        assert model_parts[0].function_call is not None
        fc = model_parts[0].function_call
        assert fc.name == "do_x"
        assert fc.args == {"y": 42}
        # User turn carries the function_response part.
        assert contents[2].role == "user"
        user_parts = contents[2].parts or []
        assert user_parts[0].function_response is not None


class TestFunctionResponseNameCorrelation:
    """Gemini correlates a FunctionResponse to its FunctionCall by name.

    On Gemini 2.x the model returns a separate opaque ``id`` on the
    ``FunctionCall``, so Layer 1 stores that id as ``call_id`` (distinct
    from the tool name). The replayed ``FunctionResponse.name`` must still
    equal the originating ``FunctionCall.name`` — recovered from the paired
    ``function_call`` item — not the opaque id.
    """

    def test_response_name_matches_call_name_when_call_id_is_opaque(self) -> None:
        items: list[Any] = [
            {"type": "message", "role": "user", "content": "weather?"},
            {
                "type": "function_call",
                "call_id": "abc-opaque-id",
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            },
            # No "name" key on the result — mirrors FunctionToolCallResultParam,
            # which has no name field. call_id is the opaque id, not the name.
            {"type": "function_call_output", "call_id": "abc-opaque-id", "output": '{"temp": 72}'},
        ]
        _system, contents = GeminiConverter.items_to_contents(items)
        call_part = (contents[1].parts or [])[0]
        resp_part = (contents[2].parts or [])[0]
        assert call_part.function_call is not None
        assert resp_part.function_response is not None
        # The correlation contract: FunctionResponse.name == FunctionCall.name.
        assert resp_part.function_response.name == "get_weather"
        assert call_part.function_call.name == "get_weather"
        # The id correlation is preserved too.
        assert resp_part.function_response.id == "abc-opaque-id"
        assert call_part.function_call.id == "abc-opaque-id"

    def test_response_name_falls_back_to_call_id_without_matching_call(self) -> None:
        # No preceding function_call and no name key: fall back to call_id.
        items: list[Any] = [
            {"type": "function_call_output", "call_id": "orphan-id", "output": "{}"},
        ]
        _system, contents = GeminiConverter.items_to_contents(items)
        resp_part = (contents[0].parts or [])[0]
        assert resp_part.function_response is not None
        assert resp_part.function_response.name == "orphan-id"


class TestStringImageSource:
    """A plain-string image source (LLMInputImage.image_url) may be a remote
    URL or an inline base64 ``data:`` URI; each needs a distinct Gemini Part.
    """

    def test_data_uri_becomes_inline_blob(self) -> None:
        part = GeminiConverter._convert_content_part(
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}
        )
        assert part is not None
        # A data URI is inline bytes, NOT a fetchable file_uri reference.
        assert part.inline_data is not None
        assert part.file_data is None
        assert part.inline_data.mime_type == "image/png"

    def test_remote_url_infers_mime_from_extension(self) -> None:
        part = GeminiConverter._convert_content_part(
            {"type": "input_image", "image_url": "https://example.com/cat.png"}
        )
        assert part is not None
        assert part.file_data is not None
        # MIME inferred from the path, not hardcoded to image/jpeg.
        assert part.file_data.mime_type == "image/png"

    def test_remote_url_without_extension_falls_back_to_jpeg(self) -> None:
        part = GeminiConverter._convert_content_part({"type": "input_image", "image_url": "https://example.com/image"})
        assert part is not None
        assert part.file_data is not None
        assert part.file_data.mime_type == "image/jpeg"


class TestConvertToolChoice:
    def test_auto(self) -> None:
        tc = GeminiConverter.convert_tool_choice("auto", tools_present=True)
        assert tc is not None
        assert tc.function_calling_config is not None
        assert tc.function_calling_config.mode is not None
        assert tc.function_calling_config.mode.value == "AUTO"

    def test_required_maps_to_any(self) -> None:
        tc = GeminiConverter.convert_tool_choice("required", tools_present=True)
        assert tc is not None
        assert tc.function_calling_config is not None
        assert tc.function_calling_config.mode is not None
        assert tc.function_calling_config.mode.value == "ANY"

    def test_named_tool(self) -> None:
        tc = GeminiConverter.convert_tool_choice("lookup", tools_present=True)
        assert tc is not None
        assert tc.function_calling_config is not None
        assert tc.function_calling_config.allowed_function_names == ["lookup"]

    def test_no_tools_returns_none(self) -> None:
        assert GeminiConverter.convert_tool_choice("auto", tools_present=False) is None


class TestConvertTools:
    def test_function_tool(self) -> None:
        @function_tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tools = GeminiConverter.convert_tools([add])
        assert tools is not None
        assert len(tools) == 1
        assert tools[0].function_declarations is not None
        assert len(tools[0].function_declarations) == 1
        decl = tools[0].function_declarations[0]
        assert decl.name == "add"


class TestResponseToLLMResponse:
    def test_text_response(self) -> None:
        resp = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(role="model", parts=[Part.from_text(text="Hello!")]),
                    finish_reason=FinishReason.STOP,
                )
            ],
            response_id="resp_1",
            model_version="gemini-2.5-flash",
            usage_metadata=GenerateContentResponseUsageMetadata(
                prompt_token_count=10,
                candidates_token_count=5,
                total_token_count=15,
            ),
        )
        llm_resp = GeminiConverter.response_to_llm_response(resp)
        assert llm_resp.content == "Hello!"
        assert llm_resp.response_id == "resp_1"
        assert llm_resp.finish_reason == "STOP"
        assert llm_resp.usage is not None
        assert llm_resp.usage.input_tokens == 10
        assert llm_resp.usage.output_tokens == 5

    def test_function_call_response(self) -> None:
        resp = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        role="model",
                        parts=[
                            Part(
                                function_call=FunctionCall(
                                    id="call_a",
                                    name="weather",
                                    args={"city": "London"},
                                )
                            )
                        ],
                    ),
                    finish_reason=FinishReason.STOP,
                )
            ],
            response_id="resp_2",
            model_version="gemini-2.5-flash",
        )
        llm_resp = GeminiConverter.response_to_llm_response(resp)
        assert len(llm_resp.tool_calls) == 1
        call = llm_resp.tool_calls[0]
        assert call.name == "weather"
        assert call.call_id == "call_a"
        assert json.loads(call.arguments) == {"city": "London"}

    def test_thought_response(self) -> None:
        resp = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        role="model",
                        parts=[
                            Part(thought=True, text="reasoning...", thought_signature=b"sig"),
                            Part.from_text(text="The answer."),
                        ],
                    ),
                    finish_reason=FinishReason.STOP,
                )
            ],
            response_id="resp_3",
            model_version="gemini-2.5-pro",
        )
        llm_resp = GeminiConverter.response_to_llm_response(resp)
        # First part is thinking, second is text.
        from troopai.adk.types.responses.llm_response import (
            LLMResponseReasoning,
            LLMResponseText,
        )

        assert isinstance(llm_resp.response[0], LLMResponseReasoning)
        assert llm_resp.response[0].thinking == "reasoning..."
        # The opaque signature bytes are carried as base64 (lossless), not as
        # a raw utf-8 decode — b"sig" round-trips to its base64 form.
        assert llm_resp.response[0].encrypted_content == base64.b64encode(b"sig").decode("ascii")
        assert isinstance(llm_resp.response[1], LLMResponseText)
        assert cast(LLMResponseText, llm_resp.response[1]).text == "The answer."


class TestParseUsage:
    def test_basic(self) -> None:
        usage = GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
            cached_content_token_count=80,
            thoughts_token_count=20,
        )
        parsed = GeminiConverter._parse_usage(usage)
        assert parsed.input_tokens == 100
        assert parsed.output_tokens == 50
        assert parsed.total_tokens == 150
        assert parsed.input_tokens_details.cached_tokens == 80
        assert parsed.output_tokens_details.reasoning_tokens == 20


class TestThoughtSignatureRoundTrip:
    """A Gemini ``thought_signature`` is opaque bytes that must replay verbatim.

    A utf-8 decode/encode round-trip corrupts any non-utf-8 byte (the decode
    replaces it with U+FFFD, then the re-encode emits the 3-byte replacement
    sequence), so the replayed signature no longer matches — breaking Gemini's
    thinking-context validation. The converter must use base64, which is
    lossless for arbitrary bytes.
    """

    def test_non_utf8_signature_survives_response_then_replay(self) -> None:
        raw_sig = bytes([0x80, 0x00, 0xFE, 0xFF, 0x10, 0xC3, 0x28])  # invalid utf-8
        resp = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        role="model",
                        parts=[Part(thought=True, text="ponder", thought_signature=raw_sig)],
                    ),
                    finish_reason=FinishReason.STOP,
                )
            ],
            response_id="resp_sig",
            model_version="gemini-2.5-pro",
        )
        llm_resp = GeminiConverter.response_to_llm_response(resp)
        reasoning = llm_resp.response[0]
        from troopai.adk.types.responses.llm_response import LLMResponseReasoning

        assert isinstance(reasoning, LLMResponseReasoning)
        encrypted = reasoning.encrypted_content
        assert encrypted is not None
        # Stored form is valid base64 of the exact bytes.
        assert base64.b64decode(encrypted) == raw_sig

        # Replay: encrypted_content → thought_signature must equal the original.
        replay_items: list[Any] = [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "ponder"}],
                "encrypted_content": encrypted,
            }
        ]
        _system, contents = GeminiConverter.items_to_contents(replay_items)
        thought_part = (contents[0].parts or [])[0]
        assert thought_part.thought_signature == raw_sig

    def test_non_base64_encrypted_content_dropped_not_corrupting(self) -> None:
        # A stored signature that is not valid base64 cannot be replayed; it is
        # dropped rather than forwarded as corrupted bytes.
        replay_items: list[Any] = [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "x"}],
                "encrypted_content": "not!valid!base64!!!",
            }
        ]
        _system, contents = GeminiConverter.items_to_contents(replay_items)
        thought_part = (contents[0].parts or [])[0]
        assert thought_part.thought is True
        assert thought_part.thought_signature is None


class TestFunctionCallSignatureRoundTrip:
    """A thinking model attaches a ``thought_signature`` to its ``function_call``
    part; those opaque bytes must replay verbatim across multi-turn tool use.

    Dropping them (or a utf-8 decode/encode round-trip) breaks Gemini's
    thinking-context validation, so the signature rides the full
    convert → to_param → reload → replay path as lossless base64.
    """

    def test_signature_bytes_survive_full_round_trip(self) -> None:
        # The data-loss regression proper: a non-utf-8 signature must arrive at
        # the wire as the EXACT original bytes after a JSON persist + reload.
        # Pre-fix the bytes were discarded, so the replayed Part carried
        # thought_signature=None.
        raw_sig = bytes([0x80, 0x00, 0xFE, 0xFF, 0x10, 0xC3, 0x28])  # invalid utf-8
        resp = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        role="model",
                        parts=[
                            Part(
                                function_call=FunctionCall(id="c1", name="do_x", args={"y": 42}),
                                thought_signature=raw_sig,
                            )
                        ],
                    ),
                    finish_reason=FinishReason.STOP,
                )
            ],
            response_id="resp_fc_sig",
            model_version="gemini-2.5-pro",
        )

        llm_resp = GeminiConverter.response_to_llm_response(resp)
        assert len(llm_resp.tool_calls) == 1

        # to_param → JSON persist → reload → to_param (mirrors RunState replay).
        param = llm_resp.tool_calls[0].to_param()
        persisted: Any = json.loads(json.dumps(param))
        reloaded = ItemHelpers.message_to_run_items(persisted)
        assert len(reloaded) == 1
        replay_param = reloaded[0].to_param()

        _system, contents = GeminiConverter.items_to_contents([replay_param])
        model_parts = contents[0].parts or []
        assert len(model_parts) == 1
        assert model_parts[0].function_call is not None
        # base64 kept the opaque bytes lossless across the whole trip.
        assert model_parts[0].thought_signature == raw_sig

    def test_signature_stored_as_base64_on_convert(self) -> None:
        raw_sig = bytes([0x00, 0xC3, 0x28])  # invalid utf-8
        resp = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        role="model",
                        parts=[
                            Part(
                                function_call=FunctionCall(id="c1", name="do_x", args={}),
                                thought_signature=raw_sig,
                            )
                        ],
                    ),
                    finish_reason=FinishReason.STOP,
                )
            ],
            response_id="resp_fc_sig2",
            model_version="gemini-2.5-pro",
        )
        call = GeminiConverter.response_to_llm_response(resp).tool_calls[0]
        expected = base64.b64encode(raw_sig).decode("ascii")
        assert call.signature == expected
        # ``signature`` is NotRequired on the replay TypedDict; ``.get`` reads
        # it without a not-present narrowing complaint.
        assert call.to_param().get("signature") == expected

    def test_function_call_without_signature_stays_none_and_omitted(self) -> None:
        # No thought_signature (every non-Gemini provider, and Gemini without
        # thinking): signature stays None, the wire key is omitted, and replay
        # carries no thought_signature — zero effect on providers that never
        # attach one.
        resp = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        role="model",
                        parts=[Part(function_call=FunctionCall(id="c1", name="do_x", args={}))],
                    ),
                    finish_reason=FinishReason.STOP,
                )
            ],
            response_id="resp_nosig",
            model_version="gemini-2.5-flash",
        )
        call = GeminiConverter.response_to_llm_response(resp).tool_calls[0]
        assert call.signature is None
        assert "signature" not in call.to_param()

        _system, contents = GeminiConverter.items_to_contents([call.to_param()])
        part = (contents[0].parts or [])[0]
        assert part.function_call is not None
        assert part.thought_signature is None

    def test_non_base64_signature_dropped_not_corrupting(self) -> None:
        # A persisted signature that is not valid base64 cannot be replayed
        # verbatim; drop it rather than forward corrupted bytes.
        replay_items: list[Any] = [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "do_x",
                "arguments": "{}",
                "signature": "not!valid!base64!!!",
            }
        ]
        _system, contents = GeminiConverter.items_to_contents(replay_items)
        part = (contents[0].parts or [])[0]
        assert part.function_call is not None
        assert part.thought_signature is None


class TestSystemListContent:
    """System / developer messages with list content must contribute their text.

    The easy-message path previously collapsed a list content to ``""``,
    silently dropping the whole system prompt.
    """

    def test_easy_message_system_list_content_extracted(self) -> None:
        items: list[Any] = [
            {"role": "system", "content": [{"type": "input_text", "text": "Be terse."}]},
            {"role": "user", "content": "hi"},
        ]
        system, _contents = GeminiConverter.items_to_contents(items)
        assert system == "Be terse."

    def test_strict_message_system_list_multiple_text_parts_joined(self) -> None:
        items: list[Any] = [
            {
                "type": "message",
                "role": "system",
                "content": [
                    {"type": "input_text", "text": "Rule one."},
                    {"type": "text", "text": "Rule two."},
                ],
            },
        ]
        system, _contents = GeminiConverter.items_to_contents(items)
        assert system == "Rule one.\n\nRule two."

"""Tests for ``LLMResponse`` response-part types and their ``to_param`` round-trips."""

from __future__ import annotations

from troopai.adk.llms.litellm.litellm_converter import ChatCompletionConverter
from troopai.adk.types.items import ItemHelpers, ToolCallItem
from troopai.adk.types.responses.llm_response import (
    LLMResponseFunctionToolCall,
    LLMResponseReasoning,
)


class TestReasoningToParam:
    """``LLMResponseReasoning.to_param`` must not fabricate a summary."""

    def test_thinking_only_does_not_populate_summary(self) -> None:
        # A reasoning part carrying only chain-of-thought (no explicit summary,
        # the DeepSeek / Anthropic shape) must flow the text into ``content``
        # ONLY — ``summary`` stays an empty list, never a copy of ``thinking``.
        part = LLMResponseReasoning(thinking="Let me think...", summary=None)

        param = part.to_param()

        assert param["summary"] == []
        assert param.get("content") == [{"type": "reasoning_text", "text": "Let me think..."}]

    def test_thinking_only_round_trip_preserves_none_summary(self) -> None:
        # Serialize → deserialize must preserve ``summary is None``; the prior
        # behavior fabricated ``summary == thinking`` on the rebuilt item.
        part = LLMResponseReasoning(thinking="Let me think...", summary=None)

        rebuilt = ItemHelpers.message_to_run_items(part.to_param())[0]

        assert rebuilt.raw.summary is None
        assert rebuilt.raw.thinking == "Let me think..."

    def test_explicit_summary_is_preserved(self) -> None:
        # When the model exposed a real summary (OpenAI o-series), both the
        # summary and the full chain-of-thought survive the round-trip.
        part = LLMResponseReasoning(thinking="full chain", summary="short summary")

        param = part.to_param()
        rebuilt = ItemHelpers.message_to_run_items(param)[0]

        assert param["summary"] == [{"type": "summary_text", "text": "short summary"}]
        assert rebuilt.raw.summary == "short summary"
        assert rebuilt.raw.thinking == "full chain"

    def test_summary_key_always_present(self) -> None:
        # The replay param mirrors the OpenAI ``ResponseReasoningItemParam``
        # pass-through, which requires the ``summary`` key — keep it present
        # (empty list) even when there is no explicit summary.
        param = LLMResponseReasoning(thinking="cot", summary=None, id="rs_1").to_param()

        assert "summary" in param


class TestDeepSeekReasoningReplay:
    """DeepSeek reasoning replay must recover the chain-of-thought after the fix."""

    def test_reasoning_content_recovered_from_content(self) -> None:
        # DeepSeek emits reasoning_content → LLMResponseReasoning(thinking=...,
        # summary=None). After the to_param fix the text lives only in
        # ``content``, so the converter must read it from there (not ``summary``)
        # to repopulate ``reasoning_content`` on the replayed assistant message.
        reasoning = LLMResponseReasoning(thinking="deepseek chain of thought", summary=None)
        tool_call = LLMResponseFunctionToolCall(call_id="c1", name="get", arguments="{}")

        messages = ChatCompletionConverter.items_to_messages(
            [reasoning.to_param(), tool_call.to_param()],
            model="deepseek/deepseek-reasoner",
        )

        assistant = next(m for m in messages if m.get("role") == "assistant")
        assert assistant.get("reasoning_content") == "deepseek chain of thought"


class TestFunctionToolCallSignatureRoundTrip:
    """``LLMResponseFunctionToolCall.signature`` carries a thinking model's
    per-tool-call signature (base64). It must survive ``to_param`` → reload and
    stay absent when no provider attached one.
    """

    def test_signature_emitted_and_reloaded(self) -> None:
        call = LLMResponseFunctionToolCall(call_id="c1", name="get", arguments="{}", signature="c2ln")

        # ``signature`` is NotRequired on the replay TypedDict; ``.get`` reads it
        # without a not-present narrowing complaint.
        param = call.to_param()
        assert param.get("signature") == "c2ln"

        rebuilt = ItemHelpers.message_to_run_items(param)[0]
        assert isinstance(rebuilt, ToolCallItem)
        assert rebuilt.raw.signature == "c2ln"

    def test_signature_absent_by_default(self) -> None:
        # The default None path — every provider that does not attach a
        # per-tool-call signature. The key must never appear on the wire.
        call = LLMResponseFunctionToolCall(call_id="c1", name="get", arguments="{}")

        assert call.signature is None
        assert "signature" not in call.to_param()

        rebuilt = ItemHelpers.message_to_run_items(call.to_param())[0]
        assert isinstance(rebuilt, ToolCallItem)
        assert rebuilt.raw.signature is None


class TestContentJoin:
    """``LLMResponse.content`` concatenates text parts with '' (no newline).

    This must match ``ItemHelpers.text_message_output`` (which joins with '')
    so the convenience accessor and the persisted message-item text agree,
    and so structured-output JSON split across parts is not corrupted by a
    spurious newline before ``output_schema.validate_json`` runs.
    """

    def test_multiple_text_parts_join_without_newline(self) -> None:
        from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText

        response = LLMResponse(
            response_id="r",
            model="m",
            response=[LLMResponseText(text='{"a":'), LLMResponseText(text="1}")],
        )
        assert response.content == '{"a":1}'

    def test_content_matches_text_message_output(self) -> None:
        from troopai.adk.types.items import MessageOutputItem
        from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText

        parts = [LLMResponseText(text="foo"), LLMResponseText(text="bar")]
        response = LLMResponse(response_id="r", model="m", response=list(parts))
        items = ItemHelpers.response_to_run_items(response)
        message_items = [i for i in items if isinstance(i, MessageOutputItem)]
        assert response.content == ItemHelpers.text_message_output(message_items[0])

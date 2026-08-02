"""Regression tests for replay-param correctness in ``ItemHelpers``.

Covers the round-trip between Layer-3 ``RunItem`` classes and the Layer-1
replay params they emit / parse, where a lossy or fabricated field silently
corrupts a later provider turn:

- A ``MessageOutputItem`` must not replay the provider RESPONSE id in the
  message-item ``id`` slot (OpenAI Responses rejects a response id there).
- A ``redacted_thinking`` reasoning param must round-trip its opaque payload
  and ``is_redacted`` marker so re-emit stays a redacted block (Anthropic).
- A Chat-Completions assistant message with list content must not be
  stringified to a Python ``repr``.
- A reasoning param with no id must not fabricate one.
- A dict-valued ``arguments`` on an MCP approval request must be JSON-encoded.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from troopai.adk.types.items import (
    ItemHelpers,
    MCPApprovalRequestItem,
    MessageOutputItem,
    ReasoningItem,
)
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseProviderItem,
    LLMResponseReasoning,
    LLMResponseText,
)

if TYPE_CHECKING:
    from troopai.adk.types.input import LLMInputContentItem

message_to_items = ItemHelpers.message_to_run_items
response_to_items = ItemHelpers.response_to_run_items


def _as_param(data: dict[str, Any]) -> LLMInputContentItem:
    """Type a hand-built replay dict as a Layer-1 input item.

    ``message_to_run_items`` dispatches dynamically on ``dict`` keys and is
    fed loosely-typed dicts on session reload / cross-provider replay. These
    tests build such dicts directly — including deliberately-malformed shapes
    (e.g. an explicit ``id: None`` from lenient JSON) — so the input is
    genuinely dynamic; isolate the single cast here rather than per call site.
    """
    return cast("LLMInputContentItem", data)


class TestMessageOutputResponseId:
    """The provider response id is retained on the item but never replayed."""

    def test_to_param_omits_response_id(self) -> None:
        """``to_param`` must not emit the response id as the message-item id."""
        item = MessageOutputItem(
            raw=[LLMResponseText(text="hi")],
            id="resp_abc123",
            status="completed",
        )
        param = item.to_param()
        assert "id" not in param
        # The id survives on the item for ``RunResult.last_response_id``.
        assert item.id == "resp_abc123"

    def test_response_to_run_items_keeps_id_on_item_not_in_param(self) -> None:
        response = LLMResponse(
            response_id="resp_xyz",
            model="gpt-4o",
            response=[LLMResponseText(text="hi")],
        )
        items = response_to_items(response)
        msg_items = [i for i in items if isinstance(i, MessageOutputItem)]
        assert len(msg_items) == 1
        # Retained for last_response_id …
        assert msg_items[0].id == "resp_xyz"
        # … but not sent back to the LLM.
        assert "id" not in msg_items[0].to_param()


class TestRedactedThinkingReasoningParam:
    """A redacted-thinking reasoning param round-trips losslessly."""

    def test_redacted_block_round_trips(self) -> None:
        original = LLMResponseReasoning(is_redacted=True, encrypted_content="REDACTED_BLOB")
        param = original.to_param()
        # Sanity: emit is a redacted_thinking content block.
        assert param.get("content") == [{"type": "redacted_thinking", "data": "REDACTED_BLOB"}]

        restored = message_to_items(param)
        assert len(restored) == 1
        item = restored[0]
        assert isinstance(item, ReasoningItem)
        assert item.raw.is_redacted is True
        assert item.raw.encrypted_content == "REDACTED_BLOB"

        # Re-emit must again be a redacted_thinking block, not a plain one.
        assert item.to_param().get("content") == [{"type": "redacted_thinking", "data": "REDACTED_BLOB"}]

    def test_redacted_payload_recovered_from_content_when_no_top_level(self) -> None:
        """When only the content block carries the payload, it still lands in
        ``encrypted_content`` and marks the item redacted."""
        param = _as_param(
            {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "redacted_thinking", "data": "ONLY_IN_CONTENT"}],
            }
        )
        item = message_to_items(param)[0]
        assert isinstance(item, ReasoningItem)
        assert item.raw.is_redacted is True
        assert item.raw.encrypted_content == "ONLY_IN_CONTENT"

    def test_plain_reasoning_text_still_parses(self) -> None:
        param = _as_param(
            {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "let me think"}],
            }
        )
        item = message_to_items(param)[0]
        assert isinstance(item, ReasoningItem)
        assert item.raw.is_redacted is False
        assert item.raw.thinking == "let me think"


class TestReasoningIdOmission:
    """A reasoning param with no genuine id must not fabricate one."""

    def test_absent_id_stays_none(self) -> None:
        param = _as_param(
            {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "think"}],
            }
        )
        item = message_to_items(param)[0]
        assert isinstance(item, ReasoningItem)
        assert item.raw.id is None
        assert "id" not in item.to_param()

    def test_explicit_null_id_not_stringified(self) -> None:
        param = _as_param({"type": "reasoning", "summary": [], "content": [], "id": None})
        item = message_to_items(param)[0]
        assert isinstance(item, ReasoningItem)
        # Must not become the string "None".
        assert item.raw.id is None

    def test_real_id_preserved(self) -> None:
        param = _as_param({"type": "reasoning", "summary": [], "content": [], "id": "rs_123"})
        item = message_to_items(param)[0]
        assert isinstance(item, ReasoningItem)
        assert item.raw.id == "rs_123"


class TestAssistantChatCompletionsListContent:
    """A CC assistant message whose content is a list of parts is not repr'd."""

    def test_list_text_parts_extracted(self) -> None:
        msg = _as_param(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ],
            }
        )
        items = message_to_items(msg)
        msg_items = [i for i in items if isinstance(i, MessageOutputItem)]
        assert len(msg_items) == 1
        texts = [p.text for p in msg_items[0].raw if isinstance(p, LLMResponseText)]
        assert texts == ["Hello ", "world"]
        # Not the Python repr of the list.
        assert not any("'type'" in t for t in texts)

    def test_string_content_still_works(self) -> None:
        msg = _as_param({"role": "assistant", "content": "plain string"})
        items = message_to_items(msg)
        msg_items = [i for i in items if isinstance(i, MessageOutputItem)]
        assert len(msg_items) == 1
        first = msg_items[0].raw[0]
        assert isinstance(first, LLMResponseText)
        assert first.text == "plain string"


class TestMcpApprovalRequestArguments:
    """Dict-valued MCP approval arguments must be JSON, not a Python repr."""

    def test_dict_arguments_json_encoded(self) -> None:
        part = LLMResponseProviderItem(
            item_type="mcp_approval_request",
            raw={
                "id": "a1",
                "server_label": "srv",
                "name": "tool",
                "arguments": {"city": "Paris"},
            },
        )
        response = LLMResponse(response_id="r", model="m", response=[part])
        items = response_to_items(response)
        approvals = [i for i in items if isinstance(i, MCPApprovalRequestItem)]
        assert len(approvals) == 1
        args = approvals[0].raw.arguments
        # Valid JSON round-trips; a Python repr (single quotes) would not.
        assert json.loads(args) == {"city": "Paris"}

    def test_string_arguments_passed_through(self) -> None:
        part = LLMResponseProviderItem(
            item_type="mcp_approval_request",
            raw={"id": "a1", "server_label": "srv", "name": "tool", "arguments": '{"a": 1}'},
        )
        response = LLMResponse(response_id="r", model="m", response=[part])
        approvals = [i for i in response_to_items(response) if isinstance(i, MCPApprovalRequestItem)]
        assert approvals[0].raw.arguments == '{"a": 1}'

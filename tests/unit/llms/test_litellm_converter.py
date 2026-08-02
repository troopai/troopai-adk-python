"""Regression tests for ``llms/litellm/litellm_converter.py``.

Each test targets a confirmed defect in ``ChatCompletionConverter`` and is
written to FAIL on the pre-fix code and PASS on the fixed code.
"""

from __future__ import annotations

from typing import Any, cast

from troopai.adk.llms.litellm.litellm_converter import ChatCompletionConverter

# ---------------------------------------------------------------------------
# Pending content parts must not be flushed ahead of an open assistant
# tool-call buffer (turn reordering).
# ---------------------------------------------------------------------------


class TestPendingContentPartsOrdering:
    """User content arriving after a tool call must stay after the assistant turn."""

    def test_user_text_after_tool_call_followed_by_second_tool_call(self) -> None:
        # Chronological order: assistant tool_call A, then user "hi", then
        # assistant tool_call B. The "hi" user turn must NOT be emitted before
        # the assistant turn that precedes it.
        items = [
            {"type": "function_call", "call_id": "A", "name": "fn", "arguments": "{}"},
            {"type": "input_text", "text": "hi"},
            {"type": "function_call", "call_id": "B", "name": "fn", "arguments": "{}"},
        ]
        messages = ChatCompletionConverter.items_to_messages(items)

        roles = [m["role"] for m in messages]
        # The assistant turn (carrying the tool call that arrived first) must
        # come before the user "hi" turn.
        assert "assistant" in roles
        assert "user" in roles
        assert roles.index("assistant") < roles.index("user")

        # tool_call A must be on the assistant turn that precedes the user turn,
        # not bundled after it.
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        all_call_ids = [tc["id"] for m in assistant_msgs for tc in (m.get("tool_calls") or [])]
        assert "A" in all_call_ids
        assert "B" in all_call_ids

    def test_trailing_user_text_after_open_assistant_buffer(self) -> None:
        # assistant tool_call A then trailing user "bye": the assistant turn
        # must precede the user turn at end-of-input flush, too.
        items = [
            {"type": "function_call", "call_id": "A", "name": "fn", "arguments": "{}"},
            {"type": "input_text", "text": "bye"},
        ]
        messages = ChatCompletionConverter.items_to_messages(items)

        roles = [m["role"] for m in messages]
        assert roles.index("assistant") < roles.index("user")

    def test_plain_user_text_only_unaffected(self) -> None:
        # No open assistant buffer: a lone user content part still becomes a
        # single user message (no spurious empty assistant turn).
        items = [{"type": "input_text", "text": "hello"}]
        messages = ChatCompletionConverter.items_to_messages(items)
        assert [m["role"] for m in messages] == ["user"]


# ---------------------------------------------------------------------------
# fix_tool_message_ordering must not duplicate assistant text/thinking content
# across the split single-tool-call messages.
# ---------------------------------------------------------------------------


class TestFixToolMessageOrderingContentDuplication:
    """Shared narration text must appear on exactly one split assistant turn."""

    def _assistant_with_text_and_calls(self, text: str, *call_ids: str) -> dict:  # type: ignore[type-arg]
        from litellm.types.llms.openai import (
            ChatCompletionAssistantMessage,
            ChatCompletionAssistantToolCall,
            ChatCompletionToolCallFunctionChunk,
        )

        tc_list = [
            ChatCompletionAssistantToolCall(
                id=cid,
                type="function",
                function=ChatCompletionToolCallFunctionChunk(name="fn", arguments="{}"),
            )
            for cid in call_ids
        ]
        msg = ChatCompletionAssistantMessage(role="assistant")
        msg["content"] = text
        msg["tool_calls"] = tc_list
        return msg  # type: ignore[return-value]

    def _tool_result(self, call_id: str) -> dict:  # type: ignore[type-arg]
        from litellm.types.llms.openai import ChatCompletionToolMessage

        return ChatCompletionToolMessage(role="tool", tool_call_id=call_id, content="result")  # type: ignore[return-value]

    def test_content_not_duplicated_across_split_messages(self) -> None:
        asst = self._assistant_with_text_and_calls("Let me check both records", "tc1", "tc2")
        messages = [asst, self._tool_result("tc1"), self._tool_result("tc2")]

        result = ChatCompletionConverter.fix_tool_message_ordering(messages)

        assistant_msgs = [m for m in result if isinstance(m, dict) and m.get("role") == "assistant"]
        # The turn was split into two single-tool-call assistant messages.
        assert len(assistant_msgs) == 2
        # The narration text must appear on exactly ONE of the split messages.
        contents = [m.get("content") for m in assistant_msgs]
        assert contents.count("Let me check both records") == 1
        # The other split message carries no duplicated content.
        assert contents.count(None) == 1 or contents.count("") == 1 or None in contents

    def test_split_still_pairs_each_call_with_its_result(self) -> None:
        # The de-duplication must not break the call/result adjacency.
        asst = self._assistant_with_text_and_calls("narration", "tc1", "tc2")
        messages = [asst, self._tool_result("tc1"), self._tool_result("tc2")]

        result = ChatCompletionConverter.fix_tool_message_ordering(messages)

        seq = [(m.get("role"), m.get("tool_calls"), m.get("tool_call_id")) for m in result]
        # Expect: assistant(tc1), tool(tc1), assistant(tc2), tool(tc2)
        assert seq[0][0] == "assistant"
        assert seq[1] == ("tool", None, "tc1")
        assert seq[2][0] == "assistant"
        assert seq[3] == ("tool", None, "tc2")

    def test_single_tool_call_keeps_content(self) -> None:
        # A single tool call is not split, so content is preserved as-is.
        asst = self._assistant_with_text_and_calls("solo", "tc1")
        messages = [asst, self._tool_result("tc1")]

        result = ChatCompletionConverter.fix_tool_message_ordering(messages)

        assistant_msgs = [m for m in result if isinstance(m, dict) and m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].get("content") == "solo"


# ---------------------------------------------------------------------------
# Easy user messages with multimodal content must be converted to wire parts,
# not passed through raw (raw ``input_image`` parts are rejected with a 400).
# ---------------------------------------------------------------------------


class TestEasyUserMessageMultimodalConversion:
    """An easy ``{"role": "user", "content": [...]}`` list must be normalized."""

    def test_easy_user_message_image_part_converted_to_wire_shape(self) -> None:
        items: list[Any] = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look at this"},
                    {"type": "input_image", "image_url": "http://img/a.png"},
                ],
            }
        ]
        messages = cast("list[dict[str, Any]]", ChatCompletionConverter.items_to_messages(items))

        assert len(messages) == 1
        content = messages[0]["content"]
        assert isinstance(content, list)
        types = [p.get("type") for p in content]
        # Wire shape: input_image -> image_url, input_text -> text.
        assert "image_url" in types
        assert "text" in types
        assert "input_image" not in types
        img = next(p for p in content if p.get("type") == "image_url")
        assert img["image_url"]["url"] == "http://img/a.png"

    def test_easy_user_message_plain_string_still_passes_through(self) -> None:
        items: list[Any] = [{"role": "user", "content": "hello"}]
        messages = cast("list[dict[str, Any]]", ChatCompletionConverter.items_to_messages(items))
        assert messages[0]["content"] == "hello"


# ---------------------------------------------------------------------------
# Multimodal tool output must be converted to wire content parts, not
# ``str()``-serialized into a corrupt Python repr.
# ---------------------------------------------------------------------------


class TestMultimodalToolOutputPreserved:
    """A tool ``output`` list must survive as content parts under preserve mode."""

    def test_multimodal_output_preserved_as_parts_not_repr(self) -> None:
        items: list[Any] = [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [
                    {"type": "input_text", "text": "the answer"},
                    {"type": "input_image", "image_url": "http://img/b.png"},
                ],
            }
        ]
        messages = cast(
            "list[dict[str, Any]]",
            ChatCompletionConverter.items_to_messages(items, preserve_tool_output_all_content=True),
        )

        assert len(messages) == 1
        tool_msg = messages[0]
        assert tool_msg["role"] == "tool"
        content = tool_msg["content"]
        # Must NOT be a Python-repr string of the raw part dicts.
        assert not (isinstance(content, str) and content.lstrip().startswith("["))
        assert isinstance(content, list)
        types = [p.get("type") for p in content]
        assert "text" in types
        assert "image_url" in types
        text_part = next(p for p in content if p.get("type") == "text")
        assert "the answer" in text_part["text"]

    def test_multimodal_output_text_only_extracted_when_not_preserving(self) -> None:
        # Without preserve mode, text is extracted (images dropped) — never a repr.
        items: list[Any] = [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_text", "text": "just text"}],
            }
        ]
        messages = cast(
            "list[dict[str, Any]]",
            ChatCompletionConverter.items_to_messages(items, preserve_tool_output_all_content=False),
        )
        assert messages[0]["content"] == "just text"


# ---------------------------------------------------------------------------
# fix_tool_message_ordering must preserve every tool result (no id-collapse)
# and must not drop the assistant's structured thinking_blocks on split.
# ---------------------------------------------------------------------------


class TestFixToolMessageOrderingPreservesResults:
    """Distinct results sharing a tool_call_id must not overwrite each other."""

    def _assistant(self, *call_ids: str, thinking_blocks: Any = None) -> dict[str, Any]:
        msg: dict[str, Any] = {
            "role": "assistant",
            "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": "fn", "arguments": "{}"}} for cid in call_ids
            ],
        }
        if thinking_blocks is not None:
            msg["thinking_blocks"] = thinking_blocks
        return msg

    def _tool_result(self, call_id: str, content: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": call_id, "content": content}

    def test_duplicate_result_ids_are_not_collapsed(self) -> None:
        # Two calls sharing an id, each with its own result. A plain dict keyed
        # by id would drop the first result; both must survive.
        messages: list[Any] = [
            self._assistant("dup", "dup"),
            self._tool_result("dup", "first"),
            self._tool_result("dup", "second"),
        ]

        result = cast("list[dict[str, Any]]", ChatCompletionConverter.fix_tool_message_ordering(messages))

        contents = [m.get("content") for m in result if m.get("role") == "tool"]
        assert "first" in contents
        assert "second" in contents

    def test_thinking_blocks_preserved_on_first_split(self) -> None:
        blocks = [{"type": "thinking", "thinking": "reasoning", "signature": "sig-1"}]
        messages: list[Any] = [
            self._assistant("tc1", "tc2", thinking_blocks=blocks),
            self._tool_result("tc1", "r1"),
            self._tool_result("tc2", "r2"),
        ]

        result = cast("list[dict[str, Any]]", ChatCompletionConverter.fix_tool_message_ordering(messages))

        assistant_msgs = [m for m in result if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 2
        # Structured thinking (with its signature) survives on exactly one split.
        with_thinking = [m for m in assistant_msgs if m.get("thinking_blocks") is not None]
        assert len(with_thinking) == 1
        assert with_thinking[0]["thinking_blocks"] == blocks

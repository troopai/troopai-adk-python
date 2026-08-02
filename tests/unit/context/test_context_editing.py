"""Tests for ContextEditor.

Message lists are typed ``list[Any]`` because ``LLMInputContentItem`` is a
TypedDict union that pyright cannot discriminator-narrow on ``.get(...)`` /
subscript; the values are plain dicts at runtime.
"""

from typing import Any

from troopai.adk.context.context_editing import (
    _CLEARED_THINKING_PLACEHOLDER,
    _CLEARED_TOOL_PLACEHOLDER,
    ContextEditor,
)

# ── clear_tool_results ───────────────────────────────────────────────


class TestClearToolResults:
    def _make_messages(self, n_tools: int) -> list[Any]:
        """Build a Layer-1 conversation with n function_call/result pairs.

        Tool results are ``function_call_output`` items (no ``role``) — the
        format ``prepare_messages`` actually operates on. Regression guard:
        the feature previously matched Layer-2 ``role == "tool"`` and
        silently no-op'd on this Layer-1 stream.
        """
        msgs: list[Any] = [{"role": "system", "content": "You are helpful."}]
        for i in range(n_tools):
            msgs.append({"type": "function_call", "call_id": f"tc_{i}", "name": f"tool_{i}", "arguments": "{}"})
            msgs.append({"type": "function_call_output", "call_id": f"tc_{i}", "output": f"Result {i}"})
        msgs.append({"role": "assistant", "content": "Done."})
        return msgs

    def test_keeps_recent_results(self):
        msgs = self._make_messages(5)
        result = ContextEditor.clear_tool_results(msgs, keep=2)

        # 5 tool results total; keep 2 most recent, clear 3 oldest.
        tool_msgs: list[Any] = [m for m in result if m.get("type") == "function_call_output"]
        cleared = [m for m in tool_msgs if m["output"] == _CLEARED_TOOL_PLACEHOLDER]
        intact = [m for m in tool_msgs if m["output"] != _CLEARED_TOOL_PLACEHOLDER]

        assert len(cleared) == 3
        assert len(intact) == 2
        # Intact ones should be the last two tool results.
        assert intact[0]["output"] == "Result 3"
        assert intact[1]["output"] == "Result 4"

    def test_no_clearing_when_under_keep(self):
        msgs = self._make_messages(2)
        result = ContextEditor.clear_tool_results(msgs, keep=5)

        tool_msgs: list[Any] = [m for m in result if m.get("type") == "function_call_output"]
        assert all(m["output"] != _CLEARED_TOOL_PLACEHOLDER for m in tool_msgs)

    def test_exclude_tools(self):
        msgs = self._make_messages(5)
        result = ContextEditor.clear_tool_results(msgs, keep=1, exclude_tools=["tool_0"])

        # tool_0's result (call_id tc_0) is resolved by name and never cleared.
        tool_0: list[Any] = [
            m for m in result if m.get("type") == "function_call_output" and m.get("call_id") == "tc_0"
        ]
        assert len(tool_0) == 1
        assert tool_0[0]["output"] == "Result 0"

    def test_original_not_mutated(self):
        msgs = self._make_messages(3)
        original_output = [m.get("output") for m in msgs]

        ContextEditor.clear_tool_results(msgs, keep=1)

        assert [m.get("output") for m in msgs] == original_output

    def test_no_tool_messages(self):
        msgs: list[Any] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = ContextEditor.clear_tool_results(msgs, keep=1)
        assert len(result) == 3


# ── clear_thinking_blocks ────────────────────────────────────────────


class TestClearThinkingBlocks:
    def _make_thinking_messages(self, n: int) -> list[Any]:
        """Build messages with n assistant turns containing thinking blocks."""
        msgs: list[Any] = [{"role": "system", "content": "sys"}]
        for i in range(n):
            msgs.append({"role": "user", "content": f"Question {i}"})
            msgs.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": f"Thinking about {i}..."},
                        {"type": "text", "text": f"Answer {i}"},
                    ],
                }
            )
        return msgs

    def test_keeps_recent_thinking(self):
        msgs = self._make_thinking_messages(4)
        result = ContextEditor.clear_thinking_blocks(msgs, keep_turns=1)

        assistant_msgs: list[Any] = [m for m in result if m.get("role") == "assistant"]
        # Last assistant should still have thinking.
        last = assistant_msgs[-1]
        assert any(isinstance(b, dict) and b.get("type") == "thinking" for b in last["content"])

        # Earlier assistants should have their thinking replaced by the placeholder.
        for a in assistant_msgs[:-1]:
            assert all(not (isinstance(block, dict) and block.get("type") == "thinking") for block in a["content"])
            assert any(
                isinstance(block, dict)
                and block.get("type") == "text"
                and block.get("text") == _CLEARED_THINKING_PLACEHOLDER
                for block in a["content"]
            )

    def test_no_thinking_blocks_is_noop(self):
        msgs: list[Any] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = ContextEditor.clear_thinking_blocks(msgs, keep_turns=1)
        assert result == msgs

    def test_clears_standalone_reasoning_items(self):
        """Reasoning replays as standalone ``{"type": "reasoning"}`` items, not
        as content-list ``thinking`` blocks. Older ones must be dropped.

        Pre-fix: clear_thinking_blocks only scanned assistant ``content`` lists,
        so a real replay stream (reasoning items) was left untouched — a no-op.
        """
        msgs: list[Any] = [{"role": "system", "content": "sys"}]
        for i in range(3):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append(
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": f"thinking {i}"}],
                    "encrypted_content": f"sig{i}",
                }
            )
            msgs.append({"role": "assistant", "content": f"a{i}"})

        result = ContextEditor.clear_thinking_blocks(msgs, keep_turns=1)

        reasoning_items = [m for m in result if m.get("type") == "reasoning"]
        # Only the most-recent reasoning item survives; the two older ones are dropped.
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["encrypted_content"] == "sig2"
        # The surrounding user/assistant messages are all preserved.
        assert len([m for m in result if m.get("role") == "assistant"]) == 3

    def test_clears_top_level_thinking_blocks(self):
        """Assistant messages can carry thinking as a top-level ``thinking_blocks``
        list (HITL resumption / already-wire-format history). Older ones must be
        stripped; the answer content stays.

        Pre-fix: only content-list ``thinking`` blocks were matched, so a message
        whose thinking lived in the top-level field kept it forever.
        """
        msgs: list[Any] = [{"role": "system", "content": "sys"}]
        for i in range(3):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append(
                {
                    "role": "assistant",
                    "content": f"a{i}",
                    "thinking_blocks": [{"type": "thinking", "thinking": f"t{i}", "signature": f"s{i}"}],
                }
            )

        result = ContextEditor.clear_thinking_blocks(msgs, keep_turns=1)

        assistants = [m for m in result if m.get("role") == "assistant"]
        # Most-recent assistant keeps its thinking_blocks.
        assert "thinking_blocks" in assistants[-1]
        # Older assistants have thinking_blocks removed but keep their answer.
        for a in assistants[:-1]:
            assert "thinking_blocks" not in a
        assert [a["content"] for a in assistants] == ["a0", "a1", "a2"]

    def test_mixed_thinking_forms_share_one_keep_budget(self):
        """keep_turns counts thinking-bearing items across all shapes together."""
        msgs: list[Any] = [
            {"role": "user", "content": "q0"},
            {"type": "reasoning", "summary": [], "content": [], "encrypted_content": "sigR"},
            {
                "role": "assistant",
                "content": "a1",
                "thinking_blocks": [{"type": "thinking", "thinking": "t", "signature": "s"}],
            },
        ]
        result = ContextEditor.clear_thinking_blocks(msgs, keep_turns=1)
        # Two thinking-bearing items; keep 1 (the most recent = the message).
        assert not any(m.get("type") == "reasoning" for m in result)
        kept_assistant = next(m for m in result if m.get("role") == "assistant")
        assert "thinking_blocks" in kept_assistant

    def test_original_not_mutated(self):
        import copy

        msgs = self._make_thinking_messages(3)
        original = copy.deepcopy(msgs)

        ContextEditor.clear_thinking_blocks(msgs, keep_turns=1)
        assert msgs == original


# ── Regression: cast instead of bare Any in clear_tool_results ───────


class TestClearToolResultsCastType:
    """clear_tool_results must use cast() rather than widening to Any.

    This is a structural/type-correctness test: the returned list must
    contain dict-like items (not be a list of Any at the call site) so
    downstream code can access keys without spurious type errors.
    """

    def test_cleared_item_is_dict_with_correct_output(self) -> None:
        msgs: list[Any] = [
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "original"},
            {"type": "function_call", "call_id": "c2", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c2", "output": "recent"},
        ]
        result = ContextEditor.clear_tool_results(msgs, keep=1)
        tool_outputs: list[Any] = [m for m in result if m.get("type") == "function_call_output"]
        # Older item was cleared.
        assert tool_outputs[0]["output"] == "[tool result cleared to save context]"
        # Cleared item is still dict-shaped (not None or broken).
        assert isinstance(tool_outputs[0], dict)
        # Recent item preserved.
        assert tool_outputs[1]["output"] == "recent"

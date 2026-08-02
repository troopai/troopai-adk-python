"""Regression tests for ``ItemHelpers.message_to_run_items`` parsing of
assistant ``thinking_blocks``.

Focus: a ``redacted_thinking`` block whose ``data`` value is an explicit
``None`` (e.g. ``"data": null`` deserialized from a persisted/forwarded
message) must not crash the history rebuild. ``.get("data", "")`` only
defaults on an *absent* key, so an explicit ``None`` previously reached
``len(None)`` and raised ``TypeError`` — aborting conversion of the entire
message instead of treating the redacted block as empty.
"""

from __future__ import annotations

from typing import Any

from troopai.adk.types.items import ItemHelpers, ReasoningItem


class TestRedactedThinkingNullData:
    def test_redacted_thinking_with_none_data_does_not_raise(self) -> None:
        """An explicit ``data: None`` is treated as empty, not a crash."""
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": "hello",
            "thinking_blocks": [
                {"type": "redacted_thinking", "data": None},
            ],
        }

        # Before the fix this raised ``TypeError: object of type 'NoneType'
        # has no len()``.
        items = ItemHelpers.message_to_run_items(msg)

        # ``None`` data contributes no signature, so no reasoning content was
        # produced — only the assistant message survives.
        assert all(not isinstance(item, ReasoningItem) for item in items)
        assert len(items) == 1

    def test_redacted_thinking_with_real_data_preserved_as_signature(self) -> None:
        """A real ``data`` payload still round-trips into the reasoning item's
        encrypted content (signature channel)."""
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": "answer",
            "thinking_blocks": [
                {"type": "redacted_thinking", "data": "ENCRYPTED_BLOB"},
            ],
        }

        items = ItemHelpers.message_to_run_items(msg)

        reasoning = [item for item in items if isinstance(item, ReasoningItem)]
        assert len(reasoning) == 1
        assert reasoning[0].raw.encrypted_content == "ENCRYPTED_BLOB"

    def test_redacted_thinking_with_absent_data_does_not_raise(self) -> None:
        """A missing ``data`` key (default branch) stays well-behaved — the
        block contributes no signature and no reasoning item is emitted."""
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": "hi",
            "thinking_blocks": [
                {"type": "redacted_thinking"},
            ],
        }

        items = ItemHelpers.message_to_run_items(msg)

        assert all(not isinstance(item, ReasoningItem) for item in items)
        assert len(items) == 1

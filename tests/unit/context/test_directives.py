"""Tests for context directives — LLM-driven context management."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from troopai.adk.context.directives import (
    CompactDirective,
    ContextDirective,
    DirectiveStore,
    DropDirective,
    apply_directives,
)
from troopai.adk.llms.llm import LLM
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.schemas import AgentOutputSchemaBase
from troopai.adk.tools import Tool
from troopai.adk.types.input import LLMInputContentItem
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseText,
    LLMStreamEvent,
)


class _StubLLM(LLM):
    """Test stub returning a canned ``LLMResponse`` from ``acomplete``."""

    def __init__(self, text: str = "Summary of earlier conversation.") -> None:
        self._text = text
        self.call_count = 0

    # ``LLM.acomplete`` is ``@overload``-typed on ``stream`` — a concrete
    # stub cannot match both overloads simultaneously.
    async def acomplete(  # type: ignore[override]
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
        self.call_count += 1
        return LLMResponse(
            response_id="stub",
            model="stub",
            response=[LLMResponseText(text=self._text)],
        )


# =====================================================================
# DirectiveStore tests
# =====================================================================


class TestDirectiveStore:
    """Tests for DirectiveStore."""

    def test_add_and_consume(self) -> None:
        store = DirectiveStore()
        store.add(DropDirective(preserve=5))
        store.add(CompactDirective(preserve=3))

        directives = store.consume()
        assert len(directives) == 2
        assert isinstance(directives[0], DropDirective)
        assert isinstance(directives[1], CompactDirective)

    def test_consume_clears_pending(self) -> None:
        store = DirectiveStore()
        store.add(DropDirective(preserve=5))
        store.consume()
        assert store.count == 0
        assert store.consume() == []

    def test_count(self) -> None:
        store = DirectiveStore()
        assert store.count == 0
        store.add(DropDirective(preserve=5))
        assert store.count == 1
        store.add(CompactDirective(preserve=3))
        assert store.count == 2

    def test_empty_consume(self) -> None:
        store = DirectiveStore()
        assert store.consume() == []


# =====================================================================
# apply_directives tests — DropDirective
# =====================================================================


class TestApplyDropDirective:
    """Tests for apply_directives with DropDirective."""

    def test_drop_preserves_system_and_recent(self) -> None:
        messages: list[LLMInputContentItem] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
        ]
        store = DirectiveStore()
        store.add(DropDirective(preserve=2))

        result = asyncio.run(apply_directives(messages, store, _StubLLM(), "gpt-4o"))
        # System + last 2 messages
        assert len(result) == 3
        assert result[0].get("role") == "system"
        assert result[1].get("content") == "msg 4"
        assert result[2].get("content") == "msg 5"

    def test_drop_preserves_all_when_preserve_exceeds_count(self) -> None:
        messages: list[LLMInputContentItem] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        store = DirectiveStore()
        store.add(DropDirective(preserve=10))

        result = asyncio.run(apply_directives(messages, store, _StubLLM(), "gpt-4o"))
        assert len(result) == 2  # Nothing dropped

    def test_drop_without_system_message(self) -> None:
        messages: list[LLMInputContentItem] = [
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
        ]
        store = DirectiveStore()
        store.add(DropDirective(preserve=1))

        result = asyncio.run(apply_directives(messages, store, _StubLLM(), "gpt-4o"))
        assert len(result) == 1
        assert result[0].get("content") == "msg 3"

    def test_empty_store_returns_unchanged(self) -> None:
        messages: list[LLMInputContentItem] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        store = DirectiveStore()

        result = asyncio.run(apply_directives(messages, store, _StubLLM(), "gpt-4o"))
        assert result == messages


# =====================================================================
# apply_directives tests — CompactDirective
# =====================================================================


class TestApplyCompactDirective:
    """Tests for apply_directives with CompactDirective."""

    def test_compact_invokes_compactor(self) -> None:
        llm = _StubLLM(text="Summary of earlier conversation.")

        messages: list[LLMInputContentItem] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
        ]
        store = DirectiveStore()
        store.add(CompactDirective(preserve=2))

        result = asyncio.run(apply_directives(messages, store, llm, "gpt-4o"))

        # Should have: system + summary + last 2 messages
        assert llm.call_count == 1
        assert len(result) >= 3
        assert result[0].get("role") == "system"
        # ``_compaction`` is an out-of-band marker outside every TypedDict
        # variant of ``LLMInputContentItem``; launder through ``Any`` for the
        # assertion rather than suppressing the access.
        summary_msg: Any = result[1]
        assert summary_msg.get("_compaction") is True
        assert "Summary" in summary_msg["content"]

    def test_compact_skipped_when_preserve_exceeds_count(self) -> None:
        messages: list[LLMInputContentItem] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        store = DirectiveStore()
        store.add(CompactDirective(preserve=10))

        result = asyncio.run(apply_directives(messages, store, _StubLLM(), "gpt-4o"))
        assert len(result) == 2  # Nothing compacted


# =====================================================================
# Multiple directives
# =====================================================================


class TestMultipleDirectives:
    """Tests for applying multiple directives in sequence."""

    def test_drop_then_compact_skipped(self) -> None:
        """If drop reduces messages enough, compact should skip."""
        messages: list[LLMInputContentItem] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
        ]
        store = DirectiveStore()
        store.add(DropDirective(preserve=2))
        store.add(CompactDirective(preserve=5))  # preserve > remaining, should skip

        result = asyncio.run(apply_directives(messages, store, _StubLLM(), "gpt-4o"))
        # Drop keeps 2, compact skips (preserve=5 > 2 remaining)
        assert len(result) == 3  # system + 2 messages

    def test_drop_directive_removes_orphaned_tool_result(self) -> None:
        """A DropDirective that evicts a function_call must not leave its
        orphaned function_call_output behind.

        Regression: when no ContextManagementConfig is set,
        ContextManager.prepare_messages (which also strips orphans) never
        runs, so apply_directives itself must guarantee the cleanup —
        otherwise the orphaned tool result reaches the next LLM call and
        Anthropic/Gemini reject the request.
        """
        messages: list[LLMInputContentItem] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "r1"},
            {"role": "user", "content": "q2"},
        ]
        store = DirectiveStore()
        # preserve=2 keeps the last two body messages —
        # [function_call_output(c1), user(q2)] — evicting the matching
        # function_call and orphaning the output.
        store.add(DropDirective(preserve=2))

        result = asyncio.run(apply_directives(messages, store, _StubLLM(), "gpt-4o"))

        assert not any(m.get("type") == "function_call_output" for m in result)
        assert not any(m.get("type") == "function_call" for m in result)
        assert result[0].get("role") == "system"
        assert result[-1].get("content") == "q2"


# =====================================================================
# Directive type tests
# =====================================================================


class TestDirectiveTypes:
    """Tests for directive dataclasses."""

    def test_drop_directive_frozen(self) -> None:
        d = DropDirective(preserve=5)
        assert d.preserve == 5
        with pytest.raises(AttributeError):
            # Frozen-dataclass write is deliberate here — the test
            # verifies that AttributeError fires. The ignore suppresses
            # pyright's correct "cannot assign to frozen" warning
            # because that's the exact invariant the test exercises.
            d.preserve = 10  # type: ignore[misc]

    def test_compact_directive_frozen(self) -> None:
        d = CompactDirective(preserve=3)
        assert d.preserve == 3
        with pytest.raises(AttributeError):
            # Same pattern as the test above — frozen-dataclass write
            # is the AttributeError we're verifying.
            d.preserve = 7  # type: ignore[misc]

    def test_union_type(self) -> None:
        directives: list[ContextDirective] = [
            DropDirective(preserve=5),
            CompactDirective(preserve=3),
        ]
        assert len(directives) == 2


# =====================================================================
# Regression: multiple CompactDirectives in one batch
# =====================================================================


class TestMultipleCompactDirectives:
    """Regression: apply_directives must process ALL CompactDirectives,
    not early-return after the first successful one.

    Pre-fix: the loop ``return``ed immediately after the first compacted
    result, silently discarding any remaining CompactDirectives in the
    batch — asymmetric with DropDirective which applied them all.
    """

    def test_two_compact_directives_both_applied(self) -> None:
        """Two sequential CompactDirectives should each reduce the body."""
        call_log: list[int] = []

        class _CountingLLM(_StubLLM):
            async def acomplete(  # type: ignore[override]
                self,
                messages: str | list[Any],
                llm_config: Any = None,
                tools: Any = None,
                output_schema: Any = None,
                stream: bool = False,
            ) -> Any:
                call_log.append(len(messages))
                self.call_count += 1
                return LLMResponse(
                    response_id="stub",
                    model="stub",
                    response=[LLMResponseText(text=f"Summary #{self.call_count}")],
                )

        llm = _CountingLLM()

        # System + 8 body messages (4 user/assistant pairs)
        messages: list[LLMInputContentItem] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "q4"},
            {"role": "assistant", "content": "a4"},
        ]
        store = DirectiveStore()
        # First compact: preserve=4 → compacts first 4, keeps last 4.
        # Second compact: preserve=2 → compacts from summary onwards, keeps last 2.
        store.add(CompactDirective(preserve=4))
        store.add(CompactDirective(preserve=2))

        result = asyncio.run(apply_directives(messages, store, llm, "gpt-4o"))

        # Both directives must have fired.
        assert llm.call_count == 2, f"Expected 2 LLM calls (one per CompactDirective), got {llm.call_count}"
        # Final result: system + second-summary + last 2 messages.
        assert result[0].get("role") == "system"
        result_any: Any = result[1]
        assert result_any.get("_compaction") is True, "Second summary message should carry _compaction=True marker"
        assert len(result) == 4  # system + summary + 2 preserved

    def test_second_compact_skipped_when_body_too_short(self) -> None:
        """If the first compaction makes the body shorter than the second
        directive's preserve count, the second should skip gracefully.
        """

        class _CountingLLM(_StubLLM):
            async def acomplete(  # type: ignore[override]
                self,
                messages: str | list[Any],
                llm_config: Any = None,
                tools: Any = None,
                output_schema: Any = None,
                stream: bool = False,
            ) -> Any:
                self.call_count += 1
                return LLMResponse(
                    response_id="stub",
                    model="stub",
                    response=[LLMResponseText(text="Summary")],
                )

        llm = _CountingLLM()
        messages: list[LLMInputContentItem] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        store = DirectiveStore()
        # First compact: preserve=2, compacts first 2 body msgs, rebuilds to
        # [system, summary, q2, a2] — body becomes [summary, q2, a2] → 3 items.
        # Second compact: preserve=5 > 3 body items → skipped.
        store.add(CompactDirective(preserve=2))
        store.add(CompactDirective(preserve=5))

        result = asyncio.run(apply_directives(messages, store, llm, "gpt-4o"))

        # Only the first directive fired.
        assert llm.call_count == 1
        assert len(result) >= 1  # at minimum system survived

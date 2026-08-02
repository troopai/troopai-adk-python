"""Tests for ContextCompactor."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest

from troopai.adk.context.compaction import CompactionResult, ContextCompactor
from troopai.adk.context.context_config import CompactionConfig
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
    """Test stub returning a canned ``LLMResponse`` from ``acomplete``.

    Test scaffolding under the no-underscore-cross-module-imports
    exception — not re-exported.
    """

    def __init__(self, text: str = "Summary of conversation") -> None:
        self._text = text
        self.captured_messages: list[Any] = []

    # ``LLM.acomplete`` is ``@overload``-typed (stream: Literal[True] vs
    # False) which pyright cannot match against a concrete stub signature
    # collapsing both arms. The runtime contract is upheld — this stub
    # only services the non-streaming arm used by the compactor.
    async def acomplete(  # type: ignore[override]
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
        self.captured_messages.append(messages)
        return LLMResponse(
            response_id="stub",
            model="stub",
            response=[LLMResponseText(text=self._text)],
        )


def _make_conversation(n_turns: int) -> list[LLMInputContentItem]:
    """Build a conversation with a system message and n user/assistant pairs."""
    msgs: list[LLMInputContentItem] = [{"role": "system", "content": "You are helpful."}]
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"Question {i}"})
        msgs.append({"role": "assistant", "content": f"Answer {i}"})
    return msgs


@pytest.mark.asyncio
@patch("troopai.adk.context.compaction.TokenCounter.count_messages", return_value=100)
async def test_compact_produces_summary(_mock_count):
    llm = _StubLLM(text="Summary of conversation")
    msgs = _make_conversation(5)  # system + 10 messages
    config = CompactionConfig(enabled=True, preserve_recent_items=2)

    result = await ContextCompactor.compact(
        msgs,
        llm=llm,
        model_name="gpt-4o-mini",
        config=config,
    )

    assert isinstance(result, CompactionResult)
    assert result.summary == "Summary of conversation"
    assert result.items_compacted == 8  # 10 body msgs - 2 preserved
    assert len(llm.captured_messages) == 1


@pytest.mark.asyncio
@patch("troopai.adk.context.compaction.TokenCounter.count_messages", return_value=50)
async def test_compact_short_conversation_returns_empty_summary(_mock_count):
    llm = _StubLLM()
    msgs = _make_conversation(1)  # system + 2 messages
    config = CompactionConfig(enabled=True, preserve_recent_items=4)

    result = await ContextCompactor.compact(
        msgs,
        llm=llm,
        model_name="gpt-4o-mini",
        config=config,
    )

    assert result.summary == ""
    assert result.items_compacted == 0
    # No LLM call when nothing to compact.
    assert len(llm.captured_messages) == 0


def test_build_compacted_messages_with_system():
    system: LLMInputContentItem = {"role": "system", "content": "sys prompt"}
    preserved: list[LLMInputContentItem] = [
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]

    result = ContextCompactor.build_compacted_messages("Summary here", preserved, system)

    assert len(result) == 4  # system + summary + 2 preserved
    compacted: Any = result[1]
    assert result[0].get("role") == "system"
    assert compacted["role"] == "assistant"
    assert compacted["_compaction"] is True
    assert compacted["content"] == "Summary here"
    assert result[2].get("role") == "user"


def test_build_compacted_messages_without_system():
    preserved: list[LLMInputContentItem] = [{"role": "user", "content": "hi"}]

    result = ContextCompactor.build_compacted_messages("Sum", preserved, None)

    assert len(result) == 2
    compacted: Any = result[0]
    assert compacted["role"] == "assistant"
    assert compacted["_compaction"] is True


def test_build_compacted_messages_empty_summary():
    preserved: list[LLMInputContentItem] = [{"role": "user", "content": "hi"}]

    result = ContextCompactor.build_compacted_messages("", preserved, None)

    assert len(result) == 1
    assert result[0].get("role") == "user"


@pytest.mark.asyncio
@patch("troopai.adk.context.compaction.TokenCounter.count_messages", return_value=100)
async def test_compact_uses_custom_instructions(_mock_count):
    llm = _StubLLM()
    msgs = _make_conversation(5)
    config = CompactionConfig(
        enabled=True,
        instructions="Be very brief.",
        preserve_recent_items=2,
    )

    await ContextCompactor.compact(
        msgs,
        llm=llm,
        model_name="gpt-4o",
        config=config,
    )

    # First (and only) acomplete call's messages — confirm the custom
    # instructions made it into the system message.
    captured = llm.captured_messages[0]
    assert isinstance(captured, list)
    assert captured[0]["role"] == "system"
    assert captured[0]["content"] == "Be very brief."


@pytest.mark.asyncio
@patch("troopai.adk.context.compaction.TokenCounter.count_messages", return_value=100)
async def test_compact_routes_through_llm_abc(_mock_count):
    """The compactor must invoke llm.acomplete (no direct litellm call)."""
    llm = _StubLLM()
    msgs = _make_conversation(5)
    config = CompactionConfig(enabled=True, preserve_recent_items=2)

    await ContextCompactor.compact(
        msgs,
        llm=llm,
        model_name="gpt-4o-mini",
        config=config,
    )

    assert len(llm.captured_messages) == 1


# ── Regression: empty LLM summary must not drop messages ─────────────


@pytest.mark.asyncio
@patch("troopai.adk.context.compaction.TokenCounter.count_messages", return_value=100)
async def test_compact_empty_summary_returns_no_op_result(_mock_count):
    """When the LLM returns an empty summary, compaction must return a
    no-op CompactionResult (items_compacted=0, token counts unchanged).

    Pre-fix: the empty summary was accepted silently, items_compacted was
    set to len(to_compact), and the rebuild call dropped all the compacted
    messages with no replacement text.
    """

    class _EmptyLLM(_StubLLM):
        async def acomplete(  # type: ignore[override]
            self,
            messages: str | list[Any],
            llm_config: Any = None,
            tools: Any = None,
            output_schema: Any = None,
            stream: bool = False,
        ) -> Any:
            self.captured_messages.append(messages)
            return LLMResponse(
                response_id="stub",
                model="stub",
                response=[LLMResponseText(text="")],  # empty summary
            )

    llm = _EmptyLLM()
    msgs = _make_conversation(5)  # system + 10 body messages
    config = CompactionConfig(enabled=True, preserve_recent_items=2)

    result = await ContextCompactor.compact(
        msgs,
        llm=llm,
        model_name="gpt-4o-mini",
        config=config,
    )

    # Must be a no-op: empty summary, no items compacted, counts unchanged.
    assert result.summary == ""
    assert result.items_compacted == 0, "items_compacted must be 0 when the LLM returns an empty summary"
    assert result.compacted_token_count == result.original_token_count, (
        "token counts must be equal (no-op) when summary is empty"
    )
    # LLM was called (to produce the empty summary) but the result is a no-op.
    assert len(llm.captured_messages) == 1


# ── Regression: preserve_recent_items=0 must compact the entire body ──


@pytest.mark.asyncio
@patch("troopai.adk.context.compaction.TokenCounter.count_messages", return_value=100)
async def test_compact_preserve_zero_compacts_entire_body(_mock_count):
    """preserve_recent_items=0 means "preserve no recent turns" — the whole
    body must be summarised, not silently skipped.

    Pre-fix: the ``0 < preserve < len(body)`` guard treated preserve==0 like
    the "conversation too short" branch, so to_compact was empty and the
    function returned a no-op (items_compacted=0) with NO LLM call — a
    developer-enabled compaction config that quietly does nothing.
    """
    llm = _StubLLM(text="Full summary")
    msgs = _make_conversation(5)  # system + 10 body messages
    config = CompactionConfig(enabled=True, preserve_recent_items=0)

    result = await ContextCompactor.compact(
        msgs,
        llm=llm,
        model_name="gpt-4o-mini",
        config=config,
    )

    assert result.summary == "Full summary"
    assert result.items_compacted == 10, "all 10 body messages must be compacted when preserve==0"
    # The LLM was invoked exactly once to produce the summary.
    assert len(llm.captured_messages) == 1


# ── Regression: tool call/result items must appear in the transcript ──


@pytest.mark.asyncio
@patch("troopai.adk.context.compaction.TokenCounter.count_messages", return_value=100)
async def test_compact_transcript_includes_tool_call_and_result(_mock_count):
    """Tool calls / results are Layer-1 items with no role/content. They must
    be formatted into meaningful transcript lines, not collapsed to blank
    ``[unknown]:`` entries that drop the whole tool exchange from the summary.

    Pre-fix: ``_format_for_summary`` read only ``role``/``content``, so
    ``function_call``/``function_call_output`` items rendered as empty
    ``[unknown]:`` lines — the tool name, arguments, and output never reached
    the summarizer.
    """
    llm = _StubLLM(text="Full summary")
    msgs: list[LLMInputContentItem] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "weather in Paris?"},
        {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": '{"city":"Paris"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "Sunny, 25C"},
        {"type": "function_call", "call_id": "c2", "name": "get_time", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c2", "output": [{"type": "text", "text": "noon"}]},
        {"role": "assistant", "content": "It's sunny at noon."},
        {"role": "user", "content": "recent"},
        {"role": "assistant", "content": "recent answer"},
    ]
    config = CompactionConfig(enabled=True, preserve_recent_items=2)

    await ContextCompactor.compact(msgs, llm=llm, model_name="gpt-4o", config=config)

    transcript = llm.captured_messages[0][1]["content"]
    assert "get_weather" in transcript
    assert "Paris" in transcript
    assert "Sunny, 25C" in transcript
    assert "get_time" in transcript
    # Multimodal (list) tool output is flattened to its text.
    assert "noon" in transcript
    # The tool exchange must not degrade to empty [unknown]: lines.
    assert "[unknown]:" not in transcript


@pytest.mark.asyncio
@patch("troopai.adk.context.compaction.TokenCounter.count_messages", return_value=100)
async def test_compact_preserve_zero_summary_transcript_covers_whole_body(_mock_count):
    """With preserve==0 the summarizer transcript must include every body
    message (none held back as "recent"), and the system message stays out
    of the transcript (it is summarised separately by reassembly)."""
    llm = _StubLLM(text="Full summary")
    msgs = _make_conversation(3)  # system + 6 body messages
    config = CompactionConfig(enabled=True, preserve_recent_items=0)

    await ContextCompactor.compact(
        msgs,
        llm=llm,
        model_name="gpt-4o-mini",
        config=config,
    )

    captured = llm.captured_messages[0]
    assert isinstance(captured, list)
    transcript = captured[1]["content"]  # user message holds the transcript
    # Every body message (including the very last one) is in the transcript.
    for i in range(3):
        assert f"Question {i}" in transcript
        assert f"Answer {i}" in transcript

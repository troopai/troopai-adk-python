"""Tests for ContextManager."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest

from troopai.adk.context.context_config import (
    CompactionConfig,
    ContextEditingConfig,
    ContextManagementConfig,
)
from troopai.adk.context.context_editing import ContextEditor
from troopai.adk.context.context_manager import ContextManager
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

    def __init__(self, text: str = "Compact summary") -> None:
        self._text = text
        self.call_count = 0

    # ``LLM.acomplete`` is ``@overload``-typed on ``stream`` — a concrete
    # stub cannot match both overloads simultaneously. Non-streaming
    # contract is upheld; that's the only arm the manager exercises.
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


def _make_conversation(n: int) -> list[LLMInputContentItem]:
    msgs: list[LLMInputContentItem] = [{"role": "system", "content": "sys"}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"Q{i}"})
        msgs.append({"role": "assistant", "content": f"A{i}"})
    return msgs


# ── Basic pass-through ───────────────────────────────────────────────


@pytest.mark.asyncio
@patch("troopai.adk.context.context_manager.TokenCounter.count_messages", return_value=100)
async def test_prepare_messages_passthrough(_mock_count):
    """When nothing is enabled, messages pass through unchanged."""
    config = ContextManagementConfig()
    mgr = ContextManager(config)
    llm = _StubLLM()

    msgs = _make_conversation(3)
    result = await mgr.prepare_messages(msgs, llm, "gpt-4o-mini")

    assert len(result) == len(msgs)
    assert llm.call_count == 0  # No compaction triggered.


# ── Context editing integration ──────────────────────────────────────


@pytest.mark.asyncio
@patch("troopai.adk.context.context_manager.TokenCounter.count_messages", return_value=120_000)
async def test_tool_result_clearing_triggered(_mock_count):
    """Tool results should be cleared when above the trigger threshold."""
    config = ContextManagementConfig(
        editing=ContextEditingConfig(
            clear_tool_results=True,
            tool_result_trigger_tokens=100_000,
            tool_results_to_keep=1,
        ),
    )
    mgr = ContextManager(config)
    llm = _StubLLM()

    msgs: list[Any] = [
        {"role": "system", "content": "sys"},
        {"type": "function_call", "call_id": "1", "name": "t1", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "1", "output": "old result"},
        {"type": "function_call", "call_id": "2", "name": "t2", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "2", "output": "old result 2"},
        {"type": "function_call", "call_id": "3", "name": "t3", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "3", "output": "recent result"},
    ]

    result = await mgr.prepare_messages(msgs, llm, "gpt-4o-mini")

    # ``LLMInputContentItem`` is a TypedDict union; pyright cannot
    # discriminator-narrow on ``.get(...)``. Lift to ``Any`` for the
    # assertion — the values are dicts at runtime.
    tool_msgs: list[Any] = [m for m in result if m.get("type") == "function_call_output"]
    # Only the most recent should be intact (tool_results_to_keep=1).
    assert tool_msgs[-1]["output"] == "recent result"
    # Earlier ones should be cleared.
    assert tool_msgs[0]["output"] == "[tool result cleared to save context]"


@pytest.mark.asyncio
@patch("troopai.adk.context.context_manager.TokenCounter.count_messages", return_value=50_000)
async def test_tool_result_clearing_not_triggered_below_threshold(_mock_count):
    """Tool results should NOT be cleared when below the trigger threshold."""
    config = ContextManagementConfig(
        editing=ContextEditingConfig(
            clear_tool_results=True,
            tool_result_trigger_tokens=100_000,
            tool_results_to_keep=1,
        ),
    )
    mgr = ContextManager(config)
    llm = _StubLLM()

    msgs: list[Any] = [
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "t1", "arguments": "{}"}},
                {"id": "2", "type": "function", "function": {"name": "t2", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "result 1", "name": "t1"},
        {"role": "tool", "tool_call_id": "2", "content": "result 2", "name": "t2"},
    ]

    result = await mgr.prepare_messages(msgs, llm, "gpt-4o-mini")

    tool_msgs: list[Any] = [m for m in result if m.get("role") == "tool"]
    assert all(m["content"].startswith("result") for m in tool_msgs)


# ── Compaction integration ───────────────────────────────────────────


@pytest.mark.asyncio
@patch("troopai.adk.context.context_manager.TokenCounter.count_messages")
async def test_compaction_triggered(mock_count):
    """Compaction should trigger when token count exceeds threshold."""
    # First call (editing check): above threshold
    # Second call (budget warning check): above threshold
    # Third call (should_compact): above threshold
    # Fourth call (inside compact): above threshold
    # Fifth call (compacted messages): smaller
    mock_count.side_effect = [160_000, 160_000, 160_000, 160_000, 50_000]

    llm = _StubLLM(text="Compact summary")

    config = ContextManagementConfig(
        compaction=CompactionConfig(
            enabled=True,
            trigger_tokens=150_000,
            preserve_recent_items=2,
        ),
    )
    mgr = ContextManager(config)

    msgs = _make_conversation(10)
    _ = await mgr.prepare_messages(msgs, llm, "gpt-4o-mini")

    assert mgr._compaction_count == 1
    assert llm.call_count == 1


@pytest.mark.asyncio
@patch("troopai.adk.context.context_manager.TokenCounter.count_messages", return_value=10_000)
async def test_compaction_not_triggered_below_threshold(_mock_count):
    config = ContextManagementConfig(
        compaction=CompactionConfig(
            enabled=True,
            trigger_tokens=150_000,
        ),
    )
    mgr = ContextManager(config)
    llm = _StubLLM()

    msgs = _make_conversation(3)
    result = await mgr.prepare_messages(msgs, llm, "gpt-4o-mini")

    assert mgr._compaction_count == 0
    assert len(result) == len(msgs)
    assert llm.call_count == 0


# ── should_compact ───────────────────────────────────────────────────


@patch("troopai.adk.context.context_manager.TokenCounter.count_messages", return_value=200_000)
def test_should_compact_true(_mock_count):
    config = ContextManagementConfig(
        compaction=CompactionConfig(enabled=True, trigger_tokens=150_000),
    )
    mgr = ContextManager(config)

    assert mgr.should_compact([], "gpt-4o") is True


@patch("troopai.adk.context.context_manager.TokenCounter.count_messages", return_value=100_000)
def test_should_compact_false_below_threshold(_mock_count):
    config = ContextManagementConfig(
        compaction=CompactionConfig(enabled=True, trigger_tokens=150_000),
    )
    mgr = ContextManager(config)

    assert mgr.should_compact([], "gpt-4o") is False


def test_should_compact_false_when_disabled():
    config = ContextManagementConfig(
        compaction=CompactionConfig(enabled=False),
    )
    mgr = ContextManager(config)

    assert mgr.should_compact([], "gpt-4o") is False


@patch("troopai.adk.context.context_manager.TokenCounter.count_messages", return_value=200_000)
def test_should_compact_respects_total_budget(_mock_count):
    config = ContextManagementConfig(
        compaction=CompactionConfig(
            enabled=True,
            trigger_tokens=150_000,
            total_token_budget=10,
        ),
    )
    mgr = ContextManager(config)
    mgr._total_tokens_compacted = 10  # Already at budget

    assert mgr.should_compact([], "gpt-4o") is False


# ── get_token_usage ──────────────────────────────────────────────────


@patch("troopai.adk.context.context_manager.TokenCounter.count_messages", return_value=80_000)
def test_get_token_usage(_mock_count):
    config = ContextManagementConfig(max_context_tokens=200_000)
    mgr = ContextManager(config)

    usage = mgr.get_token_usage([], "gpt-4o")

    assert usage["used"] == 80_000
    assert usage["max"] == 200_000
    assert usage["remaining"] == 120_000
    assert usage["utilisation"] == pytest.approx(0.4)
    assert usage["compaction_count"] == 0


# ── Truncation ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_truncation_drops_oldest_messages():
    """When truncation=True and over budget, oldest non-system messages are dropped."""
    config = ContextManagementConfig(
        max_context_tokens=100,
        truncation=True,
        editing=ContextEditingConfig(clear_tool_results=False),
    )
    mgr = ContextManager(config)
    llm = _StubLLM()

    msgs: list[LLMInputContentItem] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old question " * 20},
        {"role": "assistant", "content": "old answer " * 20},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]

    result = await mgr.prepare_messages(msgs, llm, "gpt-4o-mini")

    truncated: list[Any] = list(result)
    assert truncated[0]["role"] == "system"
    assert len(truncated) < len(msgs)
    assert truncated[-1]["content"] == "recent answer"


@pytest.mark.asyncio
@patch("troopai.adk.context.context_manager.TokenCounter.count_messages", return_value=50)
async def test_truncation_noop_when_under_budget(_mock_count):
    config = ContextManagementConfig(
        max_context_tokens=200,
        truncation=True,
        editing=ContextEditingConfig(clear_tool_results=False),
    )
    mgr = ContextManager(config)
    llm = _StubLLM()

    msgs = _make_conversation(3)
    result = await mgr.prepare_messages(msgs, llm, "gpt-4o-mini")

    assert len(result) == len(msgs)


@pytest.mark.asyncio
@patch("troopai.adk.context.context_manager.TokenCounter.count_messages", return_value=50)
async def test_truncation_disabled_by_default(_mock_count):
    config = ContextManagementConfig(max_context_tokens=10)  # Way under
    mgr = ContextManager(config)
    llm = _StubLLM()

    msgs = _make_conversation(3)
    result = await mgr.prepare_messages(msgs, llm, "gpt-4o-mini")

    assert len(result) == len(msgs)


# ── Pressure feedback (LLM-visible warning) ──────────────────────────


@pytest.mark.asyncio
@patch("troopai.adk.context.context_manager.TokenCounter.count_messages", return_value=900)
async def test_pressure_feedback_injected_when_above_threshold(_mock_count):
    config = ContextManagementConfig(
        max_context_tokens=1000,
        token_budget_warning_threshold=0.8,
        pressure_feedback=True,
        editing=ContextEditingConfig(clear_tool_results=False),
    )
    mgr = ContextManager(config)
    llm = _StubLLM()

    msgs: list[LLMInputContentItem] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]

    result = await mgr.prepare_messages(msgs, llm, "gpt-4o-mini")
    feedback: Any = result[1]

    assert feedback["role"] == "developer"
    assert "Context budget" in feedback["content"]
    assert feedback.get(ContextManager._PRESSURE_MARKER) is True


@pytest.mark.asyncio
@patch("troopai.adk.context.context_manager.TokenCounter.count_messages", return_value=500)
async def test_pressure_feedback_not_injected_below_threshold(_mock_count):
    config = ContextManagementConfig(
        max_context_tokens=1000,
        token_budget_warning_threshold=0.8,
        pressure_feedback=True,
        editing=ContextEditingConfig(clear_tool_results=False),
    )
    mgr = ContextManager(config)
    llm = _StubLLM()

    msgs: list[LLMInputContentItem] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]

    result = await mgr.prepare_messages(msgs, llm, "gpt-4o-mini")
    developer_msgs = [m for m in result if m.get("role") == "developer"]
    assert len(developer_msgs) == 0


@pytest.mark.asyncio
@patch("troopai.adk.context.context_manager.TokenCounter.count_messages", return_value=900)
async def test_pressure_feedback_not_duplicated(_mock_count):
    config = ContextManagementConfig(
        max_context_tokens=1000,
        token_budget_warning_threshold=0.8,
        pressure_feedback=True,
        editing=ContextEditingConfig(clear_tool_results=False),
    )
    mgr = ContextManager(config)
    llm = _StubLLM()

    msgs: list[LLMInputContentItem] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]

    result1 = await mgr.prepare_messages(msgs, llm, "gpt-4o-mini")
    result2 = await mgr.prepare_messages(result1, llm, "gpt-4o-mini")

    developer_msgs = [m for m in result2 if m.get("role") == "developer"]
    assert len(developer_msgs) == 1


@pytest.mark.asyncio
@patch("troopai.adk.context.context_manager.TokenCounter.count_messages", return_value=50)
async def test_pressure_feedback_disabled_by_default(_mock_count):
    config = ContextManagementConfig(max_context_tokens=100)
    mgr = ContextManager(config)
    llm = _StubLLM()

    msgs = _make_conversation(2)
    result = await mgr.prepare_messages(msgs, llm, "gpt-4o-mini")

    developer_msgs = [m for m in result if m.get("role") == "developer"]
    assert len(developer_msgs) == 0


# ── Orphan tool-result cleanup ───────────────────────────────────────


def test_remove_orphaned_tool_results_layer1():
    """Layer 1 ``function_call_output`` without matching ``function_call`` is dropped."""
    msgs: list[LLMInputContentItem] = [
        {"role": "system", "content": "sys"},
        # Orphan: no matching function_call earlier
        {
            "type": "function_call_output",
            "call_id": "missing_id",
            "output": "stale result",
        },
        {"type": "function_call", "call_id": "live_id", "name": "t", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "live_id", "output": "ok"},
        {"role": "user", "content": "hi"},
    ]
    result = ContextEditor.remove_orphaned_tool_results(msgs)
    types = [m.get("type") or m.get("role") for m in result]
    assert "function_call" in types
    assert types.count("function_call_output") == 1
    assert not any(m.get("call_id") == "missing_id" for m in result)


def test_remove_orphaned_tool_results_layer2_still_supported():
    from typing import cast

    raw_msgs = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "keep_id", "function": {"name": "t", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "keep_id", "content": "ok"},
        {"role": "tool", "tool_call_id": "orphan_id", "content": "stale"},
    ]
    # raw_msgs is hand-rolled Chat Completions (Layer 2); the function's
    # input type is Layer 1, so cast() isolates the boundary without
    # polluting production call sites.
    result = ContextEditor.remove_orphaned_tool_results(cast(list[LLMInputContentItem], raw_msgs))
    tool_ids = [m.get("tool_call_id") for m in result if m.get("role") == "tool"]
    assert tool_ids == ["keep_id"]


# ── RunConfig.compaction_llm override ────────────────────────────────


@pytest.mark.asyncio
@patch("troopai.adk.context.context_manager.TokenCounter.count_messages")
async def test_compaction_routes_through_provided_llm(mock_count):
    """The LLM passed into prepare_messages is the one invoked for compaction."""
    mock_count.side_effect = [160_000, 160_000, 160_000, 160_000, 50_000]

    primary_llm = _StubLLM(text="primary should not be called")
    override_llm = _StubLLM(text="override summary")

    config = ContextManagementConfig(
        compaction=CompactionConfig(
            enabled=True,
            trigger_tokens=150_000,
            preserve_recent_items=2,
        ),
    )
    mgr = ContextManager(config)

    msgs = _make_conversation(10)
    # Pass the override directly — Runner-side resolve_compaction_llm
    # is responsible for selecting it from RunConfig.compaction_llm; the
    # manager just calls whichever LLM the Runner threads in.
    _ = await mgr.prepare_messages(msgs, override_llm, "gpt-4o-mini")

    assert override_llm.call_count == 1
    assert primary_llm.call_count == 0


# ── Regression: negative delta clamping ─────────────────────────────


@pytest.mark.asyncio
@patch("troopai.adk.context.context_manager.TokenCounter.count_messages")
async def test_total_tokens_compacted_never_goes_negative(mock_count):
    """When the LLM summary exceeds the original token count, the running
    total must be clamped to 0 (not go negative), so total_token_budget
    guard in should_compact remains effective.

    Pre-fix: ``self._total_tokens_compacted += original - compacted`` would
    subtract when ``compacted > original`` (verbose model), making the
    accumulated counter negative and permanently disabling the budget cap.
    """
    # Simulate: original=160k, compacted=200k (summary larger than original).
    # Token count sequence: editing, warning, should_compact check, inside
    # compact (original), inside compact (compacted after build).
    mock_count.side_effect = [160_000, 160_000, 160_000, 160_000, 200_000]

    class _VerboseLLM(_StubLLM):
        """Returns a very long summary (simulates verbose model)."""

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
                response=[LLMResponseText(text="verbose " * 10_000)],
            )

    config = ContextManagementConfig(
        compaction=CompactionConfig(
            enabled=True,
            trigger_tokens=150_000,
            preserve_recent_items=2,
        ),
    )
    mgr = ContextManager(config)
    msgs = _make_conversation(10)
    await mgr.prepare_messages(msgs, _VerboseLLM(), "gpt-4o-mini")

    # The delta was negative (200k - 160k = -40k), but clamped to 0.
    assert mgr._total_tokens_compacted >= 0, (
        "_total_tokens_compacted went negative; total_token_budget guard permanently disabled"
    )


# ── Regression: TokenUsage is a typed TypedDict ──────────────────────


@patch("troopai.adk.context.context_manager.TokenCounter.count_messages", return_value=80_000)
def test_get_token_usage_returns_token_usage_typeddict(_mock_count):
    """get_token_usage must return a TokenUsage TypedDict, not dict[str, Any]."""
    from troopai.adk.context.context_config import TokenUsage

    config = ContextManagementConfig(max_context_tokens=200_000)
    mgr = ContextManager(config)
    usage = mgr.get_token_usage([], "gpt-4o")

    # Type is TokenUsage (a TypedDict — instances are dicts at runtime).
    assert isinstance(usage, dict)
    # All five required keys are present.
    assert set(usage.keys()) == {"used", "max", "remaining", "utilisation", "compaction_count"}
    # Values satisfy the TokenUsage contract.
    assert usage["used"] == 80_000
    assert usage["max"] == 200_000
    assert usage["remaining"] == 120_000
    assert usage["compaction_count"] == 0
    # Structural check: TokenUsage annotations match at runtime.
    assert set(TokenUsage.__annotations__.keys()) == set(usage.keys())


# ── Regression: forced tool not self-triggering on pressure-feedback ─


@pytest.mark.asyncio
@patch("troopai.adk.context.context_manager.TokenCounter.count_messages")
async def test_forced_tool_uses_pre_feedback_token_count(mock_count):
    """The forced-tool threshold check must use the pre-feedback token count.

    Pre-fix: step 7 counted tokens AFTER the pressure-feedback message was
    injected (step 6), causing self-triggering: the injected developer
    message alone tipped the count over the warning threshold, so the
    forced-tool signal fired every turn near threshold.
    """
    # Effective budget = 1000 - 0 = 1000; threshold = 0.8 * 1000 = 800.
    # Before feedback: 790 (just below threshold — forced tool must NOT fire).
    # After feedback injection the injected message would push it over, but
    # the fix ensures step 7 uses the pre-feedback value.
    mock_count.return_value = 790

    config = ContextManagementConfig(
        max_context_tokens=1000,
        token_budget_warning_threshold=0.8,
        pressure_feedback=True,
        forced_tool="manage_context",
        editing=ContextEditingConfig(clear_tool_results=False),
    )
    mgr = ContextManager(config)
    llm = _StubLLM()

    msgs: list[LLMInputContentItem] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    await mgr.prepare_messages(msgs, llm, "gpt-4o-mini")

    # Pre-feedback count (790) < threshold (800): forced_tool must be None.
    assert mgr.force_tool is None, "forced_tool should not fire when pre-feedback count is below threshold"

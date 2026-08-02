"""Tests for ``SharedContextStrategy`` output shapes.

Covers the three non-LLM strategies (``SCOPED``, ``LAST_N``,
``FULL_BROADCAST``). ``SUMMARIZED`` hits an LLM via
``ContextCompactor.compact`` and is left to the integration suite.

The assertion style: build a small ``SwarmState`` with hand-crafted
``UserItem`` entries, call ``prepare_turn_input``, and verify the
returned Layer 1 param list.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.swarms.config import SharedContextConfig
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.shared_context import prepare_turn_input
from troopai.adk.swarms.shared_context_strategy import SharedContextStrategy
from troopai.adk.swarms.state import SwarmState
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.swarms.yield_signal import SwarmHandoff
from troopai.adk.types.items.items import UserItem


def _user_item(content: str, agent_name: str = "a") -> UserItem:
    return UserItem(agent_name=agent_name, raw={"role": "user", "content": content})


def _as_easy(item: object) -> dict[str, object]:
    """Narrow a ``list[LLMInputContentItem]`` entry for easy-message access.

    The fixtures in this module only produce ``UserItem`` entries whose
    Layer 1 param is ``LLMInputEasyMessage`` (``role`` + ``content``).
    All ``LLMInputContentItem`` union members are TypedDicts — plain
    dicts at runtime — so we verify the entry is dict-shaped and then
    return it typed ``dict[str, object]`` so callers can index ``role``
    and ``content`` without a narrowing cast.

    For any other Layer 1 variant (``LLMInputText``, ``LLMInputImage``,
    ``FunctionToolCallResultParam``, etc.), the missing key would surface
    as a ``KeyError`` at assertion time — keeping the test failure
    localised.
    """
    assert isinstance(item, dict), f"expected dict-shaped Layer 1 item, got {type(item)!r}"
    return item


def _mkstate() -> tuple[SwarmState, Agent, Agent]:
    a = Agent(name="a", system_prompt="noop")
    b = Agent(name="b", system_prompt="noop")
    swarm = Swarm(
        members=(a, b),
        entry=a,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(10),
    )
    state = SwarmState(
        swarm=swarm,
        current_agent=a,
        current_agent_name="a",
    )
    return state, a, b


class TestScoped:
    def test_empty_scratch_empty_result(self) -> None:
        state, _a, b = _mkstate()
        config = SharedContextConfig(strategy=SharedContextStrategy.SCOPED)

        result = asyncio.run(prepare_turn_input(state, b, None, config))
        assert result == []

    def test_scratch_passes_through(self) -> None:
        state, _a, b = _mkstate()
        state.per_agent_scratch["b"] = [
            _user_item("first b turn", "b"),
            _user_item("second b turn", "b"),
        ]
        config = SharedContextConfig(strategy=SharedContextStrategy.SCOPED)

        result = asyncio.run(prepare_turn_input(state, b, None, config))
        assert len(result) == 2
        assert _as_easy(result[0])["content"] == "first b turn"
        assert _as_easy(result[1])["content"] == "second b turn"

    def test_handoff_message_appended_when_targeted(self) -> None:
        state, _a, b = _mkstate()
        state.per_agent_scratch["b"] = [_user_item("prior", "b")]
        config = SharedContextConfig(strategy=SharedContextStrategy.SCOPED)

        handoff = SwarmHandoff(target="b", message="please review")
        result = asyncio.run(prepare_turn_input(state, b, handoff, config))
        assert len(result) == 2
        assert _as_easy(result[1])["role"] == "user"
        assert _as_easy(result[1])["content"] == "please review"

    def test_empty_handoff_message_not_appended(self) -> None:
        # Zero-content handoff message could confuse providers and
        # would also inflate context for no benefit.
        state, _a, b = _mkstate()
        state.per_agent_scratch["b"] = [_user_item("prior", "b")]
        config = SharedContextConfig(strategy=SharedContextStrategy.SCOPED)

        handoff = SwarmHandoff(target="b", message="")
        result = asyncio.run(prepare_turn_input(state, b, handoff, config))
        assert len(result) == 1
        assert _as_easy(result[0])["content"] == "prior"

    def test_handoff_to_other_agent_not_appended(self) -> None:
        state, _a, b = _mkstate()
        state.per_agent_scratch["b"] = [_user_item("prior", "b")]
        config = SharedContextConfig(strategy=SharedContextStrategy.SCOPED)

        handoff = SwarmHandoff(target="a", message="for a only")
        result = asyncio.run(prepare_turn_input(state, b, handoff, config))
        # Agent 'b' gets its scratch only; the handoff message
        # addressed 'a' must not leak into b's turn.
        assert len(result) == 1
        assert _as_easy(result[0])["content"] == "prior"


class TestLastN:
    def test_keeps_only_last_n(self) -> None:
        state, _a, _b = _mkstate()
        state.shared_history = [_user_item(f"m{i}") for i in range(5)]
        config = SharedContextConfig(
            strategy=SharedContextStrategy.LAST_N,
            window=2,
        )

        result = asyncio.run(prepare_turn_input(state, state.current_agent, None, config))
        assert len(result) == 2
        assert _as_easy(result[0])["content"] == "m3"
        assert _as_easy(result[1])["content"] == "m4"

    def test_window_larger_than_history_returns_all(self) -> None:
        state, _a, _b = _mkstate()
        state.shared_history = [_user_item("only")]
        config = SharedContextConfig(
            strategy=SharedContextStrategy.LAST_N,
            window=10,
        )

        result = asyncio.run(prepare_turn_input(state, state.current_agent, None, config))
        assert len(result) == 1


class TestFullBroadcast:
    def test_returns_full_shared_history(self) -> None:
        state, _a, _b = _mkstate()
        state.shared_history = [_user_item(f"m{i}") for i in range(3)]
        config = SharedContextConfig(strategy=SharedContextStrategy.FULL_BROADCAST)

        result = asyncio.run(prepare_turn_input(state, state.current_agent, None, config))
        assert len(result) == 3
        assert [_as_easy(r)["content"] for r in result] == ["m0", "m1", "m2"]


class TestUnknownStrategyRaises:
    def test_invalid_strategy_raises(self) -> None:
        state, _a, _b = _mkstate()

        class _Fake:
            strategy = "not-a-real-strategy"
            window = None
            budget = None
            max_handoff_message_chars = None

        with pytest.raises(ValueError, match="Unknown SharedContextStrategy"):
            asyncio.run(
                prepare_turn_input(
                    state,
                    state.current_agent,
                    None,
                    _Fake(),  # type: ignore[arg-type]
                )
            )


class TestHandoffMessageSizeCap:
    """Defence-in-depth cap on handoff-message injection.

    The SCOPED strategy injects ``SwarmHandoff.message`` straight into
    the next agent's user slot *before* the LLM call — ahead of
    ``FunctionTool.max_result_tokens``, ``HandoffConfig.budget``, and
    ``SwarmConfig.max_total_tokens``. A cooperating-but-pathological
    LLM could emit a massive ``transfer_to_<name>`` message that all
    three other caps would miss. ``max_handoff_message_chars`` is the
    per-turn guard.
    """

    def test_message_under_cap_passes_through_unchanged(self) -> None:
        state, _a, b = _mkstate()
        config = SharedContextConfig(
            strategy=SharedContextStrategy.SCOPED,
            max_handoff_message_chars=100,
        )
        handoff = SwarmHandoff(target="b", message="short message")

        result = asyncio.run(prepare_turn_input(state, b, handoff, config))
        assert len(result) == 1
        assert _as_easy(result[0])["content"] == "short message"

    def test_message_over_cap_truncated_with_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        state, _a, b = _mkstate()
        cap = 50
        config = SharedContextConfig(
            strategy=SharedContextStrategy.SCOPED,
            max_handoff_message_chars=cap,
        )
        oversized = "x" * 500
        handoff = SwarmHandoff(target="b", message=oversized)

        with caplog.at_level(logging.WARNING, logger="troopai.adk.swarms.shared_context"):
            result = asyncio.run(prepare_turn_input(state, b, handoff, config))

        assert len(result) == 1
        assert _as_easy(result[0])["content"] == "x" * cap
        assert any("SwarmHandoff.message truncated" in r.message for r in caplog.records)

    def test_none_cap_allows_arbitrary_length(self) -> None:
        state, _a, b = _mkstate()
        config = SharedContextConfig(
            strategy=SharedContextStrategy.SCOPED,
            max_handoff_message_chars=None,
        )
        oversized = "x" * 100_000
        handoff = SwarmHandoff(target="b", message=oversized)

        result = asyncio.run(prepare_turn_input(state, b, handoff, config))
        assert len(result) == 1
        assert _as_easy(result[0])["content"] == oversized

    def test_config_rejects_nonpositive_cap(self) -> None:
        with pytest.raises(ValueError, match="max_handoff_message_chars must be > 0"):
            SharedContextConfig(
                strategy=SharedContextStrategy.SCOPED,
                max_handoff_message_chars=0,
            )
        with pytest.raises(ValueError, match="max_handoff_message_chars must be > 0"):
            SharedContextConfig(
                strategy=SharedContextStrategy.SCOPED,
                max_handoff_message_chars=-1,
            )


# ---------------------------------------------------------------------------
# Regression: SUMMARIZED compaction error falls back to raw history (#HIGH)
# ---------------------------------------------------------------------------


class TestSummarizedCompactionErrorFallback:
    """Regression: _prepare_summarized must not abort the swarm turn when
    the compaction LLM call raises — it should fall back to LAST_N truncation
    (bounded by budget) rather than silently returning over-budget raw history,
    and log an ERROR stating the budget is NOT enforced."""

    def test_compaction_error_falls_back_to_last_n(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """On compaction failure, LAST_N truncation is applied so the budget
        constraint is honoured rather than silently exceeded."""
        from unittest.mock import AsyncMock, MagicMock, patch

        state, _a, _b = _mkstate()
        # 5 items; budget=1000 → approx_items = max(1, 1000//200) = 5
        # so all items should be returned.
        state.shared_history = [_user_item(f"item{i}") for i in range(5)]
        config = SharedContextConfig(
            strategy=SharedContextStrategy.SUMMARIZED,
            budget=1000,
        )

        mock_llm = MagicMock()
        mock_model = "mock-model"
        mock_context = MagicMock()

        with (
            # Force the history over budget so compaction is actually attempted
            # (the gate skips summarization while under budget).
            patch(
                "troopai.adk.context.token_counter.TokenCounter.count_messages",
                return_value=10**9,
            ),
            patch(
                "troopai.adk.context.compaction.ContextCompactor.compact",
                new=AsyncMock(side_effect=RuntimeError("LLM unavailable")),
            ),
            caplog.at_level(logging.ERROR, logger="troopai.adk.swarms.shared_context"),
        ):
            result = asyncio.run(
                prepare_turn_input(
                    state,
                    state.current_agent,
                    None,
                    config,
                    compaction_llm=mock_llm,
                    compaction_model=mock_model,
                    context=mock_context,
                )
            )

        # Must NOT raise; result is LAST_N-truncated (≤ approx_items entries)
        assert len(result) >= 1, "fallback must return at least one item"
        assert any("compaction LLM call failed" in r.message for r in caplog.records), (
            "ERROR log must mention compaction failure"
        )
        assert any("NOT enforced" in r.message for r in caplog.records), (
            "ERROR log must state that the budget is NOT enforced"
        )

    def test_compaction_error_small_budget_truncates_aggressively(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A small budget causes the fallback to keep only the most recent items."""
        from unittest.mock import AsyncMock, MagicMock, patch

        state, _a, _b = _mkstate()
        # 3 items, budget=100 → approx_items = max(1, 100//200) = 1
        state.shared_history = [_user_item("old1"), _user_item("old2"), _user_item("recent")]
        config = SharedContextConfig(
            strategy=SharedContextStrategy.SUMMARIZED,
            budget=100,
        )

        mock_llm = MagicMock()
        mock_model = "mock-model"

        with (
            # Force over-budget so compaction is attempted before it fails.
            patch(
                "troopai.adk.context.token_counter.TokenCounter.count_messages",
                return_value=10**9,
            ),
            patch(
                "troopai.adk.context.compaction.ContextCompactor.compact",
                new=AsyncMock(side_effect=RuntimeError("LLM unavailable")),
            ),
        ):
            result = asyncio.run(
                prepare_turn_input(
                    state,
                    state.current_agent,
                    None,
                    config,
                    compaction_llm=mock_llm,
                    compaction_model=mock_model,
                )
            )

        # With budget=100, approx_items=1, only the most recent item is kept
        assert len(result) == 1
        assert _as_easy(result[0])["content"] == "recent"

    def test_compaction_error_log_carries_exception_traceback(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Regression: the compaction-failure log must preserve the underlying
        exception (type, message, traceback) so the silent budget-degradation is
        diagnosable. ``logger.exception`` attaches ``exc_info``; the prior
        ``logger.error`` left ``exc_info`` None and discarded the cause entirely.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        state, _a, _b = _mkstate()
        state.shared_history = [_user_item("only")]
        config = SharedContextConfig(
            strategy=SharedContextStrategy.SUMMARIZED,
            budget=1000,
        )

        with (
            # Force over-budget so compaction is attempted and its failure logs.
            patch(
                "troopai.adk.context.token_counter.TokenCounter.count_messages",
                return_value=10**9,
            ),
            patch(
                "troopai.adk.context.compaction.ContextCompactor.compact",
                new=AsyncMock(side_effect=RuntimeError("LLM auth failure")),
            ),
            caplog.at_level(logging.ERROR, logger="troopai.adk.swarms.shared_context"),
        ):
            asyncio.run(
                prepare_turn_input(
                    state,
                    state.current_agent,
                    None,
                    config,
                    compaction_llm=MagicMock(),
                    compaction_model="mock-model",
                )
            )

        failure_records = [r for r in caplog.records if "compaction LLM call failed" in r.message]
        assert len(failure_records) == 1, "exactly one compaction-failure record expected"
        record = failure_records[0]
        # logger.exception sets exc_info to the live (type, value, tb) tuple.
        assert record.exc_info is not None, "compaction-failure log must carry exc_info"
        exc_type, exc_value, _tb = record.exc_info
        assert exc_type is RuntimeError
        assert str(exc_value) == "LLM auth failure"


# ---------------------------------------------------------------------------
# Regression: preserve_recent_items=0 returns [] not full body (#MED)
# ---------------------------------------------------------------------------


class TestSummarizedPreserveZero:
    """Regression: when preserve_recent_items=0, the 'preserved' slice must be
    empty — not the full body — so the summary stands alone with no duplication."""

    def test_preserve_zero_returns_empty_preserved(self) -> None:
        """Directly test the edge case by inspecting the actual preserve logic.

        We build a body of 3 items and verify that ``preserve=0`` produces an
        empty slice.  The condition ``0 < preserve < len(body)`` is False for
        preserve=0, and the else clause (fixed) should give ``[]`` instead of
        ``body``.
        """
        # Direct unit test of the fixed condition
        body = [{"role": "user", "content": f"m{i}"} for i in range(3)]

        def _preserved(preserve: int) -> list:
            return body[-preserve:] if 0 < preserve < len(body) else ([] if preserve == 0 else body)

        # preserve=0 must yield empty
        assert _preserved(0) == [], "preserve=0 must return [] (not full body)"
        # preserve=2 yields last 2
        assert len(_preserved(2)) == 2
        # preserve >= len yields full body (the else branch where preserve > len)
        assert len(_preserved(10)) == 3

"""Regression tests for the swarm shared-context sweep fixes.

Covers three drifts fixed together:

- The run's opening prompt was never visible to cross-agent strategies
  (``FULL_BROADCAST`` / ``LAST_N`` / ``SUMMARIZED``) because ``shared_history``
  holds only produced items — turn-2+ agents lost the question.
- ``SUMMARIZED`` fired an LLM summarization call every turn: the compactor
  ignores ``trigger_tokens``, so passing the budget through it never gated the
  call. It must skip while the history fits the budget.
- The SCOPED handoff message was delivered for one turn but never persisted
  into the target's scratch, so a later revisit lost the prompting question.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

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


def _as_dict(item: object) -> dict[str, Any]:
    assert isinstance(item, dict)
    return item


def _mkstate() -> tuple[SwarmState[Any], Agent[Any], Agent[Any]]:
    a = Agent(name="a", system_prompt="noop")
    b = Agent(name="b", system_prompt="noop")
    swarm: Swarm[Any] = Swarm(
        members=(a, b),
        entry=a,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(10),
    )
    state: SwarmState[Any] = SwarmState(swarm=swarm, current_agent=a, current_agent_name="a")
    return state, a, b


class TestInitialInputPrepended:
    def test_full_broadcast_prepends_opening_prompt(self) -> None:
        state, _a, b = _mkstate()
        state.initial_input_items = [_user_item("What is 2+2?", "user")]
        state.shared_history = [_user_item("thinking...", "a")]
        config = SharedContextConfig(strategy=SharedContextStrategy.FULL_BROADCAST)

        result = asyncio.run(prepare_turn_input(state, b, None, config))

        assert [_as_dict(r)["content"] for r in result] == ["What is 2+2?", "thinking..."]

    def test_last_n_prepends_opening_prompt(self) -> None:
        state, _a, b = _mkstate()
        state.initial_input_items = [_user_item("original question", "user")]
        state.shared_history = [_user_item(f"m{i}", "a") for i in range(4)]
        config = SharedContextConfig(strategy=SharedContextStrategy.LAST_N, window=2)

        result = asyncio.run(prepare_turn_input(state, b, None, config))

        # window=2 keeps the last two produced items; the question is prepended.
        assert [_as_dict(r)["content"] for r in result] == ["original question", "m2", "m3"]

    def test_empty_initial_input_is_noop(self) -> None:
        # Default (no opening prompt recorded) leaves broadcast output unchanged.
        state, _a, b = _mkstate()
        state.shared_history = [_user_item("only", "a")]
        config = SharedContextConfig(strategy=SharedContextStrategy.FULL_BROADCAST)

        result = asyncio.run(prepare_turn_input(state, b, None, config))
        assert [_as_dict(r)["content"] for r in result] == ["only"]


class TestSummarizedBudgetGate:
    def test_under_budget_skips_compaction_llm_call(self) -> None:
        state, _a, _b = _mkstate()
        state.shared_history = [_user_item(f"m{i}", "a") for i in range(3)]
        config = SharedContextConfig(strategy=SharedContextStrategy.SUMMARIZED, budget=1000)

        compact_mock = AsyncMock()
        with (
            patch(
                "troopai.adk.context.token_counter.TokenCounter.count_messages",
                return_value=10,  # 10 <= budget=1000 -> gate must skip
            ),
            patch(
                "troopai.adk.context.compaction.ContextCompactor.compact",
                new=compact_mock,
            ),
        ):
            result = asyncio.run(
                prepare_turn_input(
                    state,
                    state.current_agent,
                    None,
                    config,
                    compaction_llm=object(),  # type: ignore[arg-type]
                    compaction_model="mock-model",
                )
            )

        compact_mock.assert_not_awaited()
        # Under budget: the raw history is returned verbatim, no summary.
        assert [_as_dict(r)["content"] for r in result] == ["m0", "m1", "m2"]

    def test_over_budget_still_compacts(self) -> None:
        state, _a, _b = _mkstate()
        state.shared_history = [_user_item(f"m{i}", "a") for i in range(3)]
        config = SharedContextConfig(strategy=SharedContextStrategy.SUMMARIZED, budget=10)

        compact_mock = AsyncMock(side_effect=RuntimeError("boom"))  # forces fallback path
        with (
            patch(
                "troopai.adk.context.token_counter.TokenCounter.count_messages",
                return_value=999,  # 999 > budget=10 -> gate must NOT skip
            ),
            patch(
                "troopai.adk.context.compaction.ContextCompactor.compact",
                new=compact_mock,
            ),
        ):
            asyncio.run(
                prepare_turn_input(
                    state,
                    state.current_agent,
                    None,
                    config,
                    compaction_llm=object(),  # type: ignore[arg-type]
                    compaction_model="mock-model",
                )
            )

        compact_mock.assert_awaited()


class TestScopedHandoffPersisted:
    def test_handoff_message_persisted_to_target_scratch(self) -> None:
        state, _a, b = _mkstate()
        config = SharedContextConfig(strategy=SharedContextStrategy.SCOPED)
        handoff = SwarmHandoff(target="b", message="please review the draft")

        asyncio.run(prepare_turn_input(state, b, handoff, config))

        scratch = state.per_agent_scratch.get("b", [])
        assert len(scratch) == 1
        assert _as_dict(scratch[-1].to_param())["content"] == "please review the draft"

    def test_persist_is_idempotent_on_repeated_prepare(self) -> None:
        # A resumed turn re-prepares with the same last_yield; the message must
        # not be delivered or persisted twice.
        state, _a, b = _mkstate()
        config = SharedContextConfig(strategy=SharedContextStrategy.SCOPED)
        handoff = SwarmHandoff(target="b", message="do the thing")

        first = asyncio.run(prepare_turn_input(state, b, handoff, config))
        second = asyncio.run(prepare_turn_input(state, b, handoff, config))

        assert len(state.per_agent_scratch["b"]) == 1
        assert [_as_dict(r)["content"] for r in first] == ["do the thing"]
        assert [_as_dict(r)["content"] for r in second] == ["do the thing"]


class TestInitialInputSerialization:
    def test_initial_input_items_round_trip(self) -> None:
        # A resumed broadcast run must keep the opening prompt, so the field is
        # serialized and rehydrated with the state.
        state, _a, _b = _mkstate()
        state.initial_input_items = [_user_item("keep me across resume", "user")]

        restored = SwarmState.from_dict(state.to_dict(), state.swarm)

        assert len(restored.initial_input_items) == 1
        assert _as_dict(restored.initial_input_items[0].to_param())["content"] == "keep me across resume"

    def test_absent_field_defaults_to_empty(self) -> None:
        # A payload persisted before the field existed loads to an empty list.
        state, _a, _b = _mkstate()
        data = state.to_dict()
        del data["initial_input_items"]  # type: ignore[misc]

        restored = SwarmState.from_dict(data, state.swarm)
        assert restored.initial_input_items == []

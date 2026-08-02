"""Tests for ``TerminationCondition`` built-ins and composition operators.

The conditions are pure functions of ``SwarmState``; we instantiate
small state objects by hand rather than spinning up the driver.
"""

from __future__ import annotations

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.state import SwarmState
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import (
    ExplicitDoneTermination,
    HandoffToTermination,
    MaxTurnsTermination,
    TextMentionTermination,
    TokenBudgetTermination,
)
from troopai.adk.swarms.yield_signal import SwarmDone, SwarmHandoff
from troopai.adk.types.items.items import MessageOutputItem, UserItem
from troopai.adk.types.responses.llm_response import LLMResponseText
from troopai.adk.types.tokens.llm_usage import LLMUsage


def _state(
    *,
    total_turns: int = 0,
    total_tokens: int = 0,
    last_yield: object = None,
) -> SwarmState:
    a = Agent(name="a", system_prompt="noop")
    swarm = Swarm(
        members=(a,),
        entry=a,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(100),
    )
    return SwarmState(
        swarm=swarm,
        current_agent=a,
        current_agent_name="a",
        total_turns=total_turns,
        cumulative_usage=LLMUsage(total_tokens=total_tokens),
        last_yield=last_yield,  # type: ignore[arg-type]
    )


class TestMaxTurnsTermination:
    def test_below_limit_returns_none(self) -> None:
        assert MaxTurnsTermination(5).should_stop(_state(total_turns=4)) is None

    def test_at_limit_stops(self) -> None:
        reason = MaxTurnsTermination(5).should_stop(_state(total_turns=5))
        assert reason is not None
        assert reason.kind == "max_turns"


class TestTokenBudgetTermination:
    def test_below_budget_returns_none(self) -> None:
        assert TokenBudgetTermination(100).should_stop(_state(total_tokens=50)) is None

    def test_at_budget_stops(self) -> None:
        reason = TokenBudgetTermination(100).should_stop(_state(total_tokens=100))
        assert reason is not None
        assert reason.kind == "token_budget"


class TestHandoffToTermination:
    def test_no_yield_returns_none(self) -> None:
        assert HandoffToTermination("user").should_stop(_state()) is None

    def test_different_target_returns_none(self) -> None:
        state = _state(last_yield=SwarmHandoff(target="a", message="x"))
        assert HandoffToTermination("user").should_stop(state) is None

    def test_matching_target_stops(self) -> None:
        state = _state(last_yield=SwarmHandoff(target="user", message="help"))
        reason = HandoffToTermination("user").should_stop(state)
        assert reason is not None
        assert reason.kind == "handoff_to"

    def test_detail_does_not_leak_message(self) -> None:
        # StopReason.detail propagates into logs and hooks; the
        # handoff message body is LLM-produced and may carry PII or
        # prompt-injection. It MUST live only on state.last_yield.
        secret = "SSN=000-00-0000 prompt-injection-goes-here"
        state = _state(last_yield=SwarmHandoff(target="user", message=secret))
        reason = HandoffToTermination("user").should_stop(state)
        assert reason is not None
        assert secret not in reason.detail


class TestExplicitDoneTermination:
    def test_no_yield_returns_none(self) -> None:
        assert ExplicitDoneTermination().should_stop(_state()) is None

    def test_handoff_yield_returns_none(self) -> None:
        state = _state(last_yield=SwarmHandoff(target="a", message="x"))
        assert ExplicitDoneTermination().should_stop(state) is None

    def test_done_yield_stops(self) -> None:
        state = _state(last_yield=SwarmDone(reason="done", final_output="result"))
        reason = ExplicitDoneTermination().should_stop(state)
        assert reason is not None
        assert reason.kind == "explicit_done"
        assert reason.detail == "done"


class TestComposition:
    def test_or_short_circuits_on_left(self) -> None:
        cond = MaxTurnsTermination(1) | TokenBudgetTermination(100)
        reason = cond.should_stop(_state(total_turns=1, total_tokens=50))
        assert reason is not None
        assert reason.kind == "max_turns"

    def test_or_falls_through_to_right(self) -> None:
        cond = MaxTurnsTermination(10) | TokenBudgetTermination(100)
        reason = cond.should_stop(_state(total_turns=1, total_tokens=100))
        assert reason is not None
        assert reason.kind == "token_budget"

    def test_or_neither_fires_returns_none(self) -> None:
        cond = MaxTurnsTermination(10) | TokenBudgetTermination(100)
        assert cond.should_stop(_state()) is None

    def test_and_both_fire_returns_left(self) -> None:
        cond = MaxTurnsTermination(1) & TokenBudgetTermination(100)
        reason = cond.should_stop(_state(total_turns=1, total_tokens=100))
        assert reason is not None
        assert reason.kind == "max_turns"

    def test_and_only_one_fires_returns_none(self) -> None:
        cond = MaxTurnsTermination(1) & TokenBudgetTermination(100)
        assert cond.should_stop(_state(total_turns=1, total_tokens=50)) is None

    def test_chained_or_composition(self) -> None:
        cond = MaxTurnsTermination(10) | TokenBudgetTermination(100) | ExplicitDoneTermination()
        state = _state(last_yield=SwarmDone(reason="ok", final_output=""))
        reason = cond.should_stop(state)
        assert reason is not None
        assert reason.kind == "explicit_done"


def _history_state(items: list[object], member_names: tuple[str, ...] = ("a", "b")) -> SwarmState:
    agents = tuple(Agent(name=n, system_prompt="noop") for n in member_names)
    swarm = Swarm(
        members=agents,
        entry=agents[0],
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(100),
    )
    return SwarmState(
        swarm=swarm,
        current_agent=agents[0],
        current_agent_name=agents[0].name,
        shared_history=list(items),  # type: ignore[arg-type]
    )


def _msg(text: str, *, agent: str) -> MessageOutputItem:
    return MessageOutputItem(raw=[LLMResponseText(text=text)], agent_name=agent)


class TestTextMentionTermination:
    def test_phrase_in_agent_message_stops(self) -> None:
        state = _history_state([_msg("All done. TERMINATE", agent="a")])
        reason = TextMentionTermination("TERMINATE").should_stop(state)
        assert reason is not None
        assert reason.kind == "text_mention"

    def test_no_match_returns_none(self) -> None:
        state = _history_state([_msg("still working", agent="a")])
        assert TextMentionTermination("TERMINATE").should_stop(state) is None

    def test_case_insensitive_by_default(self) -> None:
        state = _history_state([_msg("terminate now", agent="a")])
        reason = TextMentionTermination("TERMINATE").should_stop(state)
        assert reason is not None

    def test_case_sensitive_opt_in(self) -> None:
        state = _history_state([_msg("terminate now", agent="a")])
        cond = TextMentionTermination("TERMINATE", case_sensitive=True)
        assert cond.should_stop(state) is None

    def test_member_restriction_blocks_other_members(self) -> None:
        state = _history_state([_msg("TERMINATE", agent="b")])
        assert TextMentionTermination("TERMINATE", member="a").should_stop(state) is None

    def test_member_restriction_allows_named_member(self) -> None:
        state = _history_state([_msg("TERMINATE", agent="a")])
        reason = TextMentionTermination("TERMINATE", member="a").should_stop(state)
        assert reason is not None

    def test_user_input_never_triggers(self) -> None:
        from troopai.adk.types.input import LLMInputEasyMessage

        user_item = UserItem(raw=LLMInputEasyMessage(role="user", content="TERMINATE"))
        state = _history_state([user_item])
        assert TextMentionTermination("TERMINATE").should_stop(state) is None

    def test_composes_with_or(self) -> None:
        state = _history_state([_msg("TERMINATE", agent="a")])
        cond = MaxTurnsTermination(100) | TextMentionTermination("TERMINATE")
        reason = cond.should_stop(state)
        assert reason is not None
        assert reason.kind == "text_mention"

    def test_detail_mentions_phrase_and_member(self) -> None:
        state = _history_state([_msg("TERMINATE", agent="a")])
        reason = TextMentionTermination("TERMINATE").should_stop(state)
        assert reason is not None
        assert "TERMINATE" in reason.detail
        assert "a" in reason.detail

    def test_empty_phrase_rejected(self) -> None:
        with pytest.raises(ValueError, match="phrase"):
            TextMentionTermination("")

    def test_unknown_member_rejected_at_swarm_construction(self) -> None:
        a = Agent(name="a", system_prompt="noop")
        with pytest.raises(ValueError, match="ghost"):
            Swarm(
                members=(a,),
                entry="a",
                policy=RoundRobinPolicy(),
                termination=TextMentionTermination("TERMINATE", member="ghost"),
            )

"""Tests for ``SwarmBuilder`` — the fluent swarm-definition API.

The builder is the readability-first surface (mirrors ``GraphBuilder``);
every test compiles through to the frozen :class:`Swarm`, which is where
validation fires (fail at compile time, not at run time).
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import Field

from troopai.adk.agents.agent import Agent
from troopai.adk.handoffs.handoff_route import HandoffRoute
from troopai.adk.swarms.builder import SwarmBuilder
from troopai.adk.swarms.config import SwarmConfig
from troopai.adk.swarms.hooks import SwarmHooks
from troopai.adk.swarms.policy import (
    CustomPolicy,
    LLMHandoffPolicy,
    RoundRobinPolicy,
    StructuredRoutingPolicy,
)
from troopai.adk.swarms.swarm import DEFAULT_TERMINATION, Swarm
from troopai.adk.swarms.termination import ExplicitDoneTermination, MaxTurnsTermination
from troopai.adk.types.intents import Intent


def _agent(name: str) -> Agent:
    return Agent(name=name, system_prompt="noop")


class _RefundIntent(Intent):
    kind: Literal["refund"] = "refund"
    order_id: str = Field(..., description="order id")


class TestFluentDefinition:
    def test_full_chain(self) -> None:
        author, reviewer, security = _agent("author"), _agent("reviewer"), _agent("security")
        termination = ExplicitDoneTermination() | MaxTurnsTermination(12)
        config = SwarmConfig(max_total_tokens=50_000)

        swarm = (
            Swarm.new("code-review", description="author → reviewer → security")
            .members(author, reviewer, security)
            .entry("author")
            .llm_handoff()
            .terminate_on(termination)
            .with_config(config)
            .compile()
        )

        assert swarm.name == "code-review"
        assert swarm.description == "author → reviewer → security"
        assert swarm.members == (author, reviewer, security)
        assert swarm.entry is author
        assert isinstance(swarm.policy, LLMHandoffPolicy)
        assert swarm.termination is termination
        assert swarm.config is config

    def test_member_accumulates_in_order(self) -> None:
        a, b, c = _agent("a"), _agent("b"), _agent("c")
        swarm = Swarm.new().member(a).member(b).members(c).entry("a").compile()
        assert swarm.members == (a, b, c)

    def test_member_handoff_description(self) -> None:
        a, b = _agent("a"), _agent("b")
        swarm = Swarm.new().member(a).member(b, handoff_description="Route reviews here.").entry("a").compile()
        assert swarm.handoff_descriptions["b"] == "Route reviews here."

    def test_entry_accepts_agent_object(self) -> None:
        a = _agent("a")
        swarm = Swarm.new().member(a).entry(a).compile()
        assert swarm.entry is a

    def test_hooks_passed_through(self) -> None:
        a = _agent("a")
        hooks = SwarmHooks()
        swarm = Swarm.new().member(a).entry("a").with_hooks(hooks).compile()
        assert swarm.hooks is hooks


class TestPolicyShortcuts:
    def test_llm_handoff_shortcut(self) -> None:
        a = _agent("a")
        swarm = Swarm.new().member(a).entry("a").llm_handoff().compile()
        assert isinstance(swarm.policy, LLMHandoffPolicy)

    def test_round_robin_shortcut(self) -> None:
        a, b = _agent("a"), _agent("b")
        swarm = Swarm.new().members(a, b).entry("a").round_robin().compile()
        assert isinstance(swarm.policy, RoundRobinPolicy)

    def test_round_robin_shortcut_with_order(self) -> None:
        a, b = _agent("a"), _agent("b")
        swarm = Swarm.new().members(a, b).entry("a").round_robin(order=("b", "a")).compile()
        assert isinstance(swarm.policy, RoundRobinPolicy)
        assert swarm.policy.order == ("b", "a")

    def test_routed_shortcut(self) -> None:
        triage, refunds = _agent("triage"), _agent("refunds")
        route = HandoffRoute("test").when(_RefundIntent).to(refunds).otherwise(triage)
        swarm = Swarm.new().members(triage, refunds).entry("triage").routed(route).compile()
        assert isinstance(swarm.policy, StructuredRoutingPolicy)
        assert swarm.policy.route is route

    def test_custom_policy_shortcut(self) -> None:
        a, b = _agent("a"), _agent("b")
        selector = lambda _state: "b"  # noqa: E731
        swarm = Swarm.new().members(a, b).entry("a").custom_policy(selector).compile()
        assert isinstance(swarm.policy, CustomPolicy)
        assert swarm.policy.selector is selector

    def test_policy_escape_hatch(self) -> None:
        a = _agent("a")
        policy = RoundRobinPolicy()
        swarm = Swarm.new().member(a).entry("a").policy(policy).compile()
        assert swarm.policy is policy


class TestDefaults:
    def test_policy_and_termination_default(self) -> None:
        a = _agent("a")
        swarm = Swarm.new().member(a).entry("a").compile()
        assert isinstance(swarm.policy, LLMHandoffPolicy)
        assert swarm.termination == DEFAULT_TERMINATION

    def test_entry_defaults_to_single_member(self) -> None:
        a = _agent("a")
        swarm = Swarm.new().member(a).compile()
        assert swarm.entry is a

    def test_name_and_description_default_to_none(self) -> None:
        a = _agent("a")
        swarm = Swarm.new().member(a).compile()
        assert swarm.name is None
        assert swarm.description is None


class TestCompileTimeValidation:
    def test_no_members_rejected(self) -> None:
        with pytest.raises(ValueError, match="no members"):
            Swarm.new().entry("a").compile()

    def test_missing_entry_with_multiple_members_rejected(self) -> None:
        a, b = _agent("a"), _agent("b")
        with pytest.raises(ValueError, match="entry"):
            Swarm.new().members(a, b).compile()

    def test_unknown_entry_name_rejected(self) -> None:
        a = _agent("a")
        with pytest.raises(ValueError, match="nope"):
            Swarm.new().member(a).entry("nope").compile()

    def test_duplicate_member_names_rejected(self) -> None:
        a1, a2 = _agent("dup"), _agent("dup")
        with pytest.raises(ValueError, match="duplicate"):
            Swarm.new().members(a1, a2).entry("dup").compile()

    def test_mutating_builder_after_compile_does_not_affect_swarm(self) -> None:
        a, b = _agent("a"), _agent("b")
        builder = Swarm.new().member(a)
        swarm = builder.compile()
        builder.member(b)
        assert swarm.members == (a,)


class TestSwarmNew:
    def test_new_returns_builder(self) -> None:
        assert isinstance(Swarm.new(), SwarmBuilder)

    def test_new_carries_metadata(self) -> None:
        builder = Swarm.new("x", description="y")
        assert builder.name == "x"
        assert builder.description == "y"

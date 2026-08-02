"""Tests for the four ``SwarmPolicy`` subclasses.

Scope:
- ``LLMHandoffPolicy``: injects ``transfer_to_<other>`` + ``swarm_done``,
  never ``transfer_to_<self>``.
- ``RoundRobinPolicy``: deterministic rotation, only ``swarm_done``
  injected.
- ``StructuredRoutingPolicy``: dispatches from
  ``state.last_structured_output`` via the wrapped ``HandoffRoute``.
- ``CustomPolicy``: selector-controlled dispatch + optional extra tools.

Agent mutation check: every test confirms that invoking the policy
does not modify ``agent.tools`` on the members (load-bearing rule).
"""

from __future__ import annotations

import asyncio
from typing import Literal

import pytest
from pydantic import Field

from troopai.adk.agents.agent import Agent
from troopai.adk.handoffs.handoff_route import HandoffRoute
from troopai.adk.run.context import RunContext
from troopai.adk.swarms.policy import (
    CustomPolicy,
    LLMHandoffPolicy,
    RoundRobinPolicy,
    StructuredRoutingPolicy,
)
from troopai.adk.swarms.state import SwarmState
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.swarms.yield_signal import SWARM_DONE_TOOL_NAME
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.intents import Intent, Respond


def _agent(name: str) -> Agent:
    return Agent(name=name, system_prompt="noop")


def _mkstate(
    members: tuple[Agent, ...],
    policy: object,
    *,
    current: Agent,
) -> SwarmState:
    swarm = Swarm(
        members=members,
        entry=members[0],
        policy=policy,  # type: ignore[arg-type]
        termination=MaxTurnsTermination(100),
    )
    return SwarmState(
        swarm=swarm,
        current_agent=current,
        current_agent_name=current.name,
    )


class TestLLMHandoffPolicy:
    def test_injects_transfer_tools_for_other_members_only(self) -> None:
        a, b, c = _agent("a"), _agent("b"), _agent("c")
        policy = LLMHandoffPolicy()
        state = _mkstate((a, b, c), policy, current=a)

        tools = policy.build_step_tools(state)
        tool_names = {t.name for t in tools}

        assert "transfer_to_b" in tool_names
        assert "transfer_to_c" in tool_names
        assert "transfer_to_a" not in tool_names
        assert SWARM_DONE_TOOL_NAME in tool_names
        assert len(tools) == 3

    def test_agent_tools_not_mutated(self) -> None:
        a, b = _agent("a"), _agent("b")
        original_tools_a = tuple(a.tools)
        original_tools_b = tuple(b.tools)

        policy = LLMHandoffPolicy()
        state = _mkstate((a, b), policy, current=a)
        _ = policy.build_step_tools(state)

        assert tuple(a.tools) == original_tools_a
        assert tuple(b.tools) == original_tools_b


class TestRoundRobinPolicy:
    def test_only_swarm_done_injected(self) -> None:
        a, b = _agent("a"), _agent("b")
        policy = RoundRobinPolicy()
        state = _mkstate((a, b), policy, current=a)

        tools = policy.build_step_tools(state)
        assert len(tools) == 1
        assert tools[0].name == SWARM_DONE_TOOL_NAME

    def test_rotation_cycles_through_members(self) -> None:
        a, b, c = _agent("a"), _agent("b"), _agent("c")
        policy = RoundRobinPolicy()
        state = _mkstate((a, b, c), policy, current=a)
        ctx = RunContext(context=None)

        # Rotation: index = total_turns % len. We just finished turn 1, so
        # next is slot 1 → "b". After turn 2 (total_turns=2), next is "c".
        state.total_turns = 1
        assert asyncio.run(policy.select_next(state, ctx)).name == "b"
        state.total_turns = 2
        assert asyncio.run(policy.select_next(state, ctx)).name == "c"
        state.total_turns = 3
        assert asyncio.run(policy.select_next(state, ctx)).name == "a"

    def test_explicit_order_overrides_member_order(self) -> None:
        a, b, c = _agent("a"), _agent("b"), _agent("c")
        # order = ("c", "a"); entry = a (members[0])
        # start_idx = rotation.index("a") = 1
        # total_turns=1: (1+1)%2=0 → "c"
        # total_turns=2: (2+1)%2=1 → "a"
        # Result: c → a → c → a (correct rotation after "a" ran as entry)
        policy = RoundRobinPolicy(order=("c", "a"))
        state = _mkstate((a, b, c), policy, current=a)
        ctx = RunContext(context=None)

        state.total_turns = 1
        assert asyncio.run(policy.select_next(state, ctx)).name == "c"
        state.total_turns = 2
        assert asyncio.run(policy.select_next(state, ctx)).name == "a"

    def test_bad_order_name_raises(self) -> None:
        a = _agent("a")
        policy = RoundRobinPolicy(order=("nonexistent",))
        state = _mkstate((a,), policy, current=a)
        ctx = RunContext(context=None)
        state.total_turns = 1
        with pytest.raises(ValueError, match="nonexistent"):
            asyncio.run(policy.select_next(state, ctx))

    def test_entry_not_members0_does_not_spin(self) -> None:
        """Regression: when entry = members[1], the formula must NOT repeat
        the entry on the first select_next call (the bug: 1 % 2 = 1 → members[1]
        again). With the offset fix, the entry's slot is the start, so the next
        agent is always a different peer on the first call."""
        a, b = _agent("a"), _agent("b")
        # entry = b = members[1]
        swarm = Swarm(
            members=(a, b),
            entry=b,
            policy=RoundRobinPolicy(),
            termination=MaxTurnsTermination(100),
        )
        state = SwarmState(swarm=swarm, current_agent=b, current_agent_name="b")
        ctx = RunContext(context=None)
        policy = RoundRobinPolicy()

        # After b's first turn (total_turns=1), select_next must give "a", not "b" again.
        state.total_turns = 1
        selected = asyncio.run(policy.select_next(state, ctx))
        assert selected.name == "a", (
            f"RoundRobinPolicy must not repeat the entry agent on first handoff; got {selected.name!r}"
        )

        # After a's turn (total_turns=2), next is "b" (back to start of rotation after offset).
        state.total_turns = 2
        selected = asyncio.run(policy.select_next(state, ctx))
        assert selected.name == "b"


# ---------------------------------------------------------------------
# StructuredRoutingPolicy
# ---------------------------------------------------------------------


class _RefundIntent(Intent):
    kind: Literal["refund"] = "refund"
    order_id: str = Field(..., description="order id")


class _OrderIntent(Intent):
    kind: Literal["order"] = "order"
    item: str = Field(..., description="item")


class TestStructuredRoutingPolicy:
    def test_only_swarm_done_injected(self) -> None:
        triage = _agent("triage")
        refunds = _agent("refunds")
        route = HandoffRoute("test").when(_RefundIntent).to(refunds).otherwise(triage)
        policy = StructuredRoutingPolicy(route=route)
        state = _mkstate((triage, refunds), policy, current=triage)
        tools = policy.build_step_tools(state)
        assert len(tools) == 1
        assert tools[0].name == SWARM_DONE_TOOL_NAME

    def test_dispatches_on_structured_output(self) -> None:
        triage = _agent("triage")
        refunds = _agent("refunds")
        orders = _agent("orders")
        route = HandoffRoute("test").when(_RefundIntent).to(refunds).when(_OrderIntent).to(orders).otherwise(triage)
        policy = StructuredRoutingPolicy(route=route)
        state = _mkstate((triage, refunds, orders), policy, current=triage)
        ctx = RunContext(context=None)

        state.last_structured_output = _RefundIntent(order_id="123")
        assert asyncio.run(policy.select_next(state, ctx)).name == "refunds"

        state.last_structured_output = _OrderIntent(item="widget")
        assert asyncio.run(policy.select_next(state, ctx)).name == "orders"

    def test_respond_intent_keeps_current_agent(self) -> None:
        triage = _agent("triage")
        refunds = _agent("refunds")
        route = HandoffRoute("test").when(_RefundIntent).to(refunds).otherwise(triage)
        policy = StructuredRoutingPolicy(route=route)
        state = _mkstate((triage, refunds), policy, current=triage)
        ctx = RunContext(context=None)

        state.last_structured_output = Respond(message="hi")
        selected = asyncio.run(policy.select_next(state, ctx))
        assert selected is triage

    def test_missing_structured_output_keeps_current_agent(self) -> None:
        """Specialists (no output_schema) leave last_structured_output=None.

        The policy MUST keep the current agent and let termination
        decide when to stop, instead of crashing the swarm. Raising
        here would make the structured-routing pattern unusable because
        specialists would fail loudly every turn after routing.
        """
        triage = _agent("triage")
        refunds = _agent("refunds")
        route = HandoffRoute("test").when(_RefundIntent).to(refunds).otherwise(triage)
        policy = StructuredRoutingPolicy(route=route)
        state = _mkstate((triage, refunds), policy, current=refunds)
        ctx = RunContext(context=None)

        state.last_structured_output = None
        selected = asyncio.run(policy.select_next(state, ctx))
        assert selected is refunds


# ---------------------------------------------------------------------
# CustomPolicy
# ---------------------------------------------------------------------


class TestCustomPolicy:
    def test_selector_drives_dispatch(self) -> None:
        a, b = _agent("a"), _agent("b")
        policy = CustomPolicy(selector=lambda _state: "b")
        state = _mkstate((a, b), policy, current=a)
        ctx = RunContext(context=None)
        assert asyncio.run(policy.select_next(state, ctx)).name == "b"

    def test_invalid_selector_raises(self) -> None:
        a = _agent("a")
        policy = CustomPolicy(selector=lambda _state: "does-not-exist")
        state = _mkstate((a,), policy, current=a)
        ctx = RunContext(context=None)
        with pytest.raises(ValueError, match="does-not-exist"):
            asyncio.run(policy.select_next(state, ctx))

    def test_extra_tools_appended_before_swarm_done(self) -> None:
        a = _agent("a")

        async def _invoker(_ctx, _raw):
            return "noop"

        extra = FunctionTool(
            name="extra",
            description="custom tool",
            schema={"type": "object", "properties": {}},
            on_invoke=_invoker,
        )
        policy = CustomPolicy(
            selector=lambda _state: "a",
            extra_tools_fn=lambda _state: [extra],
        )
        state = _mkstate((a,), policy, current=a)
        tools = policy.build_step_tools(state)
        assert [t.name for t in tools] == ["extra", SWARM_DONE_TOOL_NAME]

    def test_no_extra_tools_only_swarm_done(self) -> None:
        a = _agent("a")
        policy = CustomPolicy(selector=lambda _state: "a")
        state = _mkstate((a,), policy, current=a)
        tools = policy.build_step_tools(state)
        assert len(tools) == 1
        assert tools[0].name == SWARM_DONE_TOOL_NAME


class TestLLMHandoffDescriptions:
    def test_transfer_tool_uses_swarm_handoff_description(self) -> None:
        a, b = _agent("a"), _agent("b")
        swarm = Swarm(
            members=(a, b),
            entry="a",
            policy=LLMHandoffPolicy(),
            termination=MaxTurnsTermination(100),
            handoff_descriptions={"b": "Route code-review requests here."},
        )
        state = SwarmState(swarm=swarm, current_agent=a, current_agent_name="a")

        tools = LLMHandoffPolicy().build_step_tools(state)
        transfer = next(t for t in tools if t.name == "transfer_to_b")

        assert transfer.description is not None
        assert "Route code-review requests here." in transfer.description

    def test_transfer_tool_falls_back_to_default_description(self) -> None:
        a, b = _agent("a"), _agent("b")
        state = _mkstate((a, b), LLMHandoffPolicy(), current=a)

        tools = LLMHandoffPolicy().build_step_tools(state)
        transfer = next(t for t in tools if t.name == "transfer_to_b")

        assert transfer.description is not None
        assert "Hand the swarm conversation to the 'b' agent" in transfer.description

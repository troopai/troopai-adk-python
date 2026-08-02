"""Construction-time validation tests for ``SwarmPolicy`` subclasses.

Focused on ``RoundRobinPolicy`` rejecting a degenerate empty rotation
``order`` at construction, rather than letting it surface as an opaque
``ZeroDivisionError`` during a swarm run.
"""

from __future__ import annotations

import asyncio

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.context import RunContext
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.state import SwarmState
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination


def _agent(name: str) -> Agent:
    return Agent(name=name, system_prompt="noop")


class TestRoundRobinPolicyEmptyOrder:
    def test_empty_order_raises_at_construction(self) -> None:
        """``RoundRobinPolicy(order=())`` must fail fast with a clear
        ``ValueError`` instead of a runtime ``ZeroDivisionError``.

        Before the fix, an empty tuple passed construction; on the first
        ``select_next`` call the modulo ``(total_turns + 0) % len(())``
        raised ``ZeroDivisionError``, which the driver swallowed into an
        opaque ``policy_error`` stop.
        """
        with pytest.raises(ValueError, match="non-empty"):
            RoundRobinPolicy(order=())

    def test_none_order_still_allowed(self) -> None:
        # The default fallback path must remain unaffected.
        policy = RoundRobinPolicy(order=None)
        assert policy.order is None

    def test_non_empty_order_still_allowed(self) -> None:
        policy = RoundRobinPolicy(order=("a", "b"))
        assert policy.order == ("a", "b")

    def test_non_empty_order_rotates_without_error(self) -> None:
        """End-to-end: a valid non-empty order rotates without raising,
        guarding against an over-broad construction check."""
        a, b = _agent("a"), _agent("b")
        policy = RoundRobinPolicy(order=("a", "b"))
        swarm = Swarm(
            members=(a, b),
            entry=a,
            policy=policy,
            termination=MaxTurnsTermination(100),
        )
        state = SwarmState(swarm=swarm, current_agent=a, current_agent_name="a")
        ctx: RunContext = RunContext(context=None)

        state.total_turns = 1
        assert asyncio.run(policy.select_next(state, ctx)).name == "b"
        state.total_turns = 2
        assert asyncio.run(policy.select_next(state, ctx)).name == "a"

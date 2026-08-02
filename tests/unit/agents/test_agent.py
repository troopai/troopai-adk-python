"""Regression tests for agents/agent.py confirmed-bug fixes.

Covers:
- get_agent_graph() surfaces HandoffRoute (code-orchestrated) targets
  instead of silently dropping them.
- as_tool() rejects agent names that convert to an empty tool name
  (non-ASCII / punctuation-only) with a clear UserError.
"""

from __future__ import annotations

import pytest

from troopai.adk.agents import Agent
from troopai.adk.exceptions import UserError
from troopai.adk.handoffs import HandoffRoute
from troopai.adk.types.intents import Intent


def _agent(name: str = "TestAgent") -> Agent:
    return Agent(name=name, system_prompt="You are a test agent.")


# ── Fix 1: get_agent_graph surfaces HandoffRoute targets ─────────────


class _RefundIntent(Intent):
    kind: str = "refund"


class _BillingIntent(Intent):
    kind: str = "billing"


class TestGetAgentGraphHandoffRoute:
    """get_agent_graph() must surface code-orchestrated handoff targets."""

    def test_route_targets_are_listed(self) -> None:
        """HandoffRoute rule + otherwise targets appear in the graph."""
        refunds = _agent("Refunds")
        billing = _agent("Billing")
        general = _agent("General")

        triage = Agent(
            name="Triage",
            system_prompt="Route requests.",
            handoffs=(
                HandoffRoute("triage")
                .when(_RefundIntent)
                .to(refunds)
                .when(_BillingIntent)
                .to(billing)
                .otherwise(general)
            ),
        )

        graph = triage.get_agent_graph()
        assert graph["name"] == "Triage"
        # Before the fix this was [] — all three targets were dropped.
        assert set(graph["handoffs"]) == {"Refunds", "Billing", "General"}

    def test_route_without_otherwise(self) -> None:
        """A route with only rule targets still surfaces them."""
        refunds = _agent("Refunds")

        triage = Agent(
            name="Triage",
            system_prompt="Route.",
            handoffs=HandoffRoute("triage").when(_RefundIntent).to(refunds),
        )

        graph = triage.get_agent_graph()
        assert graph["handoffs"] == ["Refunds"]


class TestHandoffRouteAllTargets:
    """HandoffRoute.all_targets() enumerates every target."""

    def test_includes_rules_and_otherwise(self) -> None:
        refunds = _agent("Refunds")
        general = _agent("General")
        route: HandoffRoute = HandoffRoute("triage").when(_RefundIntent).to(refunds).otherwise(general)

        targets = route.all_targets()
        assert [t.target.name for t in targets] == ["Refunds", "General"]

    def test_empty_route_returns_empty(self) -> None:
        route: HandoffRoute = HandoffRoute("triage")
        assert route.all_targets() == []


# ── Fix 2: as_tool empty-name guard ──────────────────────────────────


class TestAsToolEmptyName:
    """as_tool() must reject names that snake_case to an empty string."""

    @pytest.mark.parametrize("agent_name", ["日本語", "!!!", "---"])
    def test_empty_resolved_name_raises(self, agent_name: str) -> None:
        agent = _agent(agent_name)
        with pytest.raises(UserError, match="empty tool name"):
            agent.as_tool()

    def test_explicit_tool_name_bypasses_guard(self) -> None:
        """An explicit tool_name= sidesteps the conversion entirely."""
        agent = _agent("日本語")
        tool = agent.as_tool(tool_name="translate")
        assert tool.name == "translate"

    def test_ascii_name_still_works(self) -> None:
        agent = _agent("Research Agent")
        tool = agent.as_tool()
        assert tool.name == "research_agent"

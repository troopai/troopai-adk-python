"""Tests for the Runner handoff helpers in ``handoff_helpers.py``.

Covers ``normalize_handoffs`` setup-time validation: bare Agents are
wrapped into default ``Handoff`` objects, and duplicate tool names are
rejected before any tool list is built or any LLM call is made.

Duplicate names are easy to hit accidentally: ``Handoff.get_name()``
lowercases and snake-cases the target name, so targets differing only in
case or spacing — or the same agent listed twice — collapse to one
``transfer_to_<name>`` tool name. Two same-named tools either make the
provider reject the request or make ``find_handoff_target`` route to
whichever handoff happened to be last in the list.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from troopai.adk.exceptions import HandoffDefinitionError
from troopai.adk.handoffs.handoff import Handoff
from troopai.adk.handoffs.handoff_helpers import (
    find_handoff_target,
    normalize_handoffs,
)
from troopai.adk.run.context import RunContext


def _mock_agent(name: str) -> MagicMock:
    """Build a mock Agent with the attributes Handoff touches."""
    agent = MagicMock()
    agent.name = name
    agent.description = None
    return agent


class TestNormalizeWrapping:
    """Bare Agents are wrapped; explicit Handoffs pass through."""

    async def test_bare_agent_is_wrapped(self) -> None:
        agent = _mock_agent("refunds")
        normalized = await normalize_handoffs([agent])
        assert len(normalized) == 1
        assert isinstance(normalized[0], Handoff)
        assert normalized[0].target is agent

    async def test_existing_handoff_passes_through(self) -> None:
        h = Handoff(target=_mock_agent("billing"))
        normalized = await normalize_handoffs([h])
        assert normalized == [h]

    async def test_distinct_names_are_allowed(self) -> None:
        normalized = await normalize_handoffs([_mock_agent("refunds"), _mock_agent("billing")])
        assert len(normalized) == 2


class TestDuplicateNameRejection:
    """Colliding tool names raise at normalization time."""

    async def test_same_agent_listed_twice_raises(self) -> None:
        agent = _mock_agent("refunds")
        with pytest.raises(HandoffDefinitionError) as excinfo:
            await normalize_handoffs([agent, agent])
        assert excinfo.value.handoff_name == "transfer_to_refunds"

    async def test_case_only_difference_raises(self) -> None:
        # get_name() lowercases, so "Refunds" and "refunds" collide.
        with pytest.raises(HandoffDefinitionError) as excinfo:
            await normalize_handoffs([_mock_agent("Refunds"), _mock_agent("refunds")])
        msg = str(excinfo.value)
        assert "transfer_to_refunds" in msg
        assert "Refunds" in msg

    async def test_spacing_only_difference_raises(self) -> None:
        # get_name() replaces spaces with underscores, so "Refund Agent"
        # and "Refund_Agent" both resolve to transfer_to_refund_agent.
        with pytest.raises(HandoffDefinitionError):
            await normalize_handoffs([_mock_agent("Refund Agent"), _mock_agent("Refund_Agent")])

    async def test_explicit_name_collision_raises(self) -> None:
        a = Handoff(target=_mock_agent("alpha"), name="transfer_to_x")
        b = Handoff(target=_mock_agent("beta"), name="transfer_to_x")
        with pytest.raises(HandoffDefinitionError):
            await normalize_handoffs([a, b])

    async def test_unique_explicit_name_breaks_collision(self) -> None:
        # The documented escape hatch: a custom name disambiguates two
        # targets that would otherwise collide on the auto-generated name.
        a = _mock_agent("refunds")
        b = Handoff(target=_mock_agent("refunds"), name="transfer_to_refunds_eu")
        normalized = await normalize_handoffs([a, b])
        assert len(normalized) == 2


class TestFindTargetAfterNormalization:
    """The dedup guard makes routing deterministic for distinct names."""

    async def test_find_routes_to_correct_target(self) -> None:
        refunds = _mock_agent("refunds")
        billing = _mock_agent("billing")
        normalized = await normalize_handoffs([refunds, billing])
        ctx: RunContext[dict[str, object]] = RunContext(context={})
        target = await find_handoff_target(normalized, "transfer_to_billing", ctx)
        assert target is not None
        assert target.target is billing

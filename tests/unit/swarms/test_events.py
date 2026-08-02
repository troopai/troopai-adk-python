"""Tests for the ``SwarmEvent`` dataclass hierarchy.

Each event is a ``@dataclass(frozen=True)`` with a ``type`` Literal
discriminator. We verify:

- Required fields produce a constructable event.
- Mutation raises (frozen contract).
- The ``type`` literal matches the expected discriminator string.
- The ``SwarmEvent`` union admits every built-in variant.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from troopai.adk.graphs.interrupt import Interrupt, NestedAgentInterrupt
from troopai.adk.swarms.events import (
    SwarmDoneEvent,
    SwarmEvent,
    SwarmHandoffEvent,
    SwarmStartEvent,
    SwarmTurnEndEvent,
    SwarmTurnInterruptEvent,
    SwarmTurnStartEvent,
)
from troopai.adk.swarms.stop_reason import StopReason


class TestSwarmStartEvent:
    def test_construction_and_discriminator(self) -> None:
        ev = SwarmStartEvent(entry_agent="a", member_names=("a", "b"))
        assert ev.type == "swarm_start"
        assert ev.entry_agent == "a"
        assert ev.member_names == ("a", "b")

    def test_frozen(self) -> None:
        ev = SwarmStartEvent(entry_agent="a", member_names=("a",))
        with pytest.raises(FrozenInstanceError):
            ev.entry_agent = "b"  # type: ignore[misc]


class TestSwarmTurnStartEvent:
    def test_fields(self) -> None:
        ev = SwarmTurnStartEvent(agent="writer", turn=3)
        assert ev.type == "swarm_turn_start"
        assert ev.agent == "writer"
        assert ev.turn == 3


class TestSwarmHandoffEvent:
    def test_fields(self) -> None:
        ev = SwarmHandoffEvent(
            from_agent="a",
            to_agent="b",
            message="over to you",
        )
        assert ev.type == "swarm_handoff"
        assert ev.from_agent == "a"
        assert ev.to_agent == "b"
        assert ev.message == "over to you"


class TestSwarmTurnEndEvent:
    def test_items_is_tuple(self) -> None:
        ev = SwarmTurnEndEvent(agent="a", items=())
        assert ev.type == "swarm_turn_end"
        assert ev.items == ()


class TestSwarmDoneEvent:
    def test_fields(self) -> None:
        reason = StopReason(kind="explicit_done", detail="finished")
        ev = SwarmDoneEvent(reason=reason, final_output="hello")
        assert ev.type == "swarm_done"
        assert ev.reason is reason
        assert ev.final_output == "hello"

    def test_none_final_output_allowed(self) -> None:
        reason = StopReason(kind="max_turns", detail="limit hit")
        ev = SwarmDoneEvent(reason=reason, final_output=None)
        assert ev.final_output is None


class TestSwarmTurnInterruptEvent:
    def test_construction_with_plain_interrupt(self) -> None:
        interrupt = Interrupt(
            node_id="approver",
            question="Approve?",
            kind="tool_approval",
        )
        event = SwarmTurnInterruptEvent(
            agent="approver",
            turn=3,
            interrupt=interrupt,
        )
        assert event.agent == "approver"
        assert event.turn == 3
        assert event.interrupt is interrupt
        assert event.type == "swarm_turn_interrupt"

    def test_construction_with_nested_agent_interrupt(self) -> None:
        interrupt = NestedAgentInterrupt(
            node_id="approver",
            agent_name="approver",
            tool_call_ids=("c1",),
            question="Approve?",
        )
        event = SwarmTurnInterruptEvent(
            agent="approver",
            turn=2,
            interrupt=interrupt,
        )
        assert isinstance(event.interrupt, NestedAgentInterrupt)
        assert event.interrupt.tool_call_ids == ("c1",)

    def test_event_is_frozen(self) -> None:
        interrupt = Interrupt(node_id="m", question="q", kind="generic")
        event = SwarmTurnInterruptEvent(agent="m", turn=1, interrupt=interrupt)
        with pytest.raises(FrozenInstanceError):
            event.agent = "other"  # type: ignore[misc]


class TestUnionMembership:
    @pytest.mark.parametrize(
        "ev",
        [
            SwarmStartEvent(entry_agent="a", member_names=("a",)),
            SwarmTurnStartEvent(agent="a", turn=1),
            SwarmHandoffEvent(from_agent="a", to_agent="b", message=""),
            SwarmTurnEndEvent(agent="a", items=()),
            SwarmTurnInterruptEvent(
                agent="a",
                turn=1,
                interrupt=Interrupt(node_id="a", question="q"),
            ),
            SwarmDoneEvent(
                reason=StopReason(kind="explicit_done", detail=""),
                final_output=None,
            ),
        ],
    )
    def test_events_satisfy_union(self, ev: SwarmEvent) -> None:
        # If this import + type hint accepts the event, the union is
        # well-formed. Runtime assertion: the event has the canonical
        # `type` discriminator.
        assert hasattr(ev, "type")
        assert isinstance(ev.type, str)

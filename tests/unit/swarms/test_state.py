"""Tests for ``SwarmState`` serialization + round-trip.

Mirrors the pattern used in ``test_runstate_serialization.py``: round
trip via ``to_dict``/``from_dict`` and via ``to_json``/``from_json``
(a plain ``json.dumps`` of ``to_dict()`` — no envelope, no version key).
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from pydantic import BaseModel

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs.interrupt import Interrupt, NestedAgentInterrupt
from troopai.adk.run.state import RunState
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.state import SwarmState, SwarmStateDict
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.swarms.yield_signal import SwarmDone, SwarmHandoff
from troopai.adk.types.tokens.llm_usage import LLMUsage


def _mkswarm() -> Swarm:
    a = Agent(name="a", system_prompt="noop")
    b = Agent(name="b", system_prompt="noop")
    return Swarm(
        members=(a, b),
        entry=a,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(10),
    )


def _mkstate(swarm: Swarm) -> SwarmState:
    return SwarmState(
        swarm=swarm,
        current_agent=swarm.entry,
        current_agent_name=swarm.entry.name,
        handoff_count=3,
        total_turns=5,
        cumulative_usage=LLMUsage(
            requests=7,
            total_tokens=1_234,
            input_tokens=1_000,
            output_tokens=234,
        ),
    )


class TestRoundTrip:
    def test_to_dict_from_dict_empty_yield(self) -> None:
        swarm = _mkswarm()
        state = _mkstate(swarm)
        raw = state.to_dict()
        restored = SwarmState.from_dict(raw, swarm)

        assert restored.current_agent_name == "a"
        assert restored.current_agent is swarm.entry
        assert restored.handoff_count == 3
        assert restored.total_turns == 5
        assert restored.cumulative_usage.total_tokens == 1_234
        assert restored.last_yield is None

    def test_round_trip_preserves_handoff_yield(self) -> None:
        swarm = _mkswarm()
        state = _mkstate(swarm)
        state.last_yield = SwarmHandoff(target="b", message="please continue")

        restored = SwarmState.from_dict(state.to_dict(), swarm)
        assert isinstance(restored.last_yield, SwarmHandoff)
        assert restored.last_yield.target == "b"
        assert restored.last_yield.message == "please continue"

    def test_round_trip_preserves_done_yield(self) -> None:
        swarm = _mkswarm()
        state = _mkstate(swarm)
        state.last_yield = SwarmDone(reason="ok", final_output={"answer": 42})

        restored = SwarmState.from_dict(state.to_dict(), swarm)
        assert isinstance(restored.last_yield, SwarmDone)
        assert restored.last_yield.reason == "ok"
        assert restored.last_yield.final_output == {"answer": 42}

    def test_from_dict_rejects_unknown_agent_name(self) -> None:
        swarm = _mkswarm()
        # cast: bare dict literal cannot be narrowed to SwarmStateDict by pyright
        # because empty list [] is list[Unknown]; runtime path validates the key.
        bad = cast(
            SwarmStateDict,
            {
                "current_agent_name": "unknown",
                "shared_history": [],
                "per_agent_scratch": {},
                "handoff_count": 0,
                "total_turns": 0,
                "cumulative_usage": {
                    "requests": 0,
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
                "last_yield": None,
                "pending_interrupts": {},
                "nested_agent_snapshots": {},
                "status": "running",
                "error": None,
            },
        )
        with pytest.raises(ValueError, match="unknown"):
            SwarmState.from_dict(bad, swarm)

    def test_from_dict_rejects_phantom_scratch_key(self) -> None:
        swarm = _mkswarm()
        # cast: bare dict literal with mixed-type per_agent_scratch cannot be
        # narrowed to SwarmStateDict by pyright; runtime path validates the key.
        tampered = cast(
            SwarmStateDict,
            {
                "current_agent_name": "a",
                "shared_history": [],
                "per_agent_scratch": {
                    "a": [],
                    "ghost": [],
                },
                "handoff_count": 0,
                "total_turns": 0,
                "cumulative_usage": {
                    "requests": 0,
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
                "last_yield": None,
                "pending_interrupts": {},
                "nested_agent_snapshots": {},
                "status": "running",
                "error": None,
            },
        )
        with pytest.raises(ValueError, match="phantom"):
            SwarmState.from_dict(tampered, swarm)


class TestJsonSchema:
    def test_to_json_is_bare_dict_no_version_key(self) -> None:
        swarm = _mkswarm()
        state = _mkstate(swarm)
        payload = json.loads(state.to_json())

        assert "_schema_version" not in payload
        assert "data" not in payload  # no envelope wrapper
        assert payload == state.to_dict()
        assert payload["current_agent_name"] == "a"

    def test_from_json_round_trip(self) -> None:
        swarm = _mkswarm()
        state = _mkstate(swarm)
        state.last_yield = SwarmDone(reason="ok", final_output="done")

        restored = SwarmState.from_json(state.to_json(), swarm)
        assert restored.current_agent_name == "a"
        assert isinstance(restored.last_yield, SwarmDone)
        assert restored.last_yield.reason == "ok"
        assert restored.last_yield.final_output == "done"

    def test_from_json_rejects_invalid_json(self) -> None:
        swarm = _mkswarm()
        with pytest.raises(json.JSONDecodeError):
            SwarmState.from_json("not json {[}", swarm)


class TestAdvanceTo:
    def test_advance_to_updates_name_and_seeds_scratch(self) -> None:
        swarm = _mkswarm()
        state = _mkstate(swarm)
        target = swarm.members[1]

        state.advance_to(target)
        assert state.current_agent is target
        assert state.current_agent_name == target.name
        assert target.name in state.per_agent_scratch
        assert len(state.per_agent_scratch[target.name]) == 0


def _make_swarm() -> Swarm:
    """Single-member swarm for state round-trip tests."""
    member = Agent(name="m1", system_prompt="x")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(3),
    )


class TestSwarmStateInterruptFields:
    def test_pending_interrupts_round_trips(self) -> None:
        sw = _make_swarm()
        state = SwarmState(
            swarm=sw,
            current_agent=sw.members[0],
            current_agent_name="m1",
        )
        state.pending_interrupts["m1"] = Interrupt(
            node_id="m1",
            question="approve?",
            kind="tool_approval",
        )
        restored = SwarmState.from_dict(state.to_dict(), sw)
        assert "m1" in restored.pending_interrupts
        assert restored.pending_interrupts["m1"].question == "approve?"

    def test_status_default_is_running(self) -> None:
        sw = _make_swarm()
        state = SwarmState(swarm=sw, current_agent=sw.members[0], current_agent_name="m1")
        assert state.status == "running"
        assert state.error is None

    def test_status_round_trips(self) -> None:
        sw = _make_swarm()
        state = SwarmState(swarm=sw, current_agent=sw.members[0], current_agent_name="m1")
        state.status = "interrupted"
        state.error = None
        restored = SwarmState.from_dict(state.to_dict(), sw)
        assert restored.status == "interrupted"

    def test_from_dict_rejects_nested_interrupt_without_snapshot(self) -> None:
        """A NestedAgentInterrupt in pending_interrupts must have a matching snapshot."""
        sw = _make_swarm()
        bad_payload = cast(
            SwarmStateDict,
            {
                "current_agent_name": "m1",
                "shared_history": [],
                "per_agent_scratch": {},
                "handoff_count": 0,
                "total_turns": 0,
                "cumulative_usage": {
                    "requests": 0,
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
                "last_yield": None,
                "pending_interrupts": {
                    "m1": {
                        "node_id": "m1",
                        "question": "approve tool",
                        "kind": "nested_agent_tool_approval",
                        "metadata": {},
                        "agent_name": "m1",
                        "tool_call_ids": ["c1"],
                    },
                },
                "nested_agent_snapshots": {},
                "status": "interrupted",
                "error": None,
            },
        )
        with pytest.raises(ValueError, match="nested_agent_snapshots"):
            SwarmState.from_dict(bad_payload, sw)

    def test_nested_agent_snapshots_round_trips(self) -> None:
        sw = _make_swarm()
        state = SwarmState(swarm=sw, current_agent=sw.members[0], current_agent_name="m1")
        rs = RunState(current_agent_name="m1")
        state.nested_agent_snapshots["m1"] = rs
        # Pair with a NestedAgentInterrupt so from_dict's cross-reference
        # check passes (every NestedAgentInterrupt must have a matching
        # nested_agent_snapshots entry).
        state.pending_interrupts["m1"] = NestedAgentInterrupt(
            node_id="m1",
            question="approve tool",
            agent_name="m1",
            tool_call_ids=("c1",),
        )
        restored = SwarmState.from_dict(state.to_dict(), sw)
        assert "m1" in restored.nested_agent_snapshots
        assert restored.nested_agent_snapshots["m1"].current_agent_name == "m1"


class TestSwarmIdAndResumeCounts:
    def test_default_values(self) -> None:
        sw = _make_swarm()
        state: SwarmState[Any] = SwarmState(
            swarm=sw,
            current_agent=sw.entry,
            current_agent_name=sw.entry.name,
        )
        assert state.swarm_id is None
        assert state.resume_counts == {}

    def test_to_dict_carries_swarm_id_and_resume_counts(self) -> None:
        sw = _make_swarm()
        state: SwarmState[Any] = SwarmState(
            swarm=sw,
            current_agent=sw.entry,
            current_agent_name=sw.entry.name,
        )
        state.swarm_id = "abc-123"
        state.resume_counts = {sw.entry.name: 2}

        payload = state.to_dict()
        assert payload.get("swarm_id") == "abc-123"
        assert payload.get("resume_counts") == {sw.entry.name: 2}

    def test_from_dict_rehydrates_new_fields(self) -> None:
        sw = _make_swarm()
        state: SwarmState[Any] = SwarmState(
            swarm=sw,
            current_agent=sw.entry,
            current_agent_name=sw.entry.name,
        )
        state.swarm_id = "abc-123"
        state.resume_counts = {sw.entry.name: 2}

        rehydrated = SwarmState.from_dict(state.to_dict(), sw)
        assert rehydrated.swarm_id == "abc-123"
        assert rehydrated.resume_counts == {sw.entry.name: 2}

    def test_from_dict_defaults_when_fields_absent(self) -> None:
        """A payload missing the new fields loads cleanly."""
        sw = _make_swarm()
        state: SwarmState[Any] = SwarmState(
            swarm=sw,
            current_agent=sw.entry,
            current_agent_name=sw.entry.name,
        )
        payload = dict(state.to_dict())
        payload.pop("swarm_id", None)
        payload.pop("resume_counts", None)

        rehydrated = SwarmState.from_dict(cast(SwarmStateDict, payload), sw)
        assert rehydrated.swarm_id is None
        assert rehydrated.resume_counts == {}


# ---------------------------------------------------------------------------
# Regression: status is NotRequired in SwarmStateDict — absent field defaults
# to "running" (#MED)
# ---------------------------------------------------------------------------


class TestStatusNotRequired:
    """Regression: ``status`` was declared as Required[str] in SwarmStateDict
    but from_dict used ``.get("status", "running")`` — a contradiction.
    Making ``status`` NotRequired fixes the TypedDict contract and allows
    loader-tolerance for older persisted payloads that omitted the field."""

    def test_from_dict_defaults_status_to_running_when_absent(self) -> None:
        """Payload without a 'status' key must load cleanly and default to 'running'."""
        sw = _mkswarm()
        state = _mkstate(sw)
        payload = dict(state.to_dict())
        payload.pop("status", None)

        rehydrated = SwarmState.from_dict(cast(SwarmStateDict, payload), sw)
        assert rehydrated.status == "running"

    def test_from_dict_accepts_known_status_values(self) -> None:
        sw = _mkswarm()
        state = _mkstate(sw)
        for status_val in ("running", "completed", "failed", "interrupted"):
            payload = dict(state.to_dict())
            payload["status"] = status_val
            rehydrated = SwarmState.from_dict(cast(SwarmStateDict, payload), sw)
            assert rehydrated.status == status_val

    def test_from_dict_rejects_unknown_status(self) -> None:
        sw = _mkswarm()
        state = _mkstate(sw)
        payload = dict(state.to_dict())
        payload["status"] = "bogus"
        with pytest.raises(ValueError, match="status has unknown value"):
            SwarmState.from_dict(cast(SwarmStateDict, payload), sw)


# ---------------------------------------------------------------------------
# Regression: SwarmDone.final_output may be a Pydantic model (structured
# output_schema). to_dict() must normalise it to a JSON-serialisable form so
# json.dumps (to_json + every cross-process checkpointer) does not crash an
# otherwise-successful structured-swarm completion. (#HIGH)
# ---------------------------------------------------------------------------


class _StructuredOutput(BaseModel):
    answer: str
    score: int


class TestDoneFinalOutputJsonSafe:
    def test_to_dict_normalises_pydantic_final_output(self) -> None:
        swarm = _mkswarm()
        state = _mkstate(swarm)
        state.last_yield = SwarmDone(
            reason="done",
            final_output=_StructuredOutput(answer="42", score=9),
        )

        payload = state.to_dict()
        done = payload["last_yield"]
        assert done is not None
        # The Pydantic model is dumped to a plain dict, not left raw.
        assert done["final_output"] == {"answer": "42", "score": 9}

    def test_to_json_serialises_pydantic_final_output(self) -> None:
        """The crash path: json.dumps(to_dict()) must not raise on a model."""
        swarm = _mkswarm()
        state = _mkstate(swarm)
        state.last_yield = SwarmDone(
            reason="done",
            final_output=_StructuredOutput(answer="42", score=9),
        )

        # Before the fix this raised "Object of type _StructuredOutput is not
        # JSON serializable".
        raw = state.to_json()
        reloaded = json.loads(raw)
        assert reloaded["last_yield"]["final_output"] == {"answer": "42", "score": 9}

    def test_round_trip_preserves_structured_final_output_as_dict(self) -> None:
        swarm = _mkswarm()
        state = _mkstate(swarm)
        state.last_yield = SwarmDone(
            reason="done",
            final_output=_StructuredOutput(answer="42", score=9),
        )

        restored = SwarmState.from_json(state.to_json(), swarm)
        assert isinstance(restored.last_yield, SwarmDone)
        assert restored.last_yield.final_output == {"answer": "42", "score": 9}

    def test_plain_final_output_passes_through_unchanged(self) -> None:
        swarm = _mkswarm()
        state = _mkstate(swarm)
        state.last_yield = SwarmDone(reason="done", final_output={"nested": [1, 2, 3]})

        payload = state.to_dict()
        done = payload["last_yield"]
        assert done is not None
        assert done["final_output"] == {"nested": [1, 2, 3]}

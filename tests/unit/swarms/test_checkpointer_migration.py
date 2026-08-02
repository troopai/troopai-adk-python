"""Schema-migration tolerance tests for SwarmState serialisation.

The tolerant-loader contract: persisted state payloads carry no version
field. On load, unknown keys must be ignored and absent evolutionary keys
must take their defaults.

These tests verify:

1. Unknown extra keys injected into a payload are silently ignored.
2. Absent evolutionary keys (those read via ``.get(key, default)`` in
   ``SwarmState.from_dict``) produce a valid state with the correct default.
3. An end-to-end path through ``InMemorySwarmCheckpointer`` also tolerates
   an injected unknown key in the stored payload.

Evolutionary keys confirmed in ``SwarmState.from_dict``:
  - ``status``      — ``data.get("status", "running")``
  - ``error``       — ``data.get("error")``
  - ``swarm_id``    — ``data.get("swarm_id")``
  - ``resume_counts`` — ``data.get("resume_counts", {})``

``resume_counts`` is used for the absent-field drop test because it is a
``NotRequired`` key in ``SwarmStateDict``, making it the cleanest candidate
for a "field that didn't exist in an older payload" scenario.
"""

from __future__ import annotations

from typing import Any, cast

from troopai.adk.agents.agent import Agent
from troopai.adk.swarms.checkpointer import SwarmCheckpoint
from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.state import SwarmState, SwarmStateDict
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination

# ---------------------------------------------------------------------------
# Helpers — mirroring the pattern in tests/unit/swarms/test_postgres_checkpointer.py
# ---------------------------------------------------------------------------


def _make_swarm() -> Swarm:
    """Single-member swarm for serialisation tests."""
    member = Agent(name="m1", system_prompt="test")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(3),
    )


def _make_state(swarm: Swarm, turns: int = 1) -> SwarmState:
    """Build a minimal ``SwarmState`` with a non-default turn count."""
    state = SwarmState(
        swarm=swarm,
        current_agent=swarm.members[0],
        current_agent_name=swarm.members[0].name,
    )
    state.total_turns = turns
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_swarm_state_ignores_unknown_field() -> None:
    """An extra key injected into a to_dict payload must not raise.

    Simulates a payload written by a newer version of the ADK that added
    a field not present in the current ``SwarmState``. The loader must
    silently ignore the unknown key and rehydrate the known fields intact.

    ``SwarmState.from_dict`` accepts ``SwarmStateDict`` but the runtime
    treats any extra dict key as data — the TypedDict contract is structural,
    not enforced at runtime, so unknown keys pass through without error.
    """
    swarm = _make_swarm()
    state = _make_state(swarm, turns=4)

    # Cast to dict[str, Any] to allow mutation without TypedDict friction.
    payload: dict[str, Any] = cast(dict[str, Any], state.to_dict())
    # Inject a synthetic future field.
    payload["a_future_field_xyz"] = {"unexpected": True, "extra_data": [1, 2, 3]}

    restored = SwarmState.from_dict(cast(SwarmStateDict, payload), swarm)

    assert restored.total_turns == 4
    assert restored.current_agent_name == "m1"
    assert restored.status == "running"


def test_swarm_state_absent_evolutionary_field_defaults() -> None:
    """Dropping ``resume_counts`` from the payload loads with an empty dict.

    ``resume_counts`` is a ``NotRequired`` key in ``SwarmStateDict`` and is
    read via ``data.get("resume_counts", {})`` in ``SwarmState.from_dict``,
    making it evolutionary: payloads persisted before the field existed load
    cleanly and the field defaults to ``{}``.
    """
    swarm = _make_swarm()
    state = _make_state(swarm, turns=2)

    payload: dict[str, Any] = cast(dict[str, Any], state.to_dict())
    # Simulate an older payload that predates resume_counts.
    del payload["resume_counts"]

    restored = SwarmState.from_dict(cast(SwarmStateDict, payload), swarm)

    assert restored.total_turns == 2
    # resume_counts defaults to an empty dict when the field is absent.
    assert restored.resume_counts == {}


def test_swarm_state_absent_swarm_id_defaults_to_none() -> None:
    """Dropping ``swarm_id`` from the payload loads with ``None``.

    ``swarm_id`` is a ``NotRequired`` key read via ``data.get("swarm_id")``
    in ``SwarmState.from_dict``. Payloads persisted before this field was
    added rehydrate with ``swarm_id=None``.
    """
    swarm = _make_swarm()
    state = _make_state(swarm, turns=3)
    # Ensure swarm_id is set so we can confirm it is replaced by None on drop.
    state.swarm_id = "some-run-id"

    payload: dict[str, Any] = cast(dict[str, Any], state.to_dict())
    del payload["swarm_id"]

    restored = SwarmState.from_dict(cast(SwarmStateDict, payload), swarm)

    assert restored.total_turns == 3
    assert restored.swarm_id is None


async def test_swarm_state_in_memory_checkpointer_tolerates_unknown_key() -> None:
    """End-to-end: save a payload with an injected key; rehydrate must succeed.

    Verifies that the full ``InMemorySwarmCheckpointer.save`` → ``load`` →
    ``SwarmState.from_dict`` path passes an evolved payload through without
    error, confirming the tolerance contract applies at the backend boundary.
    """
    swarm = _make_swarm()
    state = _make_state(swarm, turns=6)

    raw_payload: dict[str, Any] = cast(dict[str, Any], state.to_dict())
    # Inject an unknown key as if written by a newer writer.
    raw_payload["_future_metadata"] = {"hint": "reserved_for_future_use"}

    cp = InMemorySwarmCheckpointer()
    await cp.save(
        SwarmCheckpoint(
            thread_id="t-swarm-migration",
            state=raw_payload,
            turn=6,
        )
    )

    loaded = await cp.load("t-swarm-migration", swarm)
    assert loaded is not None
    assert loaded.turn == 6

    # Rehydrate through SwarmState.from_dict — must not raise on the unknown key.
    rehydrated = SwarmState.from_dict(cast(SwarmStateDict, loaded.state), swarm)
    assert rehydrated.total_turns == 6
    assert rehydrated.current_agent_name == "m1"

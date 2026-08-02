"""``PostgresSwarmCheckpointer`` — real-database tests via pytest-postgresql.

Requires PostgreSQL 17+ and psycopg[binary,pool]>=3.2 — both are
present in the test environment. Tests do NOT skip; they fail hard if
the infrastructure is absent.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from pytest_postgresql.factories import postgresql, postgresql_proc

from troopai.adk.agents.agent import Agent
from troopai.adk.exceptions import CheckpointConflictError
from troopai.adk.swarms.checkpointer import SwarmCheckpoint
from troopai.adk.swarms.checkpointers.postgres import PostgresSwarmCheckpointer
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.state import SwarmState, SwarmStateDict
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination

pytestmark = pytest.mark.postgres

# ---------------------------------------------------------------------------
# pytest-postgresql process + connection fixtures
# ---------------------------------------------------------------------------

postgresql_my_proc = postgresql_proc()
postgresql_my = postgresql("postgresql_my_proc")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_swarm() -> Swarm:
    """Single-member swarm fixture for checkpointer tests."""
    member = Agent(name="m1", system_prompt="x")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(3),
    )


def _make_state(swarm: Swarm, turns: int = 1) -> SwarmState:
    state = SwarmState(
        swarm=swarm,
        current_agent=swarm.members[0],
        current_agent_name=swarm.members[0].name,
    )
    state.total_turns = turns
    return state


def _ckpt(thread_id: str, state: SwarmState, turn: int) -> SwarmCheckpoint:
    return SwarmCheckpoint(
        thread_id=thread_id,
        state=cast(dict[str, Any], state.to_dict()),
        turn=turn,
    )


@pytest.fixture
def conninfo(postgresql_my) -> str:
    info = postgresql_my.info
    parts = [
        f"dbname={info.dbname}",
        f"user={info.user}",
        f"host={info.host}",
        f"port={info.port}",
    ]
    if info.password is not None and len(info.password) > 0:
        parts.append(f"password={info.password}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_save_load_round_trip(conninfo: str) -> None:
    """save then load returns a SwarmCheckpoint with the correct turn; rehydration succeeds."""
    swarm = _make_swarm()
    state = _make_state(swarm, turns=5)
    cp = PostgresSwarmCheckpointer(conninfo)

    await cp.save(_ckpt("t1", state, state.total_turns))

    loaded = await cp.load("t1", swarm)
    assert loaded is not None
    assert loaded.turn == 5

    # Rehydrate via SwarmState.from_dict — the caller's contract.
    rehydrated = SwarmState.from_dict(cast(SwarmStateDict, loaded.state), swarm)
    assert rehydrated.total_turns == 5
    assert rehydrated.current_agent_name == "m1"

    await cp.close()


async def test_load_missing_returns_none(conninfo: str) -> None:
    """load on an unknown thread_id returns None."""
    swarm = _make_swarm()
    cp = PostgresSwarmCheckpointer(conninfo)
    result = await cp.load("does-not-exist", swarm)
    assert result is None
    await cp.close()


async def test_list_and_delete(conninfo: str) -> None:
    """After save list_checkpoints contains the thread_id; after delete it is gone."""
    swarm = _make_swarm()
    state = _make_state(swarm, turns=1)
    cp = PostgresSwarmCheckpointer(conninfo)

    await cp.save(_ckpt("t1", state, 1))
    assert await cp.list_checkpoints() == ["t1"]

    await cp.delete("t1")
    assert await cp.list_checkpoints() == []

    # load after delete returns None
    assert await cp.load("t1", swarm) is None

    await cp.close()


async def test_conflict_on_stale_token(conninfo: str) -> None:
    """Concurrent writers: the stale token causes CheckpointConflictError.

    Instance A saves (gets token-A). Instance B loads (caches token-A).
    Instance A saves again (token rotates to token-B). Instance B's next
    save still holds token-A → conflict.
    """
    swarm = _make_swarm()

    cp_a = PostgresSwarmCheckpointer(conninfo, thread_id="t2")
    cp_b = PostgresSwarmCheckpointer(conninfo, thread_id="t2")

    # A's first save — row is inserted.
    state_a = _make_state(swarm, turns=1)
    await cp_a.save(_ckpt("t2", state_a, 1))

    # B loads — caches the current lock_token.
    loaded = await cp_b.load("t2", swarm)
    assert loaded is not None

    # A saves again — token rotates.
    state_a.total_turns = 2
    await cp_a.save(_ckpt("t2", state_a, 2))

    # B's save is now stale — token it cached is no longer current.
    state_b = _make_state(swarm, turns=99)
    with pytest.raises(CheckpointConflictError):
        await cp_b.save(_ckpt("t2", state_b, 99))

    await cp_a.close()
    await cp_b.close()

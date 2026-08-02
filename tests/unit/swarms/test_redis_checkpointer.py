"""``RedisSwarmCheckpointer`` — tests via fakeredis (in-process, Lua-capable).

Requires ``fakeredis[lua]`` (pulls ``lupa``) so the optimistic-locking Lua
script executes in-process. Tests do NOT skip.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from fakeredis.aioredis import FakeRedis

from troopai.adk.agents.agent import Agent
from troopai.adk.exceptions import CheckpointConflictError
from troopai.adk.swarms.checkpointer import SwarmCheckpoint
from troopai.adk.swarms.checkpointers.redis import RedisSwarmCheckpointer
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.state import SwarmState, SwarmStateDict
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_save_load_round_trip() -> None:
    """save then load returns a SwarmCheckpoint with the correct turn; rehydration succeeds."""
    swarm = _make_swarm()
    state = _make_state(swarm, turns=5)
    cp = RedisSwarmCheckpointer(client=FakeRedis())

    await cp.save(_ckpt("t1", state, state.total_turns))

    loaded = await cp.load("t1", swarm)
    assert loaded is not None
    assert loaded.turn == 5

    # Rehydrate via SwarmState.from_dict — the caller's contract.
    rehydrated = SwarmState.from_dict(cast(SwarmStateDict, loaded.state), swarm)
    assert rehydrated.total_turns == 5
    assert rehydrated.current_agent_name == "m1"


async def test_load_missing_returns_none() -> None:
    """load on an unknown thread_id returns None."""
    swarm = _make_swarm()
    cp = RedisSwarmCheckpointer(client=FakeRedis())
    assert await cp.load("does-not-exist", swarm) is None


async def test_list_and_delete() -> None:
    """After save list_checkpoints contains the thread_id; after delete it is gone."""
    swarm = _make_swarm()
    state = _make_state(swarm, turns=1)
    cp = RedisSwarmCheckpointer(client=FakeRedis())

    await cp.save(_ckpt("t1", state, 1))
    assert await cp.list_checkpoints() == ["t1"]

    await cp.delete("t1")
    assert await cp.list_checkpoints() == []

    # load after delete returns None
    assert await cp.load("t1", swarm) is None


async def test_conflict_on_stale_token() -> None:
    """Two instances on one key: B's stale-token save raises CheckpointConflictError."""
    swarm = _make_swarm()
    shared = FakeRedis()
    cp_a = RedisSwarmCheckpointer(client=shared)
    cp_b = RedisSwarmCheckpointer(client=shared)

    state = _make_state(swarm, turns=1)
    await cp_a.save(_ckpt("t2", state, 1))

    await cp_b.load("t2", swarm)  # B caches the current token

    state.total_turns = 2
    await cp_a.save(_ckpt("t2", state, 2))  # A rotates the token

    state.total_turns = 99
    with pytest.raises(CheckpointConflictError):
        await cp_b.save(_ckpt("t2", state, 99))


async def test_ttl_expiry_returns_none() -> None:
    """A key written with a TTL reads back as None after it expires."""
    swarm = _make_swarm()
    state = _make_state(swarm, turns=1)
    cp = RedisSwarmCheckpointer(client=FakeRedis(), ttl_seconds=0.05)
    await cp.save(_ckpt("t1", state, 1))
    assert await cp.load("t1", swarm) is not None
    await asyncio.sleep(0.12)
    assert await cp.load("t1", swarm) is None


async def test_save_after_expired_load_does_not_conflict() -> None:
    """load() on an expired key clears the cached token so the next save() is a clean insert."""
    swarm = _make_swarm()
    state = _make_state(swarm, turns=1)
    cp = RedisSwarmCheckpointer(client=FakeRedis(), ttl_seconds=0.05)
    await cp.save(_ckpt("t-exp", state, 1))
    await asyncio.sleep(0.12)
    assert await cp.load("t-exp", swarm) is None  # expired
    # Must NOT raise CheckpointConflictError — the token cache was cleared.
    state.total_turns = 2
    await cp.save(_ckpt("t-exp", state, 2))
    reloaded = await cp.load("t-exp", swarm)
    assert reloaded is not None
    assert reloaded.turn == 2


async def test_negative_ttl_rejected() -> None:
    """A non-positive ttl_seconds is a constructor error, not a silent no-op."""
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        RedisSwarmCheckpointer(client=FakeRedis(), ttl_seconds=-1.0)


async def test_close_only_closes_owned_client() -> None:
    """close() is a no-op for a caller-supplied client= (caller owns it)."""
    swarm = _make_swarm()
    state = _make_state(swarm, turns=1)
    client = FakeRedis()
    cp = RedisSwarmCheckpointer(client=client)
    await cp.close()  # must not close the caller's client
    # The client is still usable: cp can still save and load through it.
    await cp.save(_ckpt("t-own", state, 1))
    assert await cp.load("t-own", swarm) is not None

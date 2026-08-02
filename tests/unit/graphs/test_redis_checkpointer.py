"""``RedisCheckpointer`` — tests via fakeredis (in-process, Lua-capable).

Requires ``fakeredis[lua]`` (pulls ``lupa``) so the optimistic-locking Lua
script executes in-process. Tests do NOT skip.
"""

from __future__ import annotations

import asyncio

import pytest
from fakeredis.aioredis import FakeRedis

from troopai.adk.exceptions import CheckpointConflictError
from troopai.adk.graphs.checkpointer import GraphCheckpoint
from troopai.adk.graphs.checkpointers.redis import RedisCheckpointer
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.state import GraphState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _g() -> Graph:
    return (
        Graph.new("redis-cp")
        .node("a", lambda: "a")
        .node("b", lambda: "b")
        .edge("a", "b")
        .entry("a")
        .terminal("b")
        .compile()
    )


def _checkpoint(state: GraphState, superstep: int = 0) -> GraphCheckpoint:
    thread_id = state.thread_id if state.thread_id is not None else "t1"
    return GraphCheckpoint(
        thread_id=thread_id,
        graph_id=state.graph.id,
        state=state.to_dict(),
        superstep=superstep,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_save_load_round_trip() -> None:
    """save then load returns a GraphState with the correct superstep."""
    g = _g()
    state = GraphState(graph=g, thread_id="t1")
    state.superstep = 2
    cp = RedisCheckpointer(client=FakeRedis())
    await cp.save(_checkpoint(state, superstep=2))
    loaded = await cp.load("t1", g)
    assert loaded is not None
    assert loaded.superstep == 2


async def test_load_missing_returns_none() -> None:
    """load on an unknown thread_id returns None."""
    cp = RedisCheckpointer(client=FakeRedis())
    assert await cp.load("does-not-exist", _g()) is None


async def test_list_and_delete() -> None:
    """After save list_checkpoints contains the thread_id; after delete it is gone."""
    g = _g()
    state = GraphState(graph=g, thread_id="t1")
    cp = RedisCheckpointer(client=FakeRedis())
    await cp.save(_checkpoint(state, superstep=0))
    assert await cp.list_checkpoints() == ["t1"]
    await cp.delete("t1")
    assert await cp.list_checkpoints() == []
    assert await cp.load("t1", g) is None


async def test_graph_id_mismatch_raises() -> None:
    """load with a mismatched graph.id raises ValueError."""
    g = _g()
    state = GraphState(graph=g, thread_id="t3")
    cp = RedisCheckpointer(client=FakeRedis())
    await cp.save(
        GraphCheckpoint(
            thread_id="t3",
            graph_id="wrong-graph",
            state=state.to_dict(),
            superstep=0,
        )
    )
    with pytest.raises(ValueError, match="does not match"):
        await cp.load("t3", g)


async def test_conflict_on_stale_token() -> None:
    """Two instances on one key: B's stale-token save raises CheckpointConflictError."""
    g = _g()
    shared = FakeRedis()
    cp_a = RedisCheckpointer(client=shared)
    cp_b = RedisCheckpointer(client=shared)

    state = GraphState(graph=g, thread_id="t2")
    state.superstep = 1
    await cp_a.save(_checkpoint(state, superstep=1))

    await cp_b.load("t2", g)  # B caches the current token

    state.superstep = 2
    await cp_a.save(_checkpoint(state, superstep=2))  # A rotates the token

    state.superstep = 99
    with pytest.raises(CheckpointConflictError):
        await cp_b.save(_checkpoint(state, superstep=99))


async def test_ttl_expiry_returns_none() -> None:
    """A key written with a TTL reads back as None after it expires."""
    g = _g()
    state = GraphState(graph=g, thread_id="t1")
    cp = RedisCheckpointer(client=FakeRedis(), ttl_seconds=0.05)
    await cp.save(_checkpoint(state, superstep=1))
    assert await cp.load("t1", g) is not None
    await asyncio.sleep(0.12)
    assert await cp.load("t1", g) is None


async def test_save_after_load_on_expired_key_does_not_conflict() -> None:
    """load() on an expired key clears the cached token so the next save() is a clean insert."""
    g = _g()
    state = GraphState(graph=g, thread_id="t-exp")
    state.superstep = 1
    cp = RedisCheckpointer(client=FakeRedis(), ttl_seconds=0.05)
    await cp.save(_checkpoint(state, superstep=1))
    await asyncio.sleep(0.12)
    assert await cp.load("t-exp", g) is None  # expired
    # Must NOT raise CheckpointConflictError — the token cache was cleared.
    state.superstep = 2
    await cp.save(_checkpoint(state, superstep=2))
    reloaded = await cp.load("t-exp", g)
    assert reloaded is not None
    assert reloaded.superstep == 2


async def test_negative_ttl_rejected() -> None:
    """A non-positive ttl_seconds is a constructor error, not a silent no-op."""
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        RedisCheckpointer(client=FakeRedis(), ttl_seconds=-1.0)


async def test_close_only_closes_owned_client() -> None:
    """close() is a no-op for a caller-supplied client= (caller owns it)."""
    g = _g()
    state = GraphState(graph=g, thread_id="t-own")
    state.superstep = 1
    client = FakeRedis()
    cp = RedisCheckpointer(client=client)
    await cp.close()  # must not close the caller's client
    # The client is still usable: cp can still save and load through it.
    await cp.save(_checkpoint(state, superstep=1))
    assert await cp.load("t-own", g) is not None

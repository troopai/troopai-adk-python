"""``PostgresCheckpointer`` — real-database tests via pytest-postgresql.

Requires PostgreSQL 17+ and psycopg[binary,pool]>=3.2 — both are
present in the test environment. Tests do NOT skip; they fail hard if
the infrastructure is absent.
"""

from __future__ import annotations

import pytest
from pytest_postgresql.factories import postgresql, postgresql_proc

from troopai.adk.exceptions import CheckpointConflictError
from troopai.adk.graphs.checkpointer import GraphCheckpoint
from troopai.adk.graphs.checkpointers.postgres import PostgresCheckpointer
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.state import GraphState

pytestmark = pytest.mark.postgres

# ---------------------------------------------------------------------------
# pytest-postgresql process + connection fixtures
# ---------------------------------------------------------------------------

postgresql_my_proc = postgresql_proc()
postgresql_my = postgresql("postgresql_my_proc")


# ---------------------------------------------------------------------------
# Helpers — mirror test_sqlite_checkpointer.py exactly
# ---------------------------------------------------------------------------


def _g() -> Graph:
    return (
        Graph.new("pg-cp")
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
    """save then load returns a GraphState with the correct superstep."""
    g = _g()
    state: GraphState = GraphState(graph=g, thread_id="t1")
    state.superstep = 3
    cp = PostgresCheckpointer(conninfo)
    await cp.save(_checkpoint(state, superstep=3))
    loaded = await cp.load("t1", g)
    assert loaded is not None
    assert loaded.superstep == 3
    await cp.close()


async def test_load_missing_returns_none(conninfo: str) -> None:
    """load on an unknown thread_id returns None."""
    cp = PostgresCheckpointer(conninfo)
    result = await cp.load("does-not-exist", _g())
    assert result is None
    await cp.close()


async def test_list_and_delete(conninfo: str) -> None:
    """After save list_checkpoints contains the thread_id; after delete it is gone."""
    g = _g()
    state: GraphState = GraphState(graph=g, thread_id="t1")
    cp = PostgresCheckpointer(conninfo)
    await cp.save(_checkpoint(state, superstep=0))
    assert await cp.list_checkpoints() == ["t1"]
    await cp.delete("t1")
    assert await cp.list_checkpoints() == []
    # load after delete returns None
    assert await cp.load("t1", g) is None
    await cp.close()


async def test_conflict_on_stale_token(conninfo: str) -> None:
    """Concurrent writers: the stale token causes CheckpointConflictError.

    Instance A saves (gets token-A). Instance B loads (caches token-A).
    Instance A saves again (token rotates to token-B). Instance B's next
    save still holds token-A → conflict.
    """
    g = _g()

    cp_a = PostgresCheckpointer(conninfo)
    cp_b = PostgresCheckpointer(conninfo)

    # A's first save — row is inserted
    state_a: GraphState = GraphState(graph=g, thread_id="t2")
    state_a.superstep = 1
    await cp_a.save(_checkpoint(state_a, superstep=1))

    # B loads — caches the current lock_token
    loaded = await cp_b.load("t2", g)
    assert loaded is not None

    # A saves again — token rotates
    state_a.superstep = 2
    await cp_a.save(_checkpoint(state_a, superstep=2))

    # B's save is now stale — token it cached is no longer current
    state_b_next: GraphState = GraphState(graph=g, thread_id="t2")
    state_b_next.superstep = 99
    with pytest.raises(CheckpointConflictError):
        await cp_b.save(_checkpoint(state_b_next, superstep=99))

    await cp_a.close()
    await cp_b.close()


async def test_graph_id_mismatch_raises(conninfo: str) -> None:
    """load with a mismatched graph.id raises ValueError."""
    g = _g()
    cp = PostgresCheckpointer(conninfo)

    state: GraphState = GraphState(graph=g, thread_id="t3")
    cp2_state = GraphCheckpoint(
        thread_id="t3",
        graph_id="wrong-graph",
        state=state.to_dict(),
        superstep=0,
    )
    await cp.save(cp2_state)

    with pytest.raises(ValueError, match="does not match"):
        await cp.load("t3", g)

    await cp.close()

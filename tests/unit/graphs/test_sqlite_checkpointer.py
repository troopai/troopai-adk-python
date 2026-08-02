"""``SQLiteCheckpointer`` CRUD + cross-instance durability."""

from __future__ import annotations

import pytest

from troopai.adk.graphs.checkpointer import GraphCheckpoint
from troopai.adk.graphs.checkpointers.sqlite import SQLiteCheckpointer
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.state import GraphState


def _g() -> Graph:
    return (
        Graph.new("sqlite-cp")
        .node("a", lambda: "a")
        .node("b", lambda: "b")
        .edge("a", "b")
        .entry("a")
        .terminal("b")
        .compile()
    )


async def test_save_load_round_trip(tmp_path) -> None:
    db = str(tmp_path / "cp.db")
    g = _g()
    state: GraphState = GraphState(graph=g, thread_id="s1")
    state.superstep = 2
    cp = SQLiteCheckpointer(db)
    await cp.save(GraphCheckpoint(thread_id="s1", graph_id=g.id, state=state.to_dict(), superstep=2))
    loaded = await cp.load("s1", g)
    assert loaded is not None
    assert loaded.superstep == 2
    await cp.close()


async def test_load_missing_returns_none(tmp_path) -> None:
    cp = SQLiteCheckpointer(str(tmp_path / "cp.db"))
    assert await cp.load("nope", _g()) is None
    await cp.close()


async def test_graph_id_mismatch_raises(tmp_path) -> None:
    db = str(tmp_path / "cp.db")
    g = _g()
    cp = SQLiteCheckpointer(db)
    await cp.save(
        GraphCheckpoint(
            thread_id="s2",
            graph_id="other-graph",
            state=GraphState(graph=g, thread_id="s2").to_dict(),
            superstep=0,
        )
    )
    with pytest.raises(ValueError, match="does not match"):
        await cp.load("s2", g)
    await cp.close()


async def test_durable_across_instances(tmp_path) -> None:
    db = str(tmp_path / "cp.db")
    g = _g()
    cp1 = SQLiteCheckpointer(db)
    s3_state: GraphState = GraphState(graph=g, thread_id="s3")
    s3_state.superstep = 1
    await cp1.save(
        GraphCheckpoint(
            thread_id="s3",
            graph_id=g.id,
            state=s3_state.to_dict(),
            superstep=1,
        )
    )
    await cp1.close()
    cp2 = SQLiteCheckpointer(db)
    loaded = await cp2.load("s3", g)
    assert loaded is not None and loaded.superstep == 1
    assert await cp2.list_checkpoints() == ["s3"]
    await cp2.delete("s3")
    assert await cp2.load("s3", g) is None
    await cp2.close()

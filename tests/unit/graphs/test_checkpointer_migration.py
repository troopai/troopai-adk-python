"""Schema-migration tolerance tests for GraphState serialisation.

The tolerant-loader contract: persisted state payloads carry no version
field. On load, unknown keys must be ignored and absent evolutionary keys
must take their defaults.

These tests verify:

1. Unknown extra keys injected into a payload are silently ignored.
2. Absent evolutionary keys (those read via ``.get(key, default)`` in
   ``GraphState.from_dict``) produce a valid state with the correct default.
3. An end-to-end path through ``InMemoryCheckpointer`` also tolerates an
   injected unknown key in the stored payload.
"""

from __future__ import annotations

from typing import Any

from troopai.adk.graphs.checkpointer import GraphCheckpoint
from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.state import GraphState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph() -> Graph:
    """Minimal two-node graph for state serialisation tests."""
    return (
        Graph.new("migration-test")
        .node("a", lambda: "a")
        .node("b", lambda: "b")
        .edge("a", "b")
        .entry("a")
        .terminal("b")
        .compile()
    )


def _make_state(graph: Graph, superstep: int = 3) -> GraphState:
    """Build a ``GraphState`` with a non-default superstep for assertion."""
    state: GraphState = GraphState(graph=graph, thread_id="t1")
    state.superstep = superstep
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_graph_state_ignores_unknown_field() -> None:
    """An extra key injected into a to_dict payload must not raise.

    Simulates a payload produced by a newer version of the ADK that
    added a field not present in the current ``GraphState``. The loader
    must silently ignore the unknown key and rehydrate the known fields
    intact.
    """
    graph = _make_graph()
    state = _make_state(graph, superstep=7)

    payload: dict[str, Any] = state.to_dict()
    # Inject a synthetic future field — GraphState.from_dict must not raise.
    payload["a_future_field_xyz"] = {"unexpected": 1, "nested": [1, 2, 3]}

    restored = GraphState.from_dict(payload, graph)

    assert restored.superstep == 7
    assert restored.thread_id == "t1"
    assert restored.status == "running"


def test_graph_state_absent_evolutionary_field_defaults() -> None:
    """Dropping ``resume_counts`` from the payload must load with an empty dict.

    ``resume_counts`` is read via ``dict.get("resume_counts", {})`` in
    ``GraphState.from_dict``, making it evolutionary: payloads persisted
    before the field existed load cleanly and the field defaults to ``{}``.
    """
    graph = _make_graph()
    state = _make_state(graph, superstep=2)

    payload: dict[str, Any] = state.to_dict()
    # Simulate an older payload that predates the resume_counts field.
    del payload["resume_counts"]

    restored = GraphState.from_dict(payload, graph)

    assert restored.superstep == 2
    # resume_counts defaults to an empty dict when the field is absent.
    assert restored.resume_counts == {}


async def test_graph_state_in_memory_checkpointer_tolerates_unknown_key() -> None:
    """End-to-end: save a payload with an injected key; load must rehydrate.

    Verifies that the full ``InMemoryCheckpointer.save`` → ``load`` path
    passes an evolved payload through ``GraphState.from_dict`` without error,
    confirming the tolerance contract applies at the backend boundary too.

    ``thread_id`` in the rehydrated state comes from the serialised payload
    (``"t-migration"`` set below), not from the ``GraphCheckpoint.thread_id``
    field, which is used only as a storage key.
    """
    graph = _make_graph()
    # Build state with thread_id matching the checkpoint key so the assertion
    # on loaded.thread_id is unambiguous.
    state: GraphState = GraphState(graph=graph, thread_id="t-migration")
    state.superstep = 5

    raw_payload: dict[str, Any] = state.to_dict()
    # Inject an unknown key as if written by a newer writer.
    raw_payload["_future_metadata"] = {"schema_hint": "reserved_for_future_use"}

    cp = InMemoryCheckpointer()
    await cp.save(
        GraphCheckpoint(
            thread_id="t-migration",
            graph_id=graph.id,
            state=raw_payload,
            superstep=5,
        )
    )

    loaded = await cp.load("t-migration", graph)

    assert loaded is not None
    assert loaded.superstep == 5
    assert loaded.thread_id == "t-migration"

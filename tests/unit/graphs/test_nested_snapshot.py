"""NestedSnapshot wrapper: validity + dict round-trip."""

from __future__ import annotations

import pytest

from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.nested_snapshot import NestedSnapshot
from troopai.adk.run.state import RunState


def _make_graph_with_agent_node() -> Graph[None]:
    return Graph.new("ns-test").node("a", lambda: "ok").entry("a").terminal("a").compile()


class TestNestedSnapshotValidity:
    def test_agent_kind_requires_run_state(self) -> None:
        rs = RunState()
        snap = NestedSnapshot(kind="agent", run_state=rs)
        assert snap.kind == "agent"
        assert snap.run_state is rs
        assert snap.graph_state is None

    def test_agent_kind_rejects_graph_state(self) -> None:
        with pytest.raises(ValueError, match="kind='agent'"):
            NestedSnapshot(kind="agent", run_state=None, graph_state=None)

    def test_graph_kind_rejects_run_state(self) -> None:
        with pytest.raises(ValueError, match="kind='graph'"):
            NestedSnapshot(kind="graph", run_state=RunState())


class TestNestedSnapshotRoundTrip:
    def test_agent_kind_dict_round_trip(self) -> None:
        rs = RunState(current_agent_name="agent-t1")
        snap = NestedSnapshot(kind="agent", run_state=rs)
        graph = _make_graph_with_agent_node()
        restored = NestedSnapshot.from_dict(snap.to_dict(), graph)
        assert restored.kind == "agent"
        assert restored.run_state is not None
        assert restored.run_state.current_agent_name == "agent-t1"

    def test_unknown_kind_raises(self) -> None:
        graph = _make_graph_with_agent_node()
        with pytest.raises(ValueError, match="unknown kind"):
            NestedSnapshot.from_dict({"kind": "magic"}, graph)

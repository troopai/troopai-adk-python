"""Unit tests for :func:`graph_to_dot` and :meth:`Graph.to_dot`."""

from __future__ import annotations

from troopai.adk.graphs.graph import Graph
from troopai.adk.visualization import graph_to_dot


def _noop() -> str:
    return "noop"


class TestGraphDotStructure:
    def test_digraph_header(self) -> None:
        g = Graph.new("my-graph").node("a", _noop).entry("a").terminal("a").compile()
        out = graph_to_dot(g)
        assert out.startswith('digraph "my-graph" {')
        assert out.endswith("}")

    def test_rankdir(self) -> None:
        g = Graph.new("g").node("a", _noop).entry("a").terminal("a").compile()
        out = graph_to_dot(g, rankdir="TB")
        assert "rankdir=TB;" in out

    def test_entry_oval(self) -> None:
        g = Graph.new("g").node("a", _noop).node("b", _noop).edge("a", "b").entry("a").terminal("b").compile()
        out = graph_to_dot(g)
        assert '"a" [label="a", shape=oval];' in out

    def test_terminal_doublecircle(self) -> None:
        g = Graph.new("g").node("a", _noop).node("b", _noop).edge("a", "b").entry("a").terminal("b").compile()
        out = graph_to_dot(g)
        assert '"b" [label="b", shape=doublecircle];' in out


class TestGraphDotEdges:
    def test_simple_edge(self) -> None:
        g = Graph.new("g").node("a", _noop).node("b", _noop).edge("a", "b").entry("a").terminal("b").compile()
        out = graph_to_dot(g)
        assert '"a" -> "b";' in out

    def test_edge_label(self) -> None:
        g = (
            Graph.new("g")
            .node("a", _noop)
            .node("b", _noop)
            .edge("a", "b", label="approved")
            .entry("a")
            .terminal("b")
            .compile()
        )
        out = graph_to_dot(g)
        assert 'label="approved"' in out

    def test_conditional_edge_dashed(self) -> None:
        g = (
            Graph.new("g")
            .node("a", _noop)
            .node("b", _noop)
            .edge("a", "b", when=lambda _r: True)
            .entry("a")
            .terminal("b")
            .compile()
        )
        out = graph_to_dot(g)
        assert "style=dashed" in out


class TestGraphDotInstanceMethod:
    def test_to_dot_delegates(self) -> None:
        g = Graph.new("g").node("a", _noop).entry("a").terminal("a").compile()
        assert g.to_dot() == graph_to_dot(g)

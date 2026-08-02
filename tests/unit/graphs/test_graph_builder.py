"""Tests for ``GraphBuilder`` — builder validation and compile-time checks.

Every test in this file exercises the *builder contract*: valid inputs
compile cleanly; invalid inputs raise ``ValueError`` with actionable
messages *before* the graph ever executes.  Runtime behaviour is covered
by ``tests/integration/graphs/test_graph_run.py``.
"""

from __future__ import annotations

import pytest

from troopai.adk.graphs.graph import Graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop() -> str:
    """Zero-arg callable usable as a trivial node executable."""
    return "noop"


def _minimal_graph(graph_id: str = "test-graph") -> Graph:
    """A single-node graph (entry == terminal) — the smallest valid shape."""
    return Graph.new(graph_id).node("solo", _noop).entry("solo").terminal("solo").compile()


# ---------------------------------------------------------------------------
# Valid paths
# ---------------------------------------------------------------------------


class TestValidGraphs:
    def test_single_node_entry_and_terminal(self) -> None:
        g = _minimal_graph()
        assert g.entry == "solo"
        assert "solo" in g.terminals
        assert len(g.nodes) == 1
        assert len(g.edges) == 0

    def test_two_node_linear_compiles(self) -> None:
        g = Graph.new("two-node").node("a", _noop).node("b", _noop).pipe("a", "b").entry("a").terminal("b").compile()
        assert g.entry == "a"
        assert "b" in g.terminals
        assert len(g.nodes) == 2
        assert len(g.edges) == 1

    def test_multiple_terminals_compile(self) -> None:
        g = (
            Graph.new("multi-term")
            .node("entry", _noop)
            .node("left", _noop)
            .node("right", _noop)
            .edge("entry", "left")
            .edge("entry", "right")
            .entry("entry")
            .terminal("left", "right")
            .compile()
        )
        assert "left" in g.terminals
        assert "right" in g.terminals

    def test_entry_idempotent_same_id(self) -> None:
        """Calling .entry() twice with the same id MUST not raise."""
        g = (
            Graph.new("idem")
            .node("a", _noop)
            .entry("a")
            .entry("a")  # idempotent
            .terminal("a")
            .compile()
        )
        assert g.entry == "a"

    def test_graph_id_auto_generated_format(self) -> None:
        g = Graph.new().node("x", _noop).entry("x").terminal("x").compile()
        assert g.id.startswith("graph-")
        # 6 chars prefix + 12 hex chars = 18
        assert len(g.id) == 18

    def test_explicit_id_preserved(self) -> None:
        g = Graph.new("my-pipeline").node("n", _noop).entry("n").terminal("n").compile()
        assert g.id == "my-pipeline"


# ---------------------------------------------------------------------------
# Duplicate node id
# ---------------------------------------------------------------------------


class TestDuplicateNodeId:
    def test_duplicate_node_id_raises(self) -> None:
        builder = Graph.new("dup").node("a", _noop)
        with pytest.raises(ValueError, match="already has a node"):
            builder.node("a", _noop)


# ---------------------------------------------------------------------------
# Unknown edge endpoints
# ---------------------------------------------------------------------------


class TestEdgeValidation:
    def test_edge_unknown_source_raises(self) -> None:
        builder = Graph.new("esrc").node("b", _noop)
        with pytest.raises(ValueError, match="source"):
            builder.edge("ghost", "b")

    def test_edge_unknown_target_raises(self) -> None:
        builder = Graph.new("etgt").node("a", _noop)
        with pytest.raises(ValueError, match="target"):
            builder.edge("a", "ghost")

    def test_edge_self_loop_raises(self) -> None:
        """GraphEdge forbids self-loops at construction time."""
        builder = Graph.new("self").node("a", _noop)
        with pytest.raises(ValueError, match="self-loop|Self-loops"):
            builder.edge("a", "a")


# ---------------------------------------------------------------------------
# Pipe validation
# ---------------------------------------------------------------------------


class TestPipeValidation:
    def test_pipe_fewer_than_two_ids_raises(self) -> None:
        builder = Graph.new("pipe-err").node("a", _noop)
        with pytest.raises(ValueError, match="at least 2"):
            builder.pipe("a")

    def test_pipe_zero_args_raises(self) -> None:
        builder = Graph.new("pipe-zero")
        with pytest.raises(ValueError, match="at least 2"):
            builder.pipe()


# ---------------------------------------------------------------------------
# Entry validation
# ---------------------------------------------------------------------------


class TestEntryValidation:
    def test_entry_unknown_id_raises(self) -> None:
        builder = Graph.new("entry-unk").node("a", _noop)
        with pytest.raises(ValueError, match="not a registered"):
            builder.entry("ghost")

    def test_entry_overwrite_different_id_raises(self) -> None:
        builder = Graph.new("entry-ow").node("a", _noop).node("b", _noop).entry("a")
        with pytest.raises(ValueError, match="already has entry"):
            builder.entry("b")


# ---------------------------------------------------------------------------
# Terminal validation
# ---------------------------------------------------------------------------


class TestTerminalValidation:
    def test_terminal_unknown_id_raises(self) -> None:
        builder = Graph.new("term-unk").node("a", _noop)
        with pytest.raises(ValueError, match="not a registered"):
            builder.terminal("ghost")

    def test_terminal_accepts_multiple_known_ids(self) -> None:
        """Multi-terminal graph must compile without error."""
        g = (
            Graph.new("multi-t")
            .node("a", _noop)
            .node("b", _noop)
            .node("c", _noop)
            .edge("a", "b")
            .edge("a", "c")
            .entry("a")
            .terminal("b", "c")
            .compile()
        )
        assert len(g.terminals) == 2


# ---------------------------------------------------------------------------
# Compile-time validation errors
# ---------------------------------------------------------------------------


class TestCompileValidation:
    def test_empty_graph_raises(self) -> None:
        builder = Graph.new("empty")
        with pytest.raises(ValueError, match="no nodes"):
            builder.compile()

    def test_no_entry_declared_raises(self) -> None:
        builder = Graph.new("no-entry").node("a", _noop).terminal("a")
        with pytest.raises(ValueError, match="no entry"):
            builder.compile()

    def test_no_terminal_declared_raises(self) -> None:
        builder = Graph.new("no-term").node("a", _noop).entry("a")
        with pytest.raises(ValueError, match="no terminal"):
            builder.compile()

    def test_entry_has_incoming_edge_raises(self) -> None:
        """Entry cannot be the target of any edge."""
        builder = Graph.new("entry-in").node("a", _noop).node("b", _noop).edge("b", "a").entry("a").terminal("b")
        with pytest.raises(ValueError, match="incoming"):
            builder.compile()

    def test_terminal_has_outgoing_edge_raises(self) -> None:
        """Terminal node cannot have outgoing edges."""
        builder = (
            Graph.new("term-out")
            .node("a", _noop)
            .node("b", _noop)
            .edge("a", "b")
            .node("c", _noop)
            .edge("b", "c")
            .entry("a")
            .terminal("b")
        )
        with pytest.raises(ValueError, match="outgoing"):
            builder.compile()

    def test_unreachable_node_raises(self) -> None:
        """Nodes not reachable from entry must be rejected."""
        builder = (
            Graph.new("unreach")
            .node("a", _noop)
            .node("b", _noop)
            # "b" is an island — no edge from "a" to "b"
            .entry("a")
            .terminal("a")
            .terminal("b")
        )
        with pytest.raises(ValueError, match="reachable"):
            builder.compile()

    def test_dead_end_node_raises(self) -> None:
        """Nodes that can never reach a terminal must be rejected.

        Setup: a → b → c; a → d.  Terminal is "d".
        "c" is reachable from entry ("a") but can never reach terminal "d".
        "b" IS reachable and CAN reach "d" only if the compiler follows "b→c"
        and "a→d" — actually "b" cannot reach "d" either, so both "b" and "c"
        are dead ends and the compiler must raise.
        """
        builder = (
            Graph.new("dead")
            .node("a", _noop)
            .node("b", _noop)
            .node("c", _noop)
            .node("d", _noop)
            .edge("a", "b")
            .edge("b", "c")
            .edge("a", "d")
            .entry("a")
            .terminal("d")
        )
        with pytest.raises(ValueError, match="cannot reach any terminal"):
            builder.compile()


# ---------------------------------------------------------------------------
# Frozen graph — mutation is rejected
# ---------------------------------------------------------------------------


class TestFrozenGraph:
    def test_mutation_raises_on_frozen_graph(self) -> None:
        g = _minimal_graph()
        with pytest.raises((AttributeError, TypeError, Exception)):
            # frozen=True dataclasses raise FrozenInstanceError (subclass of AttributeError)
            g.nodes = ()  # type: ignore[misc]

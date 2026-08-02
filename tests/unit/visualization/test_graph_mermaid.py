"""Unit tests for :func:`graph_to_mermaid` and :meth:`Graph.to_mermaid`."""

from __future__ import annotations

from troopai.adk.graphs.graph import Graph
from troopai.adk.visualization import graph_to_mermaid


def _noop() -> str:
    return "noop"


class TestGraphMermaidStructure:
    def test_header_with_direction(self) -> None:
        g = Graph.new("g").node("a", _noop).entry("a").terminal("a").compile()
        out = graph_to_mermaid(g, direction="LR")
        assert out.startswith("flowchart LR\n")

    def test_top_down_direction(self) -> None:
        g = Graph.new("g").node("a", _noop).entry("a").terminal("a").compile()
        out = graph_to_mermaid(g, direction="TD")
        assert out.startswith("flowchart TD\n")

    def test_entry_node_rounded(self) -> None:
        g = Graph.new("g").node("a", _noop).entry("a").terminal("a").compile()
        out = graph_to_mermaid(g)
        # Single-node graph: entry takes precedence over terminal in shape.
        assert "a(" in out

    def test_terminal_node_stadium(self) -> None:
        g = Graph.new("g").node("a", _noop).node("b", _noop).edge("a", "b").entry("a").terminal("b").compile()
        out = graph_to_mermaid(g)
        assert "b([" in out

    def test_intermediate_node_rectangle(self) -> None:
        g = (
            Graph.new("g")
            .node("a", _noop)
            .node("mid", _noop)
            .node("z", _noop)
            .edge("a", "mid")
            .edge("mid", "z")
            .entry("a")
            .terminal("z")
            .compile()
        )
        out = graph_to_mermaid(g)
        assert 'mid["mid"]' in out


class TestGraphMermaidEdges:
    def test_simple_edge(self) -> None:
        g = Graph.new("g").node("a", _noop).node("b", _noop).edge("a", "b").entry("a").terminal("b").compile()
        out = graph_to_mermaid(g)
        assert "a --> b" in out

    def test_edge_with_label(self) -> None:
        g = (
            Graph.new("g")
            .node("a", _noop)
            .node("b", _noop)
            .edge("a", "b", label="approved")
            .entry("a")
            .terminal("b")
            .compile()
        )
        out = graph_to_mermaid(g)
        assert '|"approved"|' in out

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
        out = graph_to_mermaid(g)
        assert "-.->" in out


class TestGraphMermaidDescription:
    def test_description_used_as_label(self) -> None:
        g = Graph.new("g").node("a", _noop, description="Step A").entry("a").terminal("a").compile()
        out = graph_to_mermaid(g)
        assert '"Step A"' in out

    def test_fallback_to_node_id(self) -> None:
        g = Graph.new("g").node("a", _noop).entry("a").terminal("a").compile()
        out = graph_to_mermaid(g)
        assert '"a"' in out


class TestGraphMermaidLabelEscaping:
    """Labels with Mermaid-breaking characters must be entity-encoded.

    Mermaid quoted strings do not honour backslash escapes; a raw ``"``
    terminates the string, a raw ``#`` can start a spurious entity, and a
    raw newline breaks the flowchart line. The Mermaid emitter must encode
    these as HTML entity codes (``#quot;`` / ``#35;``) and ``<br>`` so the
    diagram stays well-formed.
    """

    def test_node_label_special_chars_escaped(self) -> None:
        # '#', '"', and a newline all appear in one label.
        g = Graph.new("g").node("n", _noop, description='a # "b"\nc').entry("n").terminal("n").compile()
        out = graph_to_mermaid(g)
        # The whole escaped label renders on one line inside the node's
        # quotes: '#' -> '#35;', '"' -> '#quot;', newline -> '<br>'.
        assert 'n("a #35; #quot;b#quot;<br>c")' in out
        # No raw double-quote leaked into the label body, and no backslash
        # escape (which Mermaid renders verbatim, leaving the string open).
        assert '\\"' not in out
        assert 'a # "b"' not in out

    def test_edge_label_special_chars_escaped(self) -> None:
        g = (
            Graph.new("g")
            .node("a", _noop)
            .node("b", _noop)
            .edge("a", "b", label='x#"y\nz')
            .entry("a")
            .terminal("b")
            .compile()
        )
        out = graph_to_mermaid(g)
        assert '|"x#35;#quot;y<br>z"|' in out
        assert '\\"' not in out


class TestGraphInstanceMethod:
    def test_to_mermaid_delegates(self) -> None:
        g = Graph.new("g").node("a", _noop).entry("a").terminal("a").compile()
        assert g.to_mermaid() == graph_to_mermaid(g)

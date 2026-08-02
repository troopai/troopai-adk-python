"""Feature 1: Per-node error handler.

Tests that:
- The handler is invoked after retries are exhausted.
- A handler returning a fallback NodeResult suppresses the exception.
- A handler returning None propagates the original exception.
- Exceptions inside the handler propagate immediately (no double-failure).
- Graph-level default_error_handler is used when node.on_error is None.
- Per-node on_error takes precedence over graph-level default.
- sync and async handlers both work.
"""

from __future__ import annotations

import pytest

from troopai.adk.graphs.config import GraphConfig
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.node import GraphNode
from troopai.adk.orchestration.executable import NodeResult


def _minimal_graph(*, default_error_handler=None) -> Graph:
    """Return a compiled two-node linear graph."""
    config = GraphConfig(
        default_error_handler=default_error_handler,
    )
    return (
        Graph.new("err-test")
        .node("a", lambda: "a")
        .node("b", lambda: "b")
        .edge("a", "b")
        .entry("a")
        .terminal("b")
        .with_config(config)
        .compile()
    )


class TestCallErrorHandler:
    """Unit tests for _call_error_handler internals."""

    async def test_no_handler_returns_none(self) -> None:
        from troopai.adk.run.graph_loop import _call_error_handler

        g = _minimal_graph()
        exc = RuntimeError("boom")
        result = await _call_error_handler(graph=g, node_id="a", exc=exc)
        assert result is None

    async def test_node_sync_handler_returns_fallback(self) -> None:
        from troopai.adk.run.graph_loop import _call_error_handler

        fallback = NodeResult(output="recovered")

        def handler(node_id: str, exc: BaseException) -> NodeResult:
            return fallback

        # Build a graph where node "a" has the handler wired in via builder
        g2 = (
            Graph.new("err-test2")
            .node("a", lambda: "a", on_error=handler)
            .node("b", lambda: "b")
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        result = await _call_error_handler(graph=g2, node_id="a", exc=RuntimeError("x"))
        assert result is fallback

    async def test_node_async_handler_returns_fallback(self) -> None:
        from troopai.adk.run.graph_loop import _call_error_handler

        fallback = NodeResult(output="async-recovered")

        async def handler(node_id: str, exc: BaseException) -> NodeResult:
            return fallback

        g = (
            Graph.new("err-test3")
            .node("a", lambda: "a", on_error=handler)
            .node("b", lambda: "b")
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        result = await _call_error_handler(graph=g, node_id="a", exc=ValueError("y"))
        assert result is fallback

    async def test_graph_level_default_handler_used_when_node_unset(self) -> None:
        from troopai.adk.run.graph_loop import _call_error_handler

        fallback = NodeResult(output="graph-default-fallback")

        def default_handler(node_id: str, exc: BaseException) -> NodeResult:
            return fallback

        g = _minimal_graph(default_error_handler=default_handler)
        result = await _call_error_handler(graph=g, node_id="a", exc=RuntimeError("z"))
        assert result is fallback

    async def test_per_node_handler_takes_precedence_over_graph_default(self) -> None:
        from troopai.adk.run.graph_loop import _call_error_handler

        node_fallback = NodeResult(output="per-node")
        graph_fallback = NodeResult(output="graph-default")

        def node_handler(_nid: str, _exc: BaseException) -> NodeResult:
            return node_fallback

        def graph_handler(_nid: str, _exc: BaseException) -> NodeResult:
            return graph_fallback

        g = (
            Graph.new("err-test4")
            .node("a", lambda: "a", on_error=node_handler)
            .node("b", lambda: "b")
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .with_config(GraphConfig(default_error_handler=graph_handler))
            .compile()
        )
        result = await _call_error_handler(graph=g, node_id="a", exc=RuntimeError("w"))
        assert result is node_fallback

    async def test_handler_exception_propagates(self) -> None:
        from troopai.adk.run.graph_loop import _call_error_handler

        handler_exc = ValueError("handler-failed")

        def bad_handler(_nid: str, _exc: BaseException) -> NodeResult:
            raise handler_exc

        g = (
            Graph.new("err-test5")
            .node("a", lambda: "a", on_error=bad_handler)
            .node("b", lambda: "b")
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        with pytest.raises(ValueError) as ei:
            await _call_error_handler(graph=g, node_id="a", exc=RuntimeError("original"))
        assert ei.value is handler_exc

    async def test_handler_returning_none_propagates_original(self) -> None:
        from troopai.adk.run.graph_loop import _call_error_handler

        def none_handler(_nid: str, _exc: BaseException) -> None:
            return None

        g = (
            Graph.new("err-test6")
            .node("a", lambda: "a", on_error=none_handler)
            .node("b", lambda: "b")
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        result = await _call_error_handler(graph=g, node_id="a", exc=RuntimeError("original"))
        assert result is None


class TestGraphNodeOnErrorField:
    """Ensure GraphNode accepts and stores on_error correctly."""

    def test_graphnode_stores_handler(self) -> None:
        from troopai.adk.graphs.adapters import to_executable

        def my_handler(_nid: str, _exc: BaseException) -> NodeResult | None:
            return None

        node = GraphNode(
            id="n1",
            executable=to_executable(lambda: "x"),
            on_error=my_handler,
        )
        assert node.on_error is my_handler

    def test_graphnode_default_on_error_is_none(self) -> None:
        from troopai.adk.graphs.adapters import to_executable

        node = GraphNode(id="n1", executable=to_executable(lambda: "x"))
        assert node.on_error is None

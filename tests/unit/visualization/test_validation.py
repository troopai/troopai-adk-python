"""Regression tests for invalid-input validation.

The emitters MUST reject invalid layout directions and detect node-id
collisions instead of producing silently-wrong diagrams.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from troopai.adk.flows import Flow, flow_listen, flow_start
from troopai.adk.graphs.graph import Graph
from troopai.adk.visualization import (
    flow_to_dot,
    flow_to_mermaid,
    graph_to_dot,
    graph_to_mermaid,
)


class _S(BaseModel):
    x: int = 0


class _MinimalFlow(Flow[_S]):
    @flow_start
    async def kickoff(self) -> None: ...


def _noop() -> str:
    return "ok"


class TestInvalidDirection:
    def test_flow_to_mermaid_rejects_unknown_direction(self) -> None:
        with pytest.raises(ValueError, match="direction must be one of"):
            flow_to_mermaid(_MinimalFlow(_S), direction="DIAGONAL")  # type: ignore[arg-type]

    def test_flow_to_dot_rejects_unknown_rankdir(self) -> None:
        with pytest.raises(ValueError, match="rankdir must be one of"):
            flow_to_dot(_MinimalFlow(_S), rankdir="DIAGONAL")  # type: ignore[arg-type]

    def test_graph_to_mermaid_rejects_unknown_direction(self) -> None:
        g = Graph.new("g").node("a", _noop).entry("a").terminal("a").compile()
        with pytest.raises(ValueError, match="direction must be one of"):
            graph_to_mermaid(g, direction="DIAGONAL")  # type: ignore[arg-type]

    def test_graph_to_dot_rejects_unknown_rankdir(self) -> None:
        g = Graph.new("g").node("a", _noop).entry("a").terminal("a").compile()
        with pytest.raises(ValueError, match="rankdir must be one of"):
            graph_to_dot(g, rankdir="DIAGONAL")  # type: ignore[arg-type]


class TestNodeIdCollision:
    def test_graph_mermaid_detects_collision(self) -> None:
        # Two node ids that differ only by a character `safe()` flattens
        # to `_` would silently merge into one diagram node.
        g = (
            Graph.new("g")
            .node("a-b", _noop)
            .node("a_b", _noop)
            .edge("a-b", "a_b")
            .entry("a-b")
            .terminal("a_b")
            .compile()
        )
        with pytest.raises(ValueError, match="collides"):
            graph_to_mermaid(g)

    def test_graph_dot_detects_collision(self) -> None:
        g = (
            Graph.new("g")
            .node("a-b", _noop)
            .node("a_b", _noop)
            .edge("a-b", "a_b")
            .entry("a-b")
            .terminal("a_b")
            .compile()
        )
        with pytest.raises(ValueError, match="collides"):
            graph_to_dot(g)


class TestFlowStartPositionalGuard:
    def test_string_positional_arg_raises_typed_error(self) -> None:
        # @flow_start("oops") would land "oops" in the `fn` positional.
        # The runtime guard should give an actionable error.
        from troopai.adk.flows.decorators import flow_start

        with pytest.raises(TypeError, match="positional argument"):
            flow_start("oops")  # type: ignore[arg-type]


class TestFlowListenDescriptionStored:
    def test_description_stored_on_listen(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

            @flow_listen("kickoff", description="My description")
            async def after(self) -> None: ...

        out = flow_to_mermaid(_F(_S))
        assert "My description" in out

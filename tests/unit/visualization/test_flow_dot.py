"""Unit tests for :func:`flow_to_dot` and :meth:`Flow.to_dot`."""

from __future__ import annotations

from pydantic import BaseModel

from troopai.adk.flows import Flow, flow_listen, flow_router, flow_start
from troopai.adk.visualization import flow_to_dot


class _S(BaseModel):
    x: int = 0


class TestFlowDotStructure:
    def test_emits_digraph_header(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

        out = flow_to_dot(_F(_S))
        assert out.startswith('digraph "flow" {')
        assert out.endswith("}")

    def test_rankdir_set(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

        out = flow_to_dot(_F(_S), rankdir="TB")
        assert "rankdir=TB;" in out

    def test_start_node_oval(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

        out = flow_to_dot(_F(_S))
        assert "shape=oval" in out
        assert '"kickoff"' in out

    def test_listen_node_box(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

            @flow_listen("kickoff")
            async def after(self) -> None: ...

        out = flow_to_dot(_F(_S))
        assert "shape=box" in out
        assert '"after"' in out

    def test_router_node_diamond(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

            @flow_router("kickoff")
            async def decide(self) -> str:
                return "branch_a"

        out = flow_to_dot(_F(_S))
        assert "shape=diamond" in out
        assert '"decide"' in out


class TestFlowDotEdges:
    def test_direct_edge(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

            @flow_listen("kickoff")
            async def after(self) -> None: ...

        out = flow_to_dot(_F(_S))
        assert '"kickoff" -> "after";' in out

    def test_and_gate(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def a(self) -> None: ...

            @flow_start
            async def b(self) -> None: ...

            @flow_listen(a & b)
            async def merged(self) -> None: ...

        out = flow_to_dot(_F(_S))
        assert 'label="AND"' in out
        assert "shape=circle" in out

    def test_or_gate(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def a(self) -> None: ...

            @flow_start
            async def b(self) -> None: ...

            @flow_listen(a | b)
            async def merged(self) -> None: ...

        out = flow_to_dot(_F(_S))
        assert 'label="OR"' in out


class TestFlowDotEscaping:
    def test_quote_escaped(self) -> None:
        class _F(Flow[_S]):
            @flow_start(description='Hello "world"')
            async def kickoff(self) -> None: ...

        out = flow_to_dot(_F(_S))
        assert '\\"world\\"' in out


class TestFlowDotInstanceMethod:
    def test_to_dot_delegates(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

        flow = _F(_S)
        assert flow.to_dot() == flow_to_dot(flow)

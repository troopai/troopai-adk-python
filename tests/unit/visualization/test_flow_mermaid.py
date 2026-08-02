"""Unit tests for :func:`flow_to_mermaid` and :meth:`Flow.to_mermaid`.

Asserts the Mermaid output shape for each topology kind: direct
trigger, AND gate, OR gate, router, custom step descriptions.
"""

from __future__ import annotations

from pydantic import BaseModel

from troopai.adk.flows import Flow, flow_listen, flow_router, flow_start
from troopai.adk.visualization import flow_to_mermaid


class _S(BaseModel):
    x: int = 0


class TestFlowMermaidDirectFlow:
    def test_emits_flowchart_header_with_direction(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

        out = flow_to_mermaid(_F(_S), direction="LR")
        assert out.startswith("flowchart LR\n")

    def test_top_down_direction(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

        out = flow_to_mermaid(_F(_S), direction="TD")
        assert out.startswith("flowchart TD\n")

    def test_start_node_uses_rounded_shape(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

        out = flow_to_mermaid(_F(_S))
        assert 'kickoff(("kickoff"))' in out

    def test_listen_node_uses_rectangle_shape(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

            @flow_listen("kickoff")
            async def after(self) -> None: ...

        out = flow_to_mermaid(_F(_S))
        assert 'after["after"]' in out

    def test_router_node_uses_diamond_shape(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

            @flow_router("kickoff")
            async def route(self) -> str:
                return "label_a"

        out = flow_to_mermaid(_F(_S))
        assert 'route{"route"}' in out

    def test_direct_edge_rendered(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

            @flow_listen("kickoff")
            async def after(self) -> None: ...

        out = flow_to_mermaid(_F(_S))
        assert "kickoff --> after" in out


class TestFlowMermaidGates:
    def test_and_gate_synthesised(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def a(self) -> None: ...

            @flow_start
            async def b(self) -> None: ...

            @flow_listen(a & b)
            async def merged(self) -> None: ...

        out = flow_to_mermaid(_F(_S))
        assert "((AND))" in out
        assert "a --> gate__" in out
        assert "b --> gate__" in out
        assert "--> merged" in out

    def test_or_gate_synthesised(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def a(self) -> None: ...

            @flow_start
            async def b(self) -> None: ...

            @flow_listen(a | b)
            async def merged(self) -> None: ...

        out = flow_to_mermaid(_F(_S))
        assert "((OR))" in out
        assert "a --> gate__" in out
        assert "b --> gate__" in out


class TestFlowMermaidDescription:
    def test_description_used_as_label(self) -> None:
        class _F(Flow[_S]):
            @flow_start(description="Seed topic")
            async def kickoff(self) -> None: ...

        out = flow_to_mermaid(_F(_S))
        assert 'kickoff(("Seed topic"))' in out

    def test_description_on_listen(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

            @flow_listen("kickoff", description="Process data")
            async def after(self) -> None: ...

        out = flow_to_mermaid(_F(_S))
        assert 'after["Process data"]' in out

    def test_fallback_to_method_name_when_no_description(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

        out = flow_to_mermaid(_F(_S))
        assert 'kickoff(("kickoff"))' in out


class TestFlowMermaidEscaping:
    def test_double_quotes_in_description_escaped(self) -> None:
        class _F(Flow[_S]):
            @flow_start(description='Said "hello"')
            async def kickoff(self) -> None: ...

        out = flow_to_mermaid(_F(_S))
        # Mermaid quoted strings do not honour backslash escapes, so a
        # double quote becomes the HTML entity ``#quot;`` — never ``\"``,
        # which Mermaid would render verbatim and leave the string open.
        assert "Said #quot;hello#quot;" in out
        assert '\\"' not in out

    def test_newline_in_description_replaced(self) -> None:
        class _F(Flow[_S]):
            @flow_start(description="line1\nline2")
            async def kickoff(self) -> None: ...

        out = flow_to_mermaid(_F(_S))
        # A raw newline MUST NOT break the node declaration line; Mermaid
        # renders ``<br>`` as an in-label line break. Look for the kickoff
        # node declaration and confirm the newline became ``<br>``.
        for line in out.split("\n"):
            if "kickoff" in line and "((" in line:
                assert "line1<br>line2" in line
                break
        else:
            raise AssertionError("kickoff node declaration not found")


class TestFlowInstanceMethod:
    def test_to_mermaid_delegates(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

        flow = _F(_S)
        assert flow.to_mermaid() == flow_to_mermaid(flow)

    def test_to_mermaid_direction_propagates(self) -> None:
        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

        flow = _F(_S)
        assert flow.to_mermaid(direction="TD").startswith("flowchart TD\n")

"""Unit tests for :func:`definition_to_mermaid` and :func:`definition_to_dot`.

Stage 3 acceptance criteria:
- A FlowDefinition renders to Mermaid / DOT without constructing or running
  a Flow instance (no Flow import or instantiation required for rendering).
- Output matches the topology encoded in the definition.
- Descriptions from StepInfo propagate to node labels.
- AND/OR gate nodes appear for the corresponding GateInfo entries.
- Invalid direction / rankdir raises ValueError.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from troopai.adk.flows import (
    And,
    FlowStepRegistry,
    Or,
    flow_listen,
    flow_router,
    flow_start,
)
from troopai.adk.flows.definition import build_flow_definition
from troopai.adk.visualization import definition_to_dot, definition_to_mermaid

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_simple_def():
    """Build a FlowDefinition for kickoff → after."""
    reg = FlowStepRegistry(
        starts=frozenset({"kickoff"}),
        listeners={"after": ("kickoff",)},
        routers={},
    )
    return build_flow_definition(reg, descriptions={"kickoff": "Start here", "after": None})


def _make_router_def():
    """Build a FlowDefinition with a router step."""
    reg = FlowStepRegistry(
        starts=frozenset({"begin"}),
        listeners={},
        routers={"route": ("begin",)},
    )
    return build_flow_definition(reg)


def _make_and_gate_def():
    """Build a FlowDefinition with an AND gate."""
    reg = FlowStepRegistry(
        starts=frozenset({"a", "b"}),
        listeners={"merged": (And(triggers=("a", "b")),)},
        routers={},
    )
    return build_flow_definition(reg)


def _make_or_gate_def():
    """Build a FlowDefinition with an OR gate."""
    reg = FlowStepRegistry(
        starts=frozenset({"a", "b"}),
        listeners={"any_first": (Or(triggers=("a", "b")),)},
        routers={},
    )
    return build_flow_definition(reg)


# ---------------------------------------------------------------------------
# definition_to_mermaid — without any Flow instance
# ---------------------------------------------------------------------------


class TestDefinitionToMermaid:
    def test_emits_flowchart_header(self) -> None:
        defn = _make_simple_def()
        out = definition_to_mermaid(defn)
        assert out.startswith("flowchart LR\n")

    def test_direction_propagates(self) -> None:
        defn = _make_simple_def()
        out = definition_to_mermaid(defn, direction="TD")
        assert out.startswith("flowchart TD\n")

    def test_start_node_rounded_shape(self) -> None:
        defn = _make_simple_def()
        out = definition_to_mermaid(defn)
        assert 'kickoff(("' in out

    def test_listen_node_rectangle_shape(self) -> None:
        defn = _make_simple_def()
        out = definition_to_mermaid(defn)
        assert 'after["' in out

    def test_router_node_diamond_shape(self) -> None:
        defn = _make_router_def()
        out = definition_to_mermaid(defn)
        assert 'route{"' in out

    def test_direct_edge_rendered(self) -> None:
        defn = _make_simple_def()
        out = definition_to_mermaid(defn)
        assert "kickoff --> after" in out

    def test_description_used_as_label(self) -> None:
        defn = _make_simple_def()
        out = definition_to_mermaid(defn)
        assert '"Start here"' in out

    def test_fallback_to_method_name_when_no_description(self) -> None:
        defn = _make_simple_def()
        out = definition_to_mermaid(defn)
        assert '"after"' in out

    def test_and_gate_node_emitted(self) -> None:
        defn = _make_and_gate_def()
        out = definition_to_mermaid(defn)
        assert "((AND))" in out
        assert "a --> gate__" in out
        assert "b --> gate__" in out
        assert "--> merged" in out

    def test_or_gate_node_emitted(self) -> None:
        defn = _make_or_gate_def()
        out = definition_to_mermaid(defn)
        assert "((OR))" in out
        assert "a --> gate__" in out
        assert "b --> gate__" in out

    def test_invalid_direction_raises(self) -> None:
        defn = _make_simple_def()
        with pytest.raises(ValueError, match="direction"):
            definition_to_mermaid(defn, direction="XX")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# definition_to_dot — without any Flow instance
# ---------------------------------------------------------------------------


class TestDefinitionToDot:
    def test_emits_digraph_header(self) -> None:
        defn = _make_simple_def()
        out = definition_to_dot(defn)
        assert out.startswith('digraph "flow" {')

    def test_rankdir_propagates(self) -> None:
        defn = _make_simple_def()
        out = definition_to_dot(defn, rankdir="TB")
        assert "rankdir=TB" in out

    def test_start_node_oval_shape(self) -> None:
        defn = _make_simple_def()
        out = definition_to_dot(defn)
        assert "shape=oval" in out

    def test_listen_node_box_shape(self) -> None:
        defn = _make_simple_def()
        out = definition_to_dot(defn)
        assert "shape=box" in out

    def test_router_node_diamond_shape(self) -> None:
        defn = _make_router_def()
        out = definition_to_dot(defn)
        assert "shape=diamond" in out

    def test_direct_edge_rendered(self) -> None:
        defn = _make_simple_def()
        out = definition_to_dot(defn)
        assert '"kickoff" -> "after"' in out

    def test_description_used_as_label(self) -> None:
        defn = _make_simple_def()
        out = definition_to_dot(defn)
        assert 'label="Start here"' in out

    def test_and_gate_circle_node(self) -> None:
        defn = _make_and_gate_def()
        out = definition_to_dot(defn)
        assert 'label="AND", shape=circle' in out

    def test_or_gate_circle_node(self) -> None:
        defn = _make_or_gate_def()
        out = definition_to_dot(defn)
        assert 'label="OR", shape=circle' in out

    def test_invalid_rankdir_raises(self) -> None:
        defn = _make_simple_def()
        with pytest.raises(ValueError, match="rankdir"):
            definition_to_dot(defn, rankdir="ZZ")  # type: ignore[arg-type]

    def test_ends_with_closing_brace(self) -> None:
        defn = _make_simple_def()
        out = definition_to_dot(defn)
        assert out.endswith("}")


# ---------------------------------------------------------------------------
# Integration: get_definition() → render without running the flow
# ---------------------------------------------------------------------------


class _S(BaseModel):
    x: int = 0


class TestGetDefinitionThenRender:
    def test_mermaid_from_get_definition_no_run(self) -> None:
        """Render Mermaid from get_definition() — no Flow execution path touched."""
        from troopai.adk.flows import Flow

        class _F(Flow[_S]):
            @flow_start(description="Seed")
            async def kickoff(self) -> None: ...

            @flow_listen("kickoff")
            async def after(self) -> None: ...

        defn = _F(_S).get_definition()
        out = definition_to_mermaid(defn)
        assert "flowchart LR" in out
        assert '"Seed"' in out
        assert "kickoff --> after" in out

    def test_dot_from_get_definition_no_run(self) -> None:
        """Render DOT from get_definition() — no Flow execution path touched."""
        from troopai.adk.flows import Flow

        class _F(Flow[_S]):
            @flow_start
            async def start(self) -> None: ...

            @flow_router("start")
            async def route(self) -> str:
                return "path_a"

        defn = _F(_S).get_definition()
        out = definition_to_dot(defn)
        assert 'digraph "flow"' in out
        assert "shape=diamond" in out

    def test_mermaid_and_gate_from_get_definition(self) -> None:
        from troopai.adk.flows import Flow

        class _F(Flow[_S]):
            @flow_start
            async def a(self) -> None: ...

            @flow_start
            async def b(self) -> None: ...

            @flow_listen(a & b)  # type: ignore[name-defined]
            async def merged(self) -> None: ...

        defn = _F(_S).get_definition()
        out = definition_to_mermaid(defn)
        assert "((AND))" in out
        assert "--> merged" in out

    def test_mermaid_or_gate_from_get_definition(self) -> None:
        from troopai.adk.flows import Flow

        class _F(Flow[_S]):
            @flow_start
            async def a(self) -> None: ...

            @flow_start
            async def b(self) -> None: ...

            @flow_listen(a | b)  # type: ignore[name-defined]
            async def any_first(self) -> None: ...

        defn = _F(_S).get_definition()
        out = definition_to_mermaid(defn)
        assert "((OR))" in out

    def test_definition_render_matches_flow_render_structure(self) -> None:
        """definition_to_mermaid and flow.to_mermaid should encode the same topology."""
        from troopai.adk.flows import Flow

        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

            @flow_listen("kickoff")
            async def after(self) -> None: ...

        flow_instance = _F(_S)
        defn = flow_instance.get_definition()

        mermaid_from_flow = flow_instance.to_mermaid()
        mermaid_from_defn = definition_to_mermaid(defn)

        # Both must encode the same step nodes and edge.
        for fragment in ["kickoff", "after", "kickoff --> after"]:
            assert fragment in mermaid_from_flow, f"Missing {fragment!r} in flow render"
            assert fragment in mermaid_from_defn, f"Missing {fragment!r} in definition render"

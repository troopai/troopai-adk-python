"""Unit tests for :class:`FlowStepRegistry` and :func:`build_transition_table`.

Covers registry validation (no flow_start, duplicates), the transition-table
dispatch indexes, and the fan-out cap enforcement.
"""

from __future__ import annotations

import pytest

from troopai.adk.flows import (
    And,
    FlowDefinitionError,
    FlowStepRegistry,
    Or,
    build_transition_table,
)


class TestFlowStepRegistry:
    def test_basic_registry(self) -> None:
        reg = FlowStepRegistry(
            starts=frozenset({"kickoff"}),
            listeners={"after": ("kickoff",)},
            routers={},
        )
        assert reg.starts == frozenset({"kickoff"})
        assert reg.listeners == {"after": ("kickoff",)}
        assert reg.step_names() == frozenset({"kickoff", "after"})

    def test_empty_starts_rejected(self) -> None:
        with pytest.raises(FlowDefinitionError, match="at least one @flow_start"):
            FlowStepRegistry(starts=frozenset(), listeners={}, routers={})

    def test_overlapping_method_names_rejected(self) -> None:
        with pytest.raises(FlowDefinitionError, match="unique"):
            FlowStepRegistry(
                starts=frozenset({"x"}),
                listeners={"x": ("kickoff",)},
                routers={},
            )


class TestBuildTransitionTable:
    def test_direct_listeners(self) -> None:
        reg = FlowStepRegistry(
            starts=frozenset({"kickoff"}),
            listeners={"after": ("kickoff",)},
            routers={},
        )
        table = build_transition_table(reg)
        assert table.starts == ("kickoff",)
        assert table.direct_listeners == {"kickoff": ("after",)}

    def test_and_gate(self) -> None:
        reg = FlowStepRegistry(
            starts=frozenset({"a", "b"}),
            listeners={"merged": (And(triggers=("a", "b")),)},
            routers={},
        )
        table = build_transition_table(reg)
        assert len(table.and_gates) == 1
        assert "a" in table.and_gates_for
        assert "b" in table.and_gates_for

    def test_or_gate(self) -> None:
        reg = FlowStepRegistry(
            starts=frozenset({"a", "b"}),
            listeners={"any": (Or(triggers=("a", "b")),)},
            routers={},
        )
        table = build_transition_table(reg)
        assert len(table.or_gates) == 1
        assert "a" in table.or_gates_for

    def test_router_indexed_for_trigger(self) -> None:
        reg = FlowStepRegistry(
            starts=frozenset({"begin"}),
            listeners={},
            routers={"route": ("begin",)},
        )
        table = build_transition_table(reg)
        assert table.routers_for == {"begin": ("route",)}

    def test_router_combinator_rejected(self) -> None:
        reg = FlowStepRegistry(
            starts=frozenset({"a", "b"}),
            listeners={},
            routers={"route": (Or(triggers=("a", "b")),)},
        )
        with pytest.raises(FlowDefinitionError, match="string triggers"):
            build_transition_table(reg)

    def test_fanout_cap_enforced(self) -> None:
        # 25 listeners on one trigger; default cap is 20.
        reg = FlowStepRegistry(
            starts=frozenset({"src"}),
            listeners={f"L{i}": ("src",) for i in range(25)},
            routers={},
        )
        with pytest.raises(FlowDefinitionError, match="exceeding max_listeners_per_step"):
            build_transition_table(reg)

    def test_fanout_cap_raised_allows_wider(self) -> None:
        reg = FlowStepRegistry(
            starts=frozenset({"src"}),
            listeners={f"L{i}": ("src",) for i in range(25)},
            routers={},
        )
        table = build_transition_table(reg, max_listeners_per_step=50)
        assert len(table.direct_listeners["src"]) == 25

    def test_zero_fanout_cap_rejected(self) -> None:
        reg = FlowStepRegistry(
            starts=frozenset({"x"}),
            listeners={},
            routers={},
        )
        with pytest.raises(FlowDefinitionError, match=">= 1"):
            build_transition_table(reg, max_listeners_per_step=0)

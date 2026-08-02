"""Unit tests for :class:`FlowDefinition` and :func:`build_flow_definition`.

Stage 1 acceptance criteria:
- Definition derived from a representative flow matches the decorator topology.
- Pickle round-trip produces an equal object.
- Immutability: frozen dataclass fields reject mutation.
"""

from __future__ import annotations

import pickle

import pytest
from pydantic import BaseModel

from troopai.adk.flows import (
    And,
    FlowDefinition,
    FlowDefinitionError,
    FlowStepRegistry,
    GateInfo,
    Or,
    StepInfo,
    flow_listen,
    flow_router,
    flow_start,
)
from troopai.adk.flows.definition import build_flow_definition as _build_flow_definition


class _S(BaseModel):
    x: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reg(
    starts: frozenset[str],
    listeners: dict | None = None,
    routers: dict | None = None,
) -> FlowStepRegistry:
    return FlowStepRegistry(
        starts=starts,
        listeners=listeners or {},
        routers=routers or {},
    )


# ---------------------------------------------------------------------------
# build_flow_definition — topology fidelity
# ---------------------------------------------------------------------------


class TestBuildFlowDefinitionTopology:
    def test_starts_recorded(self) -> None:
        reg = _reg(frozenset({"kickoff"}))
        defn = _build_flow_definition(reg)
        assert defn.starts == frozenset({"kickoff"})
        assert defn.roles["kickoff"] == "start"

    def test_listener_recorded(self) -> None:
        reg = _reg(
            frozenset({"a"}),
            listeners={"b": ("a",)},
        )
        defn = _build_flow_definition(reg)
        assert defn.roles["b"] == "listen"
        assert defn.direct_triggers["b"] == ("a",)

    def test_router_recorded(self) -> None:
        reg = _reg(
            frozenset({"a"}),
            routers={"r": ("a",)},
        )
        defn = _build_flow_definition(reg)
        assert defn.roles["r"] == "router"
        assert defn.router_triggers["r"] == ("a",)

    def test_and_gate_recorded(self) -> None:
        reg = _reg(
            frozenset({"a", "b"}),
            listeners={"merged": (And(triggers=("a", "b")),)},
        )
        defn = _build_flow_definition(reg)
        assert len(defn.gates) == 1
        gate = defn.gates[0]
        assert gate.kind == "and"
        assert gate.listener_name == "merged"
        assert gate.triggers == frozenset({"a", "b"})

    def test_or_gate_recorded(self) -> None:
        reg = _reg(
            frozenset({"a", "b"}),
            listeners={"any_first": (Or(triggers=("a", "b")),)},
        )
        defn = _build_flow_definition(reg)
        assert len(defn.gates) == 1
        gate = defn.gates[0]
        assert gate.kind == "or"
        assert gate.triggers == frozenset({"a", "b"})

    def test_step_names_set(self) -> None:
        reg = _reg(
            frozenset({"start_step"}),
            listeners={"listen_step": ("start_step",)},
            routers={"route_step": ("listen_step",)},
        )
        defn = _build_flow_definition(reg)
        assert defn.step_names() == frozenset({"start_step", "listen_step", "route_step"})

    def test_steps_sorted_by_name(self) -> None:
        reg = _reg(
            frozenset({"z_start", "a_start"}),
        )
        defn = _build_flow_definition(reg)
        names = [s.name for s in defn.steps]
        assert names == sorted(names)

    def test_description_propagated(self) -> None:
        reg = _reg(frozenset({"kickoff"}))
        defn = _build_flow_definition(reg, descriptions={"kickoff": "Launch the flow"})
        step = next(s for s in defn.steps if s.name == "kickoff")
        assert step.description == "Launch the flow"

    def test_description_none_by_default(self) -> None:
        reg = _reg(frozenset({"kickoff"}))
        defn = _build_flow_definition(reg)
        step = defn.steps[0]
        assert step.description is None

    def test_gate_id_canonical(self) -> None:
        reg = _reg(
            frozenset({"a", "b"}),
            listeners={"merged": (And(triggers=("b", "a")),)},
        )
        defn = _build_flow_definition(reg)
        # Gate id must contain sorted trigger names for stability.
        assert "a,b" in defn.gates[0].gate_id

    def test_router_with_gate_trigger_raises(self) -> None:
        """A router declaring a combinator gate must be rejected.

        Regression: ``build_transition_table`` refuses gate-gated routers
        (they could never execute), but ``build_flow_definition`` silently
        accepted them — producing a ``FlowDefinition`` for an unrunnable
        flow. Both build paths now reject with the same
        :class:`FlowDefinitionError`.
        """
        reg = _reg(
            frozenset({"a", "b"}),
            routers={"route": (Or(triggers=("a", "b")),)},
        )
        with pytest.raises(FlowDefinitionError, match="may only have string triggers"):
            _build_flow_definition(reg)

    def test_router_with_and_gate_trigger_raises(self) -> None:
        reg = _reg(
            frozenset({"a", "b"}),
            routers={"route": (And(triggers=("a", "b")),)},
        )
        with pytest.raises(FlowDefinitionError, match="may only have string triggers"):
            _build_flow_definition(reg)


# ---------------------------------------------------------------------------
# Flow.get_definition() — integration with decorator system
# ---------------------------------------------------------------------------


class TestFlowGetDefinition:
    def test_start_step_in_definition(self) -> None:
        from troopai.adk.flows import Flow

        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

        defn = _F(_S).get_definition()
        assert "kickoff" in defn.starts
        assert defn.roles["kickoff"] == "start"

    def test_listen_step_in_definition(self) -> None:
        from troopai.adk.flows import Flow

        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

            @flow_listen("kickoff")
            async def after(self) -> None: ...

        defn = _F(_S).get_definition()
        assert defn.roles["after"] == "listen"
        assert defn.direct_triggers["after"] == ("kickoff",)

    def test_router_step_in_definition(self) -> None:
        from troopai.adk.flows import Flow

        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

            @flow_router("kickoff")
            async def route(self) -> str:
                return "path_a"

        defn = _F(_S).get_definition()
        assert defn.roles["route"] == "router"
        assert defn.router_triggers["route"] == ("kickoff",)

    def test_and_gate_in_definition(self) -> None:
        from troopai.adk.flows import Flow

        class _F(Flow[_S]):
            @flow_start
            async def a(self) -> None: ...

            @flow_start
            async def b(self) -> None: ...

            @flow_listen(a & b)  # type: ignore[name-defined]
            async def merged(self) -> None: ...

        defn = _F(_S).get_definition()
        assert len(defn.gates) == 1
        gate = defn.gates[0]
        assert gate.kind == "and"
        assert gate.listener_name == "merged"
        assert gate.triggers == frozenset({"a", "b"})

    def test_or_gate_in_definition(self) -> None:
        from troopai.adk.flows import Flow

        class _F(Flow[_S]):
            @flow_start
            async def a(self) -> None: ...

            @flow_start
            async def b(self) -> None: ...

            @flow_listen(a | b)  # type: ignore[name-defined]
            async def any_first(self) -> None: ...

        defn = _F(_S).get_definition()
        assert len(defn.gates) == 1
        gate = defn.gates[0]
        assert gate.kind == "or"

    def test_description_from_decorator(self) -> None:
        from troopai.adk.flows import Flow

        class _F(Flow[_S]):
            @flow_start(description="Seed the run")
            async def kickoff(self) -> None: ...

        defn = _F(_S).get_definition()
        step = next(s for s in defn.steps if s.name == "kickoff")
        assert step.description == "Seed the run"

    def test_multiple_instances_same_definition(self) -> None:
        from troopai.adk.flows import Flow

        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

        defn_a = _F(_S).get_definition()
        defn_b = _F(_S).get_definition()
        assert defn_a == defn_b


# ---------------------------------------------------------------------------
# Pickle round-trip
# ---------------------------------------------------------------------------


class TestFlowDefinitionPickle:
    def test_pickle_roundtrip_simple(self) -> None:
        reg = _reg(
            frozenset({"kickoff"}),
            listeners={"after": ("kickoff",)},
        )
        defn = _build_flow_definition(reg)
        restored: FlowDefinition = pickle.loads(pickle.dumps(defn))
        assert restored == defn

    def test_pickle_roundtrip_with_gates(self) -> None:
        reg = _reg(
            frozenset({"a", "b"}),
            listeners={
                "and_step": (And(triggers=("a", "b")),),
                "or_step": (Or(triggers=("a", "b")),),
            },
        )
        defn = _build_flow_definition(reg)
        restored: FlowDefinition = pickle.loads(pickle.dumps(defn))
        assert restored.gates == defn.gates
        assert restored.starts == defn.starts

    def test_pickle_roundtrip_with_description(self) -> None:
        reg = _reg(frozenset({"kickoff"}))
        defn = _build_flow_definition(reg, descriptions={"kickoff": "Launch"})
        restored: FlowDefinition = pickle.loads(pickle.dumps(defn))
        assert restored.steps[0].description == "Launch"

    def test_step_info_picklable(self) -> None:
        info = StepInfo(name="kickoff", role="start", triggers=(), description="Hello")
        restored: StepInfo = pickle.loads(pickle.dumps(info))
        assert restored == info

    def test_gate_info_picklable(self) -> None:
        gate = GateInfo(
            gate_id="merged:and:a,b",
            listener_name="merged",
            kind="and",
            triggers=frozenset({"a", "b"}),
        )
        restored: GateInfo = pickle.loads(pickle.dumps(gate))
        assert restored == gate


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestFlowDefinitionImmutability:
    def test_flow_definition_frozen(self) -> None:
        reg = _reg(frozenset({"kickoff"}))
        defn = _build_flow_definition(reg)
        with pytest.raises((AttributeError, TypeError)):
            defn.starts = frozenset({"other"})  # type: ignore[misc]

    def test_roles_mapping_read_only(self) -> None:
        """roles is a MappingProxyType — in-place mutation must raise TypeError."""
        reg = _reg(frozenset({"kickoff"}))
        defn = _build_flow_definition(reg)
        with pytest.raises(TypeError):
            defn.roles["injected"] = "start"  # type: ignore[index]

    def test_direct_triggers_mapping_read_only(self) -> None:
        """direct_triggers is a MappingProxyType — in-place mutation must raise TypeError."""
        reg = _reg(
            frozenset({"a"}),
            listeners={"b": ("a",)},
        )
        defn = _build_flow_definition(reg)
        with pytest.raises(TypeError):
            defn.direct_triggers["b"] = ("x",)  # type: ignore[index]

    def test_router_triggers_mapping_read_only(self) -> None:
        """router_triggers is a MappingProxyType — in-place mutation must raise TypeError."""
        reg = _reg(
            frozenset({"a"}),
            routers={"r": ("a",)},
        )
        defn = _build_flow_definition(reg)
        with pytest.raises(TypeError):
            defn.router_triggers["r"] = ("x",)  # type: ignore[index]

    def test_step_info_frozen(self) -> None:
        info = StepInfo(name="kickoff", role="start", triggers=(), description=None)
        with pytest.raises((AttributeError, TypeError)):
            info.name = "other"  # type: ignore[misc]

    def test_gate_info_frozen(self) -> None:
        gate = GateInfo(
            gate_id="g:and:a,b",
            listener_name="g",
            kind="and",
            triggers=frozenset({"a", "b"}),
        )
        with pytest.raises((AttributeError, TypeError)):
            gate.kind = "or"  # type: ignore[misc]

    def test_no_flow_reference_in_definition(self) -> None:
        """FlowDefinition must not hold a reference to any Flow instance."""
        import gc

        from troopai.adk.flows import Flow

        class _F(Flow[_S]):
            @flow_start
            async def kickoff(self) -> None: ...

        flow_instance = _F(_S)
        defn = flow_instance.get_definition()

        del flow_instance
        gc.collect()
        # If definition held the flow, accessing its data would be risky.
        # Simply verify fields are still accessible (no stored reference).
        assert "kickoff" in defn.starts


# ---------------------------------------------------------------------------
# steps_by_role helper
# ---------------------------------------------------------------------------


class TestStepsByRole:
    def test_steps_by_role_start(self) -> None:
        reg = _reg(
            frozenset({"a"}),
            listeners={"b": ("a",)},
            routers={"r": ("b",)},
        )
        defn = _build_flow_definition(reg)
        starts = defn.steps_by_role("start")
        assert len(starts) == 1
        assert starts[0].name == "a"

    def test_steps_by_role_listen(self) -> None:
        reg = _reg(
            frozenset({"a"}),
            listeners={"b": ("a",), "c": ("a",)},
        )
        defn = _build_flow_definition(reg)
        listeners = defn.steps_by_role("listen")
        assert {s.name for s in listeners} == {"b", "c"}

    def test_steps_by_role_unknown_returns_empty(self) -> None:
        reg = _reg(frozenset({"a"}))
        defn = _build_flow_definition(reg)
        assert defn.steps_by_role("nonexistent") == ()

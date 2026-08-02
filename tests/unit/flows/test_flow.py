"""Unit tests for :class:`Flow` base class and :class:`FlowMeta`.

Covers state initialization, registry stamping, abstract-base detection,
and the no-inheritance contract.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from troopai.adk.flows import Flow, FlowDefinitionError, flow_start


class _State(BaseModel):
    counter: int = 0


class TestFlowInstantiation:
    def test_initial_state_arg(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def kickoff(self) -> None:
                self.state.counter = 1

        s = _State(counter=42)
        f = F(initial_state=s)
        assert f.state is s
        assert f.flow_id.startswith("flow-")

    def test_initial_state_accepts_class(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def kickoff(self) -> None: ...

        f = F(_State)
        assert isinstance(f.state, _State)
        assert f.state.counter == 0

    def test_initial_state_accepts_instance(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def kickoff(self) -> None: ...

        s = _State(counter=99)
        f = F(initial_state=s)
        assert f.state.counter == 99

    def test_no_initial_state_raises(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def kickoff(self) -> None: ...

        with pytest.raises(FlowDefinitionError, match="initial_state"):
            F()


class TestFlowRegistry:
    def test_registry_stamped_on_subclass(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def kickoff(self) -> None: ...

        registry = F.__flow_registry__
        assert registry is not None
        assert "kickoff" in registry.starts

    def test_abstract_base_has_no_registry(self) -> None:
        assert Flow.__flow_registry__ is None

    def test_subclass_without_decorations_raises(self) -> None:
        # A Flow subclass MUST have at least one @flow_start
        with pytest.raises(FlowDefinitionError, match="at least one @flow_start"):

            class _Bad(Flow[_State]):
                pass


class TestFlowIDIsUnique:
    def test_distinct_ids(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def kickoff(self) -> None: ...

        f1 = F(_State)
        f2 = F(_State)
        assert f1.flow_id != f2.flow_id


class TestNoInheritance:
    def test_parent_decorated_method_not_inherited(self) -> None:
        # Decorated methods on a parent are NOT registered on the child.
        class Parent(Flow[_State]):
            @flow_start
            async def parent_step(self) -> None: ...

        # Child without its own @flow_start should fail.
        with pytest.raises(FlowDefinitionError, match="at least one @flow_start"):

            class _Child(Parent):
                pass


class TestGetRegistry:
    def test_get_registry_returns_registry(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def kickoff(self) -> None: ...

        f = F(_State)
        reg = f.get_registry()
        assert "kickoff" in reg.starts

    def test_abstract_base_instantiation_raises(self) -> None:
        # Direct Flow() instantiation isn't sensible — no initial_state
        # supplied, so __init__ raises a clear error.
        with pytest.raises(FlowDefinitionError, match="initial_state"):
            Flow()  # type: ignore[type-var]

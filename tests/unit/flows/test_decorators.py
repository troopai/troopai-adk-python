"""Unit tests for ``@flow_start`` / ``@flow_listen`` / ``@flow_router`` decorators.

Covers the markers set on :class:`FlowStep`, signature validation
in :class:`FlowMeta`, and the operator-built combinators.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from troopai.adk.flows import (
    And,
    Flow,
    FlowDefinitionError,
    FlowStep,
    Or,
    flow_listen,
    flow_router,
    flow_start,
)


class _S(BaseModel):
    x: int = 0


class TestStartDecorator:
    def test_returns_flow_step(self) -> None:
        async def fn(self) -> None: ...

        step = flow_start(fn)
        assert isinstance(step, FlowStep)
        assert step.__flow_role__ == "start"
        assert step.__flow_triggers__ == ()

    def test_preserves_name(self) -> None:
        async def my_step(self) -> None: ...

        step = flow_start(my_step)
        assert step.__name__ == "my_step"


class TestListenDecorator:
    def test_string_trigger(self) -> None:
        @flow_listen("event_a")
        async def my_step(self) -> None: ...

        assert isinstance(my_step, FlowStep)
        assert my_step.__flow_role__ == "listen"
        assert my_step.__flow_triggers__ == ("event_a",)

    def test_method_ref_trigger(self) -> None:
        @flow_start
        async def upstream(self) -> None: ...

        @flow_listen(upstream)
        async def downstream(self) -> None: ...

        assert downstream.__flow_triggers__ == ("upstream",)

    def test_or_gate_trigger(self) -> None:
        @flow_start
        async def a(self) -> None: ...

        @flow_start
        async def b(self) -> None: ...

        @flow_listen(a | b)
        async def merged(self) -> None: ...

        triggers = merged.__flow_triggers__
        assert len(triggers) == 1
        gate = triggers[0]
        assert isinstance(gate, Or)
        assert set(gate.triggers) == {"a", "b"}

    def test_and_gate_trigger(self) -> None:
        @flow_start
        async def a(self) -> None: ...

        @flow_start
        async def b(self) -> None: ...

        @flow_listen(a & b)
        async def merged(self) -> None: ...

        gate = merged.__flow_triggers__[0]
        assert isinstance(gate, And)
        assert set(gate.triggers) == {"a", "b"}


class TestRouterDecorator:
    def test_returns_flow_step(self) -> None:
        @flow_router("upstream")
        async def route_fn(self) -> str:
            return "label"

        assert isinstance(route_fn, FlowStep)
        assert route_fn.__flow_role__ == "router"


class TestSignatureValidation:
    def test_non_async_rejected(self) -> None:
        with pytest.raises(FlowDefinitionError, match="async def"):

            class _Bad(Flow[_S]):
                @flow_start
                def kickoff(self) -> None: ...

    def test_extra_param_rejected(self) -> None:
        with pytest.raises(FlowDefinitionError, match="no auto-injected args"):

            class _Bad(Flow[_S]):
                @flow_start
                async def kickoff(self, prev_result) -> None: ...

    def test_classmethod_rejected(self) -> None:
        with pytest.raises(FlowDefinitionError, match="plain async method"):

            class _Bad(Flow[_S]):
                @classmethod
                @flow_start
                async def kickoff(cls) -> None: ...  # type: ignore[misc]


class TestDecoratorAtMostOneRole:
    def test_flow_step_constructed_once_per_decorator(self) -> None:
        # Two decorators applied → both produce FlowStep. The outer wins.
        @flow_listen("a")
        @flow_start
        async def fn(self) -> None: ...

        # The OUTER decorator's role wins because each decorator produces
        # a new FlowStep around the previous result. Note: This is not a
        # supported usage pattern — documented for completeness.
        assert isinstance(fn, FlowStep)

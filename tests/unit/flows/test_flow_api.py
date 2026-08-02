"""Unit tests for the Flow public API-readability surface.

Covers the introspection and ergonomics additions:

- :class:`Flow` / :class:`FlowRunResult` compact one-line ``__repr__``s.
- :class:`FlowStep` ``role`` / ``triggers`` / ``approval_policy``
  read-only properties.
- ``FLOW_ERROR_TRIGGER`` — the ``"__error__"`` route-literal constant.
- ``Flow.state_factory`` class attribute as an explicit state path.
- :class:`FlowCheckpoint` ``approve`` / ``reject`` accepting a bare
  step-name string.
- ``approval_policy=`` decorator kwarg plumbing through to
  :class:`FlowDeferredStep.policy`.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from troopai.adk.flows import (
    Flow,
    FlowApprovalPolicy,
    FlowCheckpoint,
    FlowConfig,
    FlowDeferredStep,
    FlowDefinitionError,
    FlowRunResult,
    flow_listen,
    flow_router,
    flow_start,
)
from troopai.adk.flows.executor import FlowExecutor
from troopai.adk.run.runner import Runner
from troopai.adk.types.tokens.llm_usage import LLMUsage


class _State(BaseModel):
    events: list[str] = []


class TestFlowRepr:
    def test_repr_compact_one_liner(self) -> None:
        class ResearchFlow(Flow[_State]):
            @flow_start
            async def kickoff(self) -> None: ...

            @flow_listen(kickoff)
            async def research(self) -> None: ...

            @flow_router(research)
            async def route(self) -> str:
                return "done"

        flow = ResearchFlow(_State)
        assert repr(flow) == (f"ResearchFlow(flow_id={flow.flow_id!r}, steps=3, routers=1, state=_State)")

    def test_repr_flow_without_router(self) -> None:
        class TinyFlow(Flow[_State]):
            @flow_start
            async def only(self) -> None: ...

        flow = TinyFlow(_State)
        assert repr(flow) == f"TinyFlow(flow_id={flow.flow_id!r}, steps=1, routers=0, state=_State)"

    def test_repr_never_dumps_state_fields(self) -> None:
        class SecretState(BaseModel):
            token: str = "super-secret-value"

        class F(Flow[SecretState]):
            @flow_start
            async def a(self) -> None: ...

        flow = F(SecretState)
        assert "super-secret-value" not in repr(flow)


class TestFlowRunResultRepr:
    def test_repr_compact_one_liner(self) -> None:
        result = FlowRunResult(
            final_state=_State(events=["a", "b"]),
            flow_id="flow-deadbeef",
            status="completed",
            completed_steps=("a", "b"),
            cumulative_usage=LLMUsage(),
        )
        text = repr(result)
        assert text.startswith("FlowRunResult(flow_id='flow-deadbeef', status='completed', steps=2, final_state=")

    def test_repr_caps_state_preview(self) -> None:
        result = FlowRunResult(
            final_state="y" * 500,
            flow_id="flow-1",
            status="completed",
            completed_steps=(),
            cumulative_usage=LLMUsage(),
        )
        text = repr(result)
        assert "…" in text
        assert len(text) < 200

    def test_repr_strips_newlines_from_string_state(self) -> None:
        result = FlowRunResult(
            final_state="line1\nline2",
            flow_id="flow-1",
            status="completed",
            completed_steps=(),
            cumulative_usage=LLMUsage(),
        )
        assert "\n" not in repr(result)

    async def test_repr_on_real_run(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None: ...

        result = await Runner.arun_flow(F(_State))
        assert repr(result).startswith(f"FlowRunResult(flow_id={result.flow_id!r}, status='completed', steps=1,")


class TestFlowStepRoleAndTriggers:
    def test_role_property_on_class_descriptors(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None: ...

            @flow_listen(a)
            async def b(self) -> None: ...

            @flow_router(b)
            async def r(self) -> str:
                return "done"

        assert F.a.role == "start"
        assert F.b.role == "listen"
        assert F.r.role == "router"

    def test_triggers_property_on_class_descriptors(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None: ...

            @flow_listen(a)
            async def b(self) -> None: ...

        assert F.a.triggers == ()
        assert F.b.triggers == ("a",)

    def test_properties_available_on_bound_instances(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None: ...

            @flow_listen(a)
            async def b(self) -> None: ...

        flow = F(_State)
        assert flow.a.role == "start"
        assert flow.b.role == "listen"
        assert flow.b.triggers == ("a",)


class TestFlowErrorTrigger:
    def test_constant_is_the_error_route_literal(self) -> None:
        from troopai.adk.flows import FLOW_ERROR_TRIGGER

        assert FLOW_ERROR_TRIGGER == "__error__"

    def test_constant_reexported_from_top_level_package(self) -> None:
        import troopai.adk as adk
        from troopai.adk.flows import FLOW_ERROR_TRIGGER

        assert adk.FLOW_ERROR_TRIGGER == FLOW_ERROR_TRIGGER

    async def test_error_handler_fires_via_constant(self) -> None:
        from troopai.adk.flows import FLOW_ERROR_TRIGGER

        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None:
                raise RuntimeError("boom")

            @flow_listen(FLOW_ERROR_TRIGGER)
            async def handle(self) -> None:
                self.state.events.append("handled")

        config = FlowConfig(error_policy="route_to_error_handler")
        result = await FlowExecutor(F(_State), config=config).run()

        assert result.status == "completed"
        assert result.final_state.events == ["handled"]


class TestStateFactory:
    def test_state_factory_constructs_with_zero_args(self) -> None:
        class F(Flow[_State]):
            state_factory = _State

            @flow_start
            async def a(self) -> None: ...

        flow = F()
        assert isinstance(flow.state, _State)

    def test_state_factory_called_per_instance(self) -> None:
        class F(Flow[_State]):
            state_factory = _State

            @flow_start
            async def a(self) -> None: ...

        f1, f2 = F(), F()
        assert f1.state is not f2.state

    def test_initial_state_wins_over_state_factory(self) -> None:
        class F(Flow[_State]):
            state_factory = _State

            @flow_start
            async def a(self) -> None: ...

        explicit = _State(events=["seed"])
        flow = F(initial_state=explicit)
        assert flow.state is explicit

    def test_neither_path_raises_naming_both(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None: ...

        with pytest.raises(FlowDefinitionError, match="initial_state") as exc_info:
            F()
        assert "state_factory" in str(exc_info.value)

    def test_non_callable_state_factory_raises(self) -> None:
        class F(Flow[_State]):
            state_factory = _State(events=["x"])  # type: ignore[assignment]  # intentional misconfiguration

            @flow_start
            async def a(self) -> None: ...

        with pytest.raises(FlowDefinitionError, match="state_factory"):
            F()

    def test_initial_state_wins_even_when_state_factory_invalid(self) -> None:
        """Explicit initial_state short-circuits the factory — it is never even inspected."""

        class F(Flow[_State]):
            state_factory = _State(events=["x"])  # type: ignore[assignment]  # intentional misconfiguration

            @flow_start
            async def a(self) -> None: ...

        explicit = _State(events=["seed"])
        flow = F(initial_state=explicit)
        assert flow.state is explicit


def _checkpoint_with_two_deferred() -> FlowCheckpoint:
    return FlowCheckpoint(
        flow_id="flow-1",
        completed_steps=(),
        pending_steps=("a", "b"),
        and_gate_arrivals={},
        consumed_gates=(),
        state_data="{}",
        deferred_steps=(
            FlowDeferredStep(step_name="a"),
            FlowDeferredStep(step_name="b"),
        ),
    )


class TestCheckpointDecisionByName:
    def test_approve_by_step_name(self) -> None:
        cp = _checkpoint_with_two_deferred()
        cp.approve("a")
        assert cp.decisions["a"].approved is True

    def test_reject_by_step_name(self) -> None:
        cp = _checkpoint_with_two_deferred()
        cp.reject("b", message="no")
        assert cp.decisions["b"].approved is False
        assert cp.decisions["b"].message == "no"

    def test_approve_unknown_name_raises_with_valid_names(self) -> None:
        cp = _checkpoint_with_two_deferred()
        with pytest.raises(ValueError, match="FlowCheckpoint.approve: step 'x' is not in deferred_steps") as exc_info:
            cp.approve("x")
        assert "['a', 'b']" in str(exc_info.value)

    def test_reject_unknown_name_raises_with_valid_names(self) -> None:
        cp = _checkpoint_with_two_deferred()
        with pytest.raises(ValueError, match="FlowCheckpoint.reject: step 'x' is not in deferred_steps") as exc_info:
            cp.reject("x")
        assert "['a', 'b']" in str(exc_info.value)

    def test_approve_object_still_works(self) -> None:
        cp = _checkpoint_with_two_deferred()
        cp.approve(cp.deferred_steps[0])
        assert cp.decisions["a"].approved is True

    def test_reject_object_still_works(self) -> None:
        cp = _checkpoint_with_two_deferred()
        cp.reject(cp.deferred_steps[1], message="nope")
        assert cp.decisions["b"].message == "nope"

    def test_approve_wrong_type_raises_type_error(self) -> None:
        cp = _checkpoint_with_two_deferred()
        with pytest.raises(TypeError, match="FlowCheckpoint.approve"):
            cp.approve(123)  # type: ignore[arg-type]  # intentional misuse

    def test_reject_wrong_type_raises_type_error(self) -> None:
        cp = _checkpoint_with_two_deferred()
        with pytest.raises(TypeError, match="FlowCheckpoint.reject"):
            cp.reject(None)  # type: ignore[arg-type]  # intentional misuse


class TestApprovalPolicyWiring:
    def test_policy_stored_on_step_descriptor(self) -> None:
        policy = FlowApprovalPolicy(quorum=2)

        class F(Flow[_State]):
            @flow_start(requires_approval=True, approval_policy=policy)
            async def a(self) -> None: ...

        assert F.a.approval_policy is policy

    def test_policy_available_on_bound_instance(self) -> None:
        policy = FlowApprovalPolicy(quorum=2)

        class F(Flow[_State]):
            @flow_start(requires_approval=True, approval_policy=policy)
            async def a(self) -> None: ...

        assert F(_State).a.approval_policy is policy

    def test_policy_defaults_to_none(self) -> None:
        class F(Flow[_State]):
            @flow_start(requires_approval=True)
            async def a(self) -> None: ...

        assert F.a.approval_policy is None

    def test_listen_and_router_accept_policy(self) -> None:
        policy = FlowApprovalPolicy(quorum=3)

        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None: ...

            @flow_listen(a, approval_policy=policy)
            async def b(self) -> None: ...

            @flow_router(b, approval_policy=policy)
            async def r(self) -> str:
                return "done"

        assert F.b.approval_policy is policy
        assert F.r.approval_policy is policy

    async def test_deferred_step_carries_policy(self) -> None:
        class F(Flow[_State]):
            @flow_start(requires_approval=True, approval_policy=FlowApprovalPolicy(quorum=2))
            async def a(self) -> None: ...

        result = await FlowExecutor(F(_State), config=FlowConfig()).run()

        assert result.status == "deferred"
        assert len(result.deferred_steps) == 1
        deferred = result.deferred_steps[0]
        assert deferred.policy is not None
        assert deferred.policy.quorum == 2

    async def test_deferred_step_policy_none_without_kwarg(self) -> None:
        class F(Flow[_State]):
            @flow_start(requires_approval=True)
            async def a(self) -> None: ...

        result = await FlowExecutor(F(_State), config=FlowConfig()).run()

        assert result.status == "deferred"
        assert result.deferred_steps[0].policy is None

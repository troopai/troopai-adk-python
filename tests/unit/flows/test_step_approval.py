"""Unit tests for step-level HITL + enabled / max_retries / timeout.

Mirrors the contract of the function-tool HITL surface — decisions
recorded on :class:`FlowCheckpoint` via
:meth:`FlowCheckpoint.approve` / :meth:`FlowCheckpoint.reject`, then
resumed via :meth:`Runner.arun_flow_from_checkpoint(flow, checkpoint)`.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, Field

from troopai.adk.flows import (
    Flow,
    FlowApprovalDecision,
    FlowApprovalPolicy,
    FlowCheckpoint,
    FlowDeferredStep,
    FlowStepContext,
    flow_listen,
    flow_start,
)
from troopai.adk.run.runner import Runner


class _CartState(BaseModel):
    """Tiny pydantic state object for flow tests."""

    amount: int = 0
    notes: list[str] = Field(default_factory=list)
    reviewed: bool = False
    refunded: bool = False


class TestRequiresApprovalDefersRun:
    async def test_high_amount_step_defers_with_checkpoint(self) -> None:
        def gate(ctx: FlowStepContext[_CartState]) -> bool:
            return ctx.flow_state.amount >= 1000

        class CartFlow(Flow[_CartState]):
            @flow_start
            async def intake(self) -> None:
                self.state.amount = 1500
                self.state.notes.append("intake")

            @flow_listen("intake", requires_approval=gate)
            async def refund(self) -> None:
                self.state.refunded = True

        flow = CartFlow(_CartState)
        result = await Runner.arun_flow(flow)

        assert result.status == "deferred"
        assert result.requires_action is True
        assert len(result.deferred_steps) == 1
        assert result.deferred_steps[0].step_name == "refund"
        assert result.checkpoint is not None
        assert result.checkpoint.requires_action is True
        # The refund body never ran:
        assert result.final_state.refunded is False
        # The intake body did run:
        assert result.final_state.amount == 1500

    async def test_streamed_high_amount_step_defers_with_checkpoint(self) -> None:
        """Streamed flow HITL must populate deferred_steps + checkpoint.

        Regression: ``_drive_flow_stream`` copied status / state / usage /
        error from the executor's final result but NOT ``deferred_steps`` or
        ``checkpoint``. A streamed flow that deferred therefore reported
        ``requires_action == False`` (it is ``len(deferred_steps) > 0``) with
        ``checkpoint is None`` — streamed HITL was completely unrecoverable.
        """

        def gate(ctx: FlowStepContext[_CartState]) -> bool:
            return ctx.flow_state.amount >= 1000

        class CartFlow(Flow[_CartState]):
            @flow_start
            async def intake(self) -> None:
                self.state.amount = 1500

            @flow_listen("intake", requires_approval=gate)
            async def refund(self) -> None:
                self.state.refunded = True

        flow = CartFlow(_CartState)
        result = Runner.arun_flow_streamed(flow)
        async for _ in result.stream_events():
            pass

        assert result.status == "deferred"
        assert result.requires_action is True
        assert len(result.deferred_steps) == 1
        assert result.deferred_steps[0].step_name == "refund"
        assert result.checkpoint is not None

    async def test_low_amount_step_runs_to_completion(self) -> None:
        def gate(ctx: FlowStepContext[_CartState]) -> bool:
            return ctx.flow_state.amount >= 1000

        class CartFlow(Flow[_CartState]):
            @flow_start
            async def intake(self) -> None:
                self.state.amount = 50

            @flow_listen("intake", requires_approval=gate)
            async def refund(self) -> None:
                self.state.refunded = True

        flow = CartFlow(_CartState)
        result = await Runner.arun_flow(flow)

        assert result.status == "completed"
        assert result.requires_action is False
        assert result.final_state.refunded is True


class TestCheckpointApproveReject:
    async def test_approve_then_resume_runs_the_step(self) -> None:
        class CartFlow(Flow[_CartState]):
            @flow_start
            async def intake(self) -> None:
                self.state.amount = 2000

            @flow_listen("intake", requires_approval=True)
            async def big_refund(self) -> None:
                self.state.refunded = True

        flow = CartFlow(_CartState)
        result = await Runner.arun_flow(flow)
        assert result.status == "deferred"
        assert result.checkpoint is not None

        checkpoint = result.checkpoint
        checkpoint.approve(
            result.deferred_steps[0],
            approver_id="alice@example.com",
            approver_role="ops_lead",
            reason="amount within manual-review band",
        )

        # Rebuild flow with the same state and resume.
        resumed_state = _CartState.model_validate_json(checkpoint.state_data)
        resumed_flow = CartFlow(resumed_state)
        resumed = await Runner.arun_flow_from_checkpoint(resumed_flow, checkpoint)

        assert resumed.status == "completed"
        assert resumed.final_state.refunded is True
        # Audit trail preserved.
        decision = checkpoint.decisions["big_refund"]
        assert decision.approved is True
        assert decision.approver_id == "alice@example.com"
        assert decision.approver_role == "ops_lead"
        assert decision.reason == "amount within manual-review band"
        assert decision.status == "approved"

    async def test_reject_routes_through_error_listener(self) -> None:
        class CartFlow(Flow[_CartState]):
            @flow_start
            async def intake(self) -> None:
                self.state.amount = 5000

            @flow_listen("intake", requires_approval=True)
            async def big_refund(self) -> None:
                self.state.refunded = True

            @flow_listen("__error__")
            async def on_error(self) -> None:
                self.state.notes.append("error_handled")

        from troopai.adk.flows import FlowConfig

        flow = CartFlow(_CartState)
        result = await Runner.arun_flow(
            flow,
            config=FlowConfig(error_policy="route_to_error_handler"),
        )
        assert result.checkpoint is not None

        checkpoint = result.checkpoint
        checkpoint.reject(
            result.deferred_steps[0],
            "fraud signal exceeded threshold",
            approver_id="bob@example.com",
        )

        resumed_state = _CartState.model_validate_json(checkpoint.state_data)
        resumed_flow = CartFlow(resumed_state)
        resumed = await Runner.arun_flow_from_checkpoint(
            resumed_flow,
            checkpoint,
            config=FlowConfig(error_policy="route_to_error_handler"),
        )

        assert resumed.final_state.refunded is False
        assert "error_handled" in resumed.final_state.notes
        decision = checkpoint.decisions["big_refund"]
        assert decision.approved is False
        assert decision.message == "fraud signal exceeded threshold"
        assert decision.status == "rejected"

    def test_approve_unknown_step_raises(self) -> None:
        unknown = FlowDeferredStep(step_name="ghost")
        checkpoint = FlowCheckpoint(
            flow_id="f",
            completed_steps=(),
            pending_steps=(),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data="{}",
        )
        with pytest.raises(ValueError, match="not in deferred_steps"):
            checkpoint.approve(unknown)


class TestEnabledGateSkipsStep:
    async def test_enabled_false_skips_body_and_successors(self) -> None:
        class CartFlow(Flow[_CartState]):
            @flow_start
            async def intake(self) -> None:
                self.state.amount = 1

            @flow_listen("intake", enabled=False)
            async def disabled(self) -> None:
                self.state.notes.append("ran_disabled")

            @flow_listen("disabled")
            async def after_disabled(self) -> None:
                self.state.notes.append("ran_after_disabled")

        flow = CartFlow(_CartState)
        result = await Runner.arun_flow(flow)

        assert result.status == "completed"
        assert "ran_disabled" not in result.final_state.notes
        assert "ran_after_disabled" not in result.final_state.notes


class TestRetryAndTimeout:
    async def test_max_retries_retries_until_success(self) -> None:
        attempts = {"n": 0}

        class CartFlow(Flow[_CartState]):
            @flow_start
            async def flaky(self) -> None:
                attempts["n"] += 1
                if attempts["n"] < 3:
                    raise RuntimeError("transient")
                self.state.amount = 99

        # The decorator stores max_retries; we set it via decorator kwarg.
        from troopai.adk.flows.flow_wrappers import FlowStep

        # Patch the descriptor's max_retries to ensure retry path.
        step = CartFlow.__dict__["flaky"]
        assert isinstance(step, FlowStep)
        step.__flow_max_retries__ = 3

        flow = CartFlow(_CartState)
        result = await Runner.arun_flow(flow)

        assert result.status == "completed"
        assert result.final_state.amount == 99
        assert attempts["n"] == 3

    async def test_timeout_routes_through_error_policy(self) -> None:
        class SlowFlow(Flow[_CartState]):
            @flow_start
            async def slow(self) -> None:
                await asyncio.sleep(0.5)

        from troopai.adk.flows.flow_wrappers import FlowStep

        step = SlowFlow.__dict__["slow"]
        assert isinstance(step, FlowStep)
        step.__flow_timeout__ = 0.05

        flow = SlowFlow(_CartState)
        result = await Runner.arun_flow(flow)
        assert result.status == "failed"
        assert result.error is not None and "TimeoutError" in result.error


class TestApprovalPolicyEnrichment:
    def test_policy_validates_quorum_positive(self) -> None:
        with pytest.raises(ValueError, match="quorum"):
            FlowApprovalPolicy(quorum=0)

    def test_policy_validates_deadline_positive(self) -> None:
        with pytest.raises(ValueError, match="deadline_seconds"):
            FlowApprovalPolicy(deadline_seconds=0)

    def test_policy_round_trips_on_checkpoint(self) -> None:
        policy = FlowApprovalPolicy(
            quorum=2,
            allowed_roles=("ops_lead", "compliance_lead"),
            deadline_seconds=300.0,
        )
        deferred = FlowDeferredStep(
            step_name="big_step",
            policy=policy,
            metadata={"ticket": "OPS-42"},
        )
        cp = FlowCheckpoint(
            flow_id="f",
            completed_steps=(),
            pending_steps=(),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data="{}",
            deferred_steps=(deferred,),
        )
        rehydrated = FlowCheckpoint.from_json(cp.to_json())
        out = rehydrated.deferred_steps[0]
        assert out.policy is not None
        assert out.policy.quorum == 2
        assert out.policy.allowed_roles == ("ops_lead", "compliance_lead")
        assert out.policy.deadline_seconds == 300.0
        assert out.metadata == {"ticket": "OPS-42"}


class TestDecisionRoundTrip:
    def test_decision_round_trips_through_checkpoint_json(self) -> None:
        deferred = FlowDeferredStep(step_name="big_step")
        cp = FlowCheckpoint(
            flow_id="f",
            completed_steps=(),
            pending_steps=(),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data="{}",
            deferred_steps=(deferred,),
        )
        cp.approve(deferred, approver_id="alice", reason="approved during business hours")
        rehydrated = FlowCheckpoint.from_json(cp.to_json())
        decision = rehydrated.decisions["big_step"]
        assert decision.approved is True
        assert decision.approver_id == "alice"
        assert decision.reason == "approved during business hours"
        assert decision.status == "approved"

    def test_status_expired_for_sla_flag(self) -> None:
        d = FlowApprovalDecision(
            step_name="x",
            approved=False,
            expired=True,
        )
        assert d.status == "expired"

    def test_expired_round_trips_through_json(self) -> None:
        deferred = FlowDeferredStep(step_name="x")
        cp = FlowCheckpoint(
            flow_id="f",
            completed_steps=(),
            pending_steps=(),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data="{}",
            deferred_steps=(deferred,),
        )
        cp.decisions["x"] = FlowApprovalDecision(step_name="x", approved=False, expired=True, message="SLA exceeded")
        rehydrated = FlowCheckpoint.from_json(cp.to_json())
        assert rehydrated.decisions["x"].expired is True
        assert rehydrated.decisions["x"].status == "expired"

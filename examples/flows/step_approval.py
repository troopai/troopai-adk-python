"""Step-level HITL example — approve / reject a sensitive Flow step.

Demonstrates the full HITL round-trip using the typed primitives:

1. A flow whose ``refund`` step is gated by a callable
   ``requires_approval`` over the cart amount.
2. A first run with a high-amount cart that defers, captured into a
   :class:`FlowCheckpoint`.
3. Two resume paths: one that approves the refund, one that rejects
   it with a routed message picked up by a
   ``@flow_listen(FLOW_ERROR_TRIGGER)`` handler.

The example is synthetic — no agent / LLM call — so it runs without
any provider API key. Its purpose is to exercise the HITL surface
shape, not to demonstrate a realistic agent workflow.

Mirrors the tool-HITL contract:
- decisions baked into the resume payload (the checkpoint), not a
  separate ``approvals=`` kwarg;
- :meth:`FlowCheckpoint.approve` / :meth:`FlowCheckpoint.reject`
  share the exact signature of :meth:`RunState.approve` /
  :meth:`RunState.reject`.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from troopai.adk.flows import (
    FLOW_ERROR_TRIGGER,
    Flow,
    FlowApprovalPolicy,
    FlowConfig,
    FlowStepContext,
    flow_listen,
    flow_start,
)
from troopai.adk.run.runner import Runner

logger = logging.getLogger(__name__)


class CartState(BaseModel):
    """Tiny pydantic state object the flow operates on."""

    amount: int = 0
    refunded: bool = False
    notes: list[str] = Field(default_factory=list)


def needs_human_review(ctx: FlowStepContext[CartState]) -> bool:
    """Defer the refund when the amount exceeds 1000."""
    return ctx.flow_state.amount >= 1000


class RefundFlow(Flow[CartState]):
    """Two-step flow: intake → refund (HITL-gated)."""

    @flow_start
    async def intake(self) -> None:
        self.state.amount = 2500
        self.state.notes.append(f"intake: amount={self.state.amount}")

    @flow_listen(
        "intake",
        requires_approval=needs_human_review,
        # Informational policy — rides on FlowDeferredStep.policy through
        # the checkpoint so an approval driver can enforce roles/quorum.
        # The executor does NOT evaluate it.
        approval_policy=FlowApprovalPolicy(allowed_roles=("ops_lead", "compliance_lead")),
    )
    async def refund(self) -> None:
        self.state.refunded = True
        self.state.notes.append("refund issued")

    @flow_listen(FLOW_ERROR_TRIGGER)
    async def on_error(self) -> None:
        self.state.notes.append("error_handler: refund blocked")


async def _approve_path() -> None:
    """Run the flow, approve the refund, resume to completion."""
    flow = RefundFlow(CartState)
    result = await Runner.arun_flow(flow)
    assert result.status == "deferred"
    assert result.requires_action is True
    assert result.checkpoint is not None

    logger.info("APPROVE path — initial run deferred at %r", result.deferred_steps[0].step_name)

    checkpoint = result.checkpoint
    # Decisions accept the pending step's NAME (or its FlowDeferredStep).
    checkpoint.approve(
        "refund",
        approver_id="alice@example.com",
        approver_role="ops_lead",
        reason="amount within manual-review band; standard refund",
    )

    # Rehydrate the state and resume.
    resumed_state = CartState.model_validate_json(checkpoint.state_data)
    resumed_flow = RefundFlow(resumed_state)
    resumed = await Runner.arun_flow_from_checkpoint(resumed_flow, checkpoint)

    logger.info("APPROVE path — resumed status=%s", resumed.status)
    logger.info("APPROVE path — final_state=%s", resumed.final_state.model_dump())


async def _reject_path() -> None:
    """Run the flow, reject the refund, resume so the error handler fires."""
    flow = RefundFlow(CartState)
    result = await Runner.arun_flow(
        flow,
        config=FlowConfig(error_policy="route_to_error_handler"),
    )
    assert result.checkpoint is not None
    logger.info("REJECT path — initial run deferred at %r", result.deferred_steps[0].step_name)

    checkpoint = result.checkpoint
    checkpoint.reject(
        "refund",
        "fraud signal exceeded threshold",
        approver_id="bob@example.com",
        approver_role="compliance_lead",
    )

    resumed_state = CartState.model_validate_json(checkpoint.state_data)
    resumed_flow = RefundFlow(resumed_state)
    resumed = await Runner.arun_flow_from_checkpoint(
        resumed_flow,
        checkpoint,
        config=FlowConfig(error_policy="route_to_error_handler"),
    )

    logger.info("REJECT path — resumed status=%s", resumed.status)
    logger.info("REJECT path — final_state=%s", resumed.final_state.model_dump())


async def main() -> None:
    await _approve_path()
    await _reject_path()


if __name__ == "__main__":
    asyncio.run(main())

"""Step-level governance example — rate limit + guardrails + cache on one flow.

Demonstrates the three Tier-2 polymorphic-polish attributes:

1. ``rate_limit`` — per-step sliding-window cap (60-second window).
2. ``guardrails`` — typed pre/post verdict surface (allow / reject /
   raise) attached to a step.
3. ``cache`` — declarative result cache: a hit short-circuits the
   body and restores the cached state snapshot.

Synthetic-only (no agent / LLM call) so the example runs without
any provider API key. Its purpose is to exercise the three
attribute paths end-to-end, not to demonstrate a realistic workflow.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from troopai.adk.flows import (
    Flow,
    FlowConfig,
    FlowStepCachePolicy,
    FlowStepContext,
    FlowStepGuardrails,
    FlowStepGuardrailVerdict,
    FlowStepRateLimit,
    flow_listen,
    flow_start,
)
from troopai.adk.run.runner import Runner

logger = logging.getLogger(__name__)


class PipelineState(BaseModel):
    amount: int = 0
    classification: str = ""
    notes: list[str] = []


def _amount_under_threshold(ctx: FlowStepContext[PipelineState]) -> FlowStepGuardrailVerdict:
    """Pre-step guardrail — reject when the amount is above 10_000."""
    if ctx.flow_state.amount > 10_000:
        return FlowStepGuardrailVerdict.reject_content(
            f"amount {ctx.flow_state.amount} exceeds risk threshold",
        )
    return FlowStepGuardrailVerdict.allow()


def _classification_key(ctx: FlowStepContext[PipelineState]) -> str:
    """Cache key — derives from the amount tier so repeated tiers hit the cache."""
    if ctx.flow_state.amount < 100:
        return "small"
    if ctx.flow_state.amount < 1000:
        return "medium"
    return "large"


class PipelineFlow(Flow[PipelineState]):
    @flow_start(
        description="Intake the request.",
        rate_limit=FlowStepRateLimit(rpm=120),
    )
    async def intake(self) -> None:
        self.state.notes.append(f"intake amount={self.state.amount}")

    @flow_listen(
        "intake",
        description="Risk-check the amount.",
        guardrails=FlowStepGuardrails(pre=(_amount_under_threshold,)),
    )
    async def risk_check(self) -> None:
        self.state.notes.append("risk_check passed")

    @flow_listen(
        "risk_check",
        description="Classify the request by amount tier (cached).",
        cache=FlowStepCachePolicy(cache_key_fn=_classification_key),
    )
    async def classify(self) -> None:
        # Simulated expensive computation — cached by amount tier.
        if self.state.amount < 100:
            self.state.classification = "tier_small"
        elif self.state.amount < 1000:
            self.state.classification = "tier_medium"
        else:
            self.state.classification = "tier_large"
        self.state.notes.append(f"classify ran (computed tier={self.state.classification})")

    @flow_listen("__error__")
    async def on_error(self) -> None:
        self.state.notes.append("error_handler_fired")


async def _accepted() -> None:
    logger.info("--- accepted path (amount=500) ---")
    flow = PipelineFlow(PipelineState(amount=500))
    result = await Runner.arun_flow(flow)
    logger.info(
        "status=%s classification=%s notes=%s",
        result.status,
        result.final_state.classification,
        result.final_state.notes,
    )


async def _rejected() -> None:
    logger.info("--- guardrail-rejected path (amount=50_000) ---")
    flow = PipelineFlow(PipelineState(amount=50_000))
    result = await Runner.arun_flow(
        flow,
        config=FlowConfig(error_policy="route_to_error_handler"),
    )
    logger.info("status=%s notes=%s", result.status, result.final_state.notes)


async def main() -> None:
    await _accepted()
    await _rejected()


if __name__ == "__main__":
    asyncio.run(main())

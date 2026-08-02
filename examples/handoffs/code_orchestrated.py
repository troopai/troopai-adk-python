"""
Example: Code-Orchestrated Handoffs (Intent-Based Routing)

Deterministic routing via Intent type matching — zero LLM routing tokens.
The triage LLM outputs a structured Intent, and HandoffRoute maps it to
an agent purely in Python. No ``transfer_to_*`` tools are generated.

This is the most cost-efficient handoff pattern: the triage agent's ONLY
job is to classify the user's request into an Intent type. The routing
logic is handled deterministically by your code.

Key points:
- Agent.output_schema is a Union of Intent types (the LLM outputs one).
- HandoffRoute maps Intent types → agents via ``.when().to()`` chain.
- ``.otherwise()`` catches any unmatched intents.
- Respond is a special type that means "no handoff, respond directly."

Patterns shown:
1. Basic Intent routing with three specialists.
2. Multi-intent matching (multiple intents → same agent).
3. Respond fallback (triage agent answers directly when no handoff needed).
4. Factory function shorthand.
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from typing import Literal, Union

from litellm.cost_calculator import cost_per_token
from pydantic import Field

from troopai.adk.agents import Agent
from troopai.adk.handoffs import HandoffRoute, handoff_route
from troopai.adk.llms.llm_usage import LLMUsage
from troopai.adk.run import RunConfig, Runner
from troopai.adk.run.config import DEFAULT_MODEL
from troopai.adk.types.intents import Intent, Respond
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


def print_usage(label: str, usage: LLMUsage) -> None:
    """Print token usage and estimated cost for a pattern run."""
    prompt_cost, completion_cost = cost_per_token(
        DEFAULT_MODEL,
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
    )
    total_cost = prompt_cost + completion_cost
    logger.info(f"  [{label}] Token Usage:")
    logger.info(f"    Requests:      {usage.requests}")
    logger.info(f"    Input tokens:  {usage.input_tokens:,}")
    logger.info(f"    Output tokens: {usage.output_tokens:,}")
    logger.info(f"    Total tokens:  {usage.total_tokens:,}")
    if usage.input_tokens_details.cached_tokens > 0:
        logger.info(f"    Cached tokens: {usage.input_tokens_details.cached_tokens:,}")
    logger.info(f"    Est. cost:     ${total_cost:.6f}")


# =============================================================================
# Intent definitions
# =============================================================================
# Each Intent subclass represents a classified user request.
# The LLM fills in the fields; HandoffRoute matches on the type.


class RefundIntent(Intent):
    """User wants a refund or return."""

    kind: Literal["refund"] = "refund"
    order_id: str | None = Field(None, description="Order ID if mentioned.")
    reason: str | None = Field(None, description="Reason for the refund.")


class BillingIntent(Intent):
    """User has a billing question (invoices, charges, payments)."""

    kind: Literal["billing"] = "billing"
    topic: str | None = Field(None, description="Specific billing topic.")


class CancelSubscription(Intent):
    """User wants to cancel their subscription."""

    kind: Literal["cancel_subscription"] = "cancel_subscription"
    subscription_id: str | None = Field(None, description="Subscription ID.")


class TechnicalIssue(Intent):
    """User reports a bug or technical problem."""

    kind: Literal["technical"] = "technical"
    error_message: str | None = Field(None, description="Error message if any.")


# The LLM outputs one of these types:
TriageOutput = Union[RefundIntent, BillingIntent, CancelSubscription, TechnicalIssue, Respond]


# =============================================================================
# Specialist agents
# =============================================================================

refunds_agent = Agent(
    name="Refunds Specialist",
    system_prompt="You handle refund requests. Be empathetic and resolve quickly.",
)

billing_agent = Agent(
    name="Billing Specialist",
    system_prompt="You handle billing and subscription questions. Be precise.",
)

technical_agent = Agent(
    name="Technical Support",
    system_prompt="You handle technical issues. Ask for reproduction steps.",
)

general_agent = Agent(
    name="General Support",
    system_prompt="You handle general customer inquiries.",
)


# =============================================================================
# Pattern 1: Basic Intent Routing
# =============================================================================
# Each Intent type maps to exactly one agent.

basic_triage = Agent(
    name="Basic Triage",
    system_prompt=(
        "Classify the user's request into an intent type. "
        "If the user just wants to chat or has a general question, respond directly."
    ),
    output_schema=TriageOutput,
    handoffs=(
        HandoffRoute("basic_triage")
        .when(RefundIntent)
        .to(refunds_agent)
        .when(BillingIntent)
        .to(billing_agent)
        .when(CancelSubscription)
        .to(billing_agent)
        .when(TechnicalIssue)
        .to(technical_agent)
        .otherwise(general_agent)
    ),
)


# =============================================================================
# Pattern 2: Multi-Intent Matching
# =============================================================================
# Multiple intent types can map to the same agent with .when(A, B).to(agent).
# This is cleaner than repeating .when(A).to(x).when(B).to(x).

multi_match_triage = Agent(
    name="Multi-Match Triage",
    system_prompt="Classify the user's request. Billing questions and cancellations go to the same team.",
    output_schema=TriageOutput,
    handoffs=(
        HandoffRoute("multi_match")
        .when(RefundIntent)
        .to(refunds_agent)
        .when(BillingIntent, CancelSubscription)
        .to(billing_agent)
        .when(TechnicalIssue)
        .to(technical_agent)
        .otherwise(general_agent)
    ),
)


# =============================================================================
# Pattern 3: Factory Function Shorthand
# =============================================================================
# The handoff() factory builds a HandoffRoute from (Intent, Agent) tuples.
# More concise when you don't need per-route callbacks or filters.

factory_triage = Agent(
    name="Factory Triage",
    system_prompt="Classify the user's request into an intent type.",
    output_schema=TriageOutput,
    handoffs=handoff_route(
        (RefundIntent, refunds_agent),
        (BillingIntent, billing_agent),
        (CancelSubscription, billing_agent),
        (TechnicalIssue, technical_agent),
        otherwise=general_agent,
    ),
)


# =============================================================================
# Running the examples
# =============================================================================


async def main() -> None:
    total_usage = LLMUsage()
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    # Pattern 1: Refund request → RefundIntent → Refunds Specialist
    logger.info("=" * 60)
    logger.info("Pattern 1: Basic Intent Routing")
    logger.info("=" * 60)
    result = await Runner.arun(basic_triage, "I want a refund for order #12345.", run_config=run_config)
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}")
    print_usage("Intent Routing", result.context.usage)
    total_usage = total_usage + result.context.usage
    logger.info("")

    # Pattern 1: Direct response (Respond) → no handoff
    logger.info("=" * 60)
    logger.info("Pattern 1: Direct Response (Respond)")
    logger.info("=" * 60)
    result = await Runner.arun(basic_triage, "Hello, how are you?", run_config=run_config)
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}")
    print_usage("Direct Response", result.context.usage)
    total_usage = total_usage + result.context.usage
    logger.info("")

    # Pattern 2: Cancellation → CancelSubscription → Billing Specialist
    logger.info("=" * 60)
    logger.info("Pattern 2: Multi-Intent Matching")
    logger.info("=" * 60)
    result = await Runner.arun(
        multi_match_triage,
        "I want to cancel my subscription SUB-789.",
        run_config=run_config,
    )
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}")
    print_usage("Multi-Intent", result.context.usage)
    total_usage = total_usage + result.context.usage
    logger.info("")

    # Pattern 3: Factory shorthand
    logger.info("=" * 60)
    logger.info("Pattern 3: Factory Function")
    logger.info("=" * 60)
    result = await Runner.arun(factory_triage, "My app crashes on startup.", run_config=run_config)
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}")
    print_usage("Factory Function", result.context.usage)
    total_usage = total_usage + result.context.usage
    logger.info("")

    # Grand total
    logger.info("=" * 60)
    logger.info("TOTAL USAGE (Code-Orchestrated Handoffs)")
    logger.info("=" * 60)
    print_usage("TOTAL", total_usage)


if __name__ == "__main__":
    asyncio.run(main())

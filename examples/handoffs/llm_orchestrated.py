"""
Example: LLM-Orchestrated Handoffs

The LLM decides when to transfer control via ``transfer_to_<agent>`` tools.
This is the simplest handoff pattern — just pass a list of agents (or Handoff
objects) to ``Agent.handoffs``.

Key points:
- Each target agent appears as a ``transfer_to_<name>`` function tool.
- The LLM calls the tool → Runner transfers control permanently.
- Unlike ``as_tool()``, control does NOT return to the original agent.

Patterns shown:
1. Bare agent list — minimal setup, LLM sees auto-generated descriptions.
2. Handoff objects — custom tool names, descriptions, and callbacks.
3. Dynamic enablement — handoff targets that activate conditionally.
4. Typed handoff input — Pydantic model for structured tool args + validation.
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from typing import Any

from litellm.cost_calculator import cost_per_token
from pydantic import BaseModel, Field

from troopai.adk.agents import Agent
from troopai.adk.handoffs import Handoff, handoff
from troopai.adk.handoffs.handoff_input_data import HandoffInputData
from troopai.adk.llms.llm_usage import LLMUsage
from troopai.adk.run import RunConfig, Runner
from troopai.adk.run.config import DEFAULT_MODEL
from troopai.adk.run.context import RunContext
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
# Specialist agents (handoff targets)
# =============================================================================

refunds_agent = Agent(
    name="Refunds Specialist",
    system_prompt=(
        "You handle refund requests. Ask for the order ID if not provided. Be empathetic and resolve quickly."
    ),
)

billing_agent = Agent(
    name="Billing Specialist",
    system_prompt=(
        "You handle billing questions: invoices, charges, payment methods. Be precise with numbers and dates."
    ),
)

technical_agent = Agent(
    name="Technical Support",
    system_prompt=(
        "You handle technical issues: bugs, errors, setup problems. Ask for error messages and steps to reproduce."
    ),
)


# =============================================================================
# Pattern 1: Bare Agent List (simplest)
# =============================================================================
# The LLM sees auto-generated tool names and descriptions:
#   transfer_to_refunds_specialist  — "Handoff to the Refunds Specialist agent..."
#   transfer_to_billing_specialist  — "Handoff to the Billing Specialist agent..."

simple_triage = Agent(
    name="Simple Triage",
    system_prompt=(
        "You are a customer support triage agent. Route the user to the "
        "right specialist:\n"
        "- Refund requests → Refunds Specialist\n"
        "- Billing questions → Billing Specialist\n"
        "Ask a clarifying question only if the intent is truly ambiguous."
    ),
    handoffs=[refunds_agent, billing_agent],
)


# =============================================================================
# Pattern 2: Handoff Objects (custom descriptions + callbacks)
# =============================================================================
# Handoff objects let you control the tool name, description, and hook into
# the handoff lifecycle via on_handoff callbacks.


def log_handoff(context: RunContext[Any]) -> None:
    """Callback invoked when a handoff occurs. Side-effect only.

    Uses the without-input signature (ctx) — no input data needed for
    parameterless handoffs.
    """
    logger.info("  [on_handoff] Transferring to specialist.")


configured_triage = Agent(
    name="Configured Triage",
    system_prompt="You are a customer support triage agent. Route the user to the right specialist. Be concise.",
    handoffs=[
        Handoff(
            target=refunds_agent,
            description="Transfer when the user wants a refund or return.",
            on_handoff=log_handoff,
        ),
        Handoff(
            target=billing_agent,
            name="transfer_to_billing",  # Custom tool name
            description="Transfer for invoice, charge, or payment questions.",
            on_handoff=log_handoff,
        ),
        Handoff(
            target=technical_agent,
            description="Transfer for bugs, errors, or technical setup issues.",
            on_handoff=log_handoff,
        ),
    ],
)


# =============================================================================
# Pattern 3: Dynamic Enablement
# =============================================================================
# Handoff targets can be enabled/disabled at runtime via a callback.
# Disabled targets are hidden from the LLM (or marked [UNAVAILABLE] with
# CacheStrategy.STABLE).


def is_premium_user(context: RunContext[Any], data: HandoffInputData) -> bool:
    """Only enable this handoff for premium users."""
    if context.context is None:
        return False
    return context.context.get("tier") == "premium"


priority_triage = Agent(
    name="Priority Triage",
    system_prompt=(
        "You are a customer support triage agent.\n"
        "Route the user to the right specialist. If priority support "
        "is unavailable, let them know politely."
    ),
    handoffs=[
        Handoff(
            target=refunds_agent,
            description="Transfer for refund requests.",
        ),
        Handoff(
            target=technical_agent,
            description="Priority technical support (premium users only).",
            enabled=is_premium_user,
        ),
    ],
)


# =============================================================================
# Pattern 4: Typed Handoff Input (structured tool args)
# =============================================================================
# When input_type is set, the handoff tool gets a JSON schema so the LLM
# produces structured arguments. The Handoff validates and parses them
# into a Pydantic model, passed directly to the on_handoff callback.
#
# on_handoff supports two signatures:
#   (ctx, typed_input) — receives the validated Pydantic model
#   (ctx)              — parameterless, only receives context


class EscalationInput(BaseModel):
    """Structured data the LLM provides when escalating."""

    reason: str = Field(description="Why this needs escalation.")
    priority: int = Field(description="Priority level 1-5.", ge=1, le=5)
    order_id: str | None = Field(None, description="Order ID if applicable.")


def log_escalation(ctx: RunContext[Any], input: EscalationInput) -> None:
    """Typed on_handoff — receives the validated EscalationInput directly."""
    logger.info(f"  [on_handoff] Escalating: reason={input.reason!r}, priority={input.priority}")


escalation_agent = Agent(
    name="Escalation Team",
    system_prompt="You handle escalated issues. Review the reason and priority.",
)


typed_triage = Agent(
    name="Typed Triage",
    system_prompt=(
        "You are a triage agent. For complex issues, escalate with a reason "
        "and priority (1=low, 5=critical). For simple refunds, transfer directly."
    ),
    handoffs=[
        handoff(
            target=escalation_agent,
            on_handoff=log_escalation,
            input_type=EscalationInput,
            description="Escalate complex issues with structured reason and priority.",
        ),
        Handoff(
            target=refunds_agent,
            description="Transfer for simple refund requests.",
        ),
    ],
)


# =============================================================================
# Running the examples
# =============================================================================


async def main() -> None:
    total_usage = LLMUsage()
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    # Pattern 1: Simple
    logger.info("=" * 60)
    logger.info("Pattern 1: Bare Agent List")
    logger.info("=" * 60)
    result = await Runner.arun(simple_triage, "I want a refund for order #12345.", run_config=run_config)
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}")
    print_usage("Bare Agent List", result.context.usage)
    total_usage = total_usage + result.context.usage
    logger.info("")

    # Pattern 2: Configured Handoffs
    logger.info("=" * 60)
    logger.info("Pattern 2: Handoff Objects with Callbacks")
    logger.info("=" * 60)
    result = await Runner.arun(configured_triage, "Why was I charged twice this month?", run_config=run_config)
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}")
    print_usage("Handoff Objects", result.context.usage)
    total_usage = total_usage + result.context.usage
    logger.info("")

    # Pattern 3: Dynamic Enablement (premium user)
    logger.info("=" * 60)
    logger.info("Pattern 3: Dynamic Enablement (premium)")
    logger.info("=" * 60)
    result = await Runner.arun(
        priority_triage,
        "My app keeps crashing after the update.",
        context={"tier": "premium"},
        run_config=run_config,
    )
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}")
    print_usage("Dynamic (premium)", result.context.usage)
    total_usage = total_usage + result.context.usage
    logger.info("")

    # Pattern 3b: Dynamic Enablement (free user — tech support hidden)
    logger.info("=" * 60)
    logger.info("Pattern 3: Dynamic Enablement (free tier)")
    logger.info("=" * 60)
    result = await Runner.arun(
        priority_triage,
        "My app keeps crashing after the update.",
        context={"tier": "free"},
        run_config=run_config,
    )
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}")
    print_usage("Dynamic (free)", result.context.usage)
    total_usage = total_usage + result.context.usage
    logger.info("")

    # Pattern 4: Typed handoff input
    logger.info("=" * 60)
    logger.info("Pattern 4: Typed Handoff Input")
    logger.info("=" * 60)
    result = await Runner.arun(
        typed_triage,
        "My order #999 has been stuck in processing for 3 weeks and nobody responds!",
        run_config=run_config,
    )
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}")
    print_usage("Typed Input", result.context.usage)
    total_usage = total_usage + result.context.usage
    logger.info("")

    # Grand total
    logger.info("=" * 60)
    logger.info("TOTAL USAGE (LLM-Orchestrated Handoffs)")
    logger.info("=" * 60)
    print_usage("TOTAL", total_usage)


if __name__ == "__main__":
    asyncio.run(main())

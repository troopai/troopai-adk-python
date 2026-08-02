"""
Example: Cost-Optimized Handoffs with HandoffConfig

HandoffConfig controls HOW MUCH conversation history transfers to the target
agent. This is the primary lever for reducing input token costs in multi-agent
workflows — without it, the target agent re-sends the entire prior conversation
on every turn.

HandoffConfig fields:
- ``strategy`` — WHICH messages to include:
    - "full"        → Everything (default).
    - "last_n"      → Last N messages (requires ``window``).
    - "intent_only" → Only the classified intent, no history.
    - "summary"     → LLM-generated summary of the conversation.
- ``window``  — Number of messages for the "last_n" strategy.
- ``budget``  — Max tokens of history. Applied AFTER strategy filtering.
    Default is 20_000 tokens. When the selected messages exceed the
    budget, oldest non-system messages are dropped (truncation) until
    under the cap — **no LLM call**. The system message is always
    preserved. For LLM-summarised handoffs, use ``strategy=HandoffStrategy.SUMMARY``
    explicitly. Composes with strategy (they don't overlap).

Cost impact examples:
- Agent A accumulated 50K tokens → hands off to Agent B.
  - Without config:                   Agent B starts with 50K input tokens.
  - With strategy=HandoffStrategy.LAST_N, window=5: Agent B starts with ~500 input tokens.
  - With budget=5000:                 Agent B starts with max 5K input tokens.
  - With strategy=HandoffStrategy.INTENT_ONLY:      Agent B starts with ~50 input tokens.

Patterns shown:
1. strategy=HandoffStrategy.FULL with budget (token cap on full history).
2. strategy=HandoffStrategy.LAST_N with window (fixed message window).
3. strategy=HandoffStrategy.INTENT_ONLY (zero history, intent only).
4. strategy=HandoffStrategy.SUMMARY (LLM-summarized history).
5. Composing strategy + budget (last_n with token cap).
6. Per-route config in code-orchestrated handoffs.
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from typing import Literal, Union

from pydantic import Field

from troopai.adk.agents import Agent
from troopai.adk.handoffs import (
    Handoff,
    HandoffConfig,
    HandoffRoute,
)
from troopai.adk.handoffs.handoff_strategy import HandoffStrategy
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools import function_tool
from troopai.adk.types.intents import Intent, Respond
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Tools (simulate data gathering that bloats conversation history)
# =============================================================================


@function_tool(name="search_orders", description="Search customer orders.")
def search_orders(customer_id: str) -> str:
    """Returns a large JSON payload — the kind that bloats context."""
    return (
        f'{{"customer_id": "{customer_id}", '
        f'"orders": ['
        f'{{"id": "ORD-001", "total": 29.99, "status": "delivered", "items": ["Widget A", "Widget B"]}},'
        f'{{"id": "ORD-002", "total": 149.99, "status": "shipped", "items": ["Gadget Pro"]}},'
        f'{{"id": "ORD-003", "total": 9.99, "status": "cancelled", "items": ["Accessory"]}}'
        f"]}}"
    )


@function_tool(name="get_account_history", description="Get account activity log.")
def get_account_history(account_id: str) -> str:
    """Returns a verbose activity log."""
    return (
        f'{{"account_id": "{account_id}", '
        f'"events": ['
        f'{{"date": "2024-01-15", "type": "payment", "amount": 29.99}},'
        f'{{"date": "2024-01-20", "type": "refund", "amount": -9.99}},'
        f'{{"date": "2024-02-01", "type": "payment", "amount": 149.99}},'
        f'{{"date": "2024-02-10", "type": "support_ticket", "subject": "Shipping delay"}},'
        f'{{"date": "2024-03-01", "type": "payment", "amount": 49.99}}'
        f"]}}"
    )


# =============================================================================
# Specialist agents
# =============================================================================

refunds_agent = Agent(
    name="Refunds Specialist",
    system_prompt="You handle refund requests. Be empathetic.",
)

billing_agent = Agent(
    name="Billing Specialist",
    system_prompt="You handle billing questions. Be precise.",
)

escalation_agent = Agent(
    name="Escalation Manager",
    system_prompt=(
        "You handle escalated cases. You have full authority to resolve "
        "complex issues. Review the conversation summary carefully."
    ),
)

general_agent = Agent(
    name="General Support",
    system_prompt="You handle general inquiries.",
)


# =============================================================================
# Pattern 1: Full History with Token Budget
# =============================================================================
# strategy=HandoffStrategy.FULL (default) keeps all messages, but budget=5000 caps the
# total at ~5K tokens. If history exceeds this, oldest non-system messages
# are dropped (truncation) until under the cap. No LLM call.
#
# Use when: You want the full picture but need to cap costs for long
# conversations that may have accumulated many tool results.

budget_capped_triage = Agent(
    name="Budget Capped Triage",
    system_prompt=(
        "You are a triage agent. Use tools to gather information about "
        "the customer, then transfer to the right specialist."
    ),
    tools=[search_orders, get_account_history],
    handoffs=[
        Handoff(
            target=refunds_agent,
            description="Transfer for refund requests.",
            config=HandoffConfig(
                strategy=HandoffStrategy.FULL,
                budget=5_000,  # Max 5K tokens of history transferred
            ),
        ),
        Handoff(
            target=billing_agent,
            description="Transfer for billing questions.",
            config=HandoffConfig(
                strategy=HandoffStrategy.FULL,
                budget=3_000,  # Tighter budget for billing
            ),
        ),
    ],
)


# =============================================================================
# Pattern 2: Last-N Messages Window
# =============================================================================
# strategy=HandoffStrategy.LAST_N with window=N keeps only the last N messages.
# Fast, deterministic, no LLM summarization cost. The trade-off is that
# early context (like the user's name or original request) may be lost.
#
# Use when: Recent context is most relevant (e.g., the specialist needs
# the last few exchanges but not the full triage conversation).

windowed_triage = Agent(
    name="Windowed Triage",
    system_prompt=(
        "You are a triage agent. Have a conversation with the user to "
        "understand their issue, then transfer to a specialist."
    ),
    handoffs=[
        Handoff(
            target=refunds_agent,
            description="Transfer for refund requests.",
            config=HandoffConfig(
                strategy=HandoffStrategy.LAST_N,
                window=4,  # Only last 4 messages transfer
            ),
        ),
        Handoff(
            target=billing_agent,
            description="Transfer for billing questions.",
            config=HandoffConfig(
                strategy=HandoffStrategy.LAST_N,
                window=6,  # Last 6 messages for billing (needs more context)
            ),
        ),
    ],
)


# =============================================================================
# Pattern 3: Intent Only (Zero History)
# =============================================================================
# strategy=HandoffStrategy.INTENT_ONLY passes ONLY the intent/tool_args to the target.
# The specialist starts completely fresh. Cheapest option.
#
# Use when: The specialist is stateless (e.g., a refund processor that
# just needs the order ID, not the full conversation).

intent_only_triage = Agent(
    name="Intent Only Triage",
    system_prompt=(
        "You are a triage agent. Gather all necessary details from the "
        "user (order ID, issue description), then transfer."
    ),
    tools=[search_orders],
    handoffs=[
        Handoff(
            target=refunds_agent,
            description="Transfer for refunds. Include the order ID in the transfer.",
            config=HandoffConfig(strategy=HandoffStrategy.INTENT_ONLY),
        ),
    ],
)


# =============================================================================
# Pattern 4: Summary Strategy
# =============================================================================
# strategy=HandoffStrategy.SUMMARY uses an LLM call to summarize the full conversation
# into a compact representation. Costs one extra LLM call but produces
# much smaller input for the target agent.
#
# Use when: The full conversation is long and the specialist needs broad
# context (not just recent messages), but you can't afford to transfer
# the entire history.

summary_triage = Agent(
    name="Summary Triage",
    system_prompt=(
        "You are a triage agent. Have a detailed conversation, use tools "
        "to investigate, then escalate to the manager with a summary."
    ),
    tools=[search_orders, get_account_history],
    handoffs=[
        Handoff(
            target=escalation_agent,
            description="Escalate to manager for complex issues.",
            config=HandoffConfig(strategy=HandoffStrategy.SUMMARY),
        ),
    ],
)


# =============================================================================
# Pattern 5: Composing Strategy + Budget
# =============================================================================
# strategy and budget compose — they DON'T overlap.
#   1. Strategy selects WHICH messages (e.g., last 10).
#   2. Budget caps HOW MANY TOKENS those selected messages can consume.
#
# This is useful when last_n might still be too large (e.g., window=10
# but one message contains a 5K-token tool result).

composed_triage = Agent(
    name="Composed Triage",
    system_prompt="You are a triage agent with access to data tools. Investigate thoroughly, then transfer.",
    tools=[search_orders, get_account_history],
    handoffs=[
        Handoff(
            target=refunds_agent,
            description="Transfer for refund requests.",
            config=HandoffConfig(
                strategy=HandoffStrategy.LAST_N,
                window=8,  # Start with last 8 messages
                budget=2_000,  # But cap at 2K tokens (compacts if exceeded)
            ),
        ),
    ],
)


# =============================================================================
# Pattern 6: Per-Route Config (Code-Orchestrated)
# =============================================================================
# HandoffConfig works with code-orchestrated routes too.
# Each .when().to() rule can have its own config.


class RefundIntent(Intent):
    kind: Literal["refund"] = "refund"
    order_id: str | None = None


class BillingIntent(Intent):
    kind: Literal["billing"] = "billing"


class EscalateIntent(Intent):
    kind: Literal["escalate"] = "escalate"
    severity: str = Field(default="normal", description="Issue severity level.")


TriageOutput = Union[RefundIntent, BillingIntent, EscalateIntent, Respond]

per_route_triage = Agent(
    name="Per-Route Config Triage",
    system_prompt="Classify the user's request. For escalations, note the severity.",
    # ``Union[...]`` is a ``types.UnionType`` at runtime, which the
    # framework accepts as a schema (wrapped internally in
    # ``AgentOutputSchema``). pyright's structural check can't see the
    # framework's runtime acceptance; mypy clears the site.
    output_schema=TriageOutput,  # type: ignore[arg-type]
    handoffs=(
        HandoffRoute("per_route")
        # Refunds: stateless — only need intent + order ID
        .when(RefundIntent)
        .to(
            refunds_agent,
            config=HandoffConfig(strategy=HandoffStrategy.INTENT_ONLY),
        )
        # Billing: needs some recent context
        .when(BillingIntent)
        .to(
            billing_agent,
            config=HandoffConfig(strategy=HandoffStrategy.LAST_N, window=4),
        )
        # Escalation: needs full picture but capped
        .when(EscalateIntent)
        .to(
            escalation_agent,
            config=HandoffConfig(strategy=HandoffStrategy.FULL, budget=5_000),
        )
        .otherwise(general_agent)
    ),
)


# =============================================================================
# Running the examples
# =============================================================================


async def main() -> None:
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    # Pattern 1: Budget-capped full history
    logger.info("=" * 60)
    logger.info("Pattern 1: Full History + Token Budget (budget=5000)")
    logger.info("=" * 60)
    result = await Runner.arun(
        budget_capped_triage,
        "Look up customer CUST-001's orders and help me get a refund for ORD-003.",
        run_config=run_config,
    )
    assert result.last_agent is not None
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}\n")

    # Pattern 2: Windowed history
    logger.info("=" * 60)
    logger.info("Pattern 2: Last-N Window (window=4)")
    logger.info("=" * 60)
    result = await Runner.arun(
        windowed_triage,
        "I was charged twice for my subscription this month.",
        run_config=run_config,
    )
    assert result.last_agent is not None
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}\n")

    # Pattern 3: Intent only
    logger.info("=" * 60)
    logger.info("Pattern 3: Intent Only (zero history)")
    logger.info("=" * 60)
    result = await Runner.arun(
        intent_only_triage,
        "I need a refund for order ORD-002.",
        run_config=run_config,
    )
    assert result.last_agent is not None
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}\n")

    # Pattern 4: Summary
    logger.info("=" * 60)
    logger.info("Pattern 4: Summary Strategy")
    logger.info("=" * 60)
    result = await Runner.arun(
        summary_triage,
        "I've been having issues for weeks. Look up account ACC-001 and "
        "check my orders for CUST-001. I want to escalate this.",
        run_config=run_config,
    )
    assert result.last_agent is not None
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}\n")

    # Pattern 6: Per-route config (code-orchestrated)
    logger.info("=" * 60)
    logger.info("Pattern 6: Per-Route Config (Code-Orchestrated)")
    logger.info("=" * 60)
    result = await Runner.arun(per_route_triage, "I need a refund for order ORD-001.", run_config=run_config)
    assert result.last_agent is not None
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}\n")


if __name__ == "__main__":
    asyncio.run(main())

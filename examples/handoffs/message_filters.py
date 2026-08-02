"""
Example: Handoff Message Filters

Filters transform the conversation before it reaches the target agent.
This controls what the new agent "knows" about the prior conversation — both
for privacy and for cost reduction (less context = fewer input tokens).

HandoffInputData provides temporal slicing:
- ``context`` — messages before the current agent's turn.
- ``output``  — messages generated during the current agent's turn.
- ``messages`` — property returning context + output (full view).
- ``forwarded`` — filtered subset for the next agent (None = use messages).

Built-in filters (in ``troopai.adk.handoffs.handoff_filters``):
- ``forward_intent``         — Append classified Intent as user message.
- ``remove_tool_calls``      — Strip tool call/result messages.
- ``remove_system_messages`` — Strip system messages.
- ``keep_last_n(n)``         — Keep only the last N messages.
- ``intent_only``            — Remove all messages, keep only the intent.

Filters read from ``forwarded`` if set (for compose chains), else ``messages``.
They set ``forwarded`` on the result, preserving context/output for audit.

Patterns shown:
1. Built-in filters with LLM-orchestrated handoffs.
2. Built-in filters with code-orchestrated handoffs.
3. Custom filter function — full control over what transfers.
4. Composing multiple filters into a pipeline.
5. Forward intent — specialist sees the structured classification.
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from typing import Any, Literal, Union

from troopai.adk.agents import Agent
from troopai.adk.handoffs import (
    Handoff,
    HandoffInputData,
    HandoffRoute,
)
from troopai.adk.handoffs.handoff_filters import (
    compose,
    forward_intent,
    intent_only,
    keep_last_n,
    remove_system_messages,
    remove_tool_calls,
)
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools import function_tool
from troopai.adk.types.intents import Intent, Respond
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Tools and agents
# =============================================================================


@function_tool(name="lookup_order", description="Look up order details by ID.")
def lookup_order(order_id: str) -> str:
    """Simulate an order lookup."""
    return f'{{"order_id": "{order_id}", "status": "delivered", "total": 49.99}}'


@function_tool(name="check_balance", description="Check account balance.")
def check_balance(account_id: str) -> str:
    """Simulate a balance check."""
    return f'{{"account_id": "{account_id}", "balance": 142.50, "currency": "USD"}}'


triage_agent = Agent(
    name="Triage",
    system_prompt="You are a triage agent. Gather information using tools, then hand off to the right specialist.",
    tools=[lookup_order, check_balance],
)

refunds_agent = Agent(
    name="Refunds",
    system_prompt="You handle refund requests. Be empathetic and resolve quickly.",
)

billing_agent = Agent(
    name="Billing",
    system_prompt="You handle billing questions. Be precise with numbers.",
)

general_agent = Agent(
    name="General",
    system_prompt="You handle general inquiries.",
)


# =============================================================================
# Pattern 1: Built-in Filters (LLM-orchestrated)
# =============================================================================
# remove_tool_calls strips tool call/result messages so the specialist
# doesn't see raw JSON from prior tool executions — cleaner context.

llm_filtered_triage = Agent(
    name="LLM Filtered Triage",
    system_prompt="You are a triage agent. Use tools to gather info, then transfer to the right specialist.",
    tools=[lookup_order, check_balance],
    handoffs=[
        Handoff(
            target=refunds_agent,
            description="Transfer for refund requests.",
            input_filter=remove_tool_calls,  # Strip tool messages
        ),
        Handoff(
            target=billing_agent,
            description="Transfer for billing questions.",
            input_filter=keep_last_n(4),  # Only last 4 messages
        ),
    ],
)


# =============================================================================
# Pattern 2: Built-in Filters (code-orchestrated)
# =============================================================================
# Filters work the same way with HandoffRoute — pass to .to(input_filter=...).


class RefundIntent(Intent):
    kind: Literal["refund"] = "refund"
    order_id: str | None = None


class BillingIntent(Intent):
    kind: Literal["billing"] = "billing"


TriageOutput = Union[RefundIntent, BillingIntent, Respond]

code_filtered_triage = Agent(
    name="Code Filtered Triage",
    system_prompt="Classify the user's request.",
    output_schema=TriageOutput,
    handoffs=(
        HandoffRoute("code_filtered")
        .when(RefundIntent)
        .to(refunds_agent, input_filter=remove_tool_calls)
        .when(BillingIntent)
        .to(billing_agent, input_filter=keep_last_n(4))
        .otherwise(general_agent, input_filter=intent_only)
    ),
)


# =============================================================================
# Pattern 3: Custom Filter Function
# =============================================================================
# Write your own filter for full control. Filters receive HandoffInputData
# and return a new (filtered) HandoffInputData via .clone().


def redact_pii(data: HandoffInputData) -> HandoffInputData:
    """Custom filter: redact email addresses from forwarded messages.

    Demonstrates how to write a custom filter that modifies message content
    rather than just removing messages. Reads from ``forwarded`` if set by
    a prior filter (in a compose() chain), otherwise from ``messages``.

    Handles both typed RunItem instances (preferred) and raw dicts.
    """
    import re

    from troopai.adk.handoffs.handoff_filters import source_messages
    from troopai.adk.types.items import (
        MessageOutputItem,
        SystemItem,
        UserItem,
    )

    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

    def redact_item(item: Any) -> Any:
        # Typed items — create new instances with redacted content
        if isinstance(item, UserItem) and isinstance(item.raw.get("content"), str):
            return UserItem(raw={"role": "user", "content": email_pattern.sub("[REDACTED_EMAIL]", item.raw["content"])})
        if isinstance(item, SystemItem):
            return SystemItem(
                raw={
                    "role": item.raw["role"],
                    "content": email_pattern.sub("[REDACTED_EMAIL]", str(item.raw.get("content", ""))),
                }
            )
        if isinstance(item, MessageOutputItem) and item.raw is not None:
            from troopai.adk.types.items.items import ItemHelpers
            from troopai.adk.types.responses.llm_response import LLMResponseText as _Text

            text = ItemHelpers.text_message_output(item)
            if len(text) > 0:
                redacted_content = email_pattern.sub("[REDACTED_EMAIL]", text)
                return MessageOutputItem(
                    raw=[_Text(text=redacted_content)],
                    id=item.id,
                    status=item.status,
                )
        # Raw dict fallback
        if isinstance(item, dict):
            content = item.get("content", "")
            if isinstance(content, str):
                redacted = email_pattern.sub("[REDACTED_EMAIL]", content)
                return {**item, "content": redacted}
        return item

    source = source_messages(data)
    filtered = tuple(redact_item(m) for m in source)
    return data.clone(forwarded=filtered)


custom_filtered_triage = Agent(
    name="Custom Filtered Triage",
    system_prompt="You are a triage agent. Transfer to the specialist when ready.",
    handoffs=[
        Handoff(
            target=refunds_agent,
            description="Transfer for refund requests.",
            input_filter=redact_pii,
        ),
    ],
)


# =============================================================================
# Pattern 4: Composing Filters
# =============================================================================
# The built-in ``compose()`` chains multiple filters left-to-right.
# The output of each filter becomes the input of the next.

# Remove tool calls → remove system messages → redact PII → keep last 6
strict_filter = compose(
    remove_tool_calls,
    remove_system_messages,
    redact_pii,
    keep_last_n(6),
)

composed_triage = Agent(
    name="Composed Triage",
    system_prompt="You are a triage agent. Use tools, then transfer to a specialist.",
    tools=[lookup_order, check_balance],
    handoffs=[
        Handoff(
            target=refunds_agent,
            description="Transfer for refund requests.",
            input_filter=strict_filter,
        ),
    ],
)


# =============================================================================
# Pattern 5: Forward Intent (code-orchestrated)
# =============================================================================
# The forward_intent filter appends the classified Intent as a user message
# so the specialist sees the structured classification (e.g., order_id, reason)
# without having to re-extract it from the raw user message.

forward_intent_triage = Agent(
    name="Forward Intent Triage",
    system_prompt="Classify the user's request.",
    output_schema=TriageOutput,
    handoffs=(
        HandoffRoute("forward_intent")
        .when(RefundIntent)
        .to(refunds_agent, input_filter=forward_intent)
        .when(BillingIntent)
        .to(billing_agent, input_filter=forward_intent)
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

    # Pattern 1: LLM-orchestrated with built-in filter
    logger.info("=" * 60)
    logger.info("Pattern 1: Built-in Filters (LLM-orchestrated)")
    logger.info("=" * 60)
    result = await Runner.arun(
        llm_filtered_triage,
        "I need a refund for order ORD-123. Can you look it up first?",
        run_config=run_config,
    )
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}\n")

    # Pattern 3: Custom filter (PII redaction)
    logger.info("=" * 60)
    logger.info("Pattern 3: Custom Filter (PII Redaction)")
    logger.info("=" * 60)
    result = await Runner.arun(
        custom_filtered_triage,
        "I need a refund. My email is alice@example.com, order ORD-456.",
        run_config=run_config,
    )
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}\n")

    # Pattern 4: Composed filters
    logger.info("=" * 60)
    logger.info("Pattern 4: Composed Filter Pipeline")
    logger.info("=" * 60)
    result = await Runner.arun(
        composed_triage,
        "Look up order ORD-789 and then process my refund. Email: bob@test.com",
        run_config=run_config,
    )
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}\n")

    # Pattern 5: Forward intent
    logger.info("=" * 60)
    logger.info("Pattern 5: Forward Intent (code-orchestrated)")
    logger.info("=" * 60)
    result = await Runner.arun(
        forward_intent_triage,
        "I want a refund for order ORD-555 because it arrived damaged.",
        run_config=run_config,
    )
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}\n")


if __name__ == "__main__":
    asyncio.run(main())

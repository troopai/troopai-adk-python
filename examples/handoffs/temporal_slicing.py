"""
Example: Temporal Slicing in Handoffs

HandoffInputData separates messages into temporal slices so filters and
callbacks can distinguish what happened BEFORE vs DURING the agent's turn.

Fields:
- ``context``   — messages before the current agent's turn started.
- ``output``    — messages generated during the current agent's turn.
- ``messages``  — property: context + output (complete pre-filter flat view).
- ``forwarded`` — filtered subset for the next agent (None = use messages).

Callback signatures:
- ``(ctx)``                        — side-effect only (logging, metrics).
- ``(ctx, intent)``                — access the validated intent.
- ``(ctx, data: HandoffInputData)`` — access temporal slices.

Patterns shown:
1. on_handoff with HandoffInputData — audit what the agent produced.
2. Temporal-aware filter — summarize old context, keep recent output.
3. Forwarded decoupling — audit trail vs what the next agent sees.
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from typing import Any

from troopai.adk.agents import Agent
from troopai.adk.handoffs import (
    Handoff,
    HandoffInputData,
)
from troopai.adk.handoffs.handoff_filters import (
    compose,
    remove_tool_calls,
    source_messages,
)
from troopai.adk.run import RunConfig, Runner
from troopai.adk.run.context import RunContext
from troopai.adk.tools import function_tool
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Tools
# =============================================================================


@function_tool(name="lookup_order", description="Look up order details by ID.")
def lookup_order(order_id: str) -> str:
    """Simulate an order lookup."""
    return f'{{"order_id": "{order_id}", "status": "delivered", "total": 49.99}}'


# =============================================================================
# Pattern 1: on_handoff with HandoffInputData
# =============================================================================
# The (ctx, data: HandoffInputData) callback signature gives access to
# temporal slices — context, output, and messages.

refunds_agent = Agent(
    name="Refunds",
    system_prompt="You handle refund requests. Be empathetic and resolve quickly.",
)


def audit_handoff(ctx: RunContext[Any], data: HandoffInputData) -> None:
    """Audit callback using the HandoffInputData signature.

    Has access to temporal slices — can analyze what the agent did
    (output) vs what it started with (context).

    Items in data.context / data.output are typed RunItem instances.
    Use ``isinstance`` checks for type-safe dispatch.
    """
    from troopai.adk.types.items import (
        MessageOutputItem,
        ToolCallItem,
        ToolCallOutputItem,
    )

    logger.info(
        "Handoff audit: %d context items, %d output items, %d total",
        len(data.context),
        len(data.output),
        len(data.messages),
    )

    # Log what the agent produced during its turn using typed checks
    for item in data.output:
        if isinstance(item, MessageOutputItem):
            from troopai.adk.types.items.items import ItemHelpers

            text = ItemHelpers.text_message_output(item)
            logger.info("  Output: MessageOutputItem (content=%s)", text[:40] if text else None)
        elif isinstance(item, ToolCallItem):
            logger.info("  Output: ToolCallItem (name=%s)", item.raw.name)
        elif isinstance(item, ToolCallOutputItem):
            logger.info("  Output: ToolCallOutputItem (call_id=%s)", item.raw.call_id)
        else:
            logger.info("  Output: %s", type(item).__name__)

    # Check if forwarded is set (filter was applied)
    if data.forwarded is not None:
        logger.info(
            "  Forwarded %d of %d items to next agent",
            len(data.forwarded),
            len(data.messages),
        )


audit_triage = Agent(
    name="Audit Triage",
    system_prompt="You are a triage agent. Use tools to gather info, then transfer to the refunds specialist.",
    tools=[lookup_order],
    handoffs=[
        Handoff(
            target=refunds_agent,
            description="Transfer for refund requests.",
            on_handoff=audit_handoff,
        ),
    ],
)


# =============================================================================
# Pattern 2: Temporal-aware filter
# =============================================================================
# Custom filter that keeps full output (what the agent just did)
# but trims context to only the last N messages.


def keep_recent_context(n: int = 3):
    """Keep full output but trim context to last N messages.

    Demonstrates temporal-aware filtering: the agent's recent work
    (output) is always preserved, but older context is trimmed for
    cost optimization.
    """

    def _filter(data: HandoffInputData) -> HandoffInputData:
        # Operate on the effective source — ``forwarded`` when a prior filter in
        # a compose() chain already set it (e.g. remove_tool_calls), otherwise
        # context + output — so earlier filters are respected, not discarded.
        # Split that source back into its temporal halves by membership in the
        # raw output slice: keep every output item, trim the context half to the
        # last N. (Context always precedes output, so the order stays correct.)
        source = source_messages(data)
        output_items = [item for item in source if item in data.output]
        context_items = [item for item in source if item not in data.output]
        trimmed_context = context_items[-n:] if len(context_items) > n else context_items
        forwarded = tuple(trimmed_context) + tuple(output_items)
        return data.clone(forwarded=forwarded)

    return _filter


temporal_triage = Agent(
    name="Temporal Triage",
    system_prompt="You are a triage agent. Use tools to gather info, then transfer to the refunds specialist.",
    tools=[lookup_order],
    handoffs=[
        Handoff(
            target=refunds_agent,
            description="Transfer for refund requests.",
            input_filter=keep_recent_context(2),
            on_handoff=audit_handoff,
        ),
    ],
)


# =============================================================================
# Pattern 3: Compose with temporal filter
# =============================================================================
# First strip tool calls (from the combined view), then apply temporal trimming.

clean_and_trim = compose(
    remove_tool_calls,
    keep_recent_context(3),
)

composed_triage = Agent(
    name="Composed Temporal Triage",
    system_prompt="You are a triage agent. Use tools to gather info, then transfer to the refunds specialist.",
    tools=[lookup_order],
    handoffs=[
        Handoff(
            target=refunds_agent,
            description="Transfer for refund requests.",
            input_filter=clean_and_trim,
            on_handoff=audit_handoff,
        ),
    ],
)


# =============================================================================
# Running the examples
# =============================================================================


async def main() -> None:
    prompt = "I need a refund for order ORD-123. Can you look it up first?"
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    # Pattern 1: Audit callback with temporal slices
    logger.info("=" * 60)
    logger.info("Pattern 1: on_handoff with HandoffInputData")
    logger.info("=" * 60)
    result = await Runner.arun(audit_triage, prompt, run_config=run_config)
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}\n")

    # Pattern 2: Temporal-aware filter
    logger.info("=" * 60)
    logger.info("Pattern 2: Temporal-aware filter (trim context, keep output)")
    logger.info("=" * 60)
    result = await Runner.arun(temporal_triage, prompt, run_config=run_config)
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}\n")

    # Pattern 3: Composed temporal filter
    logger.info("=" * 60)
    logger.info("Pattern 3: Composed (remove tools + trim context)")
    logger.info("=" * 60)
    result = await Runner.arun(composed_triage, prompt, run_config=run_config)
    logger.info(f"Final agent: {result.last_agent.name}")
    logger.info(f"Output: {result.final_output}\n")


if __name__ == "__main__":
    asyncio.run(main())

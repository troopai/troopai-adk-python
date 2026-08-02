"""
Example: Sub-Agent Streaming via ``on_stream``

Demonstrates using ``as_tool(on_stream=callback)`` to tap into real-time
streaming events from nested sub-agents. When ``on_stream`` is provided,
the sub-agent runs in streaming mode (``stream=True``) and forwards each
``StreamEvent`` to the callback.

A customer support supervisor delegates to a billing agent. The billing
agent's streaming tokens are displayed in real time with a ``[billing]``
prefix, giving visibility into what the sub-agent is generating.

Stream event types:
- ``raw_response_event`` — individual tokens from the LLM
- ``run_item_stream_event`` — semantic events (tool calls, outputs, messages)
- ``agent_updated_stream_event`` — agent transitions during handoffs

Key concepts shown:
- ``as_tool(on_stream=handle_stream)`` — async callback receives ``StreamEvent``
- Real-time prefixed output from sub-agent tokens
- Sub-agent tools (``get_billing_status``) work normally during streaming
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.agents import Agent
from troopai.adk.run import RunConfig, RunItemType, Runner, StreamEvent
from troopai.adk.tools.function_tool import function_tool
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Billing sub-agent tools
# =============================================================================


@function_tool
def get_billing_status(customer_id: str) -> str:
    """Look up billing status for a customer.

    Args:
        customer_id: The customer's account ID.
    """
    accounts = {
        "CUST-001": "Active | Plan: Premium | Next billing: 2025-02-15 | Amount: $49.99",
        "CUST-002": "Past due | Plan: Basic | Overdue since: 2025-01-01 | Amount: $9.99",
        "CUST-003": "Cancelled | Plan: None | Cancelled on: 2024-12-20",
    }
    return accounts.get(customer_id, f"No account found for {customer_id}")


@function_tool
def apply_discount(customer_id: str, percent: int) -> str:
    """Apply a discount to a customer's next billing cycle.

    Args:
        customer_id: The customer's account ID.
        percent: Discount percentage (1-50).
    """
    return f"Applied {percent}% discount to {customer_id}'s next billing cycle."


# =============================================================================
# Stream callback — observe sub-agent tokens in real time
# =============================================================================


async def handle_billing_stream(event: StreamEvent) -> None:
    """Process streaming events from the billing sub-agent.

    Each raw token is logged with a [billing] prefix for visibility.
    Semantic events (tool calls, outputs) are logged as they occur.
    """
    if event.type == "raw_response_event":
        # Raw LLM tokens — log with prefix for visibility
        logger.info(f"[billing] {event.data}")
    elif event.type == "run_item_stream_event":
        # Semantic events — tool calls, outputs, messages
        if event.name == RunItemType.TOOL_CALLED:
            logger.info(f"[billing] >> Tool called: {event.item['name']}")
        elif event.name == RunItemType.TOOL_OUTPUT:
            logger.info(f"[billing] >> Tool output: {event.item['output']}")
        elif event.name == RunItemType.MESSAGE_OUTPUT_CREATED:
            logger.info("[billing] >> Message created")


# =============================================================================
# Agents
# =============================================================================

billing_agent = Agent(
    name="Billing Specialist",
    system_prompt=(
        "You are a billing specialist. Help customers with billing "
        "inquiries, payment issues, and account status. Use the billing "
        "tools to look up account information and apply adjustments. "
        "Be professional and helpful."
    ),
    tools=[get_billing_status, apply_discount],
)

supervisor = Agent(
    name="Support Supervisor",
    system_prompt=(
        "You are a customer support supervisor. Route billing questions "
        "to the billing specialist using the available tool. Summarize "
        "the specialist's findings for the customer."
    ),
    tools=[
        billing_agent.as_tool(
            tool_description=(
                "Delegate billing inquiries to the billing specialist. Provide the customer's question and account ID."
            ),
            on_stream=handle_billing_stream,
        ),
    ],
)


# =============================================================================
# Running the example
# =============================================================================


async def main():
    logger.info("=" * 60)
    logger.info("Sub-Agent Streaming via on_stream")
    logger.info("=" * 60)
    logger.info("")

    prompt = (
        "Customer CUST-002 is calling about their overdue bill. "
        "Please check their account status and see if we can help."
    )
    logger.info(f"User: {prompt}\n")
    logger.info("-" * 40)
    logger.info("Streaming from billing sub-agent:")
    logger.info("-" * 40)

    result = await Runner.arun(supervisor, prompt, run_config=RunConfig(verbose=VerboseConfig()))

    logger.info(f"\n\n{'=' * 60}")
    logger.info("Final supervisor response:")
    logger.info("=" * 60)
    logger.info(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

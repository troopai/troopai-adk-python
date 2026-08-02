"""Streaming tool execution with full feature parity.

Demonstrates that streaming mode has all the same capabilities as
non-streaming:

1. **Input guardrails** — Validated before execution in both modes.
2. **Output guardrails** — Checked after execution in both modes.
3. **ExecutionAwareToolContext** — Budget-aware tools work in streaming.
4. **HITL approval events** — ``TOOL_APPROVAL_REQUESTED`` events in stream.

The streaming path uses the same ``_execute_single_tool_call()``
as non-streaming, with stream events emitted at TOOL_CALLED,
TOOL_OUTPUT, and TOOL_APPROVAL_REQUESTED.

Usage:
    python examples/tools/streaming_tools.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.agents import Agent
from troopai.adk.tools.function_tool import function_tool
from troopai.adk.tools.tool_context import ExecutionAwareToolContext
from troopai.adk.tools.tool_guardrails import (
    ToolGuardrailFunctionOutput,
    ToolGuardrails,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    tool_input_guardrail,
    tool_output_guardrail,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guardrails (work in both streaming and non-streaming)
# ---------------------------------------------------------------------------


@tool_input_guardrail()
def validate_query_length(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Reject queries that are too short."""
    query = data.context.tool_arguments.get("query", "")
    if len(query) < 3:
        return ToolGuardrailFunctionOutput.reject_content("Query too short. Please provide at least 3 characters.")
    return ToolGuardrailFunctionOutput.allow()


@tool_output_guardrail()
def redact_emails(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Replace email addresses in output with [REDACTED]."""
    import re

    output = str(data.output)
    if re.search(r"\S+@\S+\.\S+", output):
        return ToolGuardrailFunctionOutput.reject_content("Output contained email addresses — redacted for privacy.")
    return ToolGuardrailFunctionOutput.allow()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@function_tool(
    name="search",
    description="Search the knowledge base",
    guardrails=ToolGuardrails(
        input=[validate_query_length],
        output=[redact_emails],
    ),
)
def search(query: str) -> str:
    """Simulate a search."""
    return f"Results for '{query}': Document A, Document B, Document C"


@function_tool(
    name="budget_search",
    description="Budget-aware search that adapts to token usage",
)
def budget_search(ctx: ExecutionAwareToolContext, query: str) -> str:
    """Use ExecutionAwareToolContext to adapt behavior based on budget."""
    if ctx.tokens > 50_000:
        # Running low on budget — return shorter results
        return f"Quick result for '{query}': Top match only"
    return f"Full results for '{query}': Match A (0.95), Match B (0.87), Match C (0.72)"


@function_tool(
    name="deploy",
    description="Deploy a service to production",
    requires_approval=True,
)
def deploy(service: str, version: str) -> str:
    """Deploy a service. Always requires approval."""
    return f"Deployed {service} v{version} to production"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


agent = Agent(
    name="Streaming Demo",
    system_prompt="You are a helpful assistant. Use search to find information and deploy to deploy services.",
    tools=[search, budget_search, deploy],
)


async def main() -> None:
    logger.info("=== Streaming with guardrails ===")
    logger.info("Running agent with stream=True...")
    logger.info("(Requires API key — showing the pattern)\n")

    # Pattern for streaming with tool events:
    #
    # result = Runner.run(agent, "Search for Python docs", stream=True)
    # async for event in result.stream_events():
    #     if event.type == "raw_response_event":
    #         print(event.data, end="", flush=True)
    #     elif event.type == "run_item_stream_event":
    #         if event.name == RunItemType.TOOL_CALLED:
    #             print(f"\n[Tool called: {event.item['name']}]")
    #         elif event.name == RunItemType.TOOL_OUTPUT:
    #             print(f"[Tool output: {event.item['output'][:50]}...]")
    #         elif event.name == RunItemType.TOOL_APPROVAL_REQUESTED:
    #             print(f"[Approval needed: {event.item['name']}]")
    #
    # if result.requires_action:
    #     for req in result.deferred_requests.approvals:
    #         result.state.approve(req)
    #     result = Runner.run(agent, result.state, stream=True)
    #     async for event in result.stream_events():
    #         if event.type == "raw_response_event":
    #             print(event.data, end="", flush=True)

    logger.info("Streaming mode now has full feature parity:")
    logger.info("  ✓ Input guardrails (validated before execution)")
    logger.info("  ✓ Output guardrails (checked after execution)")
    logger.info("  ✓ ExecutionAwareToolContext (budget-aware tools)")
    logger.info("  ✓ HITL approval events (TOOL_APPROVAL_REQUESTED)")


if __name__ == "__main__":
    asyncio.run(main())

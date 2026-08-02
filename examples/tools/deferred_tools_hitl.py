"""Human-in-the-loop tool approval at the tool level.

Demonstrates ``FunctionTool.requires_approval``:

1. **Static approval** — ``requires_approval=True`` always defers.
2. **Conditional approval** — A callable that inspects ``ToolContext``
   to decide whether approval is needed (e.g., production vs staging).

When a tool requires approval, execution pauses and returns a
``RunResult`` with ``requires_action=True``.  The caller then
approves or rejects each deferred tool call, and resumes execution.

Usage:
    python examples/tools/deferred_tools_hitl.py
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
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools.function_tool import function_tool
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Static approval — always requires human confirmation
# ---------------------------------------------------------------------------


@function_tool(
    name="delete_record",
    description="Permanently delete a database record",
    requires_approval=True,
)
def delete_record(record_id: str) -> str:
    """Delete a record. Always requires approval."""
    return f"Record {record_id} permanently deleted."


# ---------------------------------------------------------------------------
# 2. Conditional approval — only in production
# ---------------------------------------------------------------------------


async def require_in_production(ctx: ToolContext) -> bool:
    """Only require approval when running in production."""
    environment = ctx.context.get("environment", "staging")
    return environment == "production"


@function_tool(
    name="restart_service",
    description="Restart a backend service",
    requires_approval=require_in_production,
)
def restart_service(service_name: str) -> str:
    """Restart a service. Requires approval only in production."""
    return f"Service '{service_name}' restarted successfully."


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


agent = Agent(
    name="Ops Agent",
    system_prompt=(
        "You are an operations agent. Use tools to manage infrastructure. "
        "When asked to delete records or restart services, use the appropriate tool."
    ),
    tools=[delete_record, restart_service],
)


async def main() -> None:
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    # --- Scenario 1: Static approval (always deferred) ---
    logger.info("=== Scenario 1: Static approval ===")
    result = await Runner.arun(agent, "Delete record REC-42", run_config=run_config)

    if result.requires_action:
        logger.info(f"Deferred {len(result.deferred_requests.approvals)} tool call(s):")
        for req in list(result.deferred_requests.approvals):
            logger.info(f"  Tool: {req.tool_name}, Args: {req.tool_arguments}")

            # Approve the deletion
            result.state.approve(req)
            logger.info("  → Approved")

        # Resume execution
        result = await Runner.arun(agent, result.state, run_config=run_config)
        logger.info(f"Final output: {result.final_output}\n")

    # --- Scenario 2: Conditional approval (production) ---
    logger.info("=== Scenario 2: Conditional approval (production) ===")
    result = await Runner.arun(
        agent,
        "Restart the auth-service",
        context={"environment": "production"},
        run_config=run_config,
    )

    if result.requires_action:
        logger.info("Approval required in production!")
        for req in list(result.deferred_requests.approvals):
            # Reject with a reason — the LLM sees this message
            result.state.reject(req, "Cannot restart auth-service during peak hours.")
            logger.info(f"  → Rejected: {req.tool_name}")

        # Resume with rejection
        result = await Runner.arun(agent, result.state, run_config=run_config)
        logger.info(f"Final output: {result.final_output}\n")

    # --- Scenario 3: Conditional approval (staging — no approval needed) ---
    logger.info("=== Scenario 3: No approval needed (staging) ===")
    result = await Runner.arun(
        agent,
        "Restart the auth-service",
        context={"environment": "staging"},
        run_config=run_config,
    )
    logger.info(f"Final output: {result.final_output}")
    logger.info("(No approval prompt — staging environment)\n")


if __name__ == "__main__":
    asyncio.run(main())

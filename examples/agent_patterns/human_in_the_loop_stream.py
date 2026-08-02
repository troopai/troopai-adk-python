"""
Example: HITL with Streaming

Combines real-time streaming output with human approval checkpoints.
You see agent responses and tool calls as they stream, then the run
pauses when a tool requires approval. After approval/rejection, the
run resumes — again in streaming mode.

Flow:
    1. Runner.run(stream=True) streams events in real-time
    2. When a tool needs approval, a TOOL_APPROVAL_REQUESTED event fires
    3. After stream_events() completes, check result.requires_action
    4. Approve/reject, then resume with Runner.arun(agent, result.state)
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
import sys
from pathlib import Path

from troopai.adk.agents import Agent
from troopai.adk.run import RunConfig, Runner
from troopai.adk.run.stream import RunItemType
from troopai.adk.tools.function_tool import function_tool
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.verbose import VerboseConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_mode import confirm_with_fallback

logger = logging.getLogger(__name__)


# -- Tools --------------------------------------------------------------------


@function_tool(
    name="get_weather",
    description="Get current weather for a city.",
)
def get_weather(city: str) -> str:
    return f"The weather in {city} is 22°C and sunny."


async def _needs_approval_for_dangerous_cities(ctx: ToolContext) -> bool:
    """Only require approval for certain cities (simulates dynamic policy)."""
    city = ctx.tool_arguments.get("city", "").lower()
    # Imagine these are restricted zones requiring authorization
    return city in ("area51", "chernobyl", "bermuda triangle")


@function_tool(
    name="deploy_drone",
    description="Deploy a surveillance drone to a location.",
    requires_approval=_needs_approval_for_dangerous_cities,
)
def deploy_drone(city: str, altitude: int = 100) -> str:
    return f"Drone deployed to {city} at {altitude}m altitude. Live feed active."


@function_tool(
    name="send_alert",
    description="Send an emergency alert to all personnel.",
    requires_approval=True,  # Always requires approval
)
def send_alert(message: str, severity: str = "high") -> str:
    return f"[{severity.upper()}] Alert sent to all personnel: {message}"


# -- Agent --------------------------------------------------------------------

agent = Agent(
    name="Operations",
    system_prompt=(
        "You are an operations coordinator. You can check weather, "
        "deploy drones, and send alerts. Follow user instructions precisely."
    ),
    tools=[get_weather, deploy_drone, send_alert],
)


# -- Main --------------------------------------------------------------------


async def main():
    logger.info("=" * 60)
    logger.info("HITL with Streaming")
    logger.info("=" * 60)
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    prompt = (
        "Check the weather in Paris, then deploy a drone to Area51, "
        "and send a high-severity alert about the deployment."
    )
    logger.info(f"\nPrompt: {prompt}\n")

    # First run — streamed
    result = Runner.run(agent, prompt, stream=True, run_config=run_config)

    logger.info("--- Streaming events ---")
    tokens: list[str] = []
    async for event in result.stream_events():
        match event.type:
            case "raw_response_event":
                tokens.append(event.data)

            case "run_item_stream_event":
                if event.name == RunItemType.TOOL_CALLED:
                    logger.info(f"\n  [Tool called] {event.item['name']}")

                elif event.name == RunItemType.TOOL_OUTPUT:
                    output = str(event.item.get("output", ""))
                    if len(output) > 80:
                        output = output[:80] + "..."
                    logger.info(f"  [Tool output] {output}")

                elif event.name == RunItemType.TOOL_APPROVAL_REQUESTED:
                    nested = event.item.get("nested_agent")
                    source = f" (from {nested})" if nested else ""
                    logger.info(f"  [Approval needed] {event.item['name']}{source}")

    logger.info("".join(tokens))
    logger.info("\n--- Stream complete ---\n")

    # Approval loop
    while result.requires_action:
        # When requires_action is True the framework guarantees both
        # deferred_requests and state are populated — narrow them via
        # locals for the remainder of the iteration.
        deferred = result.deferred_requests
        state = result.state
        if deferred is None or state is None:
            raise RuntimeError("requires_action was True but deferred_requests/state is None")
        logger.info(f"{len(deferred.approvals)} tool(s) need approval:\n")

        for req in list(deferred.approvals):
            logger.info(f"  Tool: {req.tool_name}")
            logger.info(f"  Args: {req.tool_arguments}")

            # Auto-approves under the batch runner / auto mode; prompts a
            # human otherwise. See examples/auto_mode.py.
            approved = confirm_with_fallback("  Approve? (y/n): ", default=True)

            if approved:
                state.approve(req)
                logger.info("  -> Approved\n")
            else:
                state.reject(req, "Operation not authorized.")
                logger.info("  -> Rejected\n")

        # Resume
        logger.info("--- Resuming ---")
        resumed = await Runner.arun(agent, state, run_config=run_config)

        # Check if more approvals are needed
        if resumed.requires_action:
            result.deferred_requests = resumed.deferred_requests
            result.state = resumed.state
        else:
            result.final_output = resumed.final_output
            result.deferred_requests = None
            result.state = None

    logger.info(f"\nFinal output:\n{result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())

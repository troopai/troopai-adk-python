"""
Example: HITL with Custom Rejection Messages

When a tool call is rejected, the rejection message is sent back to the LLM
as a tool result. This lets the model adapt its behavior — e.g., explain
the limitation, suggest an alternative, or ask the user for guidance.

This example uses a publish_announcement tool that always requires approval.
The system prompt instructs the LLM to call the tool directly, and runtime
approvals handle gating — no text-based confirmation needed.
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
from troopai.adk.tools.function_tool import function_tool
from troopai.adk.verbose import VerboseConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_mode import input_with_fallback

logger = logging.getLogger(__name__)


# -- Tools --------------------------------------------------------------------


@function_tool(
    name="publish_announcement",
    description="Publish an announcement visible to all users.",
    requires_approval=True,
)
def publish_announcement(title: str, body: str) -> str:
    return f"Published announcement '{title}' with body: {body}"


@function_tool(
    name="save_draft",
    description="Save an announcement as a draft for later review.",
)
def save_draft(title: str, body: str) -> str:
    return f"Draft saved: '{title}'. It can be reviewed and published later."


# -- Agent --------------------------------------------------------------------

agent = Agent(
    name="Communications Assistant",
    system_prompt=(
        "You help publish internal announcements. When asked to publish, "
        "call the publish_announcement tool directly. Do not ask the user "
        "for approval in plain text; runtime approvals handle that. "
        "If the tool call is rejected, offer to save as draft instead."
    ),
    tools=[publish_announcement, save_draft],
)


# -- Main --------------------------------------------------------------------


async def main():
    logger.info("=" * 60)
    logger.info("HITL: Custom Rejection Messages")
    logger.info("=" * 60)
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    prompt = (
        "Publish an announcement titled 'Office Maintenance' "
        "with body 'The office will close at 6 PM today for maintenance.'"
    )
    logger.info(f"\nPrompt: {prompt}")

    result = await Runner.arun(agent, prompt, run_config=run_config)

    # Approval loop
    while result.requires_action:
        # When requires_action is True the framework guarantees both
        # deferred_requests and state are populated — narrow them via
        # locals for the remainder of the iteration.
        deferred = result.deferred_requests
        state = result.state
        if deferred is None or state is None:
            raise RuntimeError("requires_action was True but deferred_requests/state is None")
        logger.info(f"\n--- {len(deferred.approvals)} tool(s) need approval ---")

        for req in list(deferred.approvals):
            logger.info(f"\n  Tool: {req.tool_name}")
            logger.info(f"  Args: {req.tool_arguments}")
            logger.info("")

            # Auto mode picks the default-rejection branch ("r"), the
            # teaching point of this example (the LLM receiving a rejection
            # message and adapting to a fallback tool). See examples/auto_mode.py.
            choice = (
                input_with_fallback("  (a)pprove, (r)eject with default, or (c)ustom rejection: ", "r").strip().lower()
            )

            if choice == "a":
                state.approve(req)
                logger.info("  -> Approved")
            elif choice == "c":
                custom_msg = input_with_fallback("  Custom rejection message: ", "").strip()
                if len(custom_msg) == 0:
                    custom_msg = "Publish rejected: announcements require manager sign-off. Save as draft instead."
                state.reject(req, custom_msg)
                logger.info(f"  -> Rejected with: {custom_msg}")
            else:
                # Default rejection — the LLM sees this as the tool result
                state.reject(
                    req,
                    "This action was denied by the reviewer. Consider saving as a draft for later approval.",
                )
                logger.info("  -> Rejected (default message)")

        # Resume — the LLM sees the rejection message and adapts
        logger.info("\nResuming execution...")
        result = await Runner.arun(agent, state, run_config=run_config)

    logger.info(f"\nFinal output:\n{result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())

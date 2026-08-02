"""
Example: Human-in-the-Loop (HITL) — Tool Approval

Demonstrates pausing agent execution when a tool requires human approval,
then resuming after the user approves or rejects.

Flow:
    1. Agent tries to call a tool with requires_approval=True
    2. Runner returns RunResult with requires_action=True
    3. User reviews deferred requests and approves/rejects each
    4. Runner.arun(agent, result.state) resumes from checkpoint

This works with both static (bool) and dynamic (callable) approval.
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
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.verbose import VerboseConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_mode import confirm_with_fallback

logger = logging.getLogger(__name__)


# -- Tools --------------------------------------------------------------------


@function_tool(
    name="get_weather",
    description="Get the current weather for a city.",
)
def get_weather(city: str) -> str:
    return f"The weather in {city} is 22°C and sunny."


@function_tool(
    name="send_email",
    description="Send an email to a recipient.",
    requires_approval=True,  # Static: always requires approval
)
def send_email(to: str, subject: str, body: str) -> str:
    return f"Email sent to {to} with subject '{subject}'."


async def _needs_approval_for_large_amounts(ctx: ToolContext) -> bool:
    """Dynamic approval: only require approval for transfers over $100."""
    amount = ctx.tool_arguments.get("amount", 0)
    return amount > 100


@function_tool(
    name="transfer_money",
    description="Transfer money to a recipient.",
    requires_approval=_needs_approval_for_large_amounts,  # Dynamic
)
def transfer_money(to: str, amount: float) -> str:
    return f"Transferred ${amount:.2f} to {to}."


# -- Agent --------------------------------------------------------------------

agent = Agent(
    name="Assistant",
    system_prompt=(
        "You are a helpful assistant. You can check weather, send emails, "
        "and transfer money. Use the available tools to fulfill requests "
        "directly — do not ask for confirmation in text, the approval "
        "system handles that."
    ),
    tools=[get_weather, send_email, transfer_money],
)


# -- Main --------------------------------------------------------------------


async def main():
    logger.info("=" * 60)
    logger.info("Human-in-the-Loop: Tool Approval")
    logger.info("=" * 60)
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    # This will trigger approval for send_email (static)
    # and transfer_money if amount > $100 (dynamic)
    prompt = (
        "Send an email to bob@example.com with subject 'Team Meeting' "
        "and body 'Reminder: team meeting tomorrow at 10 AM in Room 301.' "
        "Then transfer $500 to alice@payments.com."
    )
    logger.info(f"\nPrompt: {prompt}")

    result = await Runner.arun(agent, prompt, run_config=run_config)

    # Approval loop
    while result.requires_action:
        # When requires_action is True the framework guarantees both
        # deferred_requests and state are populated — pull them into
        # locals so the type checker can narrow them for the rest of
        # the iteration.
        deferred = result.deferred_requests
        state = result.state
        if deferred is None or state is None:
            raise RuntimeError("requires_action was True but deferred_requests/state is None")
        logger.info(f"\n--- {len(deferred.approvals)} tool(s) need approval ---")

        for req in list(deferred.approvals):
            logger.info(f"\n  Tool: {req.tool_name}")
            logger.info(f"  Args: {req.tool_arguments}")

            # Auto-approves under the batch runner / auto mode; prompts a
            # human otherwise. See examples/auto_mode.py.
            approved = confirm_with_fallback("  Approve? (y/n): ", default=True)

            if approved:
                state.approve(req)
                logger.info("  -> Approved")
            else:
                state.reject(req, "User denied this action.")
                logger.info("  -> Rejected")

        # Resume
        logger.info("\nResuming execution...")
        result = await Runner.arun(agent, state, run_config=run_config)

    logger.info(f"\nFinal output:\n{result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())

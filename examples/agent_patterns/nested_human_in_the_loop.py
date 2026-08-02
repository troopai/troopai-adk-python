"""
Example: Nested HITL (Human-in-the-Loop) through as_tool() boundaries

Demonstrates how tool approval requests propagate transparently from
a sub-agent back to the parent's RunResult. The user sees a flat list
of deferred requests regardless of nesting depth.

Flow:
    1. Supervisor asks Worker to delete a user
    2. Worker's delete_user tool has requires_approval=True
    3. Worker defers → AgentToolDeferral propagates through as_tool()
    4. Supervisor's RunResult surfaces the deferral
    5. User approves/rejects → Runner.arun(supervisor, result.state) resumes
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
from auto_mode import confirm_with_fallback

logger = logging.getLogger(__name__)


# -- Tools with approval requirements ----------------------------------------


@function_tool(
    name="delete_user",
    description="Permanently delete a user account. Irreversible.",
    requires_approval=True,
)
def delete_user(user_id: str) -> str:
    return f"User {user_id} has been permanently deleted."


@function_tool(
    name="list_users",
    description="List all users in the system.",
)
def list_users() -> str:
    return "Users: alice (id=1), bob (id=2), charlie (id=3)"


# -- Agents ------------------------------------------------------------------

worker = Agent(
    name="Account Manager",
    system_prompt="You manage user accounts. Use list_users to find users and delete_user to remove them when asked.",
    tools=[list_users, delete_user],
)

supervisor = Agent(
    name="Supervisor",
    system_prompt="You coordinate account management tasks. Use the account_manager tool to delegate work.",
    tools=[
        worker.as_tool(
            tool_description="Delegate account management tasks.",
        ),
    ],
)


# -- Main --------------------------------------------------------------------


async def main():
    logger.info("=" * 60)
    logger.info("Nested HITL Example")
    logger.info("=" * 60)
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    # Step 1: Run — the sub-agent will hit requires_approval on delete_user
    logger.info("\n[1] Running supervisor with: 'Delete user bob'")
    result = await Runner.arun(supervisor, "Delete user bob", run_config=run_config)

    # Step 2: Check if approval is needed
    if result.requires_action:
        # When requires_action is True the framework guarantees both
        # deferred_requests and state are populated — narrow them via
        # locals so the type checker sees non-None for this block.
        deferred = result.deferred_requests
        state = result.state
        if deferred is None or state is None:
            raise RuntimeError("requires_action was True but deferred_requests/state is None")
        logger.info(f"\n[2] Approval required! {len(deferred.approvals)} request(s):")

        for req in list(deferred.approvals):
            nested = req.metadata is not None and req.metadata.nested_agent
            agent_name = req.metadata.nested_agent_name if req.metadata is not None else "direct"
            logger.info(f"    - Tool: {req.tool_name}")
            logger.info(f"      Nested: {nested} (from: {agent_name})")
            logger.info(f"      Args: {req.tool_arguments}")

            # Auto-approves under the batch runner / auto mode; prompts a
            # human otherwise. See examples/auto_mode.py.
            approved = confirm_with_fallback("\n    Approve? (y/n): ", default=True)

            if approved:
                logger.info("    -> Approved")
                state.approve(req)
            else:
                logger.info("    -> Rejected")
                state.reject(req, "User denied the operation.")

        # Step 3: Resume execution with approvals
        logger.info("\n[3] Resuming execution...")
        result = await Runner.arun(supervisor, state, run_config=run_config)

    # Step 4: Final output
    logger.info(f"\n[4] Final output:\n{result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())

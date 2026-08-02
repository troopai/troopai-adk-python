"""Verbose output — nested HITL through ``as_tool()`` boundaries.

Shows the nested-HITL breadcrumb visualization: when an agent is
wrapped as a tool and its inner tool raises a HITL approval
requirement, the approval bubbles up through the outer agent and the
panel shows the full path ``outer_agent → delegate_tool → inner_tool``.

CrewAI cannot render this because the framework itself does not model
agents-as-tools with approval propagation. In TroopAI ADK it is
first-class: :class:`~troopai.adk.types.agents.DeferredToolCall` carries
``metadata.nested_agent=True`` at the as_tool() boundary, and the
renderer uses it to build the breadcrumb header.

Try it:

    python examples/verbose/nested_hitl.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk import Agent, RunConfig, Runner, VerboseConfig
from troopai.adk.tools.function_tool import function_tool

logger = logging.getLogger(__name__)


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


async def main() -> None:
    worker = Agent(
        name="AccountManager",
        llm="gpt-4o-mini",
        system_prompt=(
            "You manage user accounts. Use list_users to find users and delete_user to remove them when asked."
        ),
        tools=[list_users, delete_user],
    )

    supervisor = Agent(
        name="Supervisor",
        llm="gpt-4o-mini",
        system_prompt="You coordinate account management tasks. Use the account_manager tool to delegate work.",
        tools=[
            worker.as_tool(
                tool_description="Delegate account management tasks.",
            ),
        ],
    )

    verbose_cfg = VerboseConfig(mode="auto")
    run_config = RunConfig(verbose=verbose_cfg)

    # First call — the supervisor delegates; the worker's delete_user
    # defers; the deferral propagates up through the as_tool() boundary.
    # The panel renderer shows the breadcrumb.
    result = await Runner.arun(
        supervisor,
        "Please delete the user with id 2.",
        run_config=run_config,
    )

    if not result.requires_action or result.state is None:
        logger.info("No nested approval required. Final output: %s", result.final_output)
        return

    # Approve every nested deferral — in a real UI each would surface
    # with the full breadcrumb so the operator sees the call chain.
    for deferred in result.interruptions:
        meta = deferred.metadata
        nested = meta is not None and bool(meta.nested_agent)
        logger.info(
            "Approving %s (nested via as_tool(): %s)",
            deferred.tool_name,
            nested,
        )
        result.state.approve(
            deferred,
            approver_id="cli-user",
            reason="example auto-approval",
        )

    resumed = await Runner.arun(supervisor, result.state, run_config=run_config)
    logger.info("Final output after resume: %s", resumed.final_output)


if __name__ == "__main__":
    asyncio.run(main())

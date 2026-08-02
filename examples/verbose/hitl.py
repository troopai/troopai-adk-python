"""Verbose output — HITL (Human-in-the-Loop) visualization.

Demonstrates the ADK-first-class event vocabulary CrewAI lacks:
:const:`~troopai.adk.verbose.EVENT_HITL_APPROVAL_REQUESTED`,
:const:`~troopai.adk.verbose.EVENT_HITL_APPROVAL_GRANTED`,
:const:`~troopai.adk.verbose.EVENT_HITL_APPROVAL_REJECTED`.

When a tool with ``requires_approval=True`` is invoked, the runner
pauses before execution and emits a yellow HITL panel tagged
``🙋 hitl``. After ``RunState.approve(...)`` or ``.reject(...)`` and a
resume call, the pending panel closes to green (approved) or red
(rejected) with an optional audit message.

Try it:

    python examples/verbose/hitl.py

See also:

- ``examples/agent_patterns/human_in_the_loop.py`` for the base HITL
  pattern (without verbose visualization).
- ``examples/verbose/nested_hitl.py`` for approvals bubbling through
  ``as_tool()`` boundaries.
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
    name="send_email",
    description="Send an email to a recipient (requires human approval).",
    requires_approval=True,
)
def send_email(to: str, subject: str, body: str) -> str:
    return f"Email sent to {to} with subject '{subject}'."


async def main() -> None:
    agent = Agent(
        name="Assistant",
        llm="gpt-4o-mini",
        system_prompt=(
            "You are a helpful assistant. Use the send_email tool when asked "
            "to send mail — the approval system handles human confirmation."
        ),
        tools=[send_email],
    )

    verbose_cfg = VerboseConfig(mode="auto")
    run_config = RunConfig(verbose=verbose_cfg)

    # First call — runner pauses at the HITL gate
    result = await Runner.arun(
        agent,
        "Send an email to alice@example.com thanking her for the report.",
        run_config=run_config,
    )

    if not result.requires_action:
        logger.info("No approval needed. Final output: %s", result.final_output)
        return

    # Auto-approve each deferred call with an audit message. In a real
    # app this is where a UI would surface the request.
    for deferred in result.interruptions:
        logger.info(
            "HITL gate opened for %s; auto-approving with audit reason.",
            deferred.tool_name,
        )
        result.state.approve(
            deferred,
            approver_id="cli-user",
            reason="example auto-approval",
        )

    # Resume — the HITL panel closes with an ``approved`` verdict
    resumed = await Runner.arun(agent, result.state, run_config=run_config)
    logger.info("Final output after resume: %s", resumed.final_output)


if __name__ == "__main__":
    asyncio.run(main())

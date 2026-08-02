"""
Example: Salvaging a Partial Result with ``RunConfig.on_max_turns``

Demonstrates the ``on_max_turns`` handler — a recovery hook that fires
when a per-agent turn budget is exhausted. Instead of raising
``MaxTurnsExceeded``, the handler returns a best-effort final string
that becomes ``RunResult.final_output``. Handy for user-facing chat
flows where "I ran out of budget, here's what I have" is a better UX
than a traceback.

This example pairs a tight ``max_turns=3`` budget with a tool that
deliberately bounces the conversation around, forcing the budget to
trip. The handler then returns a salvage string.

Run:
    python examples/agent_patterns/on_max_turns.py
"""

from __future__ import annotations

import asyncio
import logging

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from troopai.adk.agents.agent import Agent
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.tools import function_tool
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


@function_tool
def check_status(system: str) -> str:
    """Return the (deliberately unstable) status of a system."""
    return f"status of {system} is indeterminate — retry"


async def salvage(agent, turns: int) -> str:
    """Handler invoked when the per-agent turn budget is exhausted."""
    logger.warning(
        "Agent %s hit the turn budget after %d turns; salvaging.",
        agent.name,
        turns,
    )
    return (
        f"[partial] Unable to produce a confident answer for "
        f"{agent.name} within {turns} turns. Please retry with more budget."
    )


async def main() -> None:
    agent = Agent(
        name="flaky_investigator",
        system_prompt=(
            "You investigate system health. Use check_status repeatedly until you are confident. Never stop early."
        ),
        tools=[check_status],
    )

    config = RunConfig(on_max_turns=salvage, verbose=VerboseConfig())
    result = await Runner.arun(
        agent,
        "What's the health of the web, db, and cache systems?",
        max_turns=3,
        run_config=config,
    )

    logger.info("final_output: %s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

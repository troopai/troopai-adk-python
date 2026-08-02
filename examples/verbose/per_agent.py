"""Verbose output — per-agent override example.

Shows that ``Agent.verbose`` overrides ``RunConfig.verbose`` for the
specific agent it is attached to. Useful in multi-agent flows where
the coordinator should be loud and colourful while a downstream
summariser should be silent (or vice versa).

Try it:

    python examples/verbose/per_agent.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk import Agent, EventStyle, RunConfig, Runner, VerboseConfig
from troopai.adk.handoffs import Handoff
from troopai.adk.verbose.config import EVENT_AGENT_START

logger = logging.getLogger(__name__)


async def main() -> None:
    # Run-level default: loud, cyan coordinator style.
    run_verbose = VerboseConfig()

    # Per-agent override for the summariser: silenced entirely so the
    # trace stays focused on the coordinator's orchestration steps.
    silent_verbose = VerboseConfig(enabled=False)

    # Per-agent override for the specialist: recoloured so it visually
    # stands out from the coordinator.
    specialist_verbose = VerboseConfig()
    specialist_verbose.styles[EVENT_AGENT_START] = EventStyle(
        color="bright_magenta",
        icon="●",
        prefix="specialist",
    )

    summariser = Agent(
        name="Summariser",
        llm="gpt-4o-mini",
        system_prompt="Summarise the conversation in one sentence.",
        verbose=silent_verbose,
    )

    specialist = Agent(
        name="Specialist",
        llm="gpt-4o-mini",
        system_prompt=(
            "You are a domain specialist. Answer the user's question briefly, then hand off to the Summariser."
        ),
        handoffs=[Handoff(target=summariser)],
        verbose=specialist_verbose,
    )

    coordinator = Agent(
        name="Coordinator",
        llm="gpt-4o-mini",
        system_prompt="You coordinate a multi-agent flow. Hand off to the Specialist for domain questions.",
        handoffs=[Handoff(target=specialist)],
    )

    result = await Runner.arun(
        coordinator,
        "What's a good way to cache LLM prompts across providers?",
        run_config=RunConfig(verbose=run_verbose),
    )

    logger.info("Final output: %s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

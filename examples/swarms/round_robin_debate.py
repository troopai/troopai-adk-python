"""Round-robin swarm: two agents debate a topic deterministically.

:class:`RoundRobinPolicy` cycles through the roster in order. There
are no ``transfer_to_<name>`` tools — the next speaker is chosen by
position, not by the LLM. This saves routing tokens entirely: the
policy injects **only** ``swarm_done`` so any debater can stop the
run when convinced. Termination still requires an explicit signal:
either an agent calls ``swarm_done`` or the
:class:`MaxTurnsTermination` caps the exchange.

Useful when:
- You want deterministic turn order (debates, panel reviews, fixed
  pipelines).
- You want zero LLM routing cost.
- You want the stop signal to be explicit but not by-absence.

Run with ``ANTHROPIC_API_KEY`` (or any litellm-supported key) set in
the environment.
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
from troopai.adk.run.config import DEFAULT_MODEL
from troopai.adk.run.runner import Runner
from troopai.adk.swarms import (
    ExplicitDoneTermination,
    MaxTurnsTermination,
    SharedContextConfig,
    SharedContextStrategy,
    Swarm,
    SwarmConfig,
    prompt_with_swarm_instructions,
)

logger = logging.getLogger(__name__)


# -- Members -----------------------------------------------------------
#
# Each debater sees the other's last message because we set
# SharedContextStrategy.LAST_N with window=2 — the most recent one
# message from the opposing side plus a handoff payload is enough
# for a coherent exchange without the token cost of a full broadcast.

optimist = Agent(
    name="optimist",
    system_prompt=prompt_with_swarm_instructions(
        "You argue that AI will augment human creativity. Respond to the "
        "pessimist's last point concisely. When you feel the debate has "
        "reached a productive synthesis, call `swarm_done` with a 1-line "
        "summary."
    ),
)

pessimist = Agent(
    name="pessimist",
    system_prompt=prompt_with_swarm_instructions(
        "You argue that AI will displace human creativity. Respond to the "
        "optimist's last point concisely. When you feel the debate has "
        "reached a productive synthesis, call `swarm_done` with a 1-line "
        "summary."
    ),
)


# -- Swarm assembly ----------------------------------------------------

debate_swarm: Swarm = (
    Swarm.new("debate", description="optimist ↔ pessimist")
    .members(optimist, pessimist)
    .entry("optimist")
    .round_robin()
    .terminate_on(ExplicitDoneTermination() | MaxTurnsTermination(10))
    # LAST_N with a small window — each debater sees the other's most
    # recent statement plus the implicit handoff payload the policy
    # surfaces between turns. shared_context lives on SwarmConfig so
    # budgets and context strategy travel together.
    .with_config(
        SwarmConfig(
            max_total_tokens=20_000,
            shared_context=SharedContextConfig(
                strategy=SharedContextStrategy.LAST_N,
                window=2,
            ),
        )
    )
    .compile()
)


# -- Driver ------------------------------------------------------------


async def main() -> None:
    logger.info("=" * 64)
    logger.info("Round-robin debate (optimist <-> pessimist)")
    logger.info("=" * 64)

    result = (
        await Runner.configure()
        .model(DEFAULT_MODEL)
        .swarm(debate_swarm)
        .verbose()
        .arun("Will AI augment or displace human creativity in the next decade?")
    )

    logger.info("Stopped because: %s", result.stop_reason)
    logger.info("Final output: %s", result.final_output)
    logger.info("Total turns: %d", result.total_turns)
    logger.info(
        "Tokens spent — total: %d (routing tokens = 0 because RoundRobin)",
        sum(u.total_tokens for u in result.per_member_usage.values()),
    )


if __name__ == "__main__":
    asyncio.run(main())

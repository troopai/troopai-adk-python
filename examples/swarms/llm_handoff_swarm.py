"""LLM-orchestrated swarm: author <-> reviewer <-> security auditor.

The :class:`LLMHandoffPolicy` injects a ``transfer_to_<member>`` tool
for every other member plus a ``swarm_done`` termination tool on each
turn. The active agent's LLM decides who speaks next by calling one
of the injected tools. When ``swarm_done`` is called — and *only*
then — the run is allowed to terminate via
:class:`ExplicitDoneTermination`.

This is the AutoGen/Strands parity pattern, but with the TroopAI
guardrails intact:

- **Agent config is never mutated.** The ``transfer_to_<name>`` tools
  are injected at turn dispatch, not at agent construction. Each
  member agent can be unit-tested in isolation with no swarm
  awareness.
- **No auto-injected system prompts.** Swarm awareness is opt-in:
  each member's prompt is wrapped explicitly with
  ``prompt_with_swarm_instructions`` so the LLM knows the
  ``transfer_to_<name>`` / ``swarm_done`` tools exist. Nothing is
  injected behind the scenes.
- **No termination by absence.** A member that fails to call any
  routing tool is simply given another turn. The swarm terminates
  when ``ExplicitDoneTermination`` catches a ``SwarmDone`` yield or
  when a composed budget fires.

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

author = Agent(
    name="author",
    system_prompt=prompt_with_swarm_instructions(
        "You are a Python author. Given a task, write a concise function. "
        "When done, hand off to 'reviewer'. If the reviewer or security "
        "auditor sent feedback, revise the code accordingly and hand back "
        "to them."
    ),
)

reviewer = Agent(
    name="reviewer",
    system_prompt=prompt_with_swarm_instructions(
        "You are a code reviewer. Look for correctness bugs, unclear names, "
        "and missing edge cases. When the code passes review, hand off to "
        "'security' for a security pass. If you see issues, hand off to "
        "'author' with the specific fix list."
    ),
)

security = Agent(
    name="security",
    system_prompt=prompt_with_swarm_instructions(
        "You are a security auditor. Look for injection, unsafe deserialisation, "
        "path traversal, and privilege-escalation risks. When the code is "
        "secure, first emit a short text summary of the final reviewed code "
        "(so RunResult.final_output surfaces something meaningful), then call "
        "`swarm_done` with 'code is production-ready'. If you see issues, "
        "hand back to 'author' with remediation notes."
    ),
)


# -- Swarm assembly -----------------------------------------------------

code_review_swarm: Swarm = (
    Swarm.new("code-review", description="author ↔ reviewer ↔ security auditor")
    .member(author)
    # handoff_description becomes the transfer_to_<name> tool description —
    # it tells the routing LLM *when* to pick each member.
    .member(reviewer, handoff_description="Code review: correctness, style, maintainability.")
    .member(security, handoff_description="Security audit: injection, traversal, privilege escalation.")
    .entry("author")
    .llm_handoff()
    # Composition: stop when either any agent calls swarm_done, OR 12
    # total turns elapse (safety net). Reads as the natural-language
    # "either / or" — see swarms/termination.py for the __or__ operator.
    .terminate_on(ExplicitDoneTermination() | MaxTurnsTermination(12))
    .with_config(
        SwarmConfig(
            max_handoffs=20,
            max_total_tokens=50_000,
            # FULL_BROADCAST matches AutoGen/Strands semantics: every member
            # sees the whole shared history so the reviewer and security
            # auditor actually see the code the author wrote. Switch to
            # SCOPED (the framework default) for cost-sensitive workflows
            # where each member only needs the prior handoff message.
            shared_context=SharedContextConfig(strategy=SharedContextStrategy.FULL_BROADCAST),
        )
    )
    .compile()
)


# -- Driver ------------------------------------------------------------


async def main() -> None:
    logger.info("=" * 64)
    logger.info("LLM-orchestrated swarm (author <-> reviewer <-> security)")
    logger.info("=" * 64)

    task = (
        "Write a Python function `safe_divide(a, b)` that divides two "
        "numbers, returning 0 on division-by-zero. Include a one-line "
        "docstring."
    )

    result = await Runner.configure().model(DEFAULT_MODEL).swarm(code_review_swarm).verbose().arun(task)

    logger.info("Stopped because: %s", result.stop_reason)
    logger.info("Final output: %s", result.final_output)
    logger.info("Total turns: %d", result.total_turns)
    logger.info("Handoff count: %d", result.handoff_count)
    logger.info("Per-member usage:")
    for agent_name, usage in result.per_member_usage.items():
        logger.info(
            "  %-10s  requests=%d input=%d output=%d",
            agent_name,
            usage.requests,
            usage.input_tokens,
            usage.output_tokens,
        )


if __name__ == "__main__":
    asyncio.run(main())

"""Sequential agent runs with explicit chaining.

Demonstrates how to feed one Task's output into the next Task's
prompt WITHOUT any framework-side prompt rewriting. The pattern is:

1. Run the upstream Task via ``Runner.arun_task``.
2. Read ``upstream_out.final_output``.
3. Construct the downstream Task with a description that embeds
   the prior output verbatim.
4. Run the downstream Task via ``Runner.arun_task``.

This is the canonical "chain two tasks" pattern in this codebase.
``Task.description`` is always exactly what the agent sees — the
framework never transforms prompts at runtime. ``TaskPipeline`` is
reserved for sequential runs with conditional skip + usage
aggregation; it is NOT the abstraction for cross-task data flow.

Run from the project root::

    python examples/tasks/sequential_pipeline.py

Requires ``ANTHROPIC_API_KEY`` (or another litellm-supported key) set
in the environment or ``.env``.
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
from troopai.adk.run import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.tasks import Task
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


researcher = Agent(
    name="researcher",
    system_prompt=(
        "You are a research assistant. Produce a concise factual "
        "summary (3-4 sentences) about the topic the user gives you. "
        "Stick to widely accepted facts; no speculation."
    ),
)

summariser = Agent(
    name="summariser",
    system_prompt=(
        "You are a summariser. The user message contains a research "
        "blurb. Compress it to ONE short sentence (under 20 words)."
    ),
)


async def main() -> None:
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    # Step 1: run the upstream Task. Its description IS the user prompt
    # — no template, no override, no chain_inputs.
    research_task = Task(
        description="Tell me about the Apollo 11 mission.",
        agent=researcher,
        name="research-apollo-11",
    )
    research_out = await Runner.arun_task(research_task, run_config=run_config)

    # Step 2: build the downstream Task's description from the prior
    # output. The developer chooses exactly how the data flows; the
    # framework does no auto-aggregation.
    summary_task = Task(
        description=f"Compress the following research blurb into one short sentence:\n\n{research_out.final_output}",
        agent=summariser,
        name="one-sentence-summary",
    )
    summary_out = await Runner.arun_task(summary_task, run_config=run_config)

    logger.info("=" * 60)
    logger.info("Research output: %s", research_out.final_output)
    logger.info("Summary output:  %s", summary_out.final_output)
    logger.info("=" * 60)
    if research_out.usage is not None and summary_out.usage is not None:
        total = research_out.usage.total_tokens + summary_out.usage.total_tokens
        logger.info("Cumulative usage: %d total tokens across 2 tasks", total)


if __name__ == "__main__":
    asyncio.run(main())

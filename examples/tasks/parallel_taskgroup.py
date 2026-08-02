"""Three Tasks running concurrently via :class:`TaskGroup`.

Demonstrates the parallel counterpart to :class:`TaskPipeline`. The
three tasks here share an agent but ask about different topics — the
group fans them out under :func:`asyncio.gather` semantics, then
returns a :class:`TaskGroupResult` with outputs in input order plus
cumulative LLM usage aggregated across every task.

Key knobs the example exercises:

- ``error_policy="collect_all"`` (the default) — every task runs to
  completion even if a sibling fails. Failures surface as
  :class:`TaskOutput` slots with ``error`` set.
- ``max_concurrent=2`` — bounds concurrency to two tasks at a time
  even though three are scheduled. Useful against provider rate
  limits.

Concurrency contract reminders (see ``TaskGroup`` docstring for
detail):

- ``on_task_start`` / ``on_task_end`` hooks fire **concurrently**
  here. Hook implementations that hold shared mutable state MUST
  lock — this example does not register any.
- When N parallel tasks share one ``Session``, events from each
  interleave. The example does not pass a session, so this is moot,
  but production code that does should give each task its own
  session if event ordering matters.

Run from the project root::

    python examples/tasks/parallel_taskgroup.py

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
from troopai.adk.tasks import Task, TaskGroup
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


researcher = Agent(
    name="researcher",
    system_prompt=(
        "You are a research assistant. Produce a concise factual "
        "summary (2-3 sentences) about whatever topic the user gives "
        "you. Stick to widely accepted facts; no speculation."
    ),
)


async def main() -> None:
    group = TaskGroup(
        tasks=(
            Task(
                description="Explain the Apollo 11 mission.",
                agent=researcher,
                name="apollo-11",
            ),
            Task(
                description="Explain photosynthesis.",
                agent=researcher,
                name="photosynthesis",
            ),
            Task(
                description="Explain the Pythagorean theorem.",
                agent=researcher,
                name="pythagoras",
            ),
        ),
        # Throttle to 2 concurrent provider calls — easier on rate
        # limits than firing all three at once.
        max_concurrent=2,
    )

    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())
    result = await Runner.arun_task_group(group, run_config=run_config)

    logger.info("=" * 60)
    for output in result.task_outputs:
        logger.info(
            "[%s] %s",
            output.task_name,
            output.final_output if output.error is None else f"ERROR: {output.error}",
        )
    logger.info("=" * 60)
    if result.context is not None:
        logger.info(
            "Cumulative usage across %d tasks: %d total tokens",
            len(result.task_outputs),
            result.context.usage.total_tokens,
        )


if __name__ == "__main__":
    asyncio.run(main())

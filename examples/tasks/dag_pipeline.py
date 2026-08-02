"""Diamond fan-out via ``Task.depends_on`` + ``TaskDependency`` per-edge filters.

Demonstrates the two halves of the DAG surface:

1. **Declarative DAG ordering** — ``intake`` runs first; ``facts`` and
   ``style`` run concurrently (they share an upstream); ``synthesise``
   waits for both. ``Task.depends_on`` switches
   ``Runner.arun_task_pipeline`` from declaration order to topological
   order.

2. **Per-edge forwarding via TaskDependency** — each upstream that
   contributes input to a downstream task is wrapped in
   ``TaskDependency(task=..., input_filter=...)``. The filter on each
   wrapper independently shapes that one upstream's contribution. The
   framework NEVER auto-aggregates upstream outputs — forwarding is
   opt-in via the wrapper, and different upstreams can carry
   different filters.

Pipeline topology::

    intake
    /    \\
   facts   style
    \\    /
    synthesise

Requires an LLM API key (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``).
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk import Agent, Runner, Task, TaskDependency, TaskPipeline
from troopai.adk.run import RunConfig
from troopai.adk.tasks.task_filters import forward_final_output
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

ARTICLE = (
    "Last quarter, our customer-support team handled 12,000 tickets — a 40% "
    "increase year over year. Mean resolution time dropped from 6.4 to 4.1 hours."
)


def _intake_agent() -> Agent:
    return Agent(
        name="intake",
        system_prompt="Echo the article back verbatim, no commentary.",
    )


def _facts_reviewer() -> Agent:
    return Agent(
        name="fact_reviewer",
        system_prompt="Review the prepended article for any factual claims that need verification. Be concise.",
    )


def _style_reviewer() -> Agent:
    return Agent(
        name="style_reviewer",
        system_prompt="Review the prepended article for tone, grammar, and clarity. Be concise.",
    )


def _synthesiser() -> Agent:
    return Agent(
        name="synthesiser",
        system_prompt="Combine the prepended reviewer feedback into one short editorial recommendation.",
    )


async def main() -> None:
    """Build the diamond pipeline with per-edge forwarding and run it."""
    intake = Task(
        description=ARTICLE,
        agent=_intake_agent(),
        task_id="intake",
    )

    # `facts` and `style` each receive intake's final output as a prepended
    # message via a per-edge filter. Each TaskDependency declares its own
    # filter — the article flows once, not duplicated in any description.
    facts = Task(
        description="Review the article above for any factual claims that need verification.",
        agent=_facts_reviewer(),
        task_id="facts",
        depends_on=[TaskDependency(task=intake, input_filter=forward_final_output)],
    )
    style = Task(
        description="Review the article above for tone, grammar, and clarity.",
        agent=_style_reviewer(),
        task_id="style",
        depends_on=[TaskDependency(task=intake, input_filter=forward_final_output)],
    )

    # `synthesise` receives BOTH reviewers' outputs as separate prepended
    # messages — each upstream gets its own TaskDependency with its own
    # filter (they happen to use the same one here, but they don't have to).
    synthesise = Task(
        description="Combine the reviewer feedback above into one short editorial recommendation.",
        agent=_synthesiser(),
        task_id="synthesise",
        depends_on=[
            TaskDependency(task=facts, input_filter=forward_final_output),
            TaskDependency(task=style, input_filter=forward_final_output),
        ],
    )

    pipeline = TaskPipeline(tasks=(intake, facts, style, synthesise))
    logger.info(
        "Topological levels: %s",
        [[t.task_id for t in level] for level in pipeline.topological_levels()],
    )

    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())
    result = await Runner.arun_task_pipeline(pipeline, run_config=run_config)
    for output in result.task_outputs:
        logger.info("%s: %s", output.task_id, str(output.final_output))


if __name__ == "__main__":
    asyncio.run(main())

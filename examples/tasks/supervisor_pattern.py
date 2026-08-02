"""Multi-agent supervisor pattern wrapped in a single Task.

Demonstrates the CrewAI / OpenAI-Agents-SDK supervisor pattern WITHOUT
the hidden manager-agent that CrewAI creates implicitly when
``process=hierarchical``. The supervisor here is just a regular
:class:`Agent` whose tools are sub-agents exposed via
``Agent.as_tool()`` — no framework code auto-spawns a supervisor.

Pattern shape::

    supervisor (Agent)
      ├── tools=[researcher.as_tool(), writer.as_tool()]
      └── system_prompt: "Coordinate the team. Use researcher first…"

    task = Task(description=..., agent=supervisor)
    result = await Runner.arun_task(task)

The supervisor's LLM decides on each turn whether to invoke
``researcher`` or ``writer`` (via tool calls) or to produce a final
answer. Every routing decision is visible in the run trace — there is
no hidden agent the developer didn't define.

Run from the project root::

    python examples/tasks/supervisor_pattern.py

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


# -- Worker sub-agents -------------------------------------------------

researcher = Agent(
    name="researcher",
    description=(
        "Researches a topic and returns 3-4 bullet points of factual information. Call this first when you need facts."
    ),
    system_prompt=(
        "You are a research assistant. Given a topic, return 3-4 "
        "concise bullet points of factual information. No prose; just "
        "bullets."
    ),
)

writer = Agent(
    name="writer",
    description=(
        "Turns bullet-point research into a polished one-paragraph "
        "answer. Call this after researcher has returned facts."
    ),
    system_prompt=(
        "You are a writer. Given bullet-point facts, weave them into a "
        "single polished paragraph (under 80 words). Do not invent "
        "new facts beyond what is given."
    ),
)


# -- Supervisor with sub-agents-as-tools -------------------------------

supervisor = Agent(
    name="supervisor",
    system_prompt=(
        "You coordinate a research team. Given a question:\n"
        "1. Call `researcher` to gather facts.\n"
        "2. Call `writer` with the facts to produce a polished answer.\n"
        "3. Return the final paragraph as your output.\n"
        "Do not answer from your own knowledge — always route through "
        "the team."
    ),
    tools=[
        researcher.as_tool(
            tool_name="researcher",
            tool_description="Research a topic and return bullet-point facts. Input: the topic to research.",
        ),
        writer.as_tool(
            tool_name="writer",
            tool_description="Turn bullet-point facts into a polished paragraph. Input: the bullets to write up.",
        ),
    ],
)


async def main() -> None:
    # The whole multi-agent run is one Task — observable as a single
    # unit via on_task_start / on_task_end and the verbose Task panel.
    task = Task(
        description="Write a short paragraph about the Apollo 11 mission.",
        agent=supervisor,
        name="apollo-research-supervisor",
        max_turns=8,  # explicit upper bound on the supervisor's tool-call loop
        metadata={"pattern": "supervisor", "workers": ["researcher", "writer"]},
    )
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())
    output = await Runner.arun_task(task, run_config=run_config)

    logger.info("=" * 60)
    logger.info("Task: %s (id=%s)", output.task_name, output.task_id)
    logger.info("Final output: %s", output.final_output)
    logger.info("Items generated: %d", len(output.new_items))
    if output.usage is not None:
        logger.info("Usage: %d total tokens", output.usage.total_tokens)
    logger.info("Metadata: %s", output.metadata)


if __name__ == "__main__":
    asyncio.run(main())

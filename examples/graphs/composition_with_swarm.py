"""Graph node hosting a Swarm — the signature composition pattern.

A ``Swarm`` drops in as a graph node via ``to_executable()`` auto-dispatch.
The graph loop calls ``SwarmExecutable.invoke`` which runs ``run_swarm_loop``;
per-node usage from the swarm's members is surfaced under the node's id in
``result.per_node_usage``.

Topology::

    triage (Agent)
    └── research (Swarm: researcher ↔ critic, RoundRobin, 4 turns max)
        └── writer (Agent)

Demonstrates:
- ``Swarm`` as a ``.node()`` executable (auto-wrapped in ``SwarmExecutable``)
- Nested lifecycle observability: usage attributed per graph node
- ``ExplicitDoneTermination | MaxTurnsTermination`` composable termination
- ``result.per_node_usage`` crossing primitive boundaries

Run::

    python examples/graphs/composition_with_swarm.py
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
from troopai.adk.graphs import Graph
from troopai.adk.run import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.swarms import (
    ExplicitDoneTermination,
    MaxTurnsTermination,
    RoundRobinPolicy,
    Swarm,
    prompt_with_swarm_instructions,
)
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# -- Triage -----------------------------------------------------------------

triage_agent = Agent(
    name="triage",
    system_prompt="Extract the core research question in one sentence.",
)


# -- Swarm members ----------------------------------------------------------

researcher = Agent(
    name="researcher",
    system_prompt=prompt_with_swarm_instructions(
        "Provide one concise finding on the topic. If the critic has validated, call swarm_done."
    ),
)

critic = Agent(
    name="critic",
    system_prompt=prompt_with_swarm_instructions(
        "Critique the researcher's finding in one sentence. If you agree it is solid, call swarm_done."
    ),
)

research_swarm = Swarm(
    members=(researcher, critic),
    entry=researcher,
    policy=RoundRobinPolicy(),
    termination=ExplicitDoneTermination() | MaxTurnsTermination(4),
)


# -- Writer -----------------------------------------------------------------

writer_agent = Agent(
    name="writer",
    system_prompt="Write a two-sentence summary suitable for an executive briefing.",
)


# -- Graph ------------------------------------------------------------------

pipeline: Graph = (
    Graph.new("swarm-in-graph", description="triage → research swarm → writer")
    .node("triage", triage_agent)
    .node("research", research_swarm)
    .node("writer", writer_agent)
    .pipe("triage", "research", "writer")
    .entry("triage")
    .terminal("writer")
    .compile()
)


# -- Driver -----------------------------------------------------------------


async def main() -> None:
    logger.info("=" * 64)
    logger.info("Graph with embedded Swarm node")
    logger.info("=" * 64)

    result = await Runner.arun_graph(
        pipeline,
        "What are the main risks of deploying LLMs in regulated industries?",
        run_config=RunConfig(verbose=VerboseConfig()),
    )

    logger.info("Status: %s", result.status.value)
    logger.info("Supersteps: %d", result.total_supersteps)
    logger.info("Final output:\n%s", result.final_output)

    logger.info("-" * 40)
    for node_id, usage in result.per_node_usage.items():
        logger.info(
            "Node %-10s — total=%d tokens",
            node_id,
            usage.total_tokens,
        )
    logger.info(
        "Graph total — %d tokens",
        result.cumulative_usage.total_tokens,
    )


if __name__ == "__main__":
    asyncio.run(main())

"""Simplest graph: three agents chained linearly with `.pipe()`.

Demonstrates:
- ``Graph.new()`` fluent builder with ``description``
- ``.pipe()`` sugar for sequential edges
- ``Runner.arun_graph()`` as the top-level entry point
- ``GraphRunResult.final_output`` and ``per_node_usage``

Run::

    python examples/graphs/linear.py
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
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# -- Agents -----------------------------------------------------------------

triage = Agent(
    name="triage",
    system_prompt="Classify the topic in one sentence. Be concise.",
)

writer = Agent(
    name="writer",
    system_prompt="Write a short paragraph on the topic. 3 sentences max.",
)

summarizer = Agent(
    name="summarizer",
    system_prompt="Summarize the input in one sentence. Be terse.",
)


# -- Graph ------------------------------------------------------------------

pipeline: Graph = (
    Graph.new("linear-pipeline", description="triage → writer → summarizer")
    .node("triage", triage)
    .node("writer", writer)
    .node("summarizer", summarizer)
    .pipe("triage", "writer", "summarizer")
    .entry("triage")
    .terminal("summarizer")
    .compile()
)


# -- Driver -----------------------------------------------------------------


async def main() -> None:
    logger.info("=" * 64)
    logger.info("Linear graph: triage -> writer -> summarizer")
    logger.info("=" * 64)

    result = await Runner.arun_graph(
        pipeline,
        "Quantum computing and its impact on cryptography",
        run_config=RunConfig(verbose=VerboseConfig()),
    )

    logger.info("Status: %s", result.status.value)
    logger.info("Supersteps: %d", result.total_supersteps)
    logger.info("Final output: %s", result.final_output)

    for node_id, usage in result.per_node_usage.items():
        logger.info(
            "Node %-12s — input=%d  output=%d  total=%d tokens",
            node_id,
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
        )

    logger.info(
        "Graph total — %d tokens",
        result.cumulative_usage.total_tokens,
    )


if __name__ == "__main__":
    asyncio.run(main())

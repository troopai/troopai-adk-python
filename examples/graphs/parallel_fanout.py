"""Fan-out graph: one entry agent fires N parallel agents, then an AND-join.

Demonstrates:
- BSP parallel execution — all fan-out nodes fire in the same superstep
- ``Merge.concat_text`` labelling every upstream contribution
- ``JoinSemantics.AND`` (default) — join waits for ALL parents
- Per-node token attribution via ``result.per_node_usage``

Topology::

    entry
    ├── legal    ─┐
    ├── finance  ─┤─→ synthesizer
    └── risk     ─┘

Run::

    python examples/graphs/parallel_fanout.py
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
from troopai.adk.graphs import Graph, Merge
from troopai.adk.run import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# -- Agents -----------------------------------------------------------------

entry_agent = Agent(
    name="entry",
    system_prompt="Restate the request clearly in two sentences.",
)

legal_agent = Agent(
    name="legal",
    system_prompt="Identify the key legal risk in one bullet point.",
)

finance_agent = Agent(
    name="finance",
    system_prompt="Identify the key financial risk in one bullet point.",
)

risk_agent = Agent(
    name="risk",
    system_prompt="Identify the key operational risk in one bullet point.",
)

synthesizer_agent = Agent(
    name="synthesizer",
    system_prompt=(
        "You receive analysis from legal, finance, and risk teams. Produce a two-sentence executive summary."
    ),
)


# -- Graph ------------------------------------------------------------------

fanout_graph: Graph = (
    Graph.new("parallel-fanout", description="1 → 3 parallel → 1 AND-join")
    .node("entry", entry_agent)
    .node("legal", legal_agent)
    .node("finance", finance_agent)
    .node("risk", risk_agent)
    .node("synthesizer", synthesizer_agent, merge=Merge.concat_text)
    .edge("entry", "legal")
    .edge("entry", "finance")
    .edge("entry", "risk")
    .edge("legal", "synthesizer")
    .edge("finance", "synthesizer")
    .edge("risk", "synthesizer")
    .entry("entry")
    .terminal("synthesizer")
    .compile()
)


# -- Driver -----------------------------------------------------------------


async def main() -> None:
    logger.info("=" * 64)
    logger.info("Parallel fan-out graph (AND-join, Merge.concat_text)")
    logger.info("=" * 64)

    result = await Runner.arun_graph(
        fanout_graph,
        "We are considering acquiring a small fintech startup.",
        run_config=RunConfig(verbose=VerboseConfig()),
    )

    logger.info("Status: %s", result.status.value)
    logger.info("Supersteps: %d", result.total_supersteps)
    logger.info("Final output:\n%s", result.final_output)

    logger.info("-" * 40)
    for node_id, usage in result.per_node_usage.items():
        logger.info(
            "Node %-12s — total=%d tokens",
            node_id,
            usage.total_tokens,
        )
    logger.info(
        "Graph total — %d tokens",
        result.cumulative_usage.total_tokens,
    )


if __name__ == "__main__":
    asyncio.run(main())

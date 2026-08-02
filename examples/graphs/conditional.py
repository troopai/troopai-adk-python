"""Conditional routing: a router agent dispatches to branch_a or branch_b.

Demonstrates:
- ``edge(..., when=predicate)`` — the single routing mechanism
- Predicate receives a ``NodeResult``; inspect ``final_text``
- Multiple terminals → ``result.final_output`` is a ``dict[str, Any]``
  keyed by terminal id
- Only the fired branch appears in ``result.per_node_usage``

Topology::

    classifier
    ├── technical  (when "technical" in classifier output)
    └── business   (when "business"  in classifier output)

Run::

    python examples/graphs/conditional.py
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
from troopai.adk.graphs import Graph, NodeResult
from troopai.adk.run import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# -- Agents -----------------------------------------------------------------

classifier = Agent(
    name="classifier",
    system_prompt=(
        "Classify the question as either 'technical' or 'business'. "
        "Reply with ONLY the single word — no punctuation, no explanation."
    ),
)

technical_expert = Agent(
    name="technical",
    system_prompt="Answer the question from a software-engineering perspective. Two sentences.",
)

business_expert = Agent(
    name="business",
    system_prompt="Answer the question from a business-strategy perspective. Two sentences.",
)


# -- Edge predicates --------------------------------------------------------


def is_technical(result: NodeResult) -> bool:
    text: str | None = result.final_text
    if text is None:
        return False
    return "technical" in text.lower()


def is_business(result: NodeResult) -> bool:
    text: str | None = result.final_text
    if text is None:
        return False
    return "business" in text.lower()


# -- Graph ------------------------------------------------------------------

conditional_graph: Graph = (
    Graph.new("conditional-routing", description="classifier → technical | business")
    .node("classifier", classifier)
    .node("technical", technical_expert)
    .node("business", business_expert)
    .edge("classifier", "technical", when=is_technical, label="technical-branch")
    .edge("classifier", "business", when=is_business, label="business-branch")
    .entry("classifier")
    .terminal("technical", "business")
    .compile()
)


# -- Driver -----------------------------------------------------------------


_RUN_CONFIG = RunConfig(verbose=VerboseConfig())


async def _run_question(question: str) -> None:
    logger.info("Question: %s", question)

    result = await Runner.arun_graph(conditional_graph, question, run_config=_RUN_CONFIG)

    logger.info("Status: %s", result.status.value)

    if isinstance(result.final_output, dict):
        for terminal_id, output in result.final_output.items():
            if output is not None:
                logger.info("Branch [%s]: %s", terminal_id, output)
    else:
        logger.info("Final output: %s", result.final_output)

    fired = list(result.per_node_usage.keys())
    logger.info("Nodes that fired: %s", fired)
    logger.info("-" * 48)


async def main() -> None:
    logger.info("=" * 64)
    logger.info("Conditional graph: classifier -> technical | business")
    logger.info("=" * 64)

    await _run_question("How does a neural network learn from data?")
    await _run_question("Should we expand into the European market?")


if __name__ == "__main__":
    asyncio.run(main())

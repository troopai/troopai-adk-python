"""Graph of Graphs — a compiled Graph as a node inside another Graph.

A compiled ``Graph`` implements ``Executable`` directly via
``Graph.invoke()``. The outer loop treats the inner graph indistinguishably
from any other node. Per-node usage from the inner graph rolls up under the
outer node id in ``result.per_node_usage``.

Inner graph (legal-review)::

    checker  →  approver

Outer graph::

    writer  →  legal-review (subgraph)

Demonstrates:
- Nested graph composition without adapters
- ``InMemoryCheckpointer`` attached via ``hooks=[...]`` to the outer run
- ``result.node_results["legal"].metadata["graph_id"]`` proving nesting
- ``result.per_node_usage`` attribution crossing graph boundaries

Run::

    python examples/graphs/nested_subgraph.py
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
from troopai.adk.graphs import Graph, InMemoryCheckpointer
from troopai.adk.run.runner import Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# -- Inner graph agents -----------------------------------------------------

compliance_checker = Agent(
    name="checker",
    system_prompt="Flag any compliance issues in one bullet point. Be brief.",
)

legal_approver = Agent(
    name="approver",
    system_prompt="Given the compliance check, approve or reject in one sentence.",
)


# -- Inner graph ------------------------------------------------------------

legal_review: Graph = (
    Graph.new("legal-review", description="compliance check → approval")
    .node("checker", compliance_checker)
    .node("approver", legal_approver)
    .pipe("checker", "approver")
    .entry("checker")
    .terminal("approver")
    .compile()
)


# -- Outer graph agents -----------------------------------------------------

content_writer = Agent(
    name="writer",
    system_prompt="Draft a two-sentence product announcement. Keep it factual.",
)


# -- Outer graph ------------------------------------------------------------
# ``legal_review`` is a compiled Graph — ``.node()`` calls ``to_executable``
# which detects it is already ``Executable`` and uses it as-is.

outer_pipeline: Graph = (
    Graph.new("outer-pipeline", description="writer → legal-review subgraph")
    .node("writer", content_writer)
    .node("legal", legal_review)
    .pipe("writer", "legal")
    .entry("writer")
    .terminal("legal")
    .compile()
)


# -- Driver -----------------------------------------------------------------


async def main() -> None:
    logger.info("=" * 64)
    logger.info("Nested subgraph: writer -> legal-review (Graph in Graph)")
    logger.info("=" * 64)

    checkpointer = InMemoryCheckpointer()

    result = (
        await Runner.configure()
        .graph(outer_pipeline)
        .hooks([checkpointer])
        .verbose(VerboseConfig())
        .arun(
            "We are launching a new AI-powered loan origination product.",
        )
    )

    logger.info("Status: %s", result.status.value)
    logger.info("Supersteps: %d", result.total_supersteps)
    logger.info("Final output:\n%s", result.final_output)

    legal_result = result.node_results.get("legal")
    if legal_result is not None:
        inner_graph_id = legal_result.metadata.get("graph_id")
        inner_status = legal_result.metadata.get("status")
        inner_steps = legal_result.metadata.get("total_supersteps")
        logger.info(
            "Inner graph_id=%s  status=%s  supersteps=%s",
            inner_graph_id,
            inner_status,
            inner_steps,
        )

    logger.info("-" * 40)
    for node_id, usage in result.per_node_usage.items():
        logger.info(
            "Node %-8s — total=%d tokens",
            node_id,
            usage.total_tokens,
        )
    logger.info(
        "Graph total — %d tokens",
        result.cumulative_usage.total_tokens,
    )

    saved = await checkpointer.list_checkpoints()
    logger.info("Checkpoint thread ids: %s", saved)


if __name__ == "__main__":
    asyncio.run(main())

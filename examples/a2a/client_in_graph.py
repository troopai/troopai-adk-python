"""Use a remote A2A agent as a node inside a local Graph.

Demonstrates the third A2A composition surface (alongside ``run()``
and ``as_tool()``): a remote A2A agent slots directly into a
:class:`~troopai.adk.graphs.graph.Graph` as a node, alongside local
``Agent`` / ``Swarm`` / callable nodes.

The graph below is a 2-node pipeline:

    [user prompt] → research_remote → summarise_local → [output]

* ``research_remote`` is the remote A2A agent.
* ``summarise_local`` is a plain local ``Agent`` running in-process.

Either node can be local or remote without the graph builder noticing
— ``GraphBuilder.node()`` calls ``to_executable()`` which dispatches
on type. This is the load-bearing property: A2A is just another
``Executable``.

Usage::

    pip install 'troopai-adk-python[a2a]'
    python examples/a2a/client_in_graph.py [REMOTE_URL]
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
import sys

from troopai.adk.a2a import A2AAgent
from troopai.adk.agents import Agent
from troopai.adk.graphs import Graph
from troopai.adk.run import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


async def main(remote_url: str) -> None:
    research_remote = A2AAgent(
        name="ResearchPeer",
        description="Authoritative remote knowledge base.",
        url=remote_url,
    )

    summarise_local = Agent(
        name="Summariser",
        system_prompt="You receive raw research notes from another agent and produce a tight 3-bullet summary.",
    )

    graph = (
        Graph.new("research_then_summarise", description="remote research → local summary")
        .node("research_remote", research_remote)  # auto-wraps via A2AExecutableAdapter
        .node("summarise_local", summarise_local)
        .edge("research_remote", "summarise_local")
        .entry("research_remote")
        .terminal("summarise_local")
        .compile()
    )

    try:
        result = await Runner.arun_graph(
            graph,
            user_prompt="What are the latest advances in retrieval augmentation?",
            run_config=RunConfig(verbose=VerboseConfig()),
        )
        logger.info("Final summary:\n%s", result.final_output)
        # Per-node usage attribution — the local summariser's tokens
        # surface here; the remote's tokens stay opaque (no usage
        # reported across the network boundary by design).
        logger.info("Per-node usage: %s", result.per_node_usage)
    finally:
        await research_remote.close()


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    asyncio.run(main(url))

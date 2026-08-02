"""Example: build and run a graph from a JSON config file.

Demonstrates ``load_topology(...)`` for the ``graph`` section: the file
declares nodes (each running a named agent), directed edges, an entry, and
terminals. The built ``Graph`` is then run live via ``Runner.arun_graph``.

Run::

    python examples/config/run_graph.py

Loading the graph needs no API key; the live run requires an LLM API key.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from pathlib import Path

from troopai.adk.config import load_topology
from troopai.adk.run import RunConfig, Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "graph.json"


async def main() -> None:
    topology = load_topology(CONFIG_PATH)
    if topology.graph is None:
        raise ValueError("config declared no graph")
    logger.info("Loaded graph %r over agents: %s", topology.graph.id, sorted(topology.agents))

    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    result = await Runner.arun_graph(
        topology.graph,
        "The impact of caching on LLM cost",
        run_config=RunConfig(verbose=VerboseConfig()),
    )
    logger.info("Status: %s", result.status.value)
    logger.info("Final output: %s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

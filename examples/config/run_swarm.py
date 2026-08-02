"""Example: build and run a swarm from a JSON config file.

Demonstrates ``load_topology(...)`` for the ``swarm`` section: the file
declares members, an entry agent, a routing policy, and a (possibly
composed) termination condition. The built ``Swarm`` is then run live.

Run::

    python examples/config/run_swarm.py

Loading the swarm needs no API key; the live run requires an LLM API key.
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
from troopai.adk.run import Runner

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "swarm.json"


async def main() -> None:
    topology = load_topology(CONFIG_PATH)
    if topology.swarm is None:
        raise ValueError("config declared no swarm")
    logger.info("Loaded swarm over agents: %s", sorted(topology.agents))

    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    result = (
        await Runner.configure().swarm(topology.swarm).verbose().arun("Explain what a vector database is, briefly.")
    )
    logger.info("Stopped because: %s", result.stop_reason)
    logger.info("Final output: %s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

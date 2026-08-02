"""Example: build a multi-agent topology from a JSON config file.

Demonstrates ``load_topology(...)``: a topology file declares several named
agents and how they reference each other. Here each agent lives in its own
file — ``topology.json`` points at ``triage.json`` and ``spanish.json`` via
``config_path`` (resolved relative to the topology file). ``triage`` hands off
to ``spanish`` by name; the loader's two-pass wiring resolves the reference
(and would resolve A<->B cycles the same way). The entry agent is run live.

Run::

    python examples/config/run_topology.py

Loading the topology needs no API key; the live turn requires an LLM API key.
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
from troopai.adk.handoffs import Handoff
from troopai.adk.run import RunConfig, Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "topology.json"


async def main() -> None:
    topology = load_topology(CONFIG_PATH)
    logger.info("Loaded topology with agents: %s", sorted(topology.agents))
    logger.info("  entry: %s", topology.entry)
    entry_agent = topology.agents[topology.entry] if topology.entry is not None else None
    if entry_agent is None:
        raise ValueError("topology has no entry agent")
    targets = [h.target.name if isinstance(h, Handoff) else h.name for h in (entry_agent.handoffs or [])]
    logger.info("  %s hands off to: %s", entry_agent.name, targets)

    # Spanish input should trigger the triage -> spanish handoff.
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    result = await Runner.arun(
        entry_agent,
        "Hola, ¿me puedes saludar?",
        run_config=RunConfig(verbose=VerboseConfig()),
    )
    logger.info("Final output: %s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

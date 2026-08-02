"""Example: build an Agent from a JSON config file.

Demonstrates the declarative agent-config loader. A strict, schema-validated
JSON document (``agent.json``) is turned into a real ``Agent`` via
``load_agent(...)``. The behavior the JSON cannot express declaratively — the
tool's function body and the output-schema class — lives in an importable
sibling module (``weather.py``) and is referenced from the JSON by a normal
dotted path (``weather.get_weather``, ``weather.WeatherReport``). References
resolve at load time relative to the config file's own directory, so no
``__main__:`` prefix is needed.

Run::

    python examples/config/run_config_agent.py

Loading the agent needs no API key; the final live turn calls the model and
requires an LLM API key in the environment (e.g. via a ``.env`` file).
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
from pathlib import Path

from troopai.adk.config import load_agent
from troopai.adk.run import RunConfig, Runner
from troopai.adk.verbose import VerboseConfig

# Make the sibling ``weather`` module importable for the typed-output cast
# below (the loader does the same internally while resolving the config).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from weather import WeatherReport

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "agent.json"


async def main() -> None:
    # Build the agent from JSON — no API key needed for this step.
    agent = load_agent(CONFIG_PATH)
    logger.info("Loaded agent %r from %s", agent.name, CONFIG_PATH.name)
    logger.info("  model: %s", agent.llm)
    logger.info("  tools: %s", [tool.name for tool in agent.tools])
    logger.info("  output_schema: %s", type(agent.output_schema).__name__)

    # Live turn — requires an LLM API key.
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    result = await Runner.arun(agent, "What's the weather in Paris?", run_config=RunConfig(verbose=VerboseConfig()))
    report = result.final_output_as(WeatherReport)
    logger.info("Structured result: city=%s | summary=%s", report.city, report.summary)


if __name__ == "__main__":
    asyncio.run(main())

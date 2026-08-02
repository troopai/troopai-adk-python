"""Verbose output — basic example.

Demonstrates the minimal wiring: pass ``VerboseConfig()`` to
``RunConfig.verbose`` to get coloured, icon-prefixed lifecycle output
on stderr with zero further configuration.

Try it:

    python examples/verbose/basic.py

Disable colour with the ``NO_COLOR`` environment variable, per
https://no-color.org/ :

    NO_COLOR=1 python examples/verbose/basic.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import json
import logging

from troopai.adk import Agent, RunConfig, Runner, VerboseConfig
from troopai.adk.tools.function_tool import FunctionTool

logger = logging.getLogger(__name__)


async def _get_weather_handler(ctx, raw_args: str) -> str:
    del ctx
    args = json.loads(raw_args) if len(raw_args) > 0 else {}
    city = args.get("city", "Paris")
    return f"{city}: 18°C, partly cloudy"


get_weather_tool = FunctionTool(
    name="get_weather",
    description="Return the current weather for a given city.",
    schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    on_invoke=_get_weather_handler,
)


async def main() -> None:
    agent = Agent(
        name="WeatherAssistant",
        llm="gpt-4o-mini",
        system_prompt="Answer weather questions using the get_weather tool.",
        tools=[get_weather_tool],
    )

    result = await Runner.arun(
        agent,
        "What's the weather in Paris?",
        run_config=RunConfig(verbose=VerboseConfig()),
    )

    logger.info("Final output: %s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

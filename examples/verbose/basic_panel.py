"""Verbose output — Panel-mode example.

Forces the CrewAI-faithful Rich-Panel renderer. Demonstrates the new
visual grammar: full-terminal-width panels with **event-kind border
colours** (yellow ``📋 Task Started``, magenta ``🤖 Agent Started``,
yellow ``🔧 Tool Execution Started (#N)``, green ``✅ Agent Final
Answer`` / ``📋 Task Completed``). The per-tool ``(#N)`` counter
increments across multiple calls to the same tool — matching CrewAI's
``ConsoleFormatter``.

Requires Rich to be installed:

    pip install rich

Try it:

    python examples/verbose/basic_panel.py

Compare against ``examples/verbose/basic.py`` which uses ``mode="auto"``
(and will downgrade to the line renderer in non-TTY environments like
this one when stdout is redirected). This script is the visual
showcase — run it in an interactive terminal to appreciate the
colour-coded borders.
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

    # ``mode="panel"`` pins the Rich backend; use ``mode="auto"`` in
    # production to let the environment decide (line for CI / non-TTY,
    # panel for interactive terminals).
    cfg = VerboseConfig(mode="panel")

    result = await Runner.arun(
        agent,
        "What's the weather in Paris?",
        run_config=RunConfig(verbose=cfg),
    )

    logger.info("Final output: %s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

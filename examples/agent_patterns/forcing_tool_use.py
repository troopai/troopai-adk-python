"""
Example: Forcing Tool Use & Tool Use Behavior

Demonstrates controlling when tools must be called and what happens after
tool execution using ``LLMConfig.tool_choice`` and ``tool_use_behavior``.

Five patterns are shown:

1. **"required" + run_llm_again** — Force tool call on the first turn, then
   ``reset_tool_choice`` (default) resets to ``"auto"`` so the LLM
   can respond with text. Without reset, this would loop forever.
2. **Stop on first tool** ("required" + stop_on_first_tool) — Force tool call,
   tool output becomes the final result directly. ``reset_tool_choice=False``
   since the loop exits before the next LLM call.
3. **Stop at specific tools** ("auto" + StopAtTools) — Only stop when a
   specific named tool is called; others continue normally.
4. **Custom behavior** ("required" + custom function) — Force tool call,
   custom function formats tool results into final output.
   ``reset_tool_choice=False`` since the loop exits before the next LLM call.
5. **"required" without reset** — Demonstrates ``reset_tool_choice=False``
   with ``stop_on_first_tool`` to show opt-out behavior.

``reset_tool_choice`` (default ``True`` when ``None``): After tools execute,
resets ``tool_choice`` from ``"required"`` to ``"auto"`` for the next LLM call.
This prevents infinite loops when ``"required"`` + ``"run_llm_again"``.
Set to ``False`` when using exit behaviors (``stop_on_first_tool``,
``StopAtTools``, custom function) that stop the loop before the next
LLM call.
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.agents import Agent
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.run import RunConfig, RunContext, Runner
from troopai.adk.tools.function_tool import function_tool
from troopai.adk.types.tools import (
    FunctionToolResult,
    StopAtTools,
    ToolsToFinalOutputResult,
)
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Shared tools
# =============================================================================


@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city to check weather for.
    """
    weather_data = {
        "paris": "Partly cloudy, 18°C, humidity 65%",
        "tokyo": "Sunny, 24°C, humidity 45%",
        "new york": "Rainy, 12°C, humidity 85%",
    }
    return weather_data.get(city.lower(), f"No weather data available for {city}")


@function_tool
def get_forecast(city: str, days: int) -> str:
    """Get a multi-day weather forecast for a city.

    Args:
        city: Name of the city to get forecast for.
        days: Number of days to forecast (1-7).
    """
    return f"Forecast for {city} ({days} days): Day 1: Sunny 20°C | Day 2: Cloudy 17°C | Day 3: Rain 14°C"


# =============================================================================
# Pattern 1: "required" + run_llm_again (safe with reset_tool_choice)
# =============================================================================
# Force the LLM to call a tool on the first turn. After tools execute,
# reset_tool_choice (True by default) resets to "auto" so the LLM can
# respond with text on the second turn.

required_with_reset_agent = Agent(
    name="Required Weather",
    system_prompt="You are a weather assistant. Use the weather tools to answer questions about weather conditions.",
    tools=[get_weather, get_forecast],
    # tool_choice="required" forces a tool call; reset_tool_choice defaults
    # to True (None) preventing infinite loop by switching to "auto" after
    # the first tool call.
    llm_config=LLMConfig(tool_choice="required"),
    tool_use_behavior="run_llm_again",
)


# =============================================================================
# Pattern 2: Stop on First Tool ("required" + stop_on_first_tool)
# =============================================================================
# The LLM must call a tool, and its raw output becomes the final result.
# No second LLM call — saves tokens when you just need the tool output.
# reset_tool_choice=False because the loop exits before the next LLM call.

stop_on_first_agent = Agent(
    name="Direct Weather",
    system_prompt="You are a weather assistant. Always check the weather for the city the user asks about.",
    tools=[get_weather, get_forecast],
    llm_config=LLMConfig(
        tool_choice="required",
        reset_tool_choice=False,
    ),
    tool_use_behavior="stop_on_first_tool",
)


# =============================================================================
# Pattern 3: Stop at Specific Tools ("auto" + StopAtTools)
# =============================================================================
# Only stop when ``get_forecast`` is called — ``get_weather`` results
# still go back to the LLM for a conversational response.

stop_at_forecast_agent = Agent(
    name="Selective Stop Weather",
    system_prompt=(
        "You are a weather assistant. For current weather, use get_weather "
        "and provide a friendly summary. For forecasts, use get_forecast."
    ),
    tools=[get_weather, get_forecast],
    llm_config=LLMConfig(tool_choice="auto"),
    tool_use_behavior=StopAtTools(stop_at_tool_names=["get_forecast"]),
)


# =============================================================================
# Pattern 4: Custom Behavior ("required" + custom function)
# =============================================================================
# The LLM must call a tool. A custom function processes the results and
# decides whether they become the final output.


async def format_weather_results(
    ctx: RunContext[None],
    results: list[FunctionToolResult],
) -> ToolsToFinalOutputResult:
    """Format tool results into a structured weather report."""
    parts = []
    for result in results:
        parts.append(f"[{result.name}] {result.output}")

    formatted = "\n".join(parts)
    return ToolsToFinalOutputResult(
        is_final_output=True,
        final_output=f"=== Weather Report ===\n{formatted}",
    )


custom_behavior_agent = Agent(
    name="Custom Weather",
    system_prompt="You are a weather assistant. Check the weather for the city the user asks about.",
    tools=[get_weather, get_forecast],
    llm_config=LLMConfig(
        tool_choice="required",
        reset_tool_choice=False,
    ),
    tool_use_behavior=format_weather_results,
)


# =============================================================================
# Running the examples
# =============================================================================


async def main():
    prompt = "What's the weather like in Paris?"
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    # Pattern 1: "required" + run_llm_again — safe with reset_tool_choice
    logger.info("=" * 60)
    logger.info("Pattern 1: required + run_llm_again (reset_tool_choice=True)")
    logger.info("=" * 60)
    result = await Runner.arun(required_with_reset_agent, prompt, run_config=run_config)
    logger.info(f"Output:\n{result.final_output}\n")

    # Pattern 2: Stop on first — raw tool output is the final result
    logger.info("=" * 60)
    logger.info("Pattern 2: Stop on First Tool (required + stop_on_first_tool)")
    logger.info("=" * 60)
    result = await Runner.arun(stop_on_first_agent, prompt, run_config=run_config)
    logger.info(f"Output:\n{result.final_output}\n")

    # Pattern 3: Stop at forecast — get_weather goes back to LLM,
    # but get_forecast would stop immediately
    logger.info("=" * 60)
    logger.info("Pattern 3a: StopAtTools — current weather (goes to LLM)")
    logger.info("=" * 60)
    result = await Runner.arun(stop_at_forecast_agent, prompt, run_config=run_config)
    logger.info(f"Output:\n{result.final_output}\n")

    logger.info("=" * 60)
    logger.info("Pattern 3b: StopAtTools — forecast (stops immediately)")
    logger.info("=" * 60)
    result = await Runner.arun(
        stop_at_forecast_agent,
        "Give me a 3-day forecast for Tokyo.",
        run_config=run_config,
    )
    logger.info(f"Output:\n{result.final_output}\n")

    # Pattern 4: Custom function formats the output
    logger.info("=" * 60)
    logger.info("Pattern 4: Custom Behavior (required + custom function)")
    logger.info("=" * 60)
    result = await Runner.arun(custom_behavior_agent, prompt, run_config=run_config)
    logger.info(f"Output:\n{result.final_output}\n")


if __name__ == "__main__":
    asyncio.run(main())

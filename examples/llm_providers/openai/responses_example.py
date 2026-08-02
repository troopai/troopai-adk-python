"""
Example: Native OpenAI Responses API provider.

Two snippets:

1. Basic turn with a function tool on ``OpenAIResponsesLLM("gpt-5.1")``.
2. Hosted web_search passthrough via ``LLMConfig.extra_body`` — no
   framework tool wrapper; the provider JSON is forwarded verbatim.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/llm_providers/openai/responses_example.py
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

from troopai.adk.agents import Agent
from troopai.adk.llms.openai import OpenAIResponsesConfig, OpenAIResponsesLLM
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


async def _get_weather_handler(ctx, raw_args: str) -> str:
    del ctx
    args = json.loads(raw_args) if len(raw_args) > 0 else {}
    city = args.get("city", "Paris")
    return f"The weather in {city} is 72°F and sunny."


weather_tool = FunctionTool(
    name="get_weather",
    description="Return the current weather for a city.",
    schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    on_invoke=_get_weather_handler,
)


# -- 1. Basic turn with a function tool --------------------------------------


async def basic_turn_with_tool(run_config: RunConfig) -> None:
    agent = Agent(
        name="Assistant",
        system_prompt="You are a friendly weather bot.",
        llm=OpenAIResponsesLLM("gpt-5.1"),
        tools=[weather_tool],
    )

    result = await Runner.arun(agent, "What's the weather in Paris?", run_config=run_config)
    logger.info("[Basic] %s", result.final_output)


# -- 2. Hosted web_search via extra_body -------------------------------------


async def hosted_web_search(run_config: RunConfig) -> None:
    """Pass native provider tool JSON through ``extra_body``.

    No framework ``WebSearchTool`` class exists on purpose — provider
    tool wrappers would force the framework to mirror every provider's
    tool schema. The raw JSON is forwarded verbatim by the converter.
    """
    cfg = OpenAIResponsesConfig(
        extra_body={
            "tools": [
                {
                    "type": "web_search",
                    "user_location": {"type": "approximate", "country": "US"},
                }
            ],
        },
    )
    agent = Agent(
        name="SearchBot",
        system_prompt="Answer questions using web search.",
        llm=OpenAIResponsesLLM("gpt-5.1"),
        llm_config=cfg,
    )

    result = await Runner.arun(agent, "What were the top tech headlines today?", run_config=run_config)
    logger.info("[WebSearch] %s", result.final_output)


async def main() -> None:
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())
    await basic_turn_with_tool(run_config)
    await hosted_web_search(run_config)


if __name__ == "__main__":
    asyncio.run(main())

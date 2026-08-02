"""Native Google Gemini API examples.

Six self-contained snippets exercise every feature of
``GeminiLLM`` / ``GeminiConfig``:

1. Basic non-streaming generate_content.
2. Streaming with token-by-token delta rendering.
3. Function tools (multi-turn tool-use loop).
4. Extended thinking via ``GeminiConfig.thinking_config``.
5. Structured output via native ``response_schema``.
6. Hosted Google Search via the typed ``WebSearchTool``.

Usage:
    export GEMINI_API_KEY=...
    python examples/llm_providers/gemini/gemini_example.py
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

from google.genai.types import ThinkingConfig
from pydantic import BaseModel

from troopai.adk.agents import Agent
from troopai.adk.llms.gemini import GeminiConfig, GeminiLLM
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools import function_tool
from troopai.adk.tools.hosted import WebSearchTool
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

# Pin a representative Flash model — fast and inexpensive.
MODEL = "gemini-2.5-flash"


# -- 1. Basic non-streaming -------------------------------------------------


async def basic_turn(run_config: RunConfig) -> None:
    agent = Agent(
        name="Concise",
        system_prompt="You are concise. One sentence answers only.",
        llm=GeminiLLM(model=MODEL),
    )
    result = await Runner.arun(agent, "What's the capital of France?", run_config=run_config)
    logger.info("[Basic] %s", result.final_output)
    usage = result.context.usage
    logger.info(
        "[Basic] usage: input=%d output=%d total=%d",
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
    )


# -- 2. Streaming -----------------------------------------------------------


async def streaming_turn(run_config: RunConfig) -> None:
    agent = Agent(
        name="Storyteller",
        system_prompt="You are a vivid storyteller.",
        llm=GeminiLLM(model=MODEL),
    )
    result = Runner.run(agent, "Tell me a two-sentence story about a curious fox.", stream=True, run_config=run_config)
    chunks: list[str] = []
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, str):
            chunks.append(event.data)
    logger.info("[Streamed] %s", "".join(chunks))


# -- 3. Function tools ------------------------------------------------------


@function_tool
def get_weather(city: str) -> str:
    """Return a fake weather report for *city*.

    Args:
        city: The city to look up.
    """
    return f"It's 18°C and partly cloudy in {city}."


async def tools_turn(run_config: RunConfig) -> None:
    agent = Agent(
        name="WeatherBot",
        system_prompt="Use the get_weather tool when asked about weather.",
        llm=GeminiLLM(model=MODEL),
        tools=[get_weather],
    )
    result = await Runner.arun(agent, "What's the weather in Tokyo?", run_config=run_config)
    logger.info("[Tools] %s", result.final_output)


# -- 4. Extended thinking ---------------------------------------------------


async def thinking_turn(run_config: RunConfig) -> None:
    config = GeminiConfig(
        thinking_config=ThinkingConfig(thinking_budget=2048, include_thoughts=True),
    )
    agent = Agent(
        name="Mathematician",
        system_prompt="Solve carefully. Show brief reasoning in the answer.",
        llm=GeminiLLM(model=MODEL),
        llm_config=config,
    )
    result = await Runner.arun(
        agent,
        "Two trains start 600 km apart and approach each other at 90 and 110 km/h. When do they meet?",
        run_config=run_config,
    )
    logger.info("[Thinking] %s", result.final_output)


# -- 5. Structured output ---------------------------------------------------


class Sentiment(BaseModel):
    polarity: str
    confidence: float


async def structured_output_turn(run_config: RunConfig) -> None:
    agent = Agent(
        name="SentimentAnalyzer",
        system_prompt="Classify the sentiment of the user's text.",
        llm=GeminiLLM(model=MODEL),
        output_schema=Sentiment,
    )
    result = await Runner.arun(agent, "I had a fantastic day at the park!", run_config=run_config)
    verdict = result.final_output
    logger.info("[Structured] polarity=%s confidence=%.2f", verdict.polarity, verdict.confidence)


# -- 6. Hosted Google Search via WebSearchTool ------------------------------


async def hosted_search_turn(run_config: RunConfig) -> None:
    agent = Agent(
        name="Researcher",
        system_prompt="Use Google Search to find current information.",
        llm=GeminiLLM(model=MODEL),
        tools=[WebSearchTool()],
    )
    result = await Runner.arun(agent, "What's the current weather in Fribourg, Switzerland?", run_config=run_config)
    logger.info("[HostedSearch] %s", result.final_output)


# -- main -------------------------------------------------------------------


async def main() -> None:
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())
    snippets = [
        ("basic", basic_turn),
        ("streaming", streaming_turn),
        ("tools", tools_turn),
        ("thinking", thinking_turn),
        ("structured_output", structured_output_turn),
        ("hosted_search", hosted_search_turn),
    ]
    # Run every snippet and report each; if any failed, exit non-zero at the end
    # so a broken run surfaces as FAILED instead of hiding behind a clean exit.
    failures: list[str] = []
    for name, fn in snippets:
        logger.info("\n=== %s ===", name)
        try:
            await fn(run_config)
        except Exception as exc:
            logger.exception("[%s] failed: %s", name, exc)
            failures.append(name)
    if len(failures) > 0:
        logger.error("Snippets failed: %s", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

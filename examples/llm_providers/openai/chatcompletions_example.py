"""
Example: Native OpenAI Chat Completions API provider.

Two snippets:

1. Streaming with ``OpenAIChatCompletionsLLM("gpt-4o")``.
2. Structured output via ``Agent.output_schema`` (converter resolves
   to ``response_format={"type": "json_schema", ...}``).

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/llm_providers/openai/chatcompletions_example.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from pydantic import BaseModel

from troopai.adk.agents import Agent
from troopai.adk.llms.openai import OpenAIChatCompletionsLLM
from troopai.adk.run import RunConfig, Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# -- 1. Streaming -----------------------------------------------------------


async def streaming_turn(run_config: RunConfig) -> None:
    agent = Agent(
        name="Storyteller",
        system_prompt="You are a concise storyteller.",
        llm=OpenAIChatCompletionsLLM("gpt-4o"),
    )

    result = Runner.run(agent, "Tell me a two-sentence story about a fox.", stream=True, run_config=run_config)

    chunks: list[str] = []
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, str):
            chunks.append(event.data)

    logger.info("[Streamed] %s", "".join(chunks))


# -- 2. Structured output ---------------------------------------------------


class Sentiment(BaseModel):
    polarity: str  # "positive" | "negative" | "neutral"
    confidence: float


async def structured_output_turn(run_config: RunConfig) -> None:
    agent = Agent(
        name="SentimentAnalyzer",
        system_prompt="Return a structured sentiment verdict.",
        llm=OpenAIChatCompletionsLLM("gpt-4o"),
        output_schema=Sentiment,
    )

    result = await Runner.arun(agent, "I had a fantastic day at the park!", run_config=run_config)
    verdict = result.final_output
    logger.info("[Structured] polarity=%s confidence=%.2f", verdict.polarity, verdict.confidence)


async def main() -> None:
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())
    await streaming_turn(run_config)
    await structured_output_turn(run_config)


if __name__ == "__main__":
    asyncio.run(main())

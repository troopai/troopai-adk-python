"""Native Anthropic Messages API examples.

Six self-contained snippets exercise every feature of
``AnthropicLLM`` / ``AnthropicConfig``:

1. Basic non-streaming Messages call.
2. Streaming with token-by-token delta rendering.
3. Function tools (multi-turn tool-use loop).
4. Extended thinking via ``AnthropicConfig.thinking``.
5. Structured output via the synthetic-tool pattern
   (``AgentOutputSchema``).
6. Prompt caching via ``AnthropicConfig.auto_cache_control``.
7. Framework retry policy on transient failures.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/llm_providers/anthropic/anthropic_example.py
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

from pydantic import BaseModel

from troopai.adk.agents import Agent
from troopai.adk.llms import LLMConfig
from troopai.adk.llms.anthropic import AnthropicConfig, AnthropicLLM
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools import function_tool
from troopai.adk.types.llms import LLMRetryPolicy
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

# Pin a representative Sonnet model — feel free to change.
MODEL = "claude-sonnet-4-5-20250929"


# -- 1. Basic non-streaming -------------------------------------------------


async def basic_turn(run_config: RunConfig) -> None:
    agent = Agent(
        name="Concise",
        system_prompt="You are a concise assistant. One sentence answers only.",
        llm=AnthropicLLM(model=MODEL),
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
        llm=AnthropicLLM(model=MODEL),
    )
    result = Runner.run(agent, "Tell me a two-sentence story about a curious fox.", stream=True, run_config=run_config)

    chunks: list[str] = []
    async for event in result.stream_events():
        # The Runner forwards the LLM's part_delta strings as
        # raw_response_event payloads — accumulate to render live.
        if event.type == "raw_response_event" and isinstance(event.data, str):
            chunks.append(event.data)

    logger.info("[Streamed] %s", "".join(chunks))


# -- 3. Function tools (tool-use loop) --------------------------------------


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
        llm=AnthropicLLM(model=MODEL),
        tools=[get_weather],
    )
    result = await Runner.arun(agent, "What's the weather in Tokyo?", run_config=run_config)
    logger.info("[Tools] %s", result.final_output)


# -- 4. Extended thinking ---------------------------------------------------


async def extended_thinking_turn(run_config: RunConfig) -> None:
    config = AnthropicConfig(
        thinking={"type": "enabled", "budget_tokens": 2048},
        # Anthropic requires max_tokens > thinking budget; the model
        # bumps the floor automatically when needed.
    )
    agent = Agent(
        name="Mathematician",
        system_prompt="Solve math problems carefully. Show your reasoning briefly in the answer.",
        llm=AnthropicLLM(model=MODEL),
        llm_config=config,
    )
    result = await Runner.arun(
        agent,
        "If a train leaves city A at 90 km/h and another leaves city B at 110 km/h, "
        "and the cities are 600 km apart, when do they meet?",
        run_config=run_config,
    )
    logger.info("[Thinking] %s", result.final_output)


# -- 5. Structured output ---------------------------------------------------


class Sentiment(BaseModel):
    polarity: str  # "positive" | "negative" | "neutral"
    confidence: float


async def structured_output_turn(run_config: RunConfig) -> None:
    agent = Agent(
        name="SentimentAnalyzer",
        system_prompt="Classify the sentiment of the user's text.",
        llm=AnthropicLLM(model=MODEL),
        output_schema=Sentiment,
    )
    result = await Runner.arun(agent, "I had a fantastic day at the park!", run_config=run_config)
    verdict = result.final_output
    logger.info("[Structured] polarity=%s confidence=%.2f", verdict.polarity, verdict.confidence)


# -- 6. Prompt caching ------------------------------------------------------


# Anthropic's prompt-caching minimum cacheable prefix is ~1024 tokens
# for Sonnet-class models. The repetition makes the system prompt long
# enough that the second turn produces a visible cache_read.
LONG_SYSTEM = (
    "You are a senior staff engineer answering questions about Anthropic's Messages API. "
    "Be precise, terse, and cite the relevant API surface when relevant. "
    "Prefer working code over prose. Always use the latest API conventions. "
    "When asked about parameters, list each one with its type and a one-line semantic. "
    "When asked about errors, name the exception class and its HTTP status. "
    "When asked about streaming, describe the RawMessageStreamEvent variants. "
    "When asked about tools, distinguish ToolParam from ServerToolUseBlock. "
    "When asked about caching, mention CacheControlEphemeralParam and TTL tiers. "
    "When asked about thinking, mention ThinkingConfigEnabledParam and budget_tokens. "
    "When asked about service tiers, distinguish auto vs standard_only behaviour. "
) * 16  # repeat to comfortably exceed Anthropic's caching prefix floor


async def caching_turn(run_config: RunConfig) -> None:
    config = AnthropicConfig(auto_cache_control=True, cache_control_ttl="5m")
    agent = Agent(
        name="StaffEngineer",
        system_prompt=LONG_SYSTEM,
        llm=AnthropicLLM(model=MODEL),
        llm_config=config,
    )

    # First turn populates the cache (cache_creation_input_tokens > 0).
    first = await Runner.arun(agent, "What does messages.create return when stream=True?", run_config=run_config)
    first_usage = first.context.usage
    logger.info("[Caching] first turn output: %s", first.final_output)
    logger.info(
        "[Caching] first usage: input=%d cache_creation=%d cache_read=%d",
        first_usage.input_tokens,
        first_usage.input_tokens_details.cache_creation_input_tokens,
        first_usage.input_tokens_details.cached_tokens,
    )

    # Second turn (within TTL) reads from the cache (cache_read_input_tokens > 0).
    second = await Runner.arun(agent, "And how do you set extended thinking?", run_config=run_config)
    second_usage = second.context.usage
    logger.info("[Caching] second turn output: %s", second.final_output)
    logger.info(
        "[Caching] second usage: input=%d cache_creation=%d cache_read=%d",
        second_usage.input_tokens,
        second_usage.input_tokens_details.cache_creation_input_tokens,
        second_usage.input_tokens_details.cached_tokens,
    )


# -- 7. Retry policy --------------------------------------------------------


async def retry_policy_turn(run_config: RunConfig) -> None:
    # Demonstrates the wiring; the policy is a no-op when no transient
    # error fires. Set a low max_retries so a misconfigured key fails
    # fast rather than burning the full budget.
    config = LLMConfig(
        retry_policy=LLMRetryPolicy(
            max_retries=3,
            initial_delay=0.5,
            multiplier=2.0,
            jitter=True,
        ),
    )
    agent = Agent(
        name="Resilient",
        system_prompt="Reply briefly.",
        llm=AnthropicLLM(model=MODEL),
        llm_config=config,
    )
    result = await Runner.arun(agent, "Say 'hello' once.", run_config=run_config)
    logger.info("[Retry] %s", result.final_output)


# -- main -------------------------------------------------------------------


async def main() -> None:
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())
    snippets = [
        ("basic", basic_turn),
        ("streaming", streaming_turn),
        ("tools", tools_turn),
        ("thinking", extended_thinking_turn),
        ("structured_output", structured_output_turn),
        ("caching", caching_turn),
        ("retry_policy", retry_policy_turn),
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

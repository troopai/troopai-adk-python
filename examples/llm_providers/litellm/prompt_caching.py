"""
Example: Prompt Caching with LiteLLM

Demonstrates prompt caching via LiteLLMConfig provider-specific fields.
Each provider has different caching mechanisms:

1. **Anthropic** — Cache control injection points (auto-inserted by litellm).
2. **OpenAI** — Automatic caching with optional routing hints.
3. **Gemini** — Pre-created CachedContent reference.

Ref: https://docs.litellm.ai/docs/completion/prompt_caching

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...

    python examples/llm_providers/litellm/prompt_caching.py
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from typing import cast

from litellm.types.integrations.anthropic_cache_control_hook import CacheControlInjectionPoint

from troopai.adk.agents import Agent
from troopai.adk.llms.litellm.litellm_model import LiteLLMConfig
from troopai.adk.run import RunConfig, Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

LONG_SYSTEM_PROMPT = "You are a helpful assistant. " * 100  # Needs 1024+ tokens for caching


# -- 1. Anthropic caching (injection points) ----------------------------------


async def anthropic_caching_example(run_config: RunConfig) -> None:
    """Anthropic caching via cache_control_injection_points.

    litellm's AnthropicCacheControlHook injects cache_control blocks
    at the specified message locations.

    Supported by: Anthropic, Google Gemini/Vertex AI.
    Ref: https://docs.litellm.ai/docs/completion/prompt_caching#anthropic
    """
    # LiteLLM's TypedDict requires nullable keys even when targeting by role/index.
    cache_control_points: list[CacheControlInjectionPoint] = [
        cast(
            CacheControlInjectionPoint,
            {"location": "message", "role": "system", "index": None, "control": None},
        ),
        cast(
            CacheControlInjectionPoint,
            {"location": "message", "role": None, "index": -1, "control": None},
        ),
    ]

    agent = Agent(
        name="CachedAgent",
        system_prompt=LONG_SYSTEM_PROMPT,
        llm="claude-sonnet-4-20250514",
        llm_config=LiteLLMConfig(
            cache_control_injection_points=cache_control_points,
        ),
    )

    # First call — cache miss (writes cache)
    result1 = await Runner.arun(agent, "What is 2+2?", run_config=run_config)
    logger.info("[Anthropic Cache] Call 1: %s", result1.final_output)
    details1 = result1.context.usage.input_tokens_details
    cache_creation_tokens = details1.cache_creation_input_tokens if details1 is not None else 0
    logger.info("  Cache creation tokens: %d", cache_creation_tokens)

    # Second call — cache hit
    result2 = await Runner.arun(agent, "What is 3+3?", run_config=run_config)
    logger.info("[Anthropic Cache] Call 2: %s", result2.final_output)
    details2 = result2.context.usage.input_tokens_details
    cached_tokens = details2.cached_tokens if details2 is not None else 0
    logger.info("  Cached tokens: %d", cached_tokens)


# -- 2. OpenAI caching (automatic with routing hints) -------------------------


async def openai_caching_example(run_config: RunConfig) -> None:
    """OpenAI automatic caching with optional routing hints.

    OpenAI caches automatically for 1024+ token prompts.
    prompt_cache_key improves cache hit rates.

    Supported by: OpenAI, Deepseek.
    Ref: https://docs.litellm.ai/docs/completion/prompt_caching#openai
    """
    agent = Agent(
        name="CachedAgent",
        system_prompt=LONG_SYSTEM_PROMPT,
        llm="gpt-4o",
        llm_config=LiteLLMConfig(
            prompt_cache_key="my-app-stable",
            prompt_cache_retention="24h",
        ),
    )

    result = await Runner.arun(agent, "What is 2+2?", run_config=run_config)
    logger.info("[OpenAI Cache] %s", result.final_output)


# -- Main ---------------------------------------------------------------------


async def main() -> None:
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())
    logger.info("=== Prompt Caching Examples ===\n")
    await anthropic_caching_example(run_config)
    # await openai_caching_example(run_config)


if __name__ == "__main__":
    asyncio.run(main())

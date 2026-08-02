"""
Example: Reasoning / Extended Thinking with LiteLLM

Demonstrates configuring reasoning via LiteLLMConfig provider-specific fields:

1. **Anthropic** — Extended thinking with budget control via ``thinking`` param.
2. **Gemini/OpenAI** — Reasoning effort via ``reasoning_effort`` param.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    export GEMINI_API_KEY=...

    python examples/llm_providers/litellm/reasoning.py
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.agents import Agent
from troopai.adk.llms.litellm.litellm_model import LiteLLMConfig
from troopai.adk.run import RunConfig, Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

MATH_PROBLEM = (
    "Find all positive integers n such that n^2 + 2n + 2 is divisible by n + 1. Prove your answer is complete."
)


# -- 1. Reasoning effort (provider-agnostic) ----------------------------------


async def reasoning_effort_example(run_config: RunConfig) -> None:
    """Reasoning effort works across providers — litellm maps it.

    Supported by: OpenAI, Anthropic, Gemini, DeepSeek, Bedrock.
    Ref: https://docs.litellm.ai/docs/reasoning_content
    """
    agent = Agent(
        name="MathSolver",
        system_prompt="You are an expert mathematician. Show your work.",
        llm="gemini/gemini-2.5-flash",
        llm_config=LiteLLMConfig(reasoning_effort="high"),
    )

    result = await Runner.arun(agent, MATH_PROBLEM, run_config=run_config)
    logger.info("[Reasoning Effort] %s", result.final_output)


# -- 2. Anthropic thinking budget ---------------------------------------------


async def anthropic_thinking_example(run_config: RunConfig) -> None:
    """Anthropic extended thinking with explicit token budget.

    The ``thinking`` param is Anthropic-specific. Shape:
    ``{"type": "enabled", "budget_tokens": N}``

    Ref: https://docs.litellm.ai/docs/reasoning_content
    """
    agent = Agent(
        name="MathSolver",
        system_prompt="You are an expert mathematician. Show your work.",
        llm="claude-sonnet-4-20250514",
        llm_config=LiteLLMConfig(
            max_output_tokens=16000,
            thinking={"type": "enabled", "budget_tokens": 8000},
        ),
    )

    result = await Runner.arun(agent, MATH_PROBLEM, run_config=run_config)
    logger.info("[Anthropic Thinking] %s", result.final_output)


# -- Main ---------------------------------------------------------------------


async def main() -> None:
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())
    logger.info("=== Reasoning / Extended Thinking Examples ===\n")

    # Run whichever provider you have API keys for.
    await reasoning_effort_example(run_config)
    # await anthropic_thinking_example(run_config)


if __name__ == "__main__":
    asyncio.run(main())

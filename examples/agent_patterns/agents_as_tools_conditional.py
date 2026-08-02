"""
Example: Conditional Agent Tool Enablement

Demonstrates dynamically enabling or disabling agent tools at runtime
based on context using the ``enabled`` callback on ``as_tool()``.

A multilingual assistant has three translation sub-agents. Which agents
are available depends on the user's subscription tier:

| Tier     | Available Languages    |
|----------|------------------------|
| Free     | Spanish only           |
| Basic    | Spanish + French       |
| Premium  | Spanish + French + Japanese |

When ``enabled`` returns ``False``, the tool is completely hidden from
the LLM — it doesn't appear in the tool list, so the LLM cannot call it.
This is different from validation (which rejects after the call).

Key concepts shown:
- ``as_tool(enabled=callable)`` — callable receives ``RunContext[TContext]``
- ``@dataclass`` context type for typed runtime state
- Tools hidden from LLM based on runtime context (not just static config)
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from dataclasses import dataclass

from troopai.adk.agents import Agent
from troopai.adk.run import RunConfig, RunContext, Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Context: user subscription tier
# =============================================================================


@dataclass
class UserProfile:
    """Runtime context carrying the user's subscription information."""

    name: str
    subscription: str  # "free", "basic", or "premium"


# =============================================================================
# Enablement callbacks
# =============================================================================
# These receive RunContext[UserProfile] and return bool.
# When False, the tool is invisible to the LLM.


def is_basic_or_above(ctx: RunContext[UserProfile]) -> bool:
    """Enable for basic and premium subscribers."""
    return ctx.context.subscription in ("basic", "premium")


def is_premium(ctx: RunContext[UserProfile]) -> bool:
    """Enable for premium subscribers only."""
    return ctx.context.subscription == "premium"


# =============================================================================
# Translation sub-agents
# =============================================================================

spanish_agent = Agent(
    name="Spanish Translator",
    system_prompt=(
        "You are a Spanish translator. Translate the given text to Spanish "
        "accurately and naturally. Output only the translation."
    ),
)

french_agent = Agent(
    name="French Translator",
    system_prompt=(
        "You are a French translator. Translate the given text to French "
        "accurately and naturally. Output only the translation."
    ),
)

japanese_agent = Agent(
    name="Japanese Translator",
    system_prompt=(
        "You are a Japanese translator. Translate the given text to Japanese "
        "accurately and naturally. Output only the translation."
    ),
)


# =============================================================================
# Supervisor with conditional tools
# =============================================================================

supervisor = Agent(
    name="Translation Assistant",
    system_prompt=(
        "You are a multilingual translation assistant. Use the available "
        "translation tools to help the user. If the user requests a language "
        "that is not available, let them know which languages you can "
        "translate to and suggest upgrading their subscription."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_description="Translate text to Spanish.",
            # enabled defaults to True — always available (all tiers)
        ),
        french_agent.as_tool(
            tool_description="Translate text to French.",
            enabled=is_basic_or_above,  # Basic + Premium only
        ),
        japanese_agent.as_tool(
            tool_description="Translate text to Japanese.",
            enabled=is_premium,  # Premium only
        ),
    ],
)


# =============================================================================
# Running the examples
# =============================================================================


async def main():
    prompt = "Translate 'The weather is beautiful today' to all available languages."
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    tiers = [
        UserProfile(name="Alice", subscription="free"),
        UserProfile(name="Bob", subscription="basic"),
        UserProfile(name="Carol", subscription="premium"),
    ]

    for profile in tiers:
        logger.info("=" * 60)
        logger.info(f"User: {profile.name} ({profile.subscription} tier)")
        logger.info("=" * 60)

        result = await Runner.arun(
            supervisor,
            prompt,
            context=profile,
            run_config=run_config,
        )
        logger.info(f"\nResponse:\n{result.final_output}\n")


if __name__ == "__main__":
    asyncio.run(main())

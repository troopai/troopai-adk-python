"""Deferred tool loading — hide rare tools until the LLM searches.

Demonstrates ``FunctionTool.defer_loading=True`` plus
``build_tool_search()``. Three deferred tools start hidden from the
LLM's tool list; a search tool reveals them by name / description
match; ``build_tools()`` admits revealed tools on the following call.

This example exercises the visibility plumbing directly (without
spinning up an LLM) so the lifecycle is observable. In production the
LLM does the searching and revealing autonomously.

Usage:
    python examples/tools/deferred_tool_loading.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.agents import Agent
from troopai.adk.run.llm_calls import build_tools
from troopai.adk.tools import build_tool_search, function_tool
from troopai.adk.tools.function_tool import FunctionTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Three tools the LLM rarely needs — hidden by default
# ---------------------------------------------------------------------------


@function_tool(
    name="payment_processor",
    description="Charge a credit card via the payment gateway.",
    defer_loading=True,
)
def payment_processor(amount_cents: int) -> str:
    return f"charged {amount_cents}"


@function_tool(
    name="schema_migrator",
    description="Run a SQL schema migration.",
    defer_loading=True,
)
def schema_migrator(sql: str) -> str:
    return "migration applied"


@function_tool(
    name="cache_warmer",
    description="Pre-populate the read-through cache for a region.",
    defer_loading=True,
)
def cache_warmer(region: str) -> str:
    return f"warmed {region}"


# ---------------------------------------------------------------------------
# A common tool the LLM uses every turn — always visible
# ---------------------------------------------------------------------------


@function_tool(name="echo", description="Echo a message back.")
def echo(message: str) -> str:
    return message


# ---------------------------------------------------------------------------
# The search tool — wired against the deferred catalogue
# ---------------------------------------------------------------------------


def visible_names(tools: list) -> list[str]:
    return [t.name for t in tools if isinstance(t, FunctionTool)]


async def main() -> None:
    search = build_tool_search(
        [payment_processor, schema_migrator, cache_warmer],
    )

    agent = Agent(
        name="Worker",
        system_prompt="Use tools when helpful.",
        tools=[
            echo,
            payment_processor,
            schema_migrator,
            cache_warmer,
            search,
        ],
    )

    # ── Turn 1: only echo + tool_search are visible ────────────────
    logger.info("=== Turn 1: initial visibility ===")
    visible = await build_tools(agent)
    assert visible is not None
    logger.info("LLM sees: %s", visible_names(visible))

    # ── The LLM (simulated) calls tool_search for "credit card" ──
    logger.info("\n=== LLM calls tool_search('credit card') ===")
    assert search.on_invoke is not None
    from unittest.mock import MagicMock

    result = await search.on_invoke(MagicMock(), '{"query": "credit card", "top_k": 3}')
    logger.info("search result: %s", result)

    # ── Turn 2: payment_processor is now revealed ──────────────────
    logger.info("\n=== Turn 2: post-reveal ===")
    visible = await build_tools(agent)
    assert visible is not None
    logger.info("LLM sees: %s", visible_names(visible))

    # ── Reveal a second tool ───────────────────────────────────────
    logger.info("\n=== LLM calls tool_search('migration') ===")
    result = await search.on_invoke(MagicMock(), '{"query": "migration", "top_k": 3}')
    logger.info("search result: %s", result)

    # ── Turn 3: both payment_processor and schema_migrator visible ─
    logger.info("\n=== Turn 3: cumulative reveal ===")
    visible = await build_tools(agent)
    assert visible is not None
    logger.info("LLM sees: %s", visible_names(visible))


if __name__ == "__main__":
    asyncio.run(main())

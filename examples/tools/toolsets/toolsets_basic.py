"""Toolset composition — namespacing + filtering + combining.

Demonstrates the four mainstream toolset variants:

1. ``FunctionToolset`` — wrap a flat list of function tools.
2. ``PrefixedToolset`` — namespace every tool name (prevents
   collisions across multiple sources).
3. ``FilteredToolset`` — drop tools whose predicate returns False
   per turn, with access to the live ``RunContext``.
4. ``CombinedToolset`` — flatten multiple toolsets into one merged
   collection.

This example calls ``build_tools()`` directly so the materialised
tool list is visible without spinning up an LLM. In real use the
toolsets are passed to ``Agent(tools=[...])`` and the framework
calls ``build_tools()`` on every turn.

Usage:
    python examples/tools/toolsets/toolsets_basic.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.agents.agent import Agent
from troopai.adk.run.context import RunContext
from troopai.adk.run.llm_calls import build_tools
from troopai.adk.tools import (
    CombinedToolset,
    FunctionToolset,
    function_tool,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Define a few tools across two domains
# ---------------------------------------------------------------------------


@function_tool(name="get_temp", description="Get the current temperature")
def get_temp(city: str) -> str:
    return f"{city}: 22C"


@function_tool(name="get_conditions", description="Get current weather conditions")
def get_conditions(city: str) -> str:
    return f"{city}: sunny"


@function_tool(name="query", description="Run a SQL query")
def query(sql: str) -> str:
    return f"rows for: {sql}"


@function_tool(name="restart", description="Restart the database (admin only)")
def restart() -> str:
    return "database restarting"


# ---------------------------------------------------------------------------
# Compose toolsets
# ---------------------------------------------------------------------------


# 1. Two namespaced toolsets — names are prefixed so the LLM sees
#    weather_get_temp, weather_get_conditions, db_query, etc.
weather = FunctionToolset(tools=[get_temp, get_conditions]).prefixed("weather")
db = FunctionToolset(tools=[query]).prefixed("db")


# 2. A filtered toolset — restart is only visible when the run context
#    carries an admin role.
def admin_only(ctx: RunContext, tool) -> bool:  # type: ignore[no-untyped-def]
    """Predicate: keep the tool only for admins."""
    return ctx.context.get("role") == "admin"


admin_actions = FunctionToolset(tools=[restart]).filtered(admin_only)


# 3. Combine all of them as one entry on the agent.
all_tools = CombinedToolset(toolsets=[weather, db, admin_actions])


async def main() -> None:
    agent = Agent(
        name="Operator",
        system_prompt="Help operators with weather, the database, and admin tasks.",
        tools=[all_tools],
    )

    # As a non-admin user, restart is hidden.
    user_ctx = RunContext(context={"role": "user"})
    user_tools = await build_tools(agent, context=user_ctx)
    logger.info(
        "User-context tool list: %s",
        sorted(getattr(t, "name", "") for t in (user_tools or [])),
    )

    # As an admin, restart is exposed.
    admin_ctx = RunContext(context={"role": "admin"})
    admin_tools = await build_tools(agent, context=admin_ctx)
    logger.info(
        "Admin-context tool list: %s",
        sorted(getattr(t, "name", "") for t in (admin_tools or [])),
    )


if __name__ == "__main__":
    asyncio.run(main())

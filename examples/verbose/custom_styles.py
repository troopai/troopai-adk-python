"""Verbose output — custom styles example.

Shows how to recolour, re-icon, and mute individual events. The style
table is a mutable ``dict[str, EventStyle]``, so override any subset
and leave the rest at default.

Try it:

    python examples/verbose/custom_styles.py
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

from troopai.adk import Agent, EventStyle, RunConfig, Runner, VerboseConfig
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.verbose.config import (
    EVENT_AGENT_END,
    EVENT_AGENT_START,
    EVENT_LLM_END,
    EVENT_LLM_START,
    EVENT_TOOL_END,
    EVENT_TOOL_START,
)

logger = logging.getLogger(__name__)


async def _search_handler(ctx, raw_args: str) -> str:
    del ctx
    args = json.loads(raw_args) if len(raw_args) > 0 else {}
    query = args.get("query", "")
    return f"results for '{query}': 3 hits"


search_tool = FunctionTool(
    name="search",
    description="Run a search query and return a short summary.",
    schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    on_invoke=_search_handler,
)


def build_verbose_config() -> VerboseConfig:
    """Return a recoloured, partially-muted VerboseConfig.

    Design choices made here:
    - Mute both LLM events (noisy on tool-heavy runs).
    - Recolour tool calls to bright magenta so they pop visually.
    - Add timestamps for timing debugging.
    """
    cfg = VerboseConfig(show_timestamps=True)
    # Mute LLM events entirely by replacing with neutral styles.
    cfg.styles[EVENT_LLM_START] = EventStyle()
    cfg.styles[EVENT_LLM_END] = EventStyle()
    # Recolour tool events. ``panel_title`` overrides the CrewAI-style
    # title in the panel backend; ``icon`` + ``prefix`` drive the line
    # backend label.
    cfg.styles[EVENT_TOOL_START] = EventStyle(
        color="bright_magenta",
        icon="▶",
        prefix="tool",
        panel_title="▶ Tool Call",
    )
    cfg.styles[EVENT_TOOL_END] = EventStyle(
        color="bright_magenta",
        icon="◼",
        prefix="tool",
    )
    # Stronger agent start/end icons + custom panel title.
    cfg.styles[EVENT_AGENT_START] = EventStyle(
        color="bold cyan",
        icon="❚❚▶",
        prefix="agent",
        panel_title="❚❚▶ Agent Activating",
    )
    cfg.styles[EVENT_AGENT_END] = EventStyle(
        color="bold green",
        icon="✔",
        prefix="agent",
    )
    return cfg


async def main() -> None:
    agent = Agent(
        name="Researcher",
        llm="gpt-4o-mini",
        system_prompt="Use the search tool to answer questions concisely.",
        tools=[search_tool],
    )

    result = await Runner.arun(
        agent,
        "What does 'anthropic' mean?",
        run_config=RunConfig(verbose=build_verbose_config()),
    )

    logger.info("Final output: %s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

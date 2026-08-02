"""Streaming tool results — partial output events without LLM token cost.

A *streaming function tool* yields :class:`ToolStreamEvent` instances
instead of returning a single value. The framework drains the
iterator, surfaces each non-``"done"`` event as a
``RunItemType.TOOL_PARTIAL_OUTPUT`` event to consumers of
``Runner.arun(stream=True)``, and the LLM still sees exactly one
tool-result message — the value carried on the terminal ``"done"``
event.

This is the typical shape for long-running tools that want to show
progress (search, scraping, LLM-as-tool calls) without flooding the
LLM with chunked tool outputs.

Usage:
    python examples/tools/streaming_tool_results.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from collections.abc import AsyncIterator

from troopai.adk.agents.agent import Agent
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.run.stream import RunItemStreamEvent, RunItemType
from troopai.adk.tools import function_tool
from troopai.adk.types.tools import ToolStreamEvent
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


@function_tool(
    name="search_documents",
    description="Search the corpus for documents matching a query.",
    streaming=True,
)
async def search_documents(query: str) -> AsyncIterator[ToolStreamEvent]:
    """Stream incremental search progress, then a final summary.

    The ``"done"`` event's ``response`` becomes the single tool-result
    message the LLM sees. Everything yielded before that surfaces only
    to the run's stream consumer (this script's ``async for`` loop).
    """
    yield ToolStreamEvent(type="part_start", index=0)
    yield ToolStreamEvent(type="part_delta", delta=f"Searching '{query}'...")
    await asyncio.sleep(0.1)
    yield ToolStreamEvent(type="part_delta", delta=" scanning index...")
    await asyncio.sleep(0.1)
    yield ToolStreamEvent(type="part_delta", delta=" ranking matches...")
    await asyncio.sleep(0.1)
    yield ToolStreamEvent(type="part_end", index=0)
    yield ToolStreamEvent(
        type="done",
        response=f"Found 3 documents for '{query}': doc_a, doc_b, doc_c.",
    )


async def main() -> None:
    agent = Agent(
        name="research-agent",
        system_prompt="You are a research assistant. Use search_documents to find information.",
        tools=[search_documents],
    )

    result = Runner.run(
        agent,
        "Find documents about ADK streaming.",
        stream=True,
        run_config=RunConfig(verbose=VerboseConfig()),
    )

    partial_chunks: list[str] = []
    async for event in result.stream_events():
        if not isinstance(event, RunItemStreamEvent):
            continue
        if event.name == RunItemType.TOOL_PARTIAL_OUTPUT:
            inner = event.item["event"]
            if inner.delta is not None:
                partial_chunks.append(inner.delta)
                logger.info("[partial] %s", inner.delta)
            else:
                logger.info("[partial] (%s, index=%s)", inner.type, inner.index)
        elif event.name == RunItemType.TOOL_OUTPUT:
            logger.info("[tool_output] %s", event.item["output"])

    logger.info("--- summary ---")
    logger.info("partial chunks observed: %d", len(partial_chunks))
    logger.info("final agent output: %s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

"""Tool middleware — agent-global interceptors around tool execution.

Demonstrates:

1. ``ToolLoggingMiddleware`` — the shipped baseline that logs start/end
   of every call.
2. A custom timing middleware that records per-tool wall-clock
   duration.
3. A circuit-breaker middleware that short-circuits via
   ``ToolMiddlewareTermination``.

Each middleware is applied agent-globally via
``Agent(middleware=Middleware(tools=[...]))``, so every tool the
agent calls flows through the chain in declaration order. The
``tools`` slot on ``Middleware`` is the function-scope layer; the
``agents`` and ``llms`` slots will be added when the matching
Protocols ship.

Usage:
    python examples/tools/middleware/middleware_basic.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
import time
from typing import Any

from troopai.adk.agents.agent import Agent
from troopai.adk.agents.middleware import Middleware
from troopai.adk.run.tools_executor import maybe_wrap_with_agent_middleware
from troopai.adk.tools import (
    FunctionTool,
    ToolLoggingMiddleware,
    ToolMiddlewareTermination,
    function_tool,
)
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.types.output.function_tool_call_result import FunctionToolCallResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@function_tool(name="search", description="Search the database")
def search(query: str) -> str:
    return f"results for {query}"


@function_tool(name="summarise", description="Summarise text")
async def summarise(text: str) -> str:
    # Simulate I/O latency
    await asyncio.sleep(0.05)
    return f"summary of {text[:20]}..."


# ---------------------------------------------------------------------------
# Custom middleware: per-tool timing accumulator
# ---------------------------------------------------------------------------


class TimingMiddleware:
    """Record per-tool wall-clock latency."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    async def __call__(
        self,
        ctx: ToolContext,
        tool: FunctionTool,
        args: dict[str, Any],
        next: Any,
    ) -> FunctionToolCallResult:
        start = time.monotonic()
        result = await next(ctx, tool, args)
        duration = time.monotonic() - start
        self.totals[tool.name] = self.totals.get(tool.name, 0.0) + duration
        self.counts[tool.name] = self.counts.get(tool.name, 0) + 1
        return result

    def report(self) -> None:
        for name, total in self.totals.items():
            count = self.counts[name]
            logger.info(
                "  %s — %d calls, total %.4fs, avg %.4fs",
                name,
                count,
                total,
                total / count,
            )


# ---------------------------------------------------------------------------
# Custom middleware: circuit breaker
# ---------------------------------------------------------------------------


class CircuitBreakerMiddleware:
    """Reject calls to a tool whose name matches ``blocked_name``."""

    def __init__(self, blocked_name: str) -> None:
        self.blocked_name = blocked_name

    async def __call__(
        self,
        ctx: ToolContext,
        tool: FunctionTool,
        args: dict[str, Any],
        next: Any,
    ) -> FunctionToolCallResult:
        if tool.name == self.blocked_name:
            raise ToolMiddlewareTermination(
                FunctionToolCallResult(
                    type="function_call_output",
                    call_id=ctx.tool_call_id or "",
                    output=f"[breaker tripped for {tool.name}]",
                )
            )
        return await next(ctx, tool, args)


# ---------------------------------------------------------------------------
# Compose and run
# ---------------------------------------------------------------------------


async def main() -> None:
    timing = TimingMiddleware()
    breaker = CircuitBreakerMiddleware(blocked_name="summarise")

    agent = Agent(
        name="Service",
        system_prompt="Help users.",
        tools=[search, summarise],
        # Order: outermost first. So Logging wraps Timing wraps Breaker.
        middleware=Middleware(tools=[ToolLoggingMiddleware(), timing, breaker]),
    )

    logger.info("\n--- Calling 'search' (should pass through) ---")
    invoke = maybe_wrap_with_agent_middleware(search, agent.middleware.tools)
    ctx = ToolContext(
        tool_name="search",
        tool_call_id="c1",
        tool_arguments={"query": "demo"},
        raw_arguments='{"query":"demo"}',
    )
    out = await invoke(ctx, '{"query": "demo"}')
    logger.info("  result: %s", out)

    logger.info("\n--- Calling 'summarise' (should be tripped) ---")
    invoke2 = maybe_wrap_with_agent_middleware(summarise, agent.middleware.tools)
    ctx2 = ToolContext(
        tool_name="summarise",
        tool_call_id="c2",
        tool_arguments={"text": "x" * 50},
        raw_arguments='{"text":"' + "x" * 50 + '"}',
    )
    out2 = await invoke2(ctx2, '{"text": "' + "x" * 50 + '"}')
    logger.info("  result: %s", out2)

    logger.info("\n--- Timing report ---")
    timing.report()


if __name__ == "__main__":
    asyncio.run(main())

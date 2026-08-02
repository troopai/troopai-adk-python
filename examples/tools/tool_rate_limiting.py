"""Per-tool rate limiting — sliding-window throttle for FunctionTool.

Demonstrates ``ToolRateLimit`` with both behaviours:

1. **wait** (default) — saturated calls sleep until a slot opens; the
   LLM is unaware of throttling.
2. **error** — saturated calls return a clear rate-limit message; the
   LLM can adapt (back off, switch tools, summarise instead).

This example invokes ``acquire_rate_slot()`` directly so the timing
shape is visible without spinning up an LLM. ``time.monotonic`` is the
clock used internally; we keep wall-clock real here.

Usage:
    python examples/tools/tool_rate_limiting.py
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

from troopai.adk.tools import ToolRateLimit, function_tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A tool capped at 3 calls per minute, behaviour="error"
# ---------------------------------------------------------------------------


@function_tool(
    name="payment_api",
    description="Process a payment.",
    rate_limit=ToolRateLimit(rpm=3, behavior="error"),
)
def payment_api(amount: int) -> str:
    return f"charged {amount} cents"


# ---------------------------------------------------------------------------
# A tool capped at 60 calls per minute (1/sec), behaviour="wait"
# ---------------------------------------------------------------------------


@function_tool(
    name="search_api",
    description="Search the public knowledge base.",
    max_calls_per_minute=60,
)
def search_api(query: str) -> str:
    return f"results for {query}"


async def scenario_error() -> None:
    logger.info("=== Scenario A: behaviour=error, rpm=3 ===")
    logger.info(
        "rapid-fire 5 acquires; 4th and 5th hit the cap and return False",
    )
    for i in range(5):
        ok = await payment_api.acquire_rate_slot()
        logger.info("  call #%d: admitted=%s", i + 1, ok)


async def scenario_wait() -> None:
    logger.info("\n=== Scenario B: behaviour=wait, rpm=60 ===")
    logger.info("4 calls back-to-back; window has plenty of room")
    start = time.monotonic()
    for i in range(4):
        await search_api.acquire_rate_slot()
        logger.info("  call #%d admitted at +%.3fs", i + 1, time.monotonic() - start)


async def main() -> None:
    await scenario_error()
    await scenario_wait()


if __name__ == "__main__":
    asyncio.run(main())

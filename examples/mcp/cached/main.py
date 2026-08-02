"""Cache-vs-invalidate behaviour for MCP tool listing.

The default ``cache_tools_list=True`` keeps the converted tool list
in memory across turns, saving a ``list_tools`` round-trip per turn.
This example demonstrates a manual ``invalidate_tools_cache()`` call
forcing a re-fetch.

Run::

    python examples/mcp/cached/main.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.mcp import MCPServerStdio, MCPServerStdioParams

logger = logging.getLogger(__name__)


async def main() -> None:
    server = MCPServerStdio(
        name="cached",
        params=MCPServerStdioParams(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
        ),
        cache_tools_list=True,
    )

    async with server:
        first = await server.list_tools()
        logger.info("First fetch: %d tools", len(first))

        # Cache hit — same names, no round-trip.
        cached = await server.list_tools()
        assert [t.name for t in cached] == [t.name for t in first]
        logger.info("Cache hit: %d tools (same names)", len(cached))

        # Manual invalidation forces a re-fetch.
        server.invalidate_tools_cache()
        refreshed = await server.list_tools()
        logger.info("After invalidation: %d tools", len(refreshed))


if __name__ == "__main__":
    asyncio.run(main())

"""SSE MCP transport (MCP-spec-deprecated).

The SSE transport is **deprecated** by the MCP spec; prefer
``MCPServerStreamableHttp``. This example exists for compatibility
with servers still serving SSE.

Prerequisite: an SSE-serving MCP server reachable at
``MCP_SSE_URL`` (default ``http://localhost:4002/sse``). Boot one
locally with::

    npx -y @modelcontextprotocol/server-everything sse --port 4002

Run::

    python examples/mcp/sse/main.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
import os

from troopai.adk.mcp import MCPConnectionError, MCPServerSse, MCPServerSseParams

logger = logging.getLogger(__name__)


async def main() -> None:
    url = os.environ.get("MCP_SSE_URL", "http://localhost:4002/sse")
    server = MCPServerSse(
        name="sse-demo",
        params=MCPServerSseParams(url=url),
    )
    try:
        async with server:
            tools = await server.list_tools()
            logger.info("Tools advertised: %d", len(tools))
    except MCPConnectionError:
        logger.error(
            "Could not reach the SSE MCP server at %s. Start one with: "
            "npx -y @modelcontextprotocol/server-everything sse --port 4002 "
            "(or override the URL via MCP_SSE_URL).",
            url,
        )


if __name__ == "__main__":
    asyncio.run(main())

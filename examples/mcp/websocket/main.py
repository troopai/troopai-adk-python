"""WebSocket MCP transport.

Requires the optional ``websockets`` package. Run an MCP server with
WebSocket transport before starting this example, or replace
``MCP_WS_URL`` with a deployed endpoint.

Prerequisite: a WebSocket-serving MCP server reachable at
``MCP_WS_URL`` (default ``ws://localhost:4001/mcp``).

Run::

    python examples/mcp/websocket/main.py
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

from troopai.adk.mcp import MCPConnectionError, MCPServerWebsocket, MCPServerWebsocketParams

logger = logging.getLogger(__name__)


async def main() -> None:
    url = os.environ.get("MCP_WS_URL", "ws://localhost:4001/mcp")
    server = MCPServerWebsocket(
        name="ws-demo",
        params=MCPServerWebsocketParams(url=url),
    )
    try:
        async with server:
            tools = await server.list_tools()
            logger.info("Tools advertised: %d", len(tools))
    except MCPConnectionError:
        logger.error(
            "Could not reach the WebSocket MCP server at %s. Start a "
            "WebSocket-serving MCP server, or override the URL via MCP_WS_URL.",
            url,
        )


if __name__ == "__main__":
    asyncio.run(main())

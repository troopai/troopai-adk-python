"""Streamable HTTP MCP server example.

Connects to an MCP server reachable over HTTP (POST + SSE for
server-pushed messages). Replace ``MCP_URL`` with the endpoint of
your own deployment, or run the reference server in streamable-HTTP
mode locally::

    npx -y @modelcontextprotocol/server-everything streamableHttp

then export ``MCP_URL=http://localhost:3001/mcp`` (the reference
server binds 3001 by default) and run this example.

Run::

    python examples/mcp/streamable_http/main.py
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

from troopai.adk.agents.agent import Agent
from troopai.adk.mcp import MCPConnectionError, MCPServerStreamableHttp, MCPServerStreamableHttpParams
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools.toolsets import MCPToolset
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


async def main() -> None:
    url = os.environ.get("MCP_URL", "http://localhost:4000/mcp")

    server = MCPServerStreamableHttp(
        name="http-demo",
        params=MCPServerStreamableHttpParams(url=url),
    )

    agent = Agent(
        name="MCP HTTP Demo",
        system_prompt="Use the MCP tools available over HTTP.",
        tools=[MCPToolset(server=server)],
        llm="claude-haiku-4-5",
    )

    try:
        result = await Runner.arun(agent, "List the available tools.", run_config=RunConfig(verbose=VerboseConfig()))
        logger.info("Final output:\n%s", result.final_output)
    except MCPConnectionError:
        logger.error(
            "Could not reach the HTTP MCP server at %s. Start one with: "
            "npx -y @modelcontextprotocol/server-everything streamableHttp "
            "(default port 3001 → MCP_URL=http://localhost:3001/mcp).",
            url,
        )


if __name__ == "__main__":
    asyncio.run(main())

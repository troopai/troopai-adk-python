"""Stdio MCP server example.

Spawns ``@modelcontextprotocol/server-everything`` (the upstream
reference test server) and exposes its tools to an Agent. The
``MCPToolset`` is dropped into ``Agent.tools`` like any other
toolset; the Runner takes care of connecting on first use and
disposing on run end.

Prerequisites:
    - ``npx`` on PATH (Node.js)
    - An ``ANTHROPIC_API_KEY`` (or any LiteLLM-compatible provider key)
      in ``.env`` or the environment.

Run::

    python examples/mcp/stdio/main.py
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
from troopai.adk.mcp import MCPServerStdio, MCPServerStdioParams
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools.toolsets import MCPToolset
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


async def main() -> None:
    server = MCPServerStdio(
        name="everything",
        params=MCPServerStdioParams(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
        ),
    )

    agent = Agent(
        name="MCP Stdio Demo",
        system_prompt=(
            "You have access to a small set of demo tools from the MCP "
            "server-everything reference server. Use any one of them to "
            "complete the user's request."
        ),
        tools=[MCPToolset(server=server)],
        llm="claude-haiku-4-5",
    )

    result = await Runner.arun(
        agent,
        "List the tools you have and pick one to call.",
        run_config=RunConfig(verbose=VerboseConfig()),
    )
    logger.info("Final output:\n%s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

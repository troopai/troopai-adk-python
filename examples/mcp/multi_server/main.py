"""Multi-server MCP example with name prefixing.

Two MCP servers wrapped as ``MCPToolset`` instances coexist in a
single agent's ``tools`` list. ``.prefixed("svc")`` namespaces each
server's tools so duplicate names across servers do not trigger
``ToolsetNameConflictError``.

Run::

    python examples/mcp/multi_server/main.py
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


def _stdio(name: str) -> MCPServerStdio:
    return MCPServerStdio(
        name=name,
        params=MCPServerStdioParams(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
        ),
    )


async def main() -> None:
    a_toolset = MCPToolset(server=_stdio("a")).prefixed("a")
    b_toolset = MCPToolset(server=_stdio("b")).prefixed("b")

    agent = Agent(
        name="MCP Multi Demo",
        system_prompt=(
            "Two MCP demo servers are attached. Tools are prefixed with "
            "``a_`` and ``b_`` so you can tell them apart. List the tools "
            "you can see."
        ),
        tools=[a_toolset, b_toolset],
        llm="claude-haiku-4-5",
    )

    result = await Runner.arun(agent, "List the available tools.", run_config=RunConfig(verbose=VerboseConfig()))
    logger.info("Final output:\n%s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

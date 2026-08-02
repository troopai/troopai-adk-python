"""Surface MCP resources and prompts as agent-callable tools.

``use_mcp_resources=True`` adds a synthetic ``read_<server>_resource``
tool the LLM can invoke to fetch a resource by URI.
``expose_prompts_as_tools=True`` converts every server prompt to a
``FunctionTool``.

Run::

    python examples/mcp/resources_and_prompts/main.py
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
        name="Resources & Prompts Demo",
        system_prompt=(
            "Inspect the available tools, including the synthetic "
            "``read_everything_resource`` and any ``prompt_*`` tools."
        ),
        tools=[
            MCPToolset(
                server=server,
                use_mcp_resources=True,
                expose_prompts_as_tools=True,
            ),
        ],
        llm="claude-haiku-4-5",
    )
    result = await Runner.arun(
        agent,
        "List all your tools and explain what each does.",
        run_config=RunConfig(verbose=VerboseConfig()),
    )
    logger.info("Final output:\n%s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

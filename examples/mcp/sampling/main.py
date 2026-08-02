"""MCP sampling: server calls back into the host LLM.

When ``MCPServerStdio(... llm=my_llm)`` is set, the underlying
``ClientSession`` advertises the sampling capability and forwards
``sampling/createMessage`` requests to ``my_llm.acomplete``.

Run::

    python examples/mcp/sampling/main.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.llms.litellm.litellm_model import LiteLLM
from troopai.adk.mcp import MCPServerStdio, MCPServerStdioParams

logger = logging.getLogger(__name__)


async def main() -> None:
    server = MCPServerStdio(
        name="everything",
        params=MCPServerStdioParams(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
        ),
        llm=LiteLLM(model="claude-haiku-4-5"),
    )
    async with server:
        tools = await server.list_tools()
        logger.info(
            "Connected with sampling enabled (%d tools). The server can "
            "now call back into the host LLM for chained reasoning.",
            len(tools),
        )


if __name__ == "__main__":
    asyncio.run(main())

"""Per-request auth header injection via ``header_provider``.

Demonstrates token rotation: ``header_provider`` is called fresh on
every outbound request, so the bearer token can change without
re-creating the underlying HTTP session. Useful for short-lived
credentials (STS, federated auth, OIDC).

Prerequisite: a streamable-HTTP MCP server reachable at
``MCP_URL`` (default ``http://localhost:4000/mcp``). The reference
server can be started with::

    npx -y @modelcontextprotocol/server-everything streamableHttp

(default port 3001 — set ``MCP_URL=http://localhost:3001/mcp``).

Run::

    python examples/mcp/auth_headers/main.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import itertools
import logging
import os

from troopai.adk.agents.agent import Agent
from troopai.adk.mcp import (
    HeaderProvider,
    MCPConnectionError,
    MCPServerStreamableHttp,
    MCPServerStreamableHttpParams,
)
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools.toolsets import MCPToolset
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


def _make_rotating_provider() -> HeaderProvider:
    """Return a closure that rotates through demo tokens.

    A real provider would call out to STS / OIDC / Vault here. The
    counter approximates the rotation pattern so example logs show
    distinct tokens reaching the server.
    """
    counter = itertools.count(1)

    def provider() -> dict[str, str]:
        n = next(counter)
        return {"Authorization": f"Bearer demo-token-{n}"}

    return provider


async def main() -> None:
    url = os.environ.get("MCP_URL", "http://localhost:4000/mcp")
    server = MCPServerStreamableHttp(
        name="auth-demo",
        params=MCPServerStreamableHttpParams(
            url=url,
            header_provider=_make_rotating_provider(),
        ),
    )

    agent = Agent(
        name="MCP Auth Demo",
        system_prompt="Make a couple of tool calls so the demo can rotate tokens.",
        tools=[MCPToolset(server=server)],
        llm="claude-haiku-4-5",
    )

    try:
        result = await Runner.arun(
            agent, "List the tools, then call any one of them.", run_config=RunConfig(verbose=VerboseConfig())
        )
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

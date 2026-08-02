"""Hosted MCP example — provider-side execution via OpenAI Responses API.

The OpenAI Responses API can call MCP servers directly without the
client process holding the connection. ``HostedMCPTool`` declares
the server label + URL; the ADK forwards a typed config block to
``client.responses.create``.

Prerequisites:
    - A publicly reachable MCP server. OpenAI's hosted MCP fetch
      runs from provider infrastructure, so ``localhost`` URLs are
      rejected with HTTP 424. For local development, expose the
      reference server via ``ngrok``/``cloudflared`` or set
      ``MCP_URL`` to any publicly reachable MCP endpoint::

          npx -y @modelcontextprotocol/server-everything streamableHttp
          ngrok http 3001
          export MCP_URL=https://<your-ngrok>.ngrok.app/mcp

    - ``OPENAI_API_KEY`` in ``.env`` or the environment.

Run::

    python examples/mcp/hosted_openai_responses/main.py
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
from troopai.adk.llms.openai.openai_responses_model import OpenAIResponsesLLM
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools.hosted import HostedMCPTool
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


async def main() -> None:
    server_url = os.environ.get("MCP_URL")
    if server_url is None:
        logger.error(
            "Set MCP_URL to a publicly reachable MCP server (OpenAI's hosted "
            "MCP fetch runs from provider infrastructure, so localhost URLs "
            "fail with HTTP 424). Example: expose a local server via ngrok."
        )
        return

    agent = Agent(
        name="MCP Hosted Demo",
        system_prompt="Use the MCP server provided to answer questions.",
        tools=[
            HostedMCPTool(
                server_label="everything",
                server_url=server_url,
                require_approval="never",
            ),
        ],
        llm=OpenAIResponsesLLM(model="gpt-4o-mini"),
    )

    result = await Runner.arun(agent, "What can the MCP server do?", run_config=RunConfig(verbose=VerboseConfig()))
    logger.info("Final output:\n%s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

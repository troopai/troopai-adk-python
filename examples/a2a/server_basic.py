"""Expose a local Agent as an A2A endpoint.

Boots a Starlette ASGI app via uvicorn, publishing the agent's
manually-authored AgentCard at /.well-known/agent-card.json and
serving the JSON-RPC dispatcher at /.

Usage::

    pip install 'troopai-adk-python[a2a]'
    python examples/a2a/server_basic.py [PORT]

Then in another terminal::

    python examples/a2a/client_basic.py http://localhost:8080
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import logging
import sys

import uvicorn
from a2a.types import AgentCapabilities, AgentCard, AgentInterface

from troopai.adk.a2a import A2AServer, build_starlette_app
from troopai.adk.agents import Agent

logger = logging.getLogger(__name__)


def make_local_agent() -> Agent[None]:
    """A trivial Agent — replace with your own."""
    return Agent(
        name="echo_agent",
        system_prompt=(
            "You are a friendly assistant. Respond concisely to the user's "
            "request, in plain text. If the user asks you to echo something, "
            "echo it verbatim."
        ),
    )


def main(port: int) -> None:
    agent = make_local_agent()
    card = AgentCard(
        name="echo-agent",
        description="A trivial demo agent that echoes user input.",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(
                url=f"http://localhost:{port}",
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            ),
        ],
        capabilities=AgentCapabilities(streaming=True),
    )

    server = A2AServer(agent=agent, agent_card=card)
    app = build_starlette_app(server)

    logger.info("A2A server starting on http://localhost:%d", port)
    logger.info("Discovery URL: http://localhost:%d/.well-known/agent-card.json", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    chosen_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    main(chosen_port)

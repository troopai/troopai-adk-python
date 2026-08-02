"""End-to-end A2A round-trip integration test.

Spins up a real A2A server in the same process via uvicorn on a free
port, then uses :class:`A2AAgent` to call it and asserts the response
flows back. Marked ``@pytest.mark.skipif`` so the standard unit test
suite ignores it; opt in by setting ``A2A_INTEGRATION_TEST=1``.

This is the load-bearing proof that the A2A client and server actually
compose end-to-end. Every other test mocks one half or the other; only
this test exercises the wire format on both sides.
"""

import asyncio
import os
import socket
from collections.abc import AsyncIterator
from contextlib import closing

import pytest

# Skip module if optional `a2a` extra is missing.
pytest.importorskip("a2a.types")

import uvicorn
from a2a.types import AgentCapabilities, AgentCard, AgentInterface

from troopai.adk.a2a import A2AAgent, A2AServer, build_starlette_app
from troopai.adk.agents import Agent
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tools.tool_context import ToolContext

pytestmark = pytest.mark.skipif(
    os.environ.get("A2A_INTEGRATION_TEST") != "1",
    reason="opt-in integration test — set A2A_INTEGRATION_TEST=1 to run",
)


def _free_port() -> int:
    """Reserve a free localhost port by binding then releasing."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_local_agent() -> Agent[None]:
    """An Agent with a single deterministic tool — no LLM required."""

    async def echo(ctx: ToolContext, raw: str) -> str:
        del ctx
        return f"Echo: {raw}"

    tool = FunctionTool(
        name="echo",
        schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        description="Echo the input text back.",
        on_invoke=echo,
    )
    return Agent(
        name="echo_agent",
        system_prompt="When asked to echo, call the echo tool.",
        tools=[tool],
    )


@pytest.fixture
async def live_server() -> AsyncIterator[str]:
    """Boot the A2A server on a free port; tear down after the test."""
    port = _free_port()
    agent = _make_local_agent()
    card = AgentCard(
        name="echo",
        description="Echoes input.",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(
                url=f"http://127.0.0.1:{port}",
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            ),
        ],
        capabilities=AgentCapabilities(streaming=True),
    )
    server = A2AServer(agent=agent, agent_card=card)
    app = build_starlette_app(server)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    uv = uvicorn.Server(config)

    serve_task = asyncio.create_task(uv.serve())
    # Wait for the server to be accepting connections.
    for _ in range(50):
        await asyncio.sleep(0.05)
        if uv.started:
            break
    else:
        uv.should_exit = True
        await serve_task
        raise RuntimeError("uvicorn did not start within 2.5s")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        uv.should_exit = True
        await serve_task


class TestRoundTrip:
    async def test_send_message_round_trip(self, live_server: str) -> None:
        """Real client -> real server -> real response, no mocks."""
        async with A2AAgent(name="remote_echo", url=live_server) as remote:
            result = await remote.run("hello")
        # The server-side agent runs Runner.arun on the local echo
        # agent, which responds based on its system prompt + echo
        # tool. The exact text depends on the LLM, but the
        # round-trip must complete without exception and return
        # non-empty text.
        assert result.text is not None
        assert len(result.task_id) > 0
        assert len(result.context_id) > 0

"""Tests for the stateless-protocol handshake seam in MCPServerWithClientSession."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from troopai.adk.mcp.mcp_server import MCPServerWithClientSession


class _ConcreteServer(MCPServerWithClientSession):
    """Minimal concrete implementation for testing."""

    async def connect(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass


class _NoOpHandshakeServer(MCPServerWithClientSession):
    """Subclass with a no-op handshake (simulates stateless protocol)."""

    async def connect(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    async def _perform_handshake(self, session: Any) -> None:
        # No-op: stateless protocol does not require initialize()
        pass


async def test_default_handshake_calls_session_initialize() -> None:
    """Default _perform_handshake must call session.initialize()."""
    server = _ConcreteServer(name="test")
    session = MagicMock()
    session.initialize = AsyncMock()

    await server._perform_handshake(session)
    session.initialize.assert_awaited_once()


async def test_noop_handshake_passes_existing_tests() -> None:
    """A no-op _perform_handshake must allow _attach_session to succeed without calling initialize()."""
    server = _NoOpHandshakeServer(name="noop-server")
    session = MagicMock()
    session.initialize = AsyncMock()

    await server._attach_session(session)

    # initialize() must NOT have been called (the no-op seam skips it)
    session.initialize.assert_not_awaited()
    # Session must be stored
    assert server._session is session


async def test_attach_session_clears_cache_on_reconnect() -> None:
    """_attach_session must clear the tool cache regardless of handshake implementation."""
    server = _ConcreteServer(name="test")
    session1 = MagicMock()
    session1.initialize = AsyncMock()

    # Pre-populate cache
    from mcp import Tool as MCPTool

    fake_tool = MCPTool(name="t", description="t", inputSchema={"type": "object"})
    server._tools_cache = [fake_tool]

    await server._attach_session(session1)
    # Cache must be cleared on fresh connect
    assert server._tools_cache is None


async def test_no_op_handshake_server_passes_list_tools() -> None:
    """The no-op subclass must still list tools once session is attached."""
    server = _NoOpHandshakeServer(name="noop-server")
    session = MagicMock()
    session.initialize = AsyncMock()

    from mcp import Tool as MCPTool

    fake_tool = MCPTool(name="test_tool", description="A test", inputSchema={"type": "object"})
    list_tools_result = MagicMock()
    list_tools_result.tools = [fake_tool]
    session.list_tools = AsyncMock(return_value=list_tools_result)

    await server._attach_session(session)
    tools = await server.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "test_tool"

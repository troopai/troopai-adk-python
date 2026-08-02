"""Tests for resource methods on MCPServerWithClientSession.

Feature: ``MCPServerWithClientSession`` must expose ``list_resources()``,
``list_resource_templates()``, and ``read_resource(uri)`` delegating
to the installed mcp 1.27.2 ``ClientSession``.

Covers:
- ``list_resources()`` delegates to ``session.list_resources()`` and returns result.
- ``list_resource_templates()`` delegates to ``session.list_resource_templates()`` and returns result.
- ``read_resource(uri)`` delegates to ``session.read_resource(AnyUrl(uri))`` and returns result.
- All three methods raise ``MCPConnectionError`` when the server is not connected.
- ``build_resource_tool`` in ``extras.py`` still works on top of these methods.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.mcp.exceptions import MCPConnectionError
from troopai.adk.mcp.mcp_server import MCPServerWithClientSession

# --------------------------------------------------------------- concrete stub


class _ConcreteServer(MCPServerWithClientSession):
    """Minimal concrete subclass for unit testing."""

    async def connect(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass


def _make_server(*, session: Any | None = None) -> _ConcreteServer:
    server = _ConcreteServer(name="test-server")
    if session is not None:
        server._session = session
    return server


def _make_session() -> Any:
    session = MagicMock()
    session.list_resources = AsyncMock()
    session.list_resource_templates = AsyncMock()
    session.read_resource = AsyncMock()
    return session


# -------------------------------------------------- list_resources


async def test_list_resources_delegates_to_session() -> None:
    """list_resources() delegates to ClientSession.list_resources()."""
    session = _make_session()
    fake_result = MagicMock()
    session.list_resources.return_value = fake_result

    server = _make_server(session=session)
    result = await server.list_resources()

    session.list_resources.assert_awaited_once()
    assert result is fake_result


async def test_list_resources_raises_when_not_connected() -> None:
    """list_resources() raises MCPConnectionError when server is not connected."""
    server = _make_server()  # no session

    with pytest.raises(MCPConnectionError, match="not connected"):
        await server.list_resources()


# -------------------------------------------------- list_resource_templates


async def test_list_resource_templates_delegates_to_session() -> None:
    """list_resource_templates() delegates to ClientSession.list_resource_templates()."""
    session = _make_session()
    fake_result = MagicMock()
    session.list_resource_templates.return_value = fake_result

    server = _make_server(session=session)
    result = await server.list_resource_templates()

    session.list_resource_templates.assert_awaited_once()
    assert result is fake_result


async def test_list_resource_templates_raises_when_not_connected() -> None:
    """list_resource_templates() raises MCPConnectionError when server is not connected."""
    server = _make_server()

    with pytest.raises(MCPConnectionError, match="not connected"):
        await server.list_resource_templates()


# -------------------------------------------------- read_resource


async def test_read_resource_delegates_to_session_with_anyurl() -> None:
    """read_resource(uri) passes a pydantic AnyUrl to ClientSession.read_resource()."""
    session = _make_session()
    fake_result = MagicMock()
    session.read_resource.return_value = fake_result

    server = _make_server(session=session)
    result = await server.read_resource("file:///tmp/test.txt")

    session.read_resource.assert_awaited_once()
    # The argument must be a pydantic AnyUrl, not a plain string
    call_arg = session.read_resource.call_args[0][0]
    assert str(call_arg) in ("file:///tmp/test.txt", "file:///tmp/test.txt/")
    assert result is fake_result


async def test_read_resource_raises_when_not_connected() -> None:
    """read_resource() raises MCPConnectionError when server is not connected."""
    server = _make_server()

    with pytest.raises(MCPConnectionError, match="not connected"):
        await server.read_resource("file:///tmp/test.txt")


# -------------------------------------------------- ABC base raises NotImplementedError


async def test_mcp_server_abc_list_resource_templates_raises() -> None:
    """The MCPServer ABC's list_resource_templates raises NotImplementedError by default."""
    from troopai.adk.mcp.mcp_server import MCPServer

    class _BareServer(MCPServer):
        async def connect(self) -> None:
            pass

        async def cleanup(self) -> None:
            pass

        async def list_tools(self):
            return []

        async def call_tool(self, name, arguments):
            raise NotImplementedError

        @property
        def name(self) -> str:
            return "bare"

        @property
        def capabilities(self):
            return None

        async def list_prompts(self):
            return []

        async def get_prompt(self, name, arguments=None):
            raise NotImplementedError

    with pytest.raises(NotImplementedError):
        await _BareServer().list_resource_templates()


# -------------------------------------------------- extras.build_resource_tool still works


async def test_build_resource_tool_still_works_after_server_changes() -> None:
    """extras.build_resource_tool still delegates to server.read_resource correctly."""

    from troopai.adk.mcp.extras import build_resource_tool

    session = _make_session()
    # Return a real-ish ReadResourceResult
    fake_read_result = MagicMock()
    fake_content = MagicMock()
    fake_content.text = "resource content"
    fake_content.blob = None
    fake_read_result.contents = [fake_content]
    session.read_resource.return_value = fake_read_result

    server = _make_server(session=session)
    resource_tool = build_resource_tool(server)

    # on_invoke should call server.read_resource
    ctx = MagicMock()
    import json

    assert resource_tool.on_invoke is not None
    result = await resource_tool.on_invoke(ctx, json.dumps({"uri": "file:///tmp/test.txt"}))
    session.read_resource.assert_awaited_once()
    assert "resource content" in result

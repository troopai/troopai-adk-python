"""Tests for ``troopai.adk.mcp.extras`` (resource + prompt builders)
and ``troopai.adk.mcp.websocket`` (transport happy / sad paths).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.mcp.exceptions import MCPToolCallError, UnsupportedTransportError
from troopai.adk.mcp.extras import build_prompt_tools, build_resource_tool


def _server(name: str = "svc") -> Any:
    s = MagicMock()
    s.name = name
    s.read_resource = AsyncMock()
    s.list_prompts = AsyncMock()
    s.get_prompt = AsyncMock()
    return s


# -------------------------------------------------------------- resource tool


async def test_resource_tool_returns_text_content() -> None:
    server = _server("everything")

    text_entry = MagicMock()
    text_entry.text = "hello world"
    server.read_resource.return_value = MagicMock(contents=[text_entry])

    tool = build_resource_tool(server)
    out = await tool.on_invoke(MagicMock(), '{"uri": "file://demo.txt"}')

    assert out == "hello world"
    server.read_resource.assert_awaited_once_with("file://demo.txt")
    assert tool.name == "read_everything_resource"


async def test_resource_tool_handles_binary_blob() -> None:
    server = _server("svc")

    blob_entry = MagicMock(spec=["blob", "mimeType"])
    blob_entry.blob = "BASE64=="
    blob_entry.mimeType = "image/png"
    server.read_resource.return_value = MagicMock(contents=[blob_entry])

    tool = build_resource_tool(server)
    out = await tool.on_invoke(MagicMock(), '{"uri": "img://demo"}')

    # Binary surfaces as a JSON string carrying mimeType + blobBase64
    assert "image/png" in out
    assert "BASE64==" in out


async def test_resource_tool_rejects_missing_uri() -> None:
    server = _server()
    tool = build_resource_tool(server)

    with pytest.raises(MCPToolCallError):
        await tool.on_invoke(MagicMock(), "{}")


async def test_resource_tool_rejects_invalid_json() -> None:
    server = _server()
    tool = build_resource_tool(server)

    with pytest.raises(MCPToolCallError):
        await tool.on_invoke(MagicMock(), "not-json")


# ----------------------------------------------------------------- prompt tools


async def test_build_prompt_tools_creates_one_tool_per_prompt() -> None:
    server = _server()
    arg = MagicMock()
    arg.name = "topic"
    arg.description = "topic to summarise"
    arg.required = True

    prompt = MagicMock()
    prompt.name = "summarise"
    prompt.description = "Summarise a topic"
    prompt.arguments = [arg]

    server.list_prompts.return_value = MagicMock(prompts=[prompt])

    tools = await build_prompt_tools(server)
    assert len(tools) == 1
    [t] = tools
    assert t.name == "prompt_summarise"
    assert t.schema == {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "topic to summarise"},
        },
        "required": ["topic"],
    }


async def test_prompt_tool_invokes_get_prompt() -> None:
    server = _server()
    arg = MagicMock()
    arg.name = "topic"
    arg.description = ""
    arg.required = False

    prompt = MagicMock()
    prompt.name = "free_form"
    prompt.description = ""
    prompt.arguments = [arg]
    server.list_prompts.return_value = MagicMock(prompts=[prompt])

    msg = MagicMock()
    msg.content = MagicMock()
    msg.content.text = "rendered prompt body"
    server.get_prompt.return_value = MagicMock(messages=[msg])

    [t] = await build_prompt_tools(server)
    out = await t.on_invoke(MagicMock(), '{"topic": "ai"}')

    assert out == "rendered prompt body"
    server.get_prompt.assert_awaited_once_with("free_form", {"topic": "ai"})


# --------------------------------------------------------- WebSocket transport


async def test_websocket_connect_raises_when_dependency_missing() -> None:
    """When the ``mcp.client.websocket`` module is missing (no
    ``websockets`` package), ``connect`` raises ``UnsupportedTransportError``
    instead of an obscure ``ImportError``.
    """
    from troopai.adk.mcp.websocket import MCPServerWebsocket, MCPServerWebsocketParams

    server = MCPServerWebsocket(
        name="ws-test",
        params=MCPServerWebsocketParams(url="ws://localhost:1/mcp"),
    )

    with patch.dict("sys.modules", {"mcp.client.websocket": None}), pytest.raises(UnsupportedTransportError):
        await server.connect()


# --------------------------------- _make_client_session explicit constructor


def _concrete_server(llm: Any = None, elicitation_callback: Any = None) -> Any:
    """Return a concrete MCPServerWithClientSession subclass for testing."""
    from troopai.adk.mcp.mcp_server import MCPServerWithClientSession

    class _Concrete(MCPServerWithClientSession):
        async def connect(self) -> None:
            pass

        async def cleanup(self) -> None:
            pass

    return _Concrete(
        name="test",
        llm=llm,
        elicitation_callback=elicitation_callback,
    )


def test_make_client_session_no_sampling_uses_explicit_constructor() -> None:
    """Regression: _make_client_session was building a kwargs dict and
    **-spreading it into ClientSession, which included ``sampling_capabilities``
    — a key NOT accepted by ``ClientSession.__init__``. The fix uses two
    explicit constructor paths, not a kwargs accumulator.

    Without a sampling LLM, ClientSession must be constructed without
    sampling_callback.
    """
    from mcp import ClientSession

    server = _concrete_server()
    read = MagicMock()
    write = MagicMock()

    # _make_client_session must not raise even though sampling_capabilities
    # is absent from ClientSession's signature.
    session = server._make_client_session(read, write)
    assert isinstance(session, ClientSession)


def test_make_client_session_with_sampling_no_sampling_capabilities_kwarg() -> None:
    """Regression: _make_client_session was passing ``sampling_capabilities``
    as a kwarg, which ClientSession.__init__ does not accept. With a sampling
    LLM, the session must be constructable without TypeError.
    """
    from mcp import ClientSession

    fake_llm = MagicMock()
    server = _concrete_server(llm=fake_llm)
    read = MagicMock()
    write = MagicMock()

    # This would raise TypeError before the fix because sampling_capabilities
    # was spread into ClientSession(**kwargs).
    session = server._make_client_session(read, write)
    assert isinstance(session, ClientSession)

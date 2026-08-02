"""Tests for callable approval policies on MCP-derived tools.

Feature: ``mcp_tool_to_function_tool`` and ``MCPToolset`` must accept
a per-call callable policy (tool name + arguments → bool) in addition
to the static ``bool`` they already accepted. The callable is threaded
through to the ``FunctionTool.requires_approval`` field unchanged.

Covers:
- ``mcp_tool_to_function_tool`` with a sync callable policy.
- ``mcp_tool_to_function_tool`` with an async callable policy.
- ``MCPToolset.requires_approval`` as a sync callable propagates to
  every converted ``FunctionTool``.
- ``MCPToolset.requires_approval`` as an async callable propagates.
- Static ``bool`` paths still work (regression guard).
- ``None`` default still maps to ``False`` (regression guard).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from troopai.adk.mcp.conversion import mcp_tool_to_function_tool
from troopai.adk.tools.toolsets.mcp_toolset import MCPToolset

# --------------------------------------------------------------- helpers


def _mcp_tool(name: str = "my_tool") -> Any:
    tool = MagicMock()
    tool.name = name
    tool.description = "A test tool"
    tool.inputSchema = {"type": "object", "properties": {}}
    return tool


def _server(name: str = "svc") -> Any:
    server = MagicMock()
    server.name = name
    server.call_tool = AsyncMock(
        return_value=MagicMock(
            content=[MagicMock(type="text", text="ok", spec=["type", "text"])],
            isError=False,
        )
    )
    return server


def _fake_server(*, name: str = "fake", tools: list[Any] | None = None) -> Any:
    from mcp.types import TextContent

    server = MagicMock()
    server.name = name
    tool_list = tools or [_mcp_tool("alpha"), _mcp_tool("beta")]
    server.list_tools = AsyncMock(return_value=tool_list)
    server.connect = AsyncMock()
    server.cleanup = AsyncMock()
    server.call_tool = AsyncMock(
        return_value=MagicMock(
            content=[TextContent(type="text", text="ok")],
            isError=False,
        )
    )
    return server


# ----------------------------------------- mcp_tool_to_function_tool tests


def test_conversion_callable_policy_sync_threaded_to_function_tool() -> None:
    """A sync callable policy is stored on FunctionTool.requires_approval unchanged."""
    mcp_tool = _mcp_tool()
    server = _server()

    def policy(ctx: Any) -> bool:
        return True

    ft = mcp_tool_to_function_tool(mcp_tool, server, requires_approval=policy)

    assert ft.requires_approval is policy


def test_conversion_callable_policy_async_threaded_to_function_tool() -> None:
    """An async callable policy is stored on FunctionTool.requires_approval unchanged."""
    mcp_tool = _mcp_tool()
    server = _server()

    async def policy(ctx: Any) -> bool:
        return False

    ft = mcp_tool_to_function_tool(mcp_tool, server, requires_approval=policy)

    assert ft.requires_approval is policy


def test_conversion_bool_true_still_works() -> None:
    """Static True still propagates (regression guard)."""
    mcp_tool = _mcp_tool()
    server = _server()

    ft = mcp_tool_to_function_tool(mcp_tool, server, requires_approval=True)

    assert ft.requires_approval is True


def test_conversion_bool_false_still_works() -> None:
    """Static False still propagates (regression guard)."""
    mcp_tool = _mcp_tool()
    server = _server()

    ft = mcp_tool_to_function_tool(mcp_tool, server, requires_approval=False)

    assert ft.requires_approval is False


def test_conversion_none_maps_to_false() -> None:
    """None default still maps to False (regression guard)."""
    mcp_tool = _mcp_tool()
    server = _server()

    ft = mcp_tool_to_function_tool(mcp_tool, server, requires_approval=None)

    assert ft.requires_approval is False


def test_conversion_callable_policy_also_works_in_structured_content_path() -> None:
    """Callable policy is threaded through the structured-content (artifact) path too."""
    mcp_tool = _mcp_tool()
    server = _server()

    def policy(ctx: Any) -> bool:
        return True

    ft = mcp_tool_to_function_tool(mcp_tool, server, requires_approval=policy, use_structured_content=True)

    assert ft.requires_approval is policy
    assert ft.response_format == "content_and_artifact"


# ----------------------------------------- MCPToolset tests


async def test_toolset_callable_policy_sync_propagates_to_function_tools() -> None:
    """MCPToolset.requires_approval as a sync callable is forwarded to every FunctionTool."""
    server = _fake_server()

    def policy(ctx: Any) -> bool:
        return True

    toolset = MCPToolset(server=server, auto_connect=True, requires_approval=policy)
    tools = await toolset.get_tools(None)

    assert len(tools) > 0
    for ft in tools.values():
        assert ft.requires_approval is policy, (
            f"Tool {ft.name!r} must carry the callable policy, got {ft.requires_approval!r}"
        )


async def test_toolset_callable_policy_async_propagates_to_function_tools() -> None:
    """MCPToolset.requires_approval as an async callable is forwarded to every FunctionTool."""
    server = _fake_server()

    async def policy(ctx: Any) -> bool:
        return False

    toolset = MCPToolset(server=server, auto_connect=True, requires_approval=policy)
    tools = await toolset.get_tools(None)

    assert len(tools) > 0
    for ft in tools.values():
        assert ft.requires_approval is policy


async def test_toolset_static_bool_approval_still_works() -> None:
    """Static bool=True on MCPToolset still propagates (regression guard)."""
    server = _fake_server()
    toolset = MCPToolset(server=server, auto_connect=True, requires_approval=True)
    tools = await toolset.get_tools(None)

    for ft in tools.values():
        assert ft.requires_approval is True


async def test_toolset_default_approval_is_false() -> None:
    """Default MCPToolset.requires_approval=False means no approval (regression guard)."""
    server = _fake_server()
    toolset = MCPToolset(server=server, auto_connect=True)
    tools = await toolset.get_tools(None)

    for ft in tools.values():
        assert ft.requires_approval is False


async def test_toolset_callable_policy_distinct_per_tool() -> None:
    """The same callable identity is stored on ALL converted tools, not a per-tool copy."""
    server = _fake_server()

    call_count = 0

    def counting_policy(ctx: Any) -> bool:
        nonlocal call_count
        call_count += 1
        return True

    toolset = MCPToolset(server=server, auto_connect=True, requires_approval=counting_policy)
    tools = await toolset.get_tools(None)

    # The callable itself is not called during conversion — only during execution.
    assert call_count == 0
    for ft in tools.values():
        assert ft.requires_approval is counting_policy

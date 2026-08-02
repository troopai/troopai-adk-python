"""Tests for ``troopai.adk.tools.toolsets.mcp_toolset.MCPToolset``.

Covers:
- Lazy connect on first ``get_tools`` when ``auto_connect=True``.
- ``auto_connect=False`` does not connect; ``get_tools`` propagates
  the underlying ``MCPConnectionError`` from the server.
- ``adispose`` calls ``server.cleanup`` when this toolset connected
  it, and is idempotent.
- ``adispose`` does NOT call cleanup when ``auto_connect=False``.
- ``get_tools`` after ``adispose`` returns an empty dict and warns.
- ``tool_filter`` (sync, async, and exception-raising) is applied
  with the right ``ToolFilterContext``.
- Concurrent ``get_tools`` calls invoke ``server.connect`` exactly
  once (lock works).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.mcp.exceptions import MCPConnectionError
from troopai.adk.mcp.filters import ToolFilterContext
from troopai.adk.tools.toolsets.mcp_toolset import MCPToolset


def _fake_mcp_tool(name: str) -> Any:
    tool = MagicMock()
    tool.name = name
    tool.description = f"Tool {name}"
    tool.inputSchema = {"type": "object", "properties": {}}
    return tool


def _fake_server(*, name: str = "fake", tools: list[Any] | None = None) -> Any:
    server = MagicMock()
    server.name = name
    server.list_tools = AsyncMock(return_value=tools or [_fake_mcp_tool("alpha"), _fake_mcp_tool("beta")])
    server.connect = AsyncMock()
    server.cleanup = AsyncMock()
    return server


# ----------------------------------------------------------------- lazy connect


async def test_get_tools_lazy_connects_when_auto_connect_true() -> None:
    server = _fake_server()
    toolset = MCPToolset(server=server, auto_connect=True)

    tools = await toolset.get_tools(None)

    server.connect.assert_awaited_once()
    server.list_tools.assert_awaited_once()
    assert set(tools.keys()) == {"alpha", "beta"}


async def test_get_tools_does_not_connect_when_auto_connect_false() -> None:
    server = _fake_server()
    toolset = MCPToolset(server=server, auto_connect=False)

    await toolset.get_tools(None)

    server.connect.assert_not_called()
    server.list_tools.assert_awaited_once()


async def test_concurrent_get_tools_connects_once() -> None:
    server = _fake_server()
    toolset = MCPToolset(server=server, auto_connect=True)

    await asyncio.gather(*(toolset.get_tools(None) for _ in range(5)))

    server.connect.assert_awaited_once()


# -------------------------------------------------------------------- adispose


async def test_adispose_calls_cleanup_for_auto_connect() -> None:
    server = _fake_server()
    toolset = MCPToolset(server=server, auto_connect=True)
    await toolset.get_tools(None)

    await toolset.adispose()

    server.cleanup.assert_awaited_once()


async def test_adispose_does_not_cleanup_when_auto_connect_false() -> None:
    server = _fake_server()
    toolset = MCPToolset(server=server, auto_connect=False)
    await toolset.get_tools(None)

    await toolset.adispose()

    server.cleanup.assert_not_called()


async def test_adispose_idempotent() -> None:
    server = _fake_server()
    toolset = MCPToolset(server=server, auto_connect=True)
    await toolset.get_tools(None)

    await toolset.adispose()
    await toolset.adispose()

    server.cleanup.assert_awaited_once()


async def test_get_tools_after_adispose_returns_empty_dict(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = _fake_server()
    toolset = MCPToolset(server=server, auto_connect=True)
    await toolset.get_tools(None)
    await toolset.adispose()

    with caplog.at_level(logging.WARNING, logger="troopai.adk.tools.toolsets.mcp_toolset"):
        result = await toolset.get_tools(None)

    assert result == {}
    assert any("after adispose" in rec.message for rec in caplog.records)


async def test_adispose_swallows_cleanup_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = _fake_server()
    server.cleanup.side_effect = RuntimeError("boom")
    toolset = MCPToolset(server=server, auto_connect=True)
    await toolset.get_tools(None)

    with caplog.at_level(logging.WARNING, logger="troopai.adk.tools.toolsets.mcp_toolset"):
        await toolset.adispose()  # MUST NOT raise

    assert any("cleanup failed" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------- filters


async def test_filter_excludes_tool_when_predicate_returns_false() -> None:
    server = _fake_server()
    captured_contexts: list[ToolFilterContext] = []

    def keep_only_alpha(ctx: ToolFilterContext, tool: Any) -> bool:
        captured_contexts.append(ctx)
        return tool.name == "alpha"

    toolset = MCPToolset(server=server, auto_connect=True, tool_filter=keep_only_alpha)

    tools = await toolset.get_tools(None)

    assert set(tools.keys()) == {"alpha"}
    assert len(captured_contexts) == 2
    assert all(c.server_name == "fake" for c in captured_contexts)


async def test_async_filter_is_awaited() -> None:
    server = _fake_server()

    async def async_filter(ctx: ToolFilterContext, tool: Any) -> bool:
        await asyncio.sleep(0)
        return True

    toolset = MCPToolset(server=server, auto_connect=True, tool_filter=async_filter)

    tools = await toolset.get_tools(None)

    assert set(tools.keys()) == {"alpha", "beta"}


async def test_filter_exception_excludes_tool_fail_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = _fake_server()

    def buggy_filter(ctx: ToolFilterContext, tool: Any) -> bool:
        raise ValueError("filter is broken")

    toolset = MCPToolset(server=server, auto_connect=True, tool_filter=buggy_filter)

    with caplog.at_level(logging.WARNING, logger="troopai.adk.tools.toolsets.mcp_toolset"):
        result = await toolset.get_tools(None)

    assert result == {}  # All excluded
    assert any("filter raised" in rec.message for rec in caplog.records)


async def test_filter_also_excludes_resource_and_prompt_surfaces() -> None:
    # SECURITY: the synthetic resource-read + prompt tools are tool surfaces a
    # denied server must not expose. They were appended after the filter loop
    # and bypassed tool_filter entirely. A deny-all filter must yield no tools.
    server = _fake_server()
    server.list_prompts = AsyncMock(return_value=MagicMock(prompts=[]))

    def deny_all(_ctx: ToolFilterContext, _tool: Any) -> bool:
        return False

    toolset = MCPToolset(
        server=server,
        auto_connect=True,
        tool_filter=deny_all,
        use_mcp_resources=True,
        expose_prompts_as_tools=True,
    )

    tools = await toolset.get_tools(None)

    assert tools == {}


async def test_resource_surface_present_without_filter() -> None:
    # Positive control: with no filter, the resource tool IS exposed (so the
    # deny-all test above is meaningful, not vacuously empty).
    server = _fake_server()
    server.list_tools = AsyncMock(return_value=[])
    toolset = MCPToolset(server=server, auto_connect=True, use_mcp_resources=True)

    tools = await toolset.get_tools(None)

    assert len(tools) == 1  # exactly the resource-read tool


# ------------------------------------------------------------- error pass-through


async def test_get_tools_propagates_server_connect_error() -> None:
    server = _fake_server()
    server.connect.side_effect = MCPConnectionError("can't connect")
    toolset = MCPToolset(server=server, auto_connect=True)

    with pytest.raises(MCPConnectionError):
        await toolset.get_tools(None)

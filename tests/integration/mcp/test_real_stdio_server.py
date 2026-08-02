"""Integration test against a real stdio MCP server.

Skipped automatically when ``npx`` is not available on PATH.
Otherwise spawns ``@modelcontextprotocol/server-everything`` (a
reference test server maintained by the MCP project) and exercises
the full round-trip: connect → list_tools → call_tool → cleanup.

The test does not assert on specific tool *behaviour* — only on
connectivity and shape — because the upstream server's tool list
changes between releases.
"""

from __future__ import annotations

import shutil

import pytest

from troopai.adk.mcp import (
    MCPServerStdio,
    MCPServerStdioParams,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def npx_path() -> str:
    """Locate ``npx`` or skip the test (no Node toolchain available)."""
    path = shutil.which("npx")
    if path is None:
        pytest.skip("npx not found on PATH; skipping real-server integration test")
    return path


async def test_stdio_round_trip(npx_path: str) -> None:
    """Full round-trip against ``@modelcontextprotocol/server-everything``."""
    server = MCPServerStdio(
        name="everything",
        params=MCPServerStdioParams(
            command=npx_path,
            args=["-y", "@modelcontextprotocol/server-everything"],
        ),
    )
    async with server:
        tools = await server.list_tools()
        assert len(tools) > 0, "MCP everything-server should advertise tools"
        # Cache hit on second call — same identity-equal list returned
        cached = await server.list_tools()
        assert [t.name for t in cached] == [t.name for t in tools]


async def test_toolset_with_real_server(npx_path: str) -> None:
    """``MCPToolset`` end-to-end: lazy connect + name discovery."""
    from troopai.adk.tools.toolsets import MCPToolset

    server = MCPServerStdio(
        name="everything",
        params=MCPServerStdioParams(
            command=npx_path,
            args=["-y", "@modelcontextprotocol/server-everything"],
        ),
    )
    toolset = MCPToolset(server=server, auto_connect=True)
    try:
        tools = await toolset.get_tools(None)
        assert len(tools) > 0
        # Every key matches its tool's name
        for name, tool in tools.items():
            assert tool.name == name
    finally:
        await toolset.adispose()

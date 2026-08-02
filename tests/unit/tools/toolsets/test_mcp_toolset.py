"""Tests for ``troopai.adk.tools.toolsets.mcp_toolset.MCPToolset``.

Regression coverage for the HITL approval-policy forwarding contract:
``MCPToolset.requires_approval`` is documented to apply to *every*
converted ``FunctionTool``. Before the fix, the policy was forwarded to
regular MCP tools but silently dropped for the synthetic
``read_<server>_resource`` tool (``use_mcp_resources=True``) and for the
``prompt_<name>`` tools (``expose_prompts_as_tools=True``), so an LLM
could trigger those server-backed actions with no approval gate.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.tools.toolsets import MCPToolset


def _server(name: str = "svc") -> Any:
    s = MagicMock()
    s.name = name
    s.connect = AsyncMock()
    s.cleanup = AsyncMock()
    s.list_tools = AsyncMock(return_value=[])
    s.read_resource = AsyncMock()
    s.get_prompt = AsyncMock()
    return s


def _prompt(name: str = "summarise") -> Any:
    prompt = MagicMock()
    prompt.name = name
    prompt.description = "Summarise a topic"
    prompt.arguments = []
    return prompt


# ----------------------------------------------------- resource tool forwarding


async def test_resource_tool_inherits_static_requires_approval() -> None:
    """A static ``requires_approval=True`` MUST reach the synthetic
    resource-read tool, not just the regular MCP tools."""
    server = _server("everything")
    toolset = MCPToolset(server=server, use_mcp_resources=True, requires_approval=True)

    tools = await toolset.get_tools(None)
    resource_tool = tools["read_everything_resource"]

    assert resource_tool.requires_approval is True
    assert await resource_tool.check_requires_approval(MagicMock(spec=ToolContext)) is True


async def test_resource_tool_inherits_callable_requires_approval() -> None:
    """A callable approval policy MUST be forwarded by identity to the
    resource tool so per-call HITL decisions still apply."""
    server = _server("everything")

    def policy(ctx: ToolContext[Any]) -> bool:
        del ctx
        return True

    toolset = MCPToolset(server=server, use_mcp_resources=True, requires_approval=policy)

    tools = await toolset.get_tools(None)
    resource_tool = tools["read_everything_resource"]

    assert resource_tool.requires_approval is policy
    assert await resource_tool.check_requires_approval(MagicMock(spec=ToolContext)) is True


async def test_resource_tool_default_is_no_approval() -> None:
    """Default (no policy set) keeps the resource tool ungated."""
    server = _server("everything")
    toolset = MCPToolset(server=server, use_mcp_resources=True)

    tools = await toolset.get_tools(None)
    resource_tool = tools["read_everything_resource"]

    assert resource_tool.requires_approval is False


# ------------------------------------------------------- prompt tool forwarding


async def test_prompt_tools_inherit_static_requires_approval() -> None:
    """A static ``requires_approval=True`` MUST reach every prompt tool."""
    server = _server("svc")
    server.list_prompts = AsyncMock(return_value=MagicMock(prompts=[_prompt("summarise")]))
    toolset = MCPToolset(server=server, expose_prompts_as_tools=True, requires_approval=True)

    tools = await toolset.get_tools(None)
    prompt_tool = tools["prompt_summarise"]

    assert prompt_tool.requires_approval is True
    assert await prompt_tool.check_requires_approval(MagicMock(spec=ToolContext)) is True


async def test_prompt_tools_inherit_callable_requires_approval() -> None:
    """A callable approval policy MUST be forwarded by identity to each
    prompt tool."""
    server = _server("svc")
    server.list_prompts = AsyncMock(return_value=MagicMock(prompts=[_prompt("summarise")]))

    def policy(ctx: ToolContext[Any]) -> bool:
        del ctx
        return True

    toolset = MCPToolset(server=server, expose_prompts_as_tools=True, requires_approval=policy)

    tools = await toolset.get_tools(None)
    prompt_tool = tools["prompt_summarise"]

    assert prompt_tool.requires_approval is policy


async def test_prompt_tools_default_is_no_approval() -> None:
    """Default (no policy set) keeps prompt tools ungated."""
    server = _server("svc")
    server.list_prompts = AsyncMock(return_value=MagicMock(prompts=[_prompt("summarise")]))
    toolset = MCPToolset(server=server, expose_prompts_as_tools=True)

    tools = await toolset.get_tools(None)
    prompt_tool = tools["prompt_summarise"]

    assert prompt_tool.requires_approval is False

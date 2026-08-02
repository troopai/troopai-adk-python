"""Integration tests: toolsets flowing through ``Agent`` + ``build_tools``.

Covers the surfaces where toolsets actually meet the rest of the
framework:

- ``Agent.tools`` accepting toolset entries
- ``Agent.__post_init__`` deferring name-uniqueness to materialisation
- ``build_tools()`` materialising toolsets and detecting cross-toolset
  conflicts
- ``resolve_function_tool`` walking toolsets to find tools by their
  post-rename / post-prefix name
- Construction-time dependency validation walking ``FunctionToolset.tools``
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.exceptions import ToolDependencyError, ToolsetNameConflictError
from troopai.adk.run.context import RunContext
from troopai.adk.run.llm_calls import build_tools, resolve_function_tool
from troopai.adk.tools import (
    CombinedToolset,
    FunctionTool,
    FunctionToolset,
    function_tool,
)


@function_tool(name="get_temp", description="Get temperature")
def get_temp(city: str) -> str:
    return f"{city}: 22C"


@function_tool(name="query", description="Run SQL")
def query(sql: str) -> str:
    return f"rows for {sql}"


class TestAgentAcceptsToolsets:
    def test_agent_post_init_accepts_toolset(self) -> None:
        ts = FunctionToolset(tools=[get_temp])
        agent = Agent(name="X", system_prompt="test", tools=[ts])
        assert isinstance(agent.tools[0], FunctionToolset)

    def test_agent_post_init_validates_static_dependencies_inside_toolset(self) -> None:
        # A FunctionTool inside a FunctionToolset still gets dependency
        # validation at construction time.
        bad_tool = FunctionTool(
            name="needs_env",
            schema={"type": "object", "properties": {}},
            requires_env=("TROOPAI_TOOLSET_TEST_MISSING",),
        )
        ts = FunctionToolset(tools=[bad_tool])
        with patch.dict("os.environ", {}, clear=True), pytest.raises(ToolDependencyError):
            Agent(name="X", system_prompt="test", tools=[ts])

    def test_agent_post_init_skips_live_wrappers_for_dep_validation(self) -> None:
        # Live wrappers (filtered/prefixed) cannot statically introspect
        # tool requirements, so dependency validation defers to runtime.
        # A wrapped toolset with a missing env var should NOT raise at
        # agent construction.
        bad_tool = FunctionTool(
            name="needs_env",
            schema={"type": "object", "properties": {}},
            requires_env=("TROOPAI_TOOLSET_TEST_MISSING",),
        )
        wrapped = FunctionToolset(tools=[bad_tool]).prefixed("ns")
        with patch.dict("os.environ", {}, clear=True):
            # No exception expected — the wrapper hides the inner tools
            # from the static validator.
            agent = Agent(name="X", system_prompt="test", tools=[wrapped])
        assert agent.tools[0] is wrapped


class TestBuildToolsMaterialisation:
    async def test_materialises_function_toolset(self) -> None:
        ts = FunctionToolset(tools=[get_temp])
        agent = Agent(name="X", system_prompt="test", tools=[ts])
        out = await build_tools(agent)
        assert out is not None
        assert [getattr(t, "name", "") for t in out] == ["get_temp"]

    async def test_materialises_prefixed_toolset(self) -> None:
        ts = FunctionToolset(tools=[get_temp]).prefixed("weather")
        agent = Agent(name="X", system_prompt="test", tools=[ts])
        out = await build_tools(agent)
        assert out is not None
        assert [getattr(t, "name", "") for t in out] == ["weather_get_temp"]

    async def test_materialises_combined_toolset(self) -> None:
        weather = FunctionToolset(tools=[get_temp]).prefixed("weather")
        db = FunctionToolset(tools=[query]).prefixed("db")
        agent = Agent(
            name="X",
            system_prompt="test",
            tools=[CombinedToolset(toolsets=[weather, db])],
        )
        out = await build_tools(agent)
        assert out is not None
        assert sorted(getattr(t, "name", "") for t in out) == ["db_query", "weather_get_temp"]

    async def test_filtered_reacts_to_context(self) -> None:
        def is_admin(ctx: RunContext[Any], tool: FunctionTool) -> bool:
            return ctx.context.get("role") == "admin"

        admin_only = FunctionToolset(tools=[get_temp]).filtered(is_admin)
        agent = Agent(name="X", system_prompt="test", tools=[admin_only])

        admin_tools = await build_tools(agent, context=RunContext(context={"role": "admin"}))
        user_tools = await build_tools(agent, context=RunContext(context={"role": "user"}))
        assert admin_tools is not None
        assert [getattr(t, "name", "") for t in admin_tools] == ["get_temp"]
        assert user_tools is None  # build_tools returns None for empty list

    async def test_standalone_and_toolset_coexist(self) -> None:
        ts = FunctionToolset(tools=[get_temp]).prefixed("weather")
        agent = Agent(name="X", system_prompt="test", tools=[query, ts])
        out = await build_tools(agent)
        assert out is not None
        assert sorted(getattr(t, "name", "") for t in out) == ["query", "weather_get_temp"]


class TestNameConflictDetection:
    async def test_two_toolsets_same_name_raises(self) -> None:
        ts1 = FunctionToolset(tools=[get_temp]).prefixed("ns")
        ts2 = FunctionToolset(tools=[get_temp]).prefixed("ns")
        agent = Agent(name="X", system_prompt="test", tools=[ts1, ts2])
        with pytest.raises(ToolsetNameConflictError) as exc_info:
            await build_tools(agent)
        assert "ns_get_temp" in exc_info.value.conflicts
        assert len(exc_info.value.conflicts["ns_get_temp"]) == 2

    async def test_toolset_collides_with_standalone_raises(self) -> None:
        # standalone get_temp + toolset that produces get_temp
        agent = Agent(
            name="X",
            system_prompt="test",
            tools=[get_temp, FunctionToolset(tools=[get_temp])],
        )
        with pytest.raises(ToolsetNameConflictError) as exc_info:
            await build_tools(agent)
        sources = exc_info.value.conflicts["get_temp"]
        assert any("agent.tools" in s for s in sources)
        assert any("FunctionToolset" in s for s in sources)

    async def test_two_standalone_with_same_name_does_not_raise(self) -> None:
        # Pre-existing behavior: standalone-vs-standalone is NOT raised
        # by build_tools (only logged elsewhere). Only toolset conflicts
        # raise.
        @function_tool(name="get_temp", description="alt")
        def alt_temp(city: str) -> str:
            return f"alt {city}"

        agent = Agent(name="X", system_prompt="test", tools=[get_temp, alt_temp])
        # Does not raise
        out = await build_tools(agent)
        assert out is not None


class TestResolveFunctionToolWithToolsets:
    async def test_resolves_through_prefixed_toolset(self) -> None:
        ts = FunctionToolset(tools=[get_temp]).prefixed("weather")
        agent = Agent(name="X", system_prompt="test", tools=[ts])
        found = await resolve_function_tool(agent, "weather_get_temp")
        assert found is not None
        assert found.name == "weather_get_temp"

    async def test_resolves_through_filtered_toolset_with_context(self) -> None:
        def keep(ctx: RunContext[Any], tool: FunctionTool) -> bool:
            return ctx.context.get("env") == "prod"

        ts = FunctionToolset(tools=[get_temp]).filtered(keep)
        agent = Agent(name="X", system_prompt="test", tools=[ts])

        ctx_prod = RunContext(context={"env": "prod"})
        ctx_dev = RunContext(context={"env": "dev"})

        assert await resolve_function_tool(agent, "get_temp", ctx_prod) is not None
        assert await resolve_function_tool(agent, "get_temp", ctx_dev) is None

    async def test_returns_none_when_unknown(self) -> None:
        ts = FunctionToolset(tools=[get_temp]).prefixed("weather")
        agent = Agent(name="X", system_prompt="test", tools=[ts])
        assert await resolve_function_tool(agent, "nonexistent") is None

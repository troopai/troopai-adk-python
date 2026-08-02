"""Unit tests for the concrete toolset variants.

Covers ``FunctionToolset``, ``PrefixedToolset``, ``RenamedToolset``,
``FilteredToolset``, ``CombinedToolset``, and ``WrapperToolset`` in
isolation. Cross-cutting integration with ``build_tools`` lives in
``test_toolset_integration.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from troopai.adk.run.context import RunContext
from troopai.adk.tools import (
    CombinedToolset,
    FilteredToolset,
    FunctionTool,
    FunctionToolset,
    PrefixedToolset,
    RenamedToolset,
    ToolCachePolicy,
    WrapperToolset,
    function_tool,
)


@function_tool(name="get_temp", description="Get temperature")
def get_temp(city: str) -> str:
    return f"{city}: 22C"


@function_tool(name="get_conditions", description="Get conditions")
def get_conditions(city: str) -> str:
    return f"{city}: sunny"


@function_tool(name="query", description="Run SQL")
def query(sql: str) -> str:
    return f"rows for {sql}"


class TestFunctionToolset:
    async def test_returns_tools_keyed_by_name(self) -> None:
        ts = FunctionToolset(tools=[get_temp, get_conditions])
        out = await ts.get_tools()
        assert sorted(out.keys()) == ["get_conditions", "get_temp"]
        assert all(isinstance(t, FunctionTool) for t in out.values())

    async def test_empty_toolset_returns_empty_dict(self) -> None:
        ts = FunctionToolset(tools=[])
        assert await ts.get_tools() == {}

    async def test_duplicate_name_within_toolset_warns_and_last_wins(self, caplog: pytest.LogCaptureFixture) -> None:
        # Two tools with name="get_temp" — last definition wins.
        @function_tool(name="get_temp", description="Different impl")
        def shadowed(city: str) -> str:
            return f"shadowed {city}"

        ts = FunctionToolset(tools=[get_temp, shadowed])
        with caplog.at_level("WARNING"):
            out = await ts.get_tools()
        assert "duplicate" in caplog.text.lower()
        assert out["get_temp"].description == "Different impl"


class TestPrefixedToolset:
    async def test_default_separator_underscore(self) -> None:
        inner = FunctionToolset(tools=[get_temp, get_conditions])
        ts = PrefixedToolset(wrapped=inner, prefix="weather")
        out = await ts.get_tools()
        assert sorted(out.keys()) == ["weather_get_conditions", "weather_get_temp"]
        assert out["weather_get_temp"].name == "weather_get_temp"

    async def test_custom_separator(self) -> None:
        inner = FunctionToolset(tools=[get_temp])
        ts = PrefixedToolset(wrapped=inner, prefix="ns", separator="-")
        out = await ts.get_tools()
        assert "ns-get_temp" in out

    def test_empty_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="prefix cannot be empty"):
            PrefixedToolset(wrapped=FunctionToolset(tools=[]), prefix="")

    async def test_clone_preserves_internal_state(self) -> None:
        # Cache hits should still work after rename: the cache dict is
        # shared between original and clone.
        cached_tool = FunctionTool(
            name="x",
            schema={"type": "object", "properties": {}},
            cache=ToolCachePolicy(scope="process"),
        )
        cached_tool.set_cached("args", "value")
        inner = FunctionToolset(tools=[cached_tool])
        ts = PrefixedToolset(wrapped=inner, prefix="ns")
        out = await ts.get_tools()
        assert out["ns_x"].get_cached("args") == "value"

    async def test_builder_method_on_base(self) -> None:
        inner = FunctionToolset(tools=[get_temp])
        ts = inner.prefixed("weather")
        assert isinstance(ts, PrefixedToolset)
        out = await ts.get_tools()
        assert "weather_get_temp" in out


class TestRenamedToolset:
    async def test_renames_subset(self) -> None:
        inner = FunctionToolset(tools=[get_temp, get_conditions])
        ts = RenamedToolset(wrapped=inner, name_map={"get_temp": "temperature"})
        out = await ts.get_tools()
        assert "temperature" in out
        assert "get_conditions" in out  # not in map → unchanged
        assert "get_temp" not in out

    def test_empty_map_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            RenamedToolset(wrapped=FunctionToolset(tools=[]), name_map={})

    def test_empty_target_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            RenamedToolset(wrapped=FunctionToolset(tools=[]), name_map={"a": ""})

    async def test_builder_method_on_base(self) -> None:
        inner = FunctionToolset(tools=[get_temp])
        ts = inner.renamed({"get_temp": "temp"})
        assert isinstance(ts, RenamedToolset)
        out = await ts.get_tools()
        assert "temp" in out


class TestFilteredToolset:
    async def test_predicate_drops_tools(self) -> None:
        inner = FunctionToolset(tools=[get_temp, get_conditions])

        def keep_temp(ctx: Any, tool: FunctionTool) -> bool:
            return tool.name == "get_temp"

        ts = FilteredToolset(wrapped=inner, filter_func=keep_temp)
        ctx = RunContext(context={})
        out = await ts.get_tools(ctx)
        assert list(out.keys()) == ["get_temp"]

    async def test_predicate_with_context(self) -> None:
        inner = FunctionToolset(tools=[get_temp])

        def is_admin(ctx: RunContext[Any], tool: FunctionTool) -> bool:
            return ctx.context.get("role") == "admin"

        ts = FilteredToolset(wrapped=inner, filter_func=is_admin)

        admin_out = await ts.get_tools(RunContext(context={"role": "admin"}))
        user_out = await ts.get_tools(RunContext(context={"role": "user"}))
        assert "get_temp" in admin_out
        assert user_out == {}

    async def test_async_predicate(self) -> None:
        inner = FunctionToolset(tools=[get_temp])

        async def keep(ctx: Any, tool: FunctionTool) -> bool:
            return True

        ts = FilteredToolset(wrapped=inner, filter_func=keep)
        out = await ts.get_tools(RunContext(context={}))
        assert "get_temp" in out

    async def test_no_context_skips_filter(self) -> None:
        # When ctx is None (construction-time validation), the filter
        # is bypassed — predicates that need ctx will get one once
        # execution begins.
        inner = FunctionToolset(tools=[get_temp])

        def always_drop(ctx: Any, tool: FunctionTool) -> bool:
            return False

        ts = FilteredToolset(wrapped=inner, filter_func=always_drop)
        out = await ts.get_tools(None)
        assert "get_temp" in out


class TestCombinedToolset:
    async def test_merges_children(self) -> None:
        a = FunctionToolset(tools=[get_temp])
        b = FunctionToolset(tools=[query])
        ts = CombinedToolset(toolsets=[a, b])
        out = await ts.get_tools()
        assert sorted(out.keys()) == ["get_temp", "query"]

    async def test_empty_returns_empty_dict(self) -> None:
        assert await CombinedToolset(toolsets=[]).get_tools() == {}

    async def test_last_writer_wins_within_combined(self) -> None:
        @function_tool(name="get_temp", description="alt")
        def alt_temp(city: str) -> str:
            return f"alt {city}"

        a = FunctionToolset(tools=[get_temp])
        b = FunctionToolset(tools=[alt_temp])
        out = await CombinedToolset(toolsets=[a, b]).get_tools()
        assert out["get_temp"].description == "alt"

    async def test_builder_method_on_base(self) -> None:
        a = FunctionToolset(tools=[get_temp])
        b = FunctionToolset(tools=[query])
        ts = a.combined_with(b)
        assert isinstance(ts, CombinedToolset)
        out = await ts.get_tools()
        assert sorted(out.keys()) == ["get_temp", "query"]


class TestWrapperToolset:
    async def test_default_passthrough(self) -> None:
        inner = FunctionToolset(tools=[get_temp])
        ts = WrapperToolset(wrapped=inner)
        out = await ts.get_tools()
        assert list(out.keys()) == ["get_temp"]

    async def test_subclass_can_override_get_tools(self) -> None:
        from typing import override

        class TaggedToolset(WrapperToolset):
            @override
            async def get_tools(self, ctx: Any = None) -> dict[str, FunctionTool]:
                inner = await self.wrapped.get_tools(ctx)
                return {f"tagged_{k}": v for k, v in inner.items()}

        ts = TaggedToolset(wrapped=FunctionToolset(tools=[get_temp]))
        out = await ts.get_tools()
        assert "tagged_get_temp" in out


class TestNestedComposition:
    async def test_prefixed_then_renamed(self) -> None:
        inner = FunctionToolset(tools=[get_temp])
        ts = inner.prefixed("weather").renamed({"weather_get_temp": "temp"})
        out = await ts.get_tools()
        assert list(out.keys()) == ["temp"]
        assert out["temp"].name == "temp"

    async def test_filtered_then_prefixed(self) -> None:
        inner = FunctionToolset(tools=[get_temp, get_conditions])

        def keep_temp(ctx: Any, tool: FunctionTool) -> bool:
            return tool.name == "get_temp"

        ts = inner.filtered(keep_temp).prefixed("weather")
        out = await ts.get_tools(RunContext(context={}))
        assert list(out.keys()) == ["weather_get_temp"]

    async def test_combined_of_prefixed(self) -> None:
        weather = FunctionToolset(tools=[get_temp]).prefixed("weather")
        db = FunctionToolset(tools=[query]).prefixed("db")
        combo = CombinedToolset(toolsets=[weather, db])
        out = await combo.get_tools()
        assert sorted(out.keys()) == ["db_query", "weather_get_temp"]

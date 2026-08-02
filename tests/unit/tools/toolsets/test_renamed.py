"""Regression tests for ``RenamedToolset`` collision detection.

A ``name_map`` whose targets collapse two distinct source tools onto a
single name used to silently drop one of them: ``get_tools`` returned a
dict with one key, so the downstream conflict check in ``build_tools``
(which inspects the post-collapse keys) could never see the collision.
These tests pin that the collapse now raises at materialisation time.
"""

from __future__ import annotations

import pytest

from troopai.adk.tools import FunctionToolset, RenamedToolset, function_tool


@function_tool(name="a", description="Tool A")
def tool_a() -> str:
    return "a"


@function_tool(name="b", description="Tool B")
def tool_b() -> str:
    return "b"


class TestRenamedCollision:
    async def test_duplicate_target_value_raises(self) -> None:
        # name_map maps two distinct source names onto the same target;
        # before the fix this silently dropped tool_a, leaving {"x": ...}.
        inner = FunctionToolset(tools=[tool_a, tool_b])
        ts = RenamedToolset(wrapped=inner, name_map={"a": "x", "b": "x"})
        with pytest.raises(ValueError, match="rename collision"):
            await ts.get_tools()

    async def test_rename_onto_passthrough_name_raises(self) -> None:
        # Renaming "a" -> "b" collides with the pass-through tool "b";
        # before the fix one of the two "b" entries silently won.
        inner = FunctionToolset(tools=[tool_a, tool_b])
        ts = RenamedToolset(wrapped=inner, name_map={"a": "b"})
        with pytest.raises(ValueError, match="rename collision"):
            await ts.get_tools()

    async def test_collision_message_names_target(self) -> None:
        inner = FunctionToolset(tools=[tool_a, tool_b])
        ts = RenamedToolset(wrapped=inner, name_map={"a": "x", "b": "x"})
        with pytest.raises(ValueError, match="'x'"):
            await ts.get_tools()

    async def test_distinct_targets_still_pass(self) -> None:
        # No collapse: both tools survive with their renamed keys.
        inner = FunctionToolset(tools=[tool_a, tool_b])
        ts = RenamedToolset(wrapped=inner, name_map={"a": "x", "b": "y"})
        out = await ts.get_tools()
        assert sorted(out.keys()) == ["x", "y"]

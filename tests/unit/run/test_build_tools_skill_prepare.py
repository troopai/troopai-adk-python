"""Tests pinning the skill-tool ``prepare()`` exclusion contract in ``build_tools``.

A ``FunctionTool.prepare`` callable may return ``None`` to hide the tool for a
given LLM step (e.g. a permission/quota gate). The regular-tool path treats a
``prepare()`` *exception* the same way — the tool is excluded rather than
offered with its original schema, so a failing gate never silently exposes a
tool it meant to hide.

These tests pin that the skill-tool path behaves identically: when a skill's
``FunctionTool`` has a ``prepare`` callable that raises, the tool MUST be
excluded from the built tool list, never appended with the unmodified original
schema.
"""

from types import SimpleNamespace
from typing import Any

from troopai.adk.run.llm_calls import build_tools
from troopai.adk.skills.skill import Skill
from troopai.adk.tools.function_tool import function_tool


def _make_agent(skills: list[Skill]) -> Any:
    """Minimal agent-like stub accepted by ``build_tools``."""
    return SimpleNamespace(
        name="test_agent",
        tools=[],
        llm=None,
        llm_config=None,
        skills=skills,
        handoffs=None,
        output_schema=None,
        system_prompt="test",
    )


def _raising_prepare(ctx: Any, tool: Any) -> Any:
    raise KeyError("context state missing")


class TestSkillToolPrepareException:
    async def test_skill_tool_excluded_when_prepare_raises(self) -> None:
        """A skill tool whose ``prepare`` raises is dropped, not offered."""

        @function_tool(name="secret_search", prepare=_raising_prepare)
        def secret_search(query: str) -> str:
            return "results"

        skill = Skill(name="gated", description="gated skill", tools=[secret_search])
        agent = _make_agent([skill])

        result = await build_tools(agent)

        # prepare() raised → the tool must be excluded entirely (same as the
        # regular-tool path and same as prepare returning None). It must NOT be
        # appended with its original schema.
        assert result is None or all(t.name != "secret_search" for t in result)

    async def test_skill_tool_excluded_matches_prepare_returning_none(self) -> None:
        """A raising prepare and a None-returning prepare yield the same exclusion."""

        def _none_prepare(ctx: Any, tool: Any) -> Any:
            return None

        @function_tool(name="quota_tool", prepare=_none_prepare)
        def quota_tool(query: str) -> str:
            return "results"

        skill = Skill(name="gated", description="gated skill", tools=[quota_tool])
        agent = _make_agent([skill])

        result = await build_tools(agent)

        assert result is None or all(t.name != "quota_tool" for t in result)

    async def test_other_skill_tools_survive_when_one_prepare_raises(self) -> None:
        """A raising prepare excludes only its own tool, not its siblings."""

        @function_tool(name="raising_tool", prepare=_raising_prepare)
        def raising_tool(query: str) -> str:
            return "results"

        @function_tool(name="healthy_tool")
        def healthy_tool(query: str) -> str:
            return "ok"

        skill = Skill(
            name="mixed",
            description="mixed skill",
            tools=[raising_tool, healthy_tool],
        )
        agent = _make_agent([skill])

        result = await build_tools(agent)

        assert result is not None
        names = {t.name for t in result}
        assert "raising_tool" not in names
        assert "healthy_tool" in names

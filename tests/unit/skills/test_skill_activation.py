"""Tests for EAGER and LAZY skill activation in the Runner."""

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.context import RunContext
from troopai.adk.run.loop import resolve_system_prompt
from troopai.adk.skills import Skill, SkillActivation
from troopai.adk.tools import FunctionTool


@pytest.fixture
def review_skill() -> Skill:
    tool = FunctionTool(
        name="analyze_code",
        schema={"type": "object", "properties": {"code": {"type": "string"}}},
        description="Analyze code for issues",
    )
    return Skill(
        name="code-review",
        description="Expert code review",
        instructions="When reviewing code:\n1. Check security\n2. Check performance",
        tools=[tool],
    )


@pytest.fixture
def debug_skill() -> Skill:
    return Skill(
        name="debugging",
        description="Systematic debugging",
        instructions="Debug step by step:\n1. Reproduce\n2. Isolate\n3. Fix",
    )


@pytest.fixture
def ctx() -> RunContext:
    return RunContext(context=None)


class TestEagerActivation:
    """EAGER mode: instructions injected at run start."""

    @pytest.mark.asyncio
    async def test_instructions_in_prompt(
        self,
        review_skill: Skill,
        ctx: RunContext,
    ) -> None:
        agent = Agent(
            name="Dev",
            system_prompt="Help developers.",
            skills=[review_skill],
            skill_activation=SkillActivation.EAGER,
        )
        prompt = await resolve_system_prompt(agent, ctx)
        assert "## Available Skills" in prompt
        assert "### code-review" in prompt
        assert "Check security" in prompt

    @pytest.mark.asyncio
    async def test_multiple_skills(
        self,
        review_skill: Skill,
        debug_skill: Skill,
        ctx: RunContext,
    ) -> None:
        agent = Agent(
            name="Dev",
            system_prompt="Help.",
            skills=[review_skill, debug_skill],
            skill_activation=SkillActivation.EAGER,
        )
        prompt = await resolve_system_prompt(agent, ctx)
        assert "### code-review" in prompt
        assert "### debugging" in prompt

    @pytest.mark.asyncio
    async def test_disabled_skill_excluded(self, ctx: RunContext) -> None:
        disabled = Skill(
            name="hidden",
            description="Hidden",
            instructions="Should not appear",
            enabled=False,
        )
        agent = Agent(
            name="Dev",
            system_prompt="Help.",
            skills=[disabled],
        )
        prompt = await resolve_system_prompt(agent, ctx)
        assert "### hidden" not in prompt

    @pytest.mark.asyncio
    async def test_skill_without_instructions(self, ctx: RunContext) -> None:
        tool = FunctionTool(
            name="tool_only",
            schema={"type": "object", "properties": {}},
            description="A tool",
        )
        no_instructions = Skill(
            name="tools-only",
            description="Only provides tools",
            tools=[tool],
        )
        agent = Agent(
            name="Dev",
            system_prompt="Help.",
            skills=[no_instructions],
        )
        prompt = await resolve_system_prompt(agent, ctx)
        # No instructions → no skills section
        assert "## Available Skills" not in prompt


class TestLazyActivation:
    """LAZY mode: instructions NOT in prompt at start."""

    @pytest.mark.asyncio
    async def test_instructions_not_in_initial_prompt(
        self,
        review_skill: Skill,
        ctx: RunContext,
    ) -> None:
        agent = Agent(
            name="Dev",
            system_prompt="Help developers.",
            skills=[review_skill],
            skill_activation=SkillActivation.LAZY,
        )
        prompt = await resolve_system_prompt(agent, ctx)
        assert "## Available Skills" not in prompt
        assert "### code-review" not in prompt

    @pytest.mark.asyncio
    async def test_base_prompt_unchanged(
        self,
        review_skill: Skill,
        ctx: RunContext,
    ) -> None:
        agent = Agent(
            name="Dev",
            system_prompt="Help developers.",
            skills=[review_skill],
            skill_activation=SkillActivation.LAZY,
        )
        prompt = await resolve_system_prompt(agent, ctx)
        assert prompt == "Help developers."

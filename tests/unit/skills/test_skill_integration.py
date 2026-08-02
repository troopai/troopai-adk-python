"""Tests for skill integration with build_tools and governance."""

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.context import RunContext
from troopai.adk.run.llm_calls import _apply_skill_governance, build_tools
from troopai.adk.skills import Skill, SkillGovernance
from troopai.adk.tools import FunctionTool


@pytest.fixture
def ctx() -> RunContext:
    return RunContext(context=None)


@pytest.fixture
def governed_skill() -> Skill:
    tool = FunctionTool(
        name="analyze",
        schema={"type": "object", "properties": {}},
        description="Analyze code",
    )
    return Skill(
        name="review",
        description="Code review",
        tools=[tool],
        governance=SkillGovernance(
            timeout=15.0,
            max_result_tokens=500,
            max_retries=2,
        ),
    )


class TestBuildToolsWithSkills:
    """Test that build_tools includes skill tools."""

    @pytest.mark.asyncio
    async def test_skill_tools_included(
        self,
        governed_skill: Skill,
        ctx: RunContext,
    ) -> None:
        agent = Agent(
            name="Dev",
            system_prompt="Help.",
            skills=[governed_skill],
        )
        tools = await build_tools(agent, context=ctx)
        assert tools is not None
        tool_names = [getattr(t, "name", None) for t in tools]
        assert "analyze" in tool_names

    @pytest.mark.asyncio
    async def test_skill_governance_applied(
        self,
        governed_skill: Skill,
        ctx: RunContext,
    ) -> None:
        agent = Agent(
            name="Dev",
            system_prompt="Help.",
            skills=[governed_skill],
        )
        tools = await build_tools(agent, context=ctx)
        assert tools is not None
        analyze = next(t for t in tools if getattr(t, "name", None) == "analyze")
        # Narrow to FunctionTool — build_tools returns a union including hosted/builtin tools.
        assert isinstance(analyze, FunctionTool)
        assert analyze.timeout == 15.0
        assert analyze.max_result_tokens == 500
        assert analyze.max_retries == 2

    @pytest.mark.asyncio
    async def test_tool_own_governance_takes_precedence(
        self,
        ctx: RunContext,
    ) -> None:
        """Tool's own governance should override skill governance."""
        tool = FunctionTool(
            name="fast_analyze",
            schema={"type": "object", "properties": {}},
            description="Fast analyze",
            timeout=5.0,  # Tool's own timeout
        )
        skill = Skill(
            name="review",
            description="Review",
            tools=[tool],
            governance=SkillGovernance(timeout=30.0),  # Skill governance
        )
        agent = Agent(
            name="Dev",
            system_prompt="Help.",
            skills=[skill],
        )
        tools = await build_tools(agent, context=ctx)
        assert tools is not None
        fast = next(t for t in tools if getattr(t, "name", None) == "fast_analyze")
        # Narrow to FunctionTool — build_tools returns a union including hosted/builtin tools.
        assert isinstance(fast, FunctionTool)
        assert fast.timeout == 5.0  # Tool's own value preserved

    @pytest.mark.asyncio
    async def test_disabled_skill_excluded(self, ctx: RunContext) -> None:
        tool = FunctionTool(
            name="hidden_tool",
            schema={"type": "object", "properties": {}},
            description="Hidden",
        )
        skill = Skill(
            name="hidden",
            description="Hidden skill",
            tools=[tool],
            enabled=False,
        )
        agent = Agent(
            name="Dev",
            system_prompt="Help.",
            skills=[skill],
        )
        tools = await build_tools(agent, context=ctx)
        # Either None or empty (no tools)
        if tools is not None:
            tool_names = [getattr(t, "name", None) for t in tools]
            assert "hidden_tool" not in tool_names

    @pytest.mark.asyncio
    async def test_agent_tools_and_skill_tools_combined(
        self,
        ctx: RunContext,
    ) -> None:
        agent_tool = FunctionTool(
            name="agent_tool",
            schema={"type": "object", "properties": {}},
            description="Agent's own tool",
        )
        skill_tool = FunctionTool(
            name="skill_tool",
            schema={"type": "object", "properties": {}},
            description="Skill's tool",
        )
        skill = Skill(name="s", description="s", tools=[skill_tool])
        agent = Agent(
            name="Dev",
            system_prompt="Help.",
            tools=[agent_tool],
            skills=[skill],
        )
        tools = await build_tools(agent, context=ctx)
        assert tools is not None
        names = [getattr(t, "name", None) for t in tools]
        assert "agent_tool" in names
        assert "skill_tool" in names


class TestApplySkillGovernance:
    """Test _apply_skill_governance helper."""

    def test_applies_governance_defaults(self) -> None:
        tool = FunctionTool(
            name="test",
            schema={"type": "object", "properties": {}},
        )
        skill = Skill(
            name="s",
            description="s",
            governance=SkillGovernance(
                timeout=20.0,
                max_result_tokens=1000,
                max_retries=3,
            ),
        )
        result = _apply_skill_governance(tool, skill)
        assert result.timeout == 20.0
        assert result.max_result_tokens == 1000
        assert result.max_retries == 3

    def test_preserves_tool_values(self) -> None:
        tool = FunctionTool(
            name="test",
            schema={"type": "object", "properties": {}},
            timeout=5.0,
            max_retries=1,
        )
        skill = Skill(
            name="s",
            description="s",
            governance=SkillGovernance(timeout=30.0, max_retries=10),
        )
        result = _apply_skill_governance(tool, skill)
        assert result.timeout == 5.0  # Tool's own
        assert result.max_retries == 1  # Tool's own

    def test_no_governance_returns_same(self) -> None:
        tool = FunctionTool(
            name="test",
            schema={"type": "object", "properties": {}},
        )
        skill = Skill(name="s", description="s")
        result = _apply_skill_governance(tool, skill)
        assert result is tool  # Same object, not a copy


class TestDuplicateToolNameValidation:
    """Test that Agent validates tool name uniqueness."""

    def test_duplicate_agent_and_skill_tool(self) -> None:
        tool = FunctionTool(
            name="analyze",
            schema={"type": "object", "properties": {}},
        )
        skill = Skill(name="s", description="s", tools=[tool])
        with pytest.raises(ValueError, match="Duplicate tool name"):
            Agent(
                name="Dev",
                system_prompt="Help.",
                tools=[tool],
                skills=[skill],
            )

    def test_no_duplicate_passes(self) -> None:
        tool1 = FunctionTool(
            name="agent_tool",
            schema={"type": "object", "properties": {}},
        )
        tool2 = FunctionTool(
            name="skill_tool",
            schema={"type": "object", "properties": {}},
        )
        skill = Skill(name="s", description="s", tools=[tool2])
        # Should not raise
        agent = Agent(
            name="Dev",
            system_prompt="Help.",
            tools=[tool1],
            skills=[skill],
        )
        assert len(agent.skills) == 1

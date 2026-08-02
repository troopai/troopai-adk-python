"""Tests for FunctionTool.max_retries (LLM-driven retry budget).

Verifies that:
- max_retries validation works correctly
- tool_failure_counts tracking in _execute_tool_calls
- _build_tools filters exhausted tools
- Budget semantics: None (no limit), 0 (no retries), N (N retries)
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.tools.function_tool import FunctionTool


def _names(result: Any) -> list[str]:
    """Extract FunctionTool names from a build_tools result.

    ``build_tools`` returns ``list[FunctionTool | BuiltinTool | HostedTool[...]] | None``;
    these tests only construct ``FunctionTool``s, so we narrow with
    ``isinstance`` and skip None to keep pyright happy.
    """
    if result is None:
        return []
    return [t.name for t in result if isinstance(t, FunctionTool)]


# ── FunctionTool.max_retries validation ─────────────────────────────


class TestMaxRetriesValidation:
    """Test max_retries field validation in __post_init__."""

    def test_default_is_none(self):
        """Default max_retries is None — load-bearing for skill-governance precedence.

        See ``FunctionTool.max_retries`` docstring: a bounded default
        would silently override ``SkillGovernance.max_retries`` for
        any tool inside a governed skill — a hidden cost change for
        developers who configured governance explicitly.
        """
        tool = FunctionTool(
            name="test",
            description="test",
            schema={"type": "object"},
        )
        assert tool.max_retries is None

    def test_zero_is_valid(self):
        """max_retries=0 means no retries allowed."""
        tool = FunctionTool(
            name="test",
            description="test",
            schema={"type": "object"},
            max_retries=0,
        )
        assert tool.max_retries == 0

    def test_positive_is_valid(self):
        """max_retries=3 allows 3 retries."""
        tool = FunctionTool(
            name="test",
            description="test",
            schema={"type": "object"},
            max_retries=3,
        )
        assert tool.max_retries == 3

    def test_negative_raises_value_error(self):
        """Negative max_retries should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            FunctionTool(
                name="test",
                description="test",
                schema={"type": "object"},
                max_retries=-1,
            )


# ── build_tools filtering ──────────────────────────────────


class TestBuildToolsFiltering:
    """Test that _build_tools filters exhausted tools."""

    @pytest.mark.asyncio
    async def test_no_failure_counts_includes_all_tools(self):
        """When tool_failure_counts is None, all tools are included."""
        from troopai.adk.run.llm_calls import build_tools

        agent = MagicMock()
        tool = FunctionTool(
            name="my_tool",
            description="test",
            schema={"type": "object"},
            max_retries=2,
        )
        tool.check_enabled = AsyncMock(return_value=True)
        agent.tools = [tool]
        agent.handoffs = None

        result = await build_tools(agent, tool_failure_counts=None)
        tool_names = _names(result)
        assert "my_tool" in tool_names

    @pytest.mark.asyncio
    async def test_exhausted_tool_filtered_out(self):
        """Tool with failures > max_retries is excluded from build_tools."""
        from troopai.adk.run.llm_calls import build_tools

        agent = MagicMock()
        tool = FunctionTool(
            name="flaky_tool",
            description="test",
            schema={"type": "object"},
            max_retries=2,
        )
        tool.check_enabled = AsyncMock(return_value=True)
        agent.tools = [tool]
        agent.handoffs = None

        # 3 failures > max_retries of 2 → should be filtered
        # When all tools are filtered, _build_tools returns None
        failure_counts = {"flaky_tool": 3}
        result = await build_tools(agent, tool_failure_counts=failure_counts)
        assert "flaky_tool" not in _names(result)

    @pytest.mark.asyncio
    async def test_tool_at_budget_still_included(self):
        """Tool with failures == max_retries is still available (budget not exceeded)."""
        from troopai.adk.run.llm_calls import build_tools

        agent = MagicMock()
        tool = FunctionTool(
            name="tool_a",
            description="test",
            schema={"type": "object"},
            max_retries=2,
        )
        tool.check_enabled = AsyncMock(return_value=True)
        agent.tools = [tool]
        agent.handoffs = None

        # Exactly at budget (2 failures, max_retries=2) → still included
        # Because the check is > not >=. After 2 failures, the 3rd call
        # is what triggers the disable.
        failure_counts = {"tool_a": 2}
        result = await build_tools(agent, tool_failure_counts=failure_counts)
        tool_names = _names(result)
        assert "tool_a" in tool_names

    @pytest.mark.asyncio
    async def test_tool_exceeding_budget_filtered(self):
        """Tool with failures > max_retries is filtered out."""
        from troopai.adk.run.llm_calls import build_tools

        agent = MagicMock()
        tool = FunctionTool(
            name="tool_a",
            description="test",
            schema={"type": "object"},
            max_retries=2,
        )
        tool.check_enabled = AsyncMock(return_value=True)
        agent.tools = [tool]
        agent.handoffs = None

        # Exceeds budget → filtered
        # When all tools are filtered, _build_tools returns None
        failure_counts = {"tool_a": 3}
        result = await build_tools(agent, tool_failure_counts=failure_counts)
        assert "tool_a" not in _names(result)

    @pytest.mark.asyncio
    async def test_none_max_retries_never_filtered(self):
        """Tool with max_retries=None is never filtered regardless of failures."""
        from troopai.adk.run.llm_calls import build_tools

        agent = MagicMock()
        tool = FunctionTool(
            name="unlimited_tool",
            description="test",
            schema={"type": "object"},
            max_retries=None,
        )
        tool.check_enabled = AsyncMock(return_value=True)
        agent.tools = [tool]
        agent.handoffs = None

        failure_counts = {"unlimited_tool": 100}
        result = await build_tools(agent, tool_failure_counts=failure_counts)
        tool_names = _names(result)
        assert "unlimited_tool" in tool_names

    @pytest.mark.asyncio
    async def test_zero_max_retries_filtered_after_first_failure(self):
        """Tool with max_retries=0 is filtered after first failure."""
        from troopai.adk.run.llm_calls import build_tools

        agent = MagicMock()
        tool = FunctionTool(
            name="no_retry_tool",
            description="test",
            schema={"type": "object"},
            max_retries=0,
        )
        tool.check_enabled = AsyncMock(return_value=True)
        agent.tools = [tool]
        agent.handoffs = None

        # 1 failure > max_retries of 0 → filtered
        # When all tools are filtered, _build_tools returns None
        failure_counts = {"no_retry_tool": 1}
        result = await build_tools(agent, tool_failure_counts=failure_counts)
        assert "no_retry_tool" not in _names(result)

    @pytest.mark.asyncio
    async def test_mixed_tools_selective_filtering(self):
        """Only exhausted tools are filtered; healthy tools remain."""
        from troopai.adk.run.llm_calls import build_tools

        agent = MagicMock()
        exhausted_tool = FunctionTool(
            name="exhausted",
            description="test",
            schema={"type": "object"},
            max_retries=1,
        )
        exhausted_tool.check_enabled = AsyncMock(return_value=True)

        healthy_tool = FunctionTool(
            name="healthy",
            description="test",
            schema={"type": "object"},
            max_retries=5,
        )
        healthy_tool.check_enabled = AsyncMock(return_value=True)

        unlimited_tool = FunctionTool(
            name="unlimited",
            description="test",
            schema={"type": "object"},
            max_retries=None,
        )
        unlimited_tool.check_enabled = AsyncMock(return_value=True)

        agent.tools = [exhausted_tool, healthy_tool, unlimited_tool]
        agent.handoffs = None

        failure_counts = {"exhausted": 5, "healthy": 2, "unlimited": 10}
        result = await build_tools(agent, tool_failure_counts=failure_counts)
        tool_names = _names(result)

        assert "exhausted" not in tool_names  # 5 > 1
        assert "healthy" in tool_names  # 2 <= 5
        assert "unlimited" in tool_names  # None → never filtered

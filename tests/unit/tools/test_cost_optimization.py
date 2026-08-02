"""Tests for cost optimization features.

Covers:
- FunctionTool.max_result_tokens validation
- minify_json()
- apply_result_limits()
- CacheStrategy.STABLE in _build_tools()
- ExecutionAwareToolContext construction
- Handoff.budget field
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.context.context_config import CacheStrategy
from troopai.adk.run.cost import apply_result_limits, minify_json
from troopai.adk.run.llm_calls import build_tools
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tools.tool_context import ExecutionAwareToolContext, ToolContext

# ── FunctionTool.max_result_tokens validation ────────────────────────


class TestMaxResultTokensValidation:
    """Test max_result_tokens field validation in __post_init__."""

    def test_none_is_default(self) -> None:
        """Default max_result_tokens is None (no limit)."""
        tool = FunctionTool(
            name="test",
            description="test",
            schema={"type": "object"},
        )
        assert tool.max_result_tokens is None

    def test_positive_is_valid(self) -> None:
        """Positive max_result_tokens is accepted."""
        tool = FunctionTool(
            name="test",
            description="test",
            schema={"type": "object"},
            max_result_tokens=500,
        )
        assert tool.max_result_tokens == 500

    def test_zero_raises_value_error(self) -> None:
        """max_result_tokens=0 should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            FunctionTool(
                name="test",
                description="test",
                schema={"type": "object"},
                max_result_tokens=0,
            )

    def test_negative_raises_value_error(self) -> None:
        """Negative max_result_tokens should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            FunctionTool(
                name="test",
                description="test",
                schema={"type": "object"},
                max_result_tokens=-10,
            )


# ── minify_json ──────────────────────────────────────────────


class TestMinifyJson:
    """Test JSON minification of tool results."""

    def test_minifies_json_object(self) -> None:
        """Pretty-printed JSON should be compacted."""

        pretty = json.dumps({"name": "Alice", "age": 30}, indent=2)
        minified = minify_json(pretty)
        assert minified == '{"name":"Alice","age":30}'

    def test_minifies_json_array(self) -> None:
        """JSON arrays should be compacted."""

        pretty = json.dumps([1, 2, 3], indent=2)
        minified = minify_json(pretty)
        assert minified == "[1,2,3]"

    def test_preserves_non_json(self) -> None:
        """Non-JSON strings should pass through unchanged."""

        text = "This is not JSON"
        assert minify_json(text) == text

    def test_preserves_unicode(self) -> None:
        """Unicode in JSON values should be preserved (no ascii escaping)."""

        original = json.dumps({"name": "Ségolène"}, ensure_ascii=True)
        minified = minify_json(original)
        assert "Ségolène" in minified

    def test_handles_empty_string(self) -> None:
        """Empty string should pass through."""

        assert minify_json("") == ""

    def test_handles_nested_json(self) -> None:
        """Nested JSON should be fully minified."""

        pretty = json.dumps({"a": {"b": [1, 2], "c": "d"}}, indent=4)
        minified = minify_json(pretty)
        assert " " not in minified
        assert minified == '{"a":{"b":[1,2],"c":"d"}}'


# ── apply_result_limits ──────────────────────────────────────


class TestApplyResultLimits:
    """Test tool result truncation based on max_result_tokens."""

    def test_no_limit_passes_through(self) -> None:
        """Tool with max_result_tokens=None returns result unchanged."""

        tool = FunctionTool(
            name="test",
            description="test",
            schema={"type": "object"},
            max_result_tokens=None,
        )
        result = "x" * 10000
        assert apply_result_limits(result, tool, "gpt-4o-mini") == result

    @patch("troopai.adk.context.token_counter.TokenCounter.count_text", return_value=100)
    def test_within_budget_passes_through(self, _mock_count: MagicMock) -> None:
        """Result within budget passes through unchanged."""

        tool = FunctionTool(
            name="test",
            description="test",
            schema={"type": "object"},
            max_result_tokens=500,
        )
        result = "short result"
        assert apply_result_limits(result, tool, "gpt-4o-mini") == result

    @patch("troopai.adk.context.token_counter.TokenCounter.count_text", return_value=1000)
    def test_over_budget_truncated(self, _mock_count: MagicMock) -> None:
        """Result exceeding budget is truncated with suffix."""

        tool = FunctionTool(
            name="test",
            description="test",
            schema={"type": "object"},
            max_result_tokens=200,
        )
        result = "x" * 5000
        truncated = apply_result_limits(result, tool, "gpt-4o-mini")

        assert len(truncated) < len(result)
        assert "[Result truncated:" in truncated
        assert "1000" in truncated  # original token count
        assert "200" in truncated  # max_result_tokens

    @patch(
        "troopai.adk.context.token_counter.TokenCounter.count_text",
        side_effect=lambda text, model: max(1, len(text) // 4),
    )
    def test_truncation_respects_token_budget(self, _mock_count: MagicMock) -> None:
        """Truncated content + suffix stays within max_result_tokens.

        The counter is mocked at 4 chars/token, so the binary search over
        character prefixes must land near (but never above) the budget.
        """

        tool = FunctionTool(
            name="test",
            description="test",
            schema={"type": "object"},
            max_result_tokens=100,
        )
        result = "x" * 5000
        truncated = apply_result_limits(result, tool, "gpt-4o-mini")

        content_before_suffix = truncated.split("\n[Result truncated:")[0]
        suffix_part = truncated[len(content_before_suffix) :]
        # Content tokens + reserved suffix tokens fit the budget under the
        # mocked counter (the documented conservative contract)…
        assert len(content_before_suffix) // 4 + len(suffix_part) // 4 <= 100
        # …and the search did not collapse to a trivially small prefix.
        assert len(content_before_suffix) > 300
        assert "[Result truncated:" in truncated

    @patch(
        "troopai.adk.context.token_counter.TokenCounter.count_text",
        side_effect=lambda text, model: max(1, len(text)),
    )
    def test_truncation_cjk_one_token_per_char(self, _mock_count: MagicMock) -> None:
        """CJK-like text (1 token/char) is not over-truncated 4x anymore."""

        tool = FunctionTool(
            name="test",
            description="test",
            schema={"type": "object"},
            max_result_tokens=100,
        )
        result = "\u6f22" * 5000
        truncated = apply_result_limits(result, tool, "gpt-4o-mini")

        # Under a 1-token-per-char counter the entire payload must stay
        # within the budget — the old 4-chars/token slice blew through it.
        assert len(truncated) <= 100


# ── CacheStrategy.STABLE in _build_tools ─────────────────────────────


class TestCacheStrategyStable:
    """Test that CacheStrategy.STABLE marks disabled tools as [UNAVAILABLE]."""

    @pytest.mark.asyncio
    async def test_stable_keeps_disabled_tool(self) -> None:
        """Disabled tool stays in list with [UNAVAILABLE] prefix."""

        agent = MagicMock()
        tool = FunctionTool(
            name="my_tool",
            description="Does something",
            schema={"type": "object"},
            enabled=False,
        )
        agent.tools = [tool]
        agent.handoffs = None

        result = await build_tools(
            agent,
            cache_strategy=CacheStrategy.STABLE,
        )
        assert result is not None
        assert len(result) == 1
        # Narrow to FunctionTool — build_tools returns a union including hosted tools.
        first = result[0]
        assert isinstance(first, FunctionTool)
        assert first.name == "my_tool"
        assert first.description is not None
        assert first.description.startswith("[UNAVAILABLE]")

    @pytest.mark.asyncio
    async def test_stable_keeps_exhausted_tool(self) -> None:
        """Exhausted tool stays in list with [UNAVAILABLE] prefix."""

        agent = MagicMock()
        tool = FunctionTool(
            name="flaky",
            description="Flaky tool",
            schema={"type": "object"},
            max_retries=1,
        )
        tool.check_enabled = AsyncMock(return_value=True)
        agent.tools = [tool]
        agent.handoffs = None

        result = await build_tools(
            agent,
            tool_failure_counts={"flaky": 5},
            cache_strategy=CacheStrategy.STABLE,
        )
        assert result is not None
        assert len(result) == 1
        first = result[0]
        assert isinstance(first, FunctionTool)
        assert first.description is not None
        assert "[UNAVAILABLE]" in first.description

    @pytest.mark.asyncio
    async def test_stable_preserves_enabled_tool_description(self) -> None:
        """Enabled tool keeps its original description."""

        agent = MagicMock()
        tool = FunctionTool(
            name="my_tool",
            description="Does something",
            schema={"type": "object"},
            enabled=True,
        )
        tool.check_enabled = AsyncMock(return_value=True)
        agent.tools = [tool]
        agent.handoffs = None

        result = await build_tools(
            agent,
            cache_strategy=CacheStrategy.STABLE,
        )
        assert result is not None
        first = result[0]
        assert isinstance(first, FunctionTool)
        assert first.description == "Does something"

    @pytest.mark.asyncio
    async def test_none_strategy_removes_disabled_tool(self) -> None:
        """Default (NONE) strategy removes disabled tools entirely."""

        agent = MagicMock()
        tool = FunctionTool(
            name="my_tool",
            description="Does something",
            schema={"type": "object"},
            enabled=False,
        )
        agent.tools = [tool]
        agent.handoffs = None

        result = await build_tools(
            agent,
            cache_strategy=CacheStrategy.NONE,
        )
        # All tools removed → None
        assert result is None

    @pytest.mark.asyncio
    async def test_mixed_tools_stable(self) -> None:
        """Mix of enabled and disabled tools with STABLE strategy."""

        agent = MagicMock()
        enabled_tool = FunctionTool(
            name="enabled",
            description="Enabled tool",
            schema={"type": "object"},
            enabled=True,
        )
        enabled_tool.check_enabled = AsyncMock(return_value=True)

        disabled_tool = FunctionTool(
            name="disabled",
            description="Disabled tool",
            schema={"type": "object"},
            enabled=False,
        )

        agent.tools = [enabled_tool, disabled_tool]
        agent.handoffs = None

        result = await build_tools(
            agent,
            cache_strategy=CacheStrategy.STABLE,
        )
        assert result is not None
        assert len(result) == 2

        names: dict[str, str] = {t.name: t.description or "" for t in result if isinstance(t, FunctionTool)}
        assert names["enabled"] == "Enabled tool"
        assert names["disabled"].startswith("[UNAVAILABLE]")


# ── ExecutionAwareToolContext ─────────────────────────────────────────


class TestExecutionAwareToolContext:
    """Test ExecutionAwareToolContext subclass."""

    def test_is_subclass_of_tool_context(self) -> None:
        """ExecutionAwareToolContext inherits from ToolContext."""
        assert issubclass(ExecutionAwareToolContext, ToolContext)

    def test_default_values(self) -> None:
        """Default execution state values are zero/None."""
        ctx = ExecutionAwareToolContext(
            tool_name="test",
            tool_call_id="tc_1",
            tool_arguments={},
            raw_arguments="{}",
        )
        assert ctx.usage is None
        assert ctx.turns == 0
        assert ctx.messages == 0
        assert ctx.tokens == 0

    def test_with_execution_state(self) -> None:
        """Can construct with execution state snapshots."""
        from troopai.adk.llms.llm_usage import LLMUsage

        usage = LLMUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        ctx = ExecutionAwareToolContext(
            tool_name="test",
            tool_call_id="tc_1",
            tool_arguments={"q": "hello"},
            raw_arguments='{"q": "hello"}',
            usage=usage,
            turns=3,
            messages=10,
            tokens=5000,
        )
        assert ctx.usage is not None
        assert ctx.usage.total_tokens == 150
        assert ctx.turns == 3
        assert ctx.messages == 10
        assert ctx.tokens == 5000

    def test_isinstance_checks(self) -> None:
        """ExecutionAwareToolContext passes isinstance for both types."""
        ctx = ExecutionAwareToolContext(
            tool_name="test",
            tool_call_id="tc_1",
            tool_arguments={},
            raw_arguments="{}",
        )
        assert isinstance(ctx, ToolContext)
        assert isinstance(ctx, ExecutionAwareToolContext)

    def test_inherits_base_fields(self) -> None:
        """All base ToolContext fields are accessible."""
        ctx = ExecutionAwareToolContext(
            tool_name="test_tool",
            tool_call_id="tc_123",
            tool_arguments={"key": "val"},
            raw_arguments='{"key": "val"}',
            context={"user_id": "alice"},
        )
        assert ctx.tool_name == "test_tool"
        assert ctx.tool_call_id == "tc_123"
        assert ctx.context == {"user_id": "alice"}


# ── FunctionTool.execution_aware ─────────────────────────────────────


class TestExecutionAwareFlag:
    """Test execution_aware field on FunctionTool."""

    def test_default_is_false(self) -> None:
        """Default execution_aware is False."""
        tool = FunctionTool(
            name="test",
            description="test",
            schema={"type": "object"},
        )
        assert tool.execution_aware is False

    def test_can_set_to_true(self) -> None:
        """execution_aware can be set to True explicitly."""
        tool = FunctionTool(
            name="test",
            description="test",
            schema={"type": "object"},
            execution_aware=True,
        )
        assert tool.execution_aware is True


# ── Handoff.budget ───────────────────────────────────────────────────


class TestHandoffBudget:
    """Test HandoffConfig.budget field."""

    def test_default_is_bounded(self) -> None:
        """Default budget is 20_000 — bounded by truncation (no LLM call).

        The budget field accepts a ``TokenBudget`` (which exposes
        the drop-policy knob) or a bare ``int`` (normalized in
        ``__post_init__`` to ``TokenBudget(max_tokens=<int>,
        drop_policy="preserve_system")``). Default 20_000.
        """
        from troopai.adk.handoffs.handoff_config import HandoffConfig
        from troopai.adk.tools.token_budget import TokenBudget

        config = HandoffConfig()
        assert isinstance(config.budget, TokenBudget)
        assert config.budget.max_tokens == 20_000
        assert config.budget.drop_policy == "preserve_system"

    def test_can_set_budget(self) -> None:
        """Budget can be set to a token count (bare int coerced) or
        an explicit TokenBudget with custom drop_policy."""
        from troopai.adk.handoffs.handoff_config import HandoffConfig
        from troopai.adk.tools.token_budget import TokenBudget

        config = HandoffConfig(budget=5_000)
        assert isinstance(config.budget, TokenBudget)
        assert config.budget.max_tokens == 5_000

        typed = HandoffConfig(
            budget=TokenBudget(max_tokens=5_000, drop_policy="oldest_first"),
        )
        assert isinstance(typed.budget, TokenBudget)
        assert typed.budget.max_tokens == 5_000
        assert typed.budget.drop_policy == "oldest_first"

    def test_handoff_inherits_config_budget(self) -> None:
        """Handoff.config.budget propagates correctly."""
        from troopai.adk.handoffs.handoff import Handoff
        from troopai.adk.handoffs.handoff_config import HandoffConfig
        from troopai.adk.tools.token_budget import TokenBudget

        agent = MagicMock()
        agent.name = "test_agent"
        handoff = Handoff(target=agent, config=HandoffConfig(budget=5_000))
        assert isinstance(handoff.config.budget, TokenBudget)
        assert handoff.config.budget.max_tokens == 5_000

    def test_is_frozen(self) -> None:
        """HandoffConfig is a frozen dataclass — budget cannot be mutated."""
        from troopai.adk.handoffs.handoff_config import HandoffConfig

        config = HandoffConfig(budget=5_000)
        with pytest.raises(AttributeError):
            # Deliberate frozen-dataclass violation under test —
            # pyright/mypy correctly flag this assignment.
            config.budget = 10_000  # type: ignore[misc]


# ── FunctionSchema.execution_aware detection ─────────────────────────


class TestFunctionSchemaDetection:
    """Test that function_schema detects ExecutionAwareToolContext."""

    def test_detects_execution_aware_context(self) -> None:
        """Function with ExecutionAwareToolContext param sets execution_aware=True."""
        from troopai.adk.schemas.function_schema import function_schema

        def my_tool(_ctx: ExecutionAwareToolContext, query: str) -> str:
            return query

        schema = function_schema(my_tool)
        assert schema.takes_context is True
        assert schema.execution_aware is True

    def test_plain_tool_context_not_execution_aware(self) -> None:
        """Function with plain ToolContext param sets execution_aware=False."""
        from troopai.adk.schemas.function_schema import function_schema

        def my_tool(_ctx: ToolContext, query: str) -> str:
            return query

        schema = function_schema(my_tool)
        assert schema.takes_context is True
        assert schema.execution_aware is False

    def test_no_context_not_execution_aware(self) -> None:
        """Function with no context param sets both to False."""
        from troopai.adk.schemas.function_schema import function_schema

        def my_tool(query: str) -> str:
            return query

        schema = function_schema(my_tool)
        assert schema.takes_context is False
        assert schema.execution_aware is False


# ── CacheStrategy enum ───────────────────────────────────────────────


class TestCacheStrategyEnum:
    """Test CacheStrategy StrEnum values."""

    def test_none_value(self) -> None:
        assert CacheStrategy.NONE == "none"

    def test_stable_value(self) -> None:
        assert CacheStrategy.STABLE == "stable"

    def test_is_str_enum(self) -> None:
        """CacheStrategy values are strings."""
        assert isinstance(CacheStrategy.NONE, str)
        assert isinstance(CacheStrategy.STABLE, str)

"""Tests for ToolContext and ExecutionAwareToolContext."""

from unittest.mock import MagicMock

from troopai.adk.tools.tool_context import (
    ExecutionAwareToolContext,
    ToolContext,
)


class TestToolContextWithContext:
    """ToolContext.with_context() returns a ToolContext."""

    def test_returns_tool_context(self) -> None:
        ctx = ToolContext(
            tool_name="search",
            tool_call_id="call_1",
            tool_arguments={"q": "hello"},
            raw_arguments='{"q": "hello"}',
            context={"user": "alice"},
        )
        new_ctx = ctx.with_context({"user": "bob"})
        assert isinstance(new_ctx, ToolContext)
        assert new_ctx.context == {"user": "bob"}
        assert new_ctx.tool_name == "search"

    def test_preserves_metadata(self) -> None:
        ctx = ToolContext(
            tool_name="t",
            tool_call_id="c",
            tool_arguments={},
            raw_arguments="{}",
            metadata={"provider": "anthropic"},
        )
        new_ctx = ctx.with_context("new_context")
        assert new_ctx.metadata == {"provider": "anthropic"}


class TestExecutionAwareToolContextWithContext:
    """ExecutionAwareToolContext.with_context() preserves execution state."""

    def test_returns_execution_aware_type(self) -> None:
        usage = MagicMock()
        ctx = ExecutionAwareToolContext(
            tool_name="search",
            tool_call_id="call_1",
            tool_arguments={},
            raw_arguments="{}",
            context="old",
            usage=usage,
            turns=3,
            messages=12,
            tokens=5000,
        )
        new_ctx = ctx.with_context("new")
        assert isinstance(new_ctx, ExecutionAwareToolContext)

    def test_preserves_usage(self) -> None:
        usage = MagicMock()
        ctx = ExecutionAwareToolContext(
            tool_name="t",
            tool_call_id="c",
            tool_arguments={},
            raw_arguments="{}",
            usage=usage,
            turns=5,
            messages=20,
            tokens=10000,
        )
        new_ctx = ctx.with_context("ctx")
        assert new_ctx.usage is usage
        assert new_ctx.turns == 5
        assert new_ctx.messages == 20
        assert new_ctx.tokens == 10000

    def test_updates_context(self) -> None:
        ctx = ExecutionAwareToolContext(
            tool_name="t",
            tool_call_id="c",
            tool_arguments={},
            raw_arguments="{}",
            context="old",
            turns=1,
        )
        new_ctx = ctx.with_context("new")
        assert new_ctx.context == "new"
        assert ctx.context == "old"  # original unchanged

    def test_preserves_base_fields(self) -> None:
        ctx = ExecutionAwareToolContext(
            tool_name="search",
            tool_call_id="call_99",
            tool_arguments={"q": "test"},
            raw_arguments='{"q":"test"}',
            metadata={"key": "val"},
            turns=2,
        )
        new_ctx = ctx.with_context(None)
        assert new_ctx.tool_name == "search"
        assert new_ctx.tool_call_id == "call_99"
        assert new_ctx.tool_arguments == {"q": "test"}
        assert new_ctx.raw_arguments == '{"q":"test"}'
        assert new_ctx.metadata == {"key": "val"}


class TestRunConfigIsInternal:
    """run_config is framework-internal: not a public field on the base context.

    The architectural invariant is that a ``ToolContext`` (what user tool
    functions receive) must not expose run-wide execution state. The
    agent-as-tool machinery still needs the parent ``RunConfig``, so it is
    threaded via a private ``_run_config`` field reachable only through the
    ``get_run_config()`` accessor — never as a public ``ctx.run_config``.
    """

    def test_base_context_has_no_public_run_config_attribute(self) -> None:
        ctx = ToolContext(
            tool_name="t",
            tool_call_id="c",
            tool_arguments={},
            raw_arguments="{}",
        )
        # The public, execution-wide leak is gone — tools cannot reach it.
        assert not hasattr(ctx, "run_config")

    def test_get_run_config_returns_threaded_config(self) -> None:
        cfg = MagicMock(name="RunConfig")
        ctx = ToolContext(
            tool_name="t",
            tool_call_id="c",
            tool_arguments={},
            raw_arguments="{}",
            _run_config=cfg,
        )
        assert ctx.get_run_config() is cfg

    def test_get_run_config_defaults_to_none(self) -> None:
        ctx = ToolContext(
            tool_name="t",
            tool_call_id="c",
            tool_arguments={},
            raw_arguments="{}",
        )
        assert ctx.get_run_config() is None

    def test_with_context_preserves_run_config(self) -> None:
        cfg = MagicMock(name="RunConfig")
        ctx = ToolContext(
            tool_name="t",
            tool_call_id="c",
            tool_arguments={},
            raw_arguments="{}",
            _run_config=cfg,
        )
        assert ctx.with_context("x").get_run_config() is cfg

    def test_execution_aware_preserves_run_config_through_with_context(self) -> None:
        cfg = MagicMock(name="RunConfig")
        ctx = ExecutionAwareToolContext(
            tool_name="t",
            tool_call_id="c",
            tool_arguments={},
            raw_arguments="{}",
            _run_config=cfg,
            turns=3,
        )
        new_ctx = ctx.with_context("x")
        assert new_ctx.get_run_config() is cfg
        assert new_ctx.turns == 3

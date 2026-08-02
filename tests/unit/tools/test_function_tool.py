import math
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from troopai.adk.tools.function_tool import FunctionTool, function_tool

# ── Helpers ──────────────────────────────────────────────────────────

MINIMAL_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _make_tool(**overrides: Any) -> FunctionTool:
    defaults: dict[str, Any] = {"name": "t", "schema": MINIMAL_SCHEMA}
    defaults.update(overrides)
    return FunctionTool(**defaults)


class _SearchInput(BaseModel):
    query: str
    max_results: int = 10


# ── TestFunctionToolPostInit ─────────────────────────────────────────


class TestFunctionToolPostInit:
    def test_valid_defaults(self) -> None:
        tool = _make_tool()
        assert tool.name == "t"
        assert tool.description is None
        assert tool.schema == MINIMAL_SCHEMA
        assert tool.max_result_tokens is None
        # Default None — load-bearing for skill-governance precedence.
        # Bounded default would silently override SkillGovernance.max_retries.
        assert tool.max_retries is None
        assert tool.timeout is None

    def test_max_result_tokens_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _make_tool(max_result_tokens=0)

    def test_max_result_tokens_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _make_tool(max_result_tokens=-5)

    def test_max_retries_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _make_tool(max_retries=-1)

    def test_timeout_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive and finite"):
            _make_tool(timeout=0.0)

    def test_timeout_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive and finite"):
            _make_tool(timeout=-2.5)

    def test_timeout_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="positive and finite"):
            _make_tool(timeout=math.inf)


# ── TestCheckEnabled ─────────────────────────────────────────────────


class TestCheckEnabled:
    @pytest.mark.asyncio
    async def test_bool_true(self) -> None:
        tool = _make_tool(enabled=True)
        assert await tool.check_enabled() is True

    @pytest.mark.asyncio
    async def test_bool_false(self) -> None:
        tool = _make_tool(enabled=False)
        assert await tool.check_enabled() is False

    @pytest.mark.asyncio
    async def test_callable_sync(self) -> None:
        checker = MagicMock(return_value=True)
        tool = _make_tool(enabled=checker)
        ctx = MagicMock()
        result = await tool.check_enabled(context=ctx)
        assert result is True
        checker.assert_called_once_with(ctx)

    @pytest.mark.asyncio
    async def test_callable_async(self) -> None:
        checker = AsyncMock(return_value=False)
        tool = _make_tool(enabled=checker)
        ctx = MagicMock()
        result = await tool.check_enabled(context=ctx)
        assert result is False
        checker.assert_called_once_with(ctx)


# ── TestCheckRequiresApproval ────────────────────────────────────────


class TestCheckRequiresApproval:
    @pytest.mark.asyncio
    async def test_bool_false(self) -> None:
        tool = _make_tool(requires_approval=False)
        ctx = MagicMock()
        assert await tool.check_requires_approval(ctx) is False

    @pytest.mark.asyncio
    async def test_bool_true(self) -> None:
        tool = _make_tool(requires_approval=True)
        ctx = MagicMock()
        assert await tool.check_requires_approval(ctx) is True

    @pytest.mark.asyncio
    async def test_callable_sync(self) -> None:
        checker = MagicMock(return_value=True)
        tool = _make_tool(requires_approval=checker)
        ctx = MagicMock()
        result = await tool.check_requires_approval(ctx)
        assert result is True
        checker.assert_called_once_with(ctx)

    @pytest.mark.asyncio
    async def test_callable_async(self) -> None:
        checker = AsyncMock(return_value=False)
        tool = _make_tool(requires_approval=checker)
        ctx = MagicMock()
        result = await tool.check_requires_approval(ctx)
        assert result is False
        checker.assert_called_once_with(ctx)


# ── TestGetJsonSchema ────────────────────────────────────────────────


class TestGetJsonSchema:
    def test_dict_schema(self) -> None:
        raw = {"type": "object", "properties": {"q": {"type": "string"}}}
        tool = _make_tool(schema=raw)
        result = tool.get_json_schema()
        assert isinstance(result, dict)
        assert result["type"] == "object"
        assert "q" in result["properties"]

    def test_pydantic_model_schema(self) -> None:
        tool = _make_tool(schema=_SearchInput)
        result = tool.get_json_schema()
        assert isinstance(result, dict)
        assert "query" in result["properties"]
        assert "max_results" in result["properties"]

    def test_dict_schema_not_mutated_by_strict_enforcement(self) -> None:
        """get_json_schema() must NOT mutate the stored dict schema.

        ``enforce_schema(STRICT)`` calls ``ensure_strict_schema`` which
        mutates the dict in-place (adds ``additionalProperties``,
        ``required``, etc.). The fix deep-copies the dict before passing
        it to ``enforce_schema`` so ``self.schema`` is never modified.
        """
        from troopai.adk.schemas import SchemaEnforcement

        original_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        # Keep a separate copy to compare against after the call.
        schema_copy = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        tool = _make_tool(
            schema=original_schema,
            schema_enforcement=SchemaEnforcement.STRICT,
        )
        result = tool.get_json_schema()

        # The returned schema has strict additions.
        assert result.get("additionalProperties") is False

        # The stored schema must be unchanged.
        assert tool.schema == schema_copy, (
            "get_json_schema() mutated the stored dict schema — self.schema no longer matches its original form"
        )
        assert "additionalProperties" not in original_schema

    def test_dict_schema_shared_object_not_cross_mutated(self) -> None:
        """Two tools sharing the same dict object must not cross-mutate."""
        from troopai.adk.schemas import SchemaEnforcement

        shared: dict[str, Any] = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }
        tool_a = _make_tool(schema=shared, schema_enforcement=SchemaEnforcement.STRICT)
        tool_b = _make_tool(schema=shared, schema_enforcement=SchemaEnforcement.STRICT)

        _result_a = tool_a.get_json_schema()
        # Calling on tool_a must not have altered the dict seen by tool_b.
        result_b = tool_b.get_json_schema()
        assert result_b.get("additionalProperties") is False
        # And the shared source is still pristine.
        assert "additionalProperties" not in shared


# ── TestDelegateAgent ────────────────────────────────────────────────


class TestDelegateAgent:
    def test_default_no_delegate(self) -> None:
        tool = _make_tool()
        assert tool.get_delegate_agent() is None

    def test_delegate_agent_set(self) -> None:
        sentinel = object()
        tool = _make_tool()
        object.__setattr__(tool, "_agent", sentinel)
        assert tool.get_delegate_agent() is sentinel


# ── TestDecoratorNameDescription ───────────────────────────────────


class TestDecoratorNameDescription:
    """Verify @function_tool resolves name/description from function when not explicit."""

    def test_name_from_function(self) -> None:
        @function_tool(parse_docstring=False)
        def my_search_tool(query: str) -> str:
            """Search the database."""
            return query

        assert my_search_tool.name == "my_search_tool"

    def test_description_none_without_docstring_parsing(self) -> None:
        @function_tool(parse_docstring=False)
        def lookup(query: str) -> str:
            """Find records in the database."""
            return query

        # Without parse_docstring, description is None (docstring not extracted)
        assert lookup.description is None

    def test_explicit_name_overrides(self) -> None:
        @function_tool(name="custom_name", parse_docstring=False)
        def my_func(query: str) -> str:
            """Do stuff."""
            return query

        assert my_func.name == "custom_name"

    def test_explicit_description_overrides(self) -> None:
        @function_tool(description="Custom description", parse_docstring=False)
        def my_func(query: str) -> str:
            """Docstring description."""
            return query

        assert my_func.description == "Custom description"

    def test_name_falls_back_to_function_name(self) -> None:
        @function_tool(parse_docstring=False)
        def another_tool(x: int) -> str:
            """Tool docstring."""
            return str(x)

        assert another_tool.name == "another_tool"


# ── TestToolErrorFunctionUsesToolContext ──────────────────────────────


class TestToolErrorFunctionContext:
    """ToolErrorFunction is typed for RunContext (its documented contract + both
    the timeout and as_tool call sites pass a RunContext). The on_invoke wrapper
    passes a ToolContext at runtime via a documented widening — error handlers
    only read ``.context`` and the exception, both present on either context.
    """

    def test_default_tool_error_function_typed_for_run_context(self) -> None:
        import typing

        from troopai.adk.run.context import RunContext
        from troopai.adk.tools.function_tool import default_tool_error_function

        hints = typing.get_type_hints(default_tool_error_function)
        # First parameter is the run context (RunContext[Any]).
        assert typing.get_origin(hints["ctx"]) is RunContext

    def test_tool_error_function_alias_uses_run_context(self) -> None:
        from troopai.adk.tools.function_tool import ToolErrorFunction

        # The alias is typed against RunContext, not ToolContext.
        assert "RunContext" in repr(ToolErrorFunction)

    async def test_error_function_receives_tool_context_at_runtime(self) -> None:
        """Error function is called with a ToolContext instance at runtime (wrapper path)."""
        from troopai.adk.run.context import RunContext
        from troopai.adk.tools.function_tool import function_tool
        from troopai.adk.tools.tool_context import ToolContext

        received_ctx: list[object] = []

        def my_error_fn(ctx: RunContext[Any], error: Exception) -> str:
            received_ctx.append(ctx)
            return "handled"

        @function_tool(on_tool_call_fails=my_error_fn)
        def boom() -> str:
            raise RuntimeError("oops")

        assert boom.on_invoke is not None
        ctx = ToolContext(
            tool_name="boom",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
        )
        result = await boom.on_invoke(ctx, "{}")
        assert result == "handled"
        assert len(received_ctx) == 1
        assert isinstance(received_ctx[0], ToolContext)


# ── TestErrorFunctionExceptionIsLogged ───────────────────────────────


class TestErrorFunctionExceptionIsLogged:
    """Finding 5: When the error function itself raises, the exception must be logged."""

    async def test_broken_error_function_is_logged_and_generic_returned(self, caplog: Any) -> None:
        import logging

        from troopai.adk.run.context import RunContext
        from troopai.adk.tools.function_tool import function_tool
        from troopai.adk.tools.tool_context import ToolContext

        def exploding_error_fn(ctx: RunContext[Any], error: Exception) -> str:
            raise RuntimeError("error function broke!")

        @function_tool(on_tool_call_fails=exploding_error_fn)
        def boom() -> str:
            raise ValueError("tool failed")

        assert boom.on_invoke is not None
        ctx = ToolContext(
            tool_name="boom",
            tool_call_id="c1",
            tool_arguments={},
            raw_arguments="{}",
        )
        with caplog.at_level(logging.WARNING, logger="troopai.adk.tools.function_tool"):
            result = await boom.on_invoke(ctx, "{}")
        # Returns generic fallback, not raw exception
        assert "error" in result.lower()
        # The error-function failure must be logged
        assert any("error function raised" in r.message for r in caplog.records)


# ── TestEnabledTypeNoBareAwaitable ────────────────────────────────────


class TestEnabledTypeAcceptsMaybeAwaitable:
    """The ``enabled`` field accepts a bool, a callable, or a (possibly-awaitable)
    bool — MaybeAwaitable[bool] is part of the union so the as_tool passthrough
    (whose enabled param is equally wide) type-checks.
    """

    def test_enabled_field_type_includes_maybe_awaitable(self) -> None:
        import dataclasses

        from troopai.adk.tools.function_tool import FunctionTool

        fields = {f.name: f for f in dataclasses.fields(FunctionTool)}
        enabled_field = fields["enabled"]
        annotation = str(enabled_field.type)
        assert "MaybeAwaitable" in annotation

    async def test_check_enabled_with_sync_callable(self) -> None:
        """The Callable arm of ``enabled`` resolves a sync predicate correctly."""
        tool = _make_tool(enabled=lambda ctx: True)
        from unittest.mock import MagicMock

        ctx = MagicMock()
        assert await tool.check_enabled(context=ctx) is True

    async def test_check_enabled_with_async_callable(self) -> None:
        """Ensure async callables (via Callable arm) still work correctly."""
        from unittest.mock import AsyncMock

        checker = AsyncMock(return_value=True)
        tool = _make_tool(enabled=checker)
        ctx = MagicMock()
        assert await tool.check_enabled(context=ctx) is True

"""Tests for :mod:`troopai.adk.workflows.temporal.tools`.

Covers:
- ``activity_tool`` preserves the wrapped function's name as the tool name.
- ``TemporalToolWrapper.should_wrap`` returns ``False`` when tool config is ``False``.
- ``TemporalToolWrapper.should_wrap`` returns ``True`` for tools not listed in tool_configs.
- ``TemporalToolWrapper.get_config`` returns the specific config when set.
- ``TemporalToolWrapper.get_config`` returns the default config for unlisted tools.
- ``activity_tool`` passes individual args to execute_activity, not a packed dict.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.tools.function_tool import FunctionTool, ToolInvokeFunction, function_tool
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.workflows.engine import ToolActivityConfig
from troopai.adk.workflows.temporal.tools import TemporalToolWrapper, activity_tool, to_durable_tool


async def _sample_activity(value: str) -> str:
    """Sample async function used as a stand-in for a Temporal activity."""
    return value


def _tool_ctx(name: str, args: dict, raw: str) -> ToolContext:
    """Build a minimal ToolContext for driving a tool's ``on_invoke``."""
    return ToolContext(tool_name=name, tool_call_id="call-1", tool_arguments=args, raw_arguments=raw)


def _invoke_of(tool: FunctionTool) -> ToolInvokeFunction:
    """Return the tool's ``on_invoke``, narrowed to non-None for the checker."""
    invoke = tool.on_invoke
    assert invoke is not None
    return invoke


class TestActivityToolPreservesName:
    def test_activity_tool_preserves_name(self) -> None:
        """The created FunctionTool carries the original function's name."""
        tool = activity_tool(_sample_activity)
        assert tool.name == "_sample_activity"


class TestToolActivityConfigDisablesWrapping:
    def test_tool_activity_config_disables_wrapping(self) -> None:
        """``should_wrap`` returns ``False`` when the tool config is ``False``."""
        wrapper = TemporalToolWrapper(tool_configs={"my_tool": False})
        assert wrapper.should_wrap("my_tool") is False


class TestToolActivityConfigDefaultWrapping:
    def test_tool_activity_config_default_wrapping(self) -> None:
        """``should_wrap`` returns ``True`` for tools not listed in tool_configs."""
        wrapper = TemporalToolWrapper()
        assert wrapper.should_wrap("unlisted_tool") is True

    def test_tool_activity_config_true_wrapping(self) -> None:
        """``should_wrap`` returns ``True`` when the tool has a ToolActivityConfig."""
        config = ToolActivityConfig(start_to_close_timeout=60, maximum_attempts=3)
        wrapper = TemporalToolWrapper(tool_configs={"my_tool": config})
        assert wrapper.should_wrap("my_tool") is True


class TestGetConfigReturnsSpecific:
    def test_get_config_returns_specific(self) -> None:
        """``get_config`` returns the specific ToolActivityConfig when set."""
        specific = ToolActivityConfig(start_to_close_timeout=120, maximum_attempts=5)
        wrapper = TemporalToolWrapper(tool_configs={"my_tool": specific})
        result = wrapper.get_config("my_tool")
        assert result is specific


class TestGetConfigReturnsDefault:
    def test_get_config_returns_default(self) -> None:
        """``get_config`` returns the default config for tools not listed in tool_configs."""
        default = ToolActivityConfig(start_to_close_timeout=45, maximum_attempts=3)
        wrapper = TemporalToolWrapper(default_config=default)
        result = wrapper.get_config("unlisted_tool")
        assert result is default

    def test_get_config_returns_default_when_config_is_false(self) -> None:
        """``get_config`` returns the default config when the tool entry is ``False``."""
        default = ToolActivityConfig(start_to_close_timeout=10, maximum_attempts=1)
        wrapper = TemporalToolWrapper(
            tool_configs={"disabled_tool": False},
            default_config=default,
        )
        result = wrapper.get_config("disabled_tool")
        assert result is default


class TestActivityToolDispatchesArgsCorrectly:
    async def test_dispatch_passes_args_not_packed_dict(self) -> None:
        """Inside a workflow, execute_activity receives the original args, not a packed dict.

        Regression: the old code did
            packed_input = {"args": args, "kwargs": kwargs}
            wf.execute_activity(fn, packed_input, ...)
        which sent a single-dict argument that no typed activity expects.
        The fix passes args directly via args=[...] keyword argument.
        """
        received_execute_activity_calls: list[tuple] = []

        async def _fake_activity(value: str) -> str:
            """A typed activity that takes a single str."""
            return value

        async def _fake_execute_activity(fn, *a, **kw):
            received_execute_activity_calls.append((fn, list(a), kw))
            return "ok"

        fake_wf = MagicMock()
        fake_wf.in_workflow.return_value = True
        fake_wf.execute_activity = _fake_execute_activity

        original_wf = sys.modules.get("temporalio.workflow")
        sys.modules["temporalio.workflow"] = fake_wf

        fake_common = MagicMock()
        fake_common.RetryPolicy = MagicMock(return_value=MagicMock())
        original_common = sys.modules.get("temporalio.common")
        sys.modules["temporalio.common"] = fake_common

        # Also need the temporalio top-level package entry
        import types

        fake_temporalio = types.ModuleType("temporalio")
        fake_temporalio.workflow = fake_wf  # type: ignore[attr-defined]
        fake_temporalio.common = fake_common  # type: ignore[attr-defined]
        original_temporalio = sys.modules.get("temporalio")
        sys.modules["temporalio"] = fake_temporalio

        try:
            # Rebuild the tool with the patched sys.modules active so that
            # the _dispatch_wrapper closure imports our fake when called.
            # We call _dispatch_wrapper directly by extracting it through the
            # closure inspection of on_invoke_tool -> _on_invoke_tool_impl -> func.
            tool = activity_tool(_fake_activity)
            # _dispatch_wrapper is the `func` held inside _on_invoke_tool_impl's closure.
            on_invoke_tool = tool.on_invoke
            on_invoke_impl = on_invoke_tool.__closure__[0].cell_contents  # type: ignore[index]
            dispatch_wrapper = on_invoke_impl.__closure__[0].cell_contents  # type: ignore[index]
            # dispatch_wrapper is the _dispatch_wrapper that calls execute_activity.
            # Call it with a single positional arg to simulate what the runner does.
            await dispatch_wrapper("hello")
        finally:
            if original_wf is None:
                sys.modules.pop("temporalio.workflow", None)
            else:
                sys.modules["temporalio.workflow"] = original_wf
            if original_common is None:
                sys.modules.pop("temporalio.common", None)
            else:
                sys.modules["temporalio.common"] = original_common
            if original_temporalio is None:
                sys.modules.pop("temporalio", None)
            else:
                sys.modules["temporalio"] = original_temporalio

        assert len(received_execute_activity_calls) == 1, "execute_activity should have been called once"
        _fn, pos_args, kw_args = received_execute_activity_calls[0]
        assert _fn is _fake_activity

        # Must NOT receive a packed {"args": ..., "kwargs": ...} dict as a positional arg.
        for pos_arg in pos_args:
            assert not isinstance(pos_arg, dict), (
                f"execute_activity received a packed dict as positional arg: {pos_arg!r}. "
                "Use args= keyword instead of packing args into a dict."
            )
        # args= keyword must carry the original call args as a list/tuple.
        assert "args" in kw_args, "execute_activity must receive individual args via args= keyword argument"
        assert list(kw_args["args"]) == ["hello"]


class TestActivityToolRejectsKeywordOnlyParameters:
    """Temporal activities accept positional args only — execute_activity has
    no channel for keyword arguments. A function with a keyword-only parameter
    would have those values silently dropped at dispatch (committing a wrong
    result to durable history) or raise for a missing required argument.
    ``activity_tool`` must reject such functions loudly at construction.
    """

    def test_keyword_only_parameter_rejected_at_construction(self) -> None:
        """A keyword-only parameter raises ValueError before any tool is built."""

        async def _kw_only_activity(*, query: str) -> str:
            return query

        with pytest.raises(ValueError, match="keyword-only"):
            activity_tool(_kw_only_activity)

    def test_keyword_only_with_default_rejected(self) -> None:
        """A keyword-only parameter with a default is also rejected.

        A default value is the dangerous case: dropping the kwarg would not
        raise but would silently run the activity with the default, committing
        a wrong result to durable history that replays forever.
        """

        async def _kw_only_default_activity(value: str, *, limit: int = 5) -> str:
            return f"{value}:{limit}"

        with pytest.raises(ValueError, match="keyword-only"):
            activity_tool(_kw_only_default_activity)

    def test_var_keyword_parameter_rejected(self) -> None:
        """A **kwargs parameter is rejected (no positional channel exists)."""

        async def _var_kw_activity(value: str, **extra: str) -> str:
            return value

        with pytest.raises(ValueError):
            activity_tool(_var_kw_activity)

    def test_positional_or_keyword_parameters_still_accepted(self) -> None:
        """Ordinary positional-or-keyword parameters remain valid."""

        async def _plain_activity(a: str, b: int) -> str:
            return f"{a}:{b}"

        tool = activity_tool(_plain_activity)
        assert tool.name == "_plain_activity"


class TestActivityToolRetriesOffByDefault:
    """Cost-conservative invariant: an activity is never re-run (re-billed)
    unless the developer explicitly raises ``maximum_attempts``.
    """

    async def test_default_retry_policy_is_single_attempt(self) -> None:
        """By default the dispatched RetryPolicy caps at one attempt (retries off).

        Regression: the default was ``maximum_attempts=2`` — retries ON —
        which silently re-bills a token-costing activity the developer never
        opted into.
        """
        import temporalio.workflow as twf

        tool = activity_tool(_sample_activity)
        with (
            patch.object(twf, "in_workflow", return_value=True),
            patch.object(twf, "execute_activity", new=AsyncMock(return_value="ok")) as exec_mock,
        ):
            ctx = _tool_ctx("_sample_activity", {"value": "hi"}, '{"value": "hi"}')
            await _invoke_of(tool)(ctx, '{"value": "hi"}')

        retry_policy = exec_mock.call_args.kwargs["retry_policy"]
        assert retry_policy.maximum_attempts == 1

    async def test_opt_in_retries_are_forwarded(self) -> None:
        """An explicit ``maximum_attempts`` is honored on dispatch."""
        import temporalio.workflow as twf

        tool = activity_tool(_sample_activity, maximum_attempts=4)
        with (
            patch.object(twf, "in_workflow", return_value=True),
            patch.object(twf, "execute_activity", new=AsyncMock(return_value="ok")) as exec_mock,
        ):
            ctx = _tool_ctx("_sample_activity", {"value": "hi"}, '{"value": "hi"}')
            await _invoke_of(tool)(ctx, '{"value": "hi"}')

        assert exec_mock.call_args.kwargs["retry_policy"].maximum_attempts == 4


async def _search(query: str) -> str:
    """A stand-in tool body with a real, non-generic signature."""
    return f"result: {query}"


class TestToDurableToolPreservesIdentity:
    """``to_durable_tool`` must keep the tool's real name and schema.

    Regression: the old ``wrap_tool`` fed the generic ``on_invoke(ctx, input)``
    closure to ``activity_tool``, collapsing every tool to the name
    ``on_invoke_tool`` and a ``(ctx, input)``-derived schema — a name
    collision plus schema degradation for the model.
    """

    def test_preserves_name(self) -> None:
        """The durable clone keeps the original tool name (not ``on_invoke_tool``)."""
        tool = function_tool(_search)
        durable = to_durable_tool(tool)
        assert durable.name == tool.name == "_search"
        assert durable.name != "on_invoke_tool"

    def test_preserves_schema(self) -> None:
        """The durable clone keeps the real parameter schema (has ``query``)."""
        tool = function_tool(_search)
        durable = to_durable_tool(tool)
        schema = durable.get_json_schema()
        assert "query" in schema.get("properties", {})
        # The generic wrapper's params must NOT leak into the schema.
        assert "input" not in schema.get("properties", {})
        assert "ctx" not in schema.get("properties", {})

    def test_rejects_tool_without_on_invoke(self) -> None:
        """A tool with ``on_invoke=None`` cannot be routed — raise clearly."""
        tool = FunctionTool(name="no_invoke", schema={"type": "object", "properties": {}}, on_invoke=None)
        with pytest.raises(ValueError, match="on_invoke"):
            to_durable_tool(tool)


class TestToDurableToolDispatch:
    """``to_durable_tool`` routes correctly by workflow context."""

    async def test_outside_workflow_calls_original(self) -> None:
        """Outside a workflow the original ``on_invoke`` runs directly."""
        tool = function_tool(_search)
        durable = to_durable_tool(tool)
        ctx = _tool_ctx("_search", {"query": "hi"}, '{"query": "hi"}')
        # temporalio is installed but we are not inside a workflow → direct call.
        result = await _invoke_of(durable)(ctx, '{"query": "hi"}')
        assert result == "result: hi"

    async def test_inside_workflow_dispatches_by_name(self) -> None:
        """Inside a workflow the call dispatches ``execute_activity`` by tool name.

        The raw JSON args string crosses the boundary (serializable); the
        run-scoped ToolContext is never shipped.
        """
        import temporalio.workflow as twf

        tool = function_tool(_search)
        durable = to_durable_tool(tool)
        with (
            patch.object(twf, "in_workflow", return_value=True),
            patch.object(twf, "execute_activity", new=AsyncMock(return_value="durable-result")) as exec_mock,
        ):
            result = await _invoke_of(durable)(MagicMock(), '{"query": "hi"}')

        assert result == "durable-result"
        # Dispatched by the tool's real name (string), not an undecorated closure.
        assert exec_mock.call_args.args[0] == "_search"
        # Args carry the serializable JSON string, not a ToolContext.
        assert exec_mock.call_args.kwargs["args"] == ['{"query": "hi"}']
        assert exec_mock.call_args.kwargs["retry_policy"].maximum_attempts == 1

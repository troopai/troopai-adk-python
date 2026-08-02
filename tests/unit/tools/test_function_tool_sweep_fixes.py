"""Regression tests for function_tool dispatch/error-handling fixes.

Covers three defects in the ``@function_tool`` decorator's ``on_invoke``
wrapper:

- The wrapper swallowed every exception into an error string, so a decorated
  tool could never reach the executor's ``ToolRetry`` handling, retry-budget
  accounting, or ``fail_on_tool_error`` policy.
- Sync tool bodies ran inline on the event loop, blocking it and defeating the
  per-tool timeout.
- A bare-awaitable ``enabled`` value was coerced with ``bool(coroutine)``
  (always truthy) instead of being awaited.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from troopai.adk.exceptions import ToolRetry
from troopai.adk.run.tools_executor import execute_tool_calls
from troopai.adk.tools.function_tool import FunctionTool, function_tool
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

MINIMAL_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _tool_ctx(name: str) -> ToolContext[Any]:
    return ToolContext(tool_name=name, tool_call_id="c1", tool_arguments={}, raw_arguments="{}")


# --- Executor-integration helpers (mirror test_tool_retry.py) ---------------


def _make_agent(tools: list[Any]) -> Any:
    from troopai.adk.agents.middleware import Middleware

    return SimpleNamespace(
        name="test_agent",
        tools=tools,
        tool_use_behavior="run_llm_again",
        handoffs=None,
        llm=None,
        hooks=None,
        middleware=Middleware(),
    )


def _make_tool_call(call_id: str, name: str) -> LLMResponseFunctionToolCall:
    return LLMResponseFunctionToolCall(call_id=call_id, name=name, arguments="{}")


def _make_ctx() -> Any:
    from troopai.adk.run.context import RunContext

    return RunContext(context=None)


def _make_hooks() -> Any:
    from troopai.adk.hooks.hooks import RunHooks

    return RunHooks()


def _config(*, fail_on_tool_error: bool) -> Any:
    from troopai.adk.run.config import RunConfig

    return RunConfig(fail_on_tool_error=fail_on_tool_error)


# --- Finding: on_invoke wrapper must not swallow control-flow / errors ------


class TestOnInvokeErrorPropagation:
    async def test_decorated_toolretry_is_reraised_not_stringified(self) -> None:
        @function_tool(parse_docstring=False)
        def query() -> str:
            raise ToolRetry("Use SELECT only")

        assert query.on_invoke is not None
        with pytest.raises(ToolRetry) as exc:
            await query.on_invoke(_tool_ctx("query"), "{}")
        assert exc.value.hint == "Use SELECT only"

    async def test_default_handler_propagates_generic_exception(self) -> None:
        @function_tool(parse_docstring=False)
        def boom() -> str:
            raise ValueError("kaboom")

        assert boom.on_invoke is not None
        with pytest.raises(ValueError, match="kaboom"):
            await boom.on_invoke(_tool_ctx("boom"), "{}")

    async def test_custom_handler_still_returns_string(self) -> None:
        def handler(ctx: Any, err: Exception) -> str:
            return f"handled: {err}"

        @function_tool(parse_docstring=False, on_tool_call_fails=handler)
        def boom() -> str:
            raise ValueError("kaboom")

        assert boom.on_invoke is not None
        result = await boom.on_invoke(_tool_ctx("boom"), "{}")
        assert result == "handled: kaboom"


class TestExecutorReachesErrorPolicyForDecoratedTools:
    async def test_toolretry_hint_reaches_llm_and_is_not_counted(self) -> None:
        @function_tool(parse_docstring=False, max_retries=1)
        def query() -> str:
            raise ToolRetry("Use SELECT only")

        counts: dict[str, int] = {}
        results, _ = await execute_tool_calls(
            agent=_make_agent([query]),
            tool_calls=[_make_tool_call("c1", "query")],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_config(fail_on_tool_error=False),
            model="gpt-4o-mini",
            tool_failure_counts=counts,
        )
        assert results[0].output == "Use SELECT only"
        assert counts.get("query", 0) == 0

    async def test_generic_error_becomes_result_and_counts_toward_budget(self) -> None:
        @function_tool(parse_docstring=False, max_retries=2)
        def boom() -> str:
            raise ValueError("kaboom")

        counts: dict[str, int] = {}
        results, _ = await execute_tool_calls(
            agent=_make_agent([boom]),
            tool_calls=[_make_tool_call("c1", "boom")],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_config(fail_on_tool_error=False),
            model="gpt-4o-mini",
            tool_failure_counts=counts,
        )
        assert "Error executing tool 'boom'" in results[0].output
        assert counts["boom"] == 1

    async def test_fail_on_tool_error_raises_for_decorated_tool(self) -> None:
        @function_tool(parse_docstring=False)
        def boom() -> str:
            raise ValueError("kaboom")

        with pytest.raises(ValueError, match="kaboom"):
            await execute_tool_calls(
                agent=_make_agent([boom]),
                tool_calls=[_make_tool_call("c1", "boom")],
                ctx_wrapper=_make_ctx(),
                hooks=_make_hooks(),
                config=_config(fail_on_tool_error=True),
                model="gpt-4o-mini",
            )


# --- Finding: sync tool bodies run off the event-loop thread ----------------


class TestSyncToolRunsOffEventLoop:
    async def test_sync_body_executes_in_worker_thread(self) -> None:
        main_ident = threading.get_ident()
        seen: dict[str, int] = {}

        @function_tool(parse_docstring=False)
        def probe() -> str:
            seen["ident"] = threading.get_ident()
            return "ok"

        assert probe.on_invoke is not None
        result = await probe.on_invoke(_tool_ctx("probe"), "{}")
        assert result == "ok"
        assert seen["ident"] != main_ident


# --- Finding: bare-awaitable ``enabled`` must be awaited ---------------------


class TestCheckEnabledBareAwaitable:
    async def test_bare_awaitable_enabled_is_awaited_not_truthy(self) -> None:
        async def _disabled() -> bool:
            return False

        tool = FunctionTool(name="t", schema=MINIMAL_SCHEMA, enabled=_disabled())
        assert await tool.check_enabled(context=MagicMock()) is False

    async def test_bare_awaitable_enabled_true(self) -> None:
        async def _enabled() -> bool:
            return True

        tool = FunctionTool(name="t", schema=MINIMAL_SCHEMA, enabled=_enabled())
        assert await tool.check_enabled(context=MagicMock()) is True

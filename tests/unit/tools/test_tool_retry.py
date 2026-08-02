"""Tests for ToolRetry exception handling."""

import pytest

from troopai.adk.exceptions import ToolRetry
from troopai.adk.run.tools_executor import execute_tool_calls
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

# ── Helpers ──────────────────────────────────────────────────────────

_call_count = 0


def _make_agent(tools):
    from types import SimpleNamespace

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


def _make_tool_call(call_id, name, args="{}"):
    return LLMResponseFunctionToolCall(call_id=call_id, name=name, arguments=args)


def _make_ctx():
    from troopai.adk.run.context import RunContext

    return RunContext(context=None)


def _make_hooks():
    from troopai.adk.hooks.hooks import RunHooks

    return RunHooks()


def _make_config():
    from troopai.adk.run.config import RunConfig

    return RunConfig(fail_on_tool_error=False)


# ── Tests ────────────────────────────────────────────────────────────


class TestToolRetry:
    def setup_method(self):
        global _call_count
        _call_count = 0

    def test_exception_has_hint(self) -> None:
        err = ToolRetry("Use SELECT instead of DROP")
        assert err.hint == "Use SELECT instead of DROP"
        assert str(err) == "Use SELECT instead of DROP"

    @pytest.mark.asyncio
    async def test_retry_returns_hint_as_result(self) -> None:
        async def handler(_ctx, _raw_args):
            raise ToolRetry("Cannot run DROP. Use SELECT only.")

        tool = FunctionTool(
            name="query_db",
            description="Query DB",
            schema={"type": "object", "properties": {}},
            on_invoke=handler,
        )
        agent = _make_agent([tool])
        tc = _make_tool_call("c1", "query_db")

        results, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        assert len(results) == 1
        assert results[0].output == "Cannot run DROP. Use SELECT only."

    @pytest.mark.asyncio
    async def test_retry_does_not_count_as_failure(self) -> None:
        """ToolRetry should NOT increment the failure count."""

        async def handler(_ctx, _raw_args):
            raise ToolRetry("Try again with different params")

        tool = FunctionTool(
            name="flaky",
            description="Flaky tool",
            schema={"type": "object", "properties": {}},
            on_invoke=handler,
            max_retries=1,
        )
        agent = _make_agent([tool])
        failure_counts: dict[str, int] = {}

        # Call it 3 times — all ToolRetry, none should count as failure
        for i in range(3):
            tc = _make_tool_call(f"c{i}", "flaky")
            await execute_tool_calls(
                agent=agent,
                tool_calls=[tc],
                ctx_wrapper=_make_ctx(),
                hooks=_make_hooks(),
                config=_make_config(),
                model="gpt-4o-mini",
                tool_failure_counts=failure_counts,
            )

        # No failures recorded
        assert failure_counts.get("flaky", 0) == 0

    @pytest.mark.asyncio
    async def test_retry_not_cached(self) -> None:
        """ToolRetry results should NOT be cached."""
        call_count = 0

        async def handler(_ctx, _raw_args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ToolRetry("Try again")
            return "success"

        tool = FunctionTool(
            name="eventual",
            description="Eventually succeeds",
            schema={"type": "object", "properties": {}},
            on_invoke=handler,
            cache=True,
        )
        agent = _make_agent([tool])

        # First call — ToolRetry
        tc1 = _make_tool_call("c1", "eventual")
        results1, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc1],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        assert results1[0].output == "Try again"

        # Second call — handler runs again (not cached), succeeds
        tc2 = _make_tool_call("c2", "eventual")
        results2, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc2],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        assert results2[0].output == "success"
        assert call_count == 2

"""Tests for parallel tool execution via asyncio.gather."""

import asyncio
import time

import pytest

from troopai.adk.run.tools_executor import execute_tool_calls
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

# ── Helpers ──────────────────────────────────────────────────────────


def _make_agent(tools):
    """Minimal agent-like object with .tools, .tool_use_behavior, and .handoffs."""
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


def _make_tool_call(call_id: str, name: str, args: str = "{}") -> LLMResponseFunctionToolCall:
    return LLMResponseFunctionToolCall(call_id=call_id, name=name, arguments=args)


async def _slow_handler(ctx, _raw_args):
    await asyncio.sleep(0.1)
    return f"done_{ctx.tool_name}"


def _make_slow_tool(name: str) -> FunctionTool:
    return FunctionTool(
        name=name,
        description=f"Slow tool {name}",
        schema={"type": "object", "properties": {}},
        on_invoke=_slow_handler,
    )


def _make_ctx():
    from troopai.adk.run.context import RunContext

    return RunContext(context=None)


def _make_hooks():
    from troopai.adk.hooks.hooks import RunHooks

    return RunHooks()


def _make_config():
    from troopai.adk.run.config import DEFAULT_RUN_CONFIG

    return DEFAULT_RUN_CONFIG


# ── Tests ────────────────────────────────────────────────────────────


class TestParallelToolExecution:
    @pytest.mark.asyncio
    async def test_parallel_faster_than_sequential(self) -> None:
        """Three 100ms tools should complete in ~100ms parallel vs ~300ms sequential."""
        tools = [_make_slow_tool(f"tool_{i}") for i in range(3)]
        agent = _make_agent(tools)
        tool_calls = [_make_tool_call(f"call_{i}", f"tool_{i}") for i in range(3)]

        ctx = _make_ctx()
        hooks = _make_hooks()
        config = _make_config()

        # Sequential
        start = time.monotonic()
        results_seq, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=tool_calls,
            ctx_wrapper=ctx,
            hooks=hooks,
            config=config,
            model="gpt-4o-mini",
            parallel=False,
        )
        seq_time = time.monotonic() - start

        # Parallel
        start = time.monotonic()
        results_par, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=tool_calls,
            ctx_wrapper=ctx,
            hooks=hooks,
            config=config,
            model="gpt-4o-mini",
            parallel=True,
        )
        par_time = time.monotonic() - start

        # Both should return 3 results
        assert len(results_seq) == 3
        assert len(results_par) == 3

        # Parallel should be meaningfully faster
        assert par_time < seq_time * 0.7, (
            f"Parallel ({par_time:.3f}s) should be faster than sequential ({seq_time:.3f}s)"
        )

    @pytest.mark.asyncio
    async def test_parallel_preserves_results(self) -> None:
        """All tool results should be present regardless of execution order."""
        tools = [_make_slow_tool(f"t{i}") for i in range(3)]
        agent = _make_agent(tools)
        tool_calls = [_make_tool_call(f"c{i}", f"t{i}") for i in range(3)]

        results, deferred = await execute_tool_calls(
            agent=agent,
            tool_calls=tool_calls,
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
            parallel=True,
        )

        assert deferred is None
        result_values = {r.output for r in results}
        assert result_values == {"done_t0", "done_t1", "done_t2"}

    @pytest.mark.asyncio
    async def test_single_tool_not_gathered(self) -> None:
        """With one tool call, parallel=True should still work (no gather)."""
        tool = _make_slow_tool("solo")
        agent = _make_agent([tool])
        tool_calls = [_make_tool_call("c0", "solo")]

        results, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=tool_calls,
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
            parallel=True,
        )

        assert len(results) == 1
        assert results[0].output == "done_solo"

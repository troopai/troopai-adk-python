"""Tests for RunConfig.max_parallel_tools concurrency cap.

Covers:
- max_parallel_tools=1 prevents overlap (serial execution under parallel=True)
- max_parallel_tools=None (default) allows genuine overlap
- max_parallel_tools<=0 raises ValueError
"""

import asyncio

import pytest

from troopai.adk.run.config import RunConfig
from troopai.adk.run.tools_executor import execute_tool_calls
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

# ---------------------------------------------------------------------------
# Shared helpers (mirror style from test_parallel_tool_execution.py)
# ---------------------------------------------------------------------------


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


def _make_tool_call(call_id: str, name: str, args: str = "{}") -> LLMResponseFunctionToolCall:
    return LLMResponseFunctionToolCall(call_id=call_id, name=name, arguments=args)


def _make_ctx():
    from troopai.adk.run.context import RunContext

    return RunContext(context=None)


def _make_hooks():
    from troopai.adk.hooks.hooks import RunHooks

    return RunHooks()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMaxParallelToolsCap:
    async def test_cap_one_serialises_execution(self) -> None:
        """With max_parallel_tools=1, three async tools must not overlap.

        Each tool increments a shared counter, asserts it equals 1 (proving no
        other tool is running concurrently), sleeps briefly, then decrements.
        Any overlap would cause the counter to reach 2, triggering an AssertionError.
        """
        active_count = 0
        overlap_detected = False

        async def guarded_handler(ctx, _raw_args):
            nonlocal active_count, overlap_detected
            active_count += 1
            if active_count > 1:
                overlap_detected = True
            await asyncio.sleep(0.02)
            active_count -= 1
            return f"ok_{ctx.tool_name}"

        tools = [
            FunctionTool(
                name=f"tool_{i}",
                description=f"Tool {i}",
                schema={"type": "object", "properties": {}},
                on_invoke=guarded_handler,
            )
            for i in range(3)
        ]
        agent = _make_agent(tools)
        tool_calls = [_make_tool_call(f"c{i}", f"tool_{i}") for i in range(3)]

        config = RunConfig(max_parallel_tools=1)

        results, deferred = await execute_tool_calls(
            agent=agent,
            tool_calls=tool_calls,
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=config,
            model="gpt-4o-mini",
            parallel=True,
        )

        assert not overlap_detected, "Tools overlapped despite max_parallel_tools=1"
        assert deferred is None
        assert len(results) == 3
        outputs = {r.output for r in results}
        assert outputs == {"ok_tool_0", "ok_tool_1", "ok_tool_2"}

    async def test_default_none_allows_overlap(self) -> None:
        """Default max_parallel_tools=None lets tools run concurrently.

        Three 30ms tools are launched with parallel=True and no cap.  We
        measure peak concurrency by tracking how many are active at once;
        it must reach > 1, proving the cap is opt-in and the default is
        genuinely unbounded.
        """
        active_count = 0
        peak_concurrency = 0

        async def tracking_handler(ctx, _raw_args):
            nonlocal active_count, peak_concurrency
            active_count += 1
            if active_count > peak_concurrency:
                peak_concurrency = active_count
            await asyncio.sleep(0.03)
            active_count -= 1
            return f"peak_ok_{ctx.tool_name}"

        tools = [
            FunctionTool(
                name=f"ptool_{i}",
                description=f"Parallel tool {i}",
                schema={"type": "object", "properties": {}},
                on_invoke=tracking_handler,
            )
            for i in range(3)
        ]
        agent = _make_agent(tools)
        tool_calls = [_make_tool_call(f"pc{i}", f"ptool_{i}") for i in range(3)]

        config = RunConfig()  # max_parallel_tools=None by default

        results, deferred = await execute_tool_calls(
            agent=agent,
            tool_calls=tool_calls,
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=config,
            model="gpt-4o-mini",
            parallel=True,
        )

        assert deferred is None
        assert len(results) == 3
        assert peak_concurrency > 1, f"Expected peak concurrency > 1 with no cap, got {peak_concurrency}"

    @pytest.mark.parametrize("bad_value", [0, -1, -100])
    @pytest.mark.asyncio
    async def test_invalid_cap_raises_value_error(self, bad_value: int) -> None:
        """max_parallel_tools=0 or negative must raise ValueError."""
        config = RunConfig(max_parallel_tools=bad_value)

        async def _dummy_invoke(ctx, _raw_args):
            return "x"

        tool = FunctionTool(
            name="dummy",
            description="Dummy",
            schema={"type": "object", "properties": {}},
            on_invoke=_dummy_invoke,
        )
        agent = _make_agent([tool])
        tool_calls = [_make_tool_call("c0", "dummy"), _make_tool_call("c1", "dummy")]

        with pytest.raises(ValueError, match="max_parallel_tools"):
            await execute_tool_calls(
                agent=agent,
                tool_calls=tool_calls,
                ctx_wrapper=_make_ctx(),
                hooks=_make_hooks(),
                config=config,
                model="gpt-4o-mini",
                parallel=True,
            )

    async def test_invalid_cap_raises_even_for_single_call(self) -> None:
        """The cap is validated on every execution, not only multi-call batches."""

        async def handler(ctx, _raw_args):
            return "ok"

        tool = FunctionTool(
            name="tool_0",
            description="Tool 0",
            schema={"type": "object", "properties": {}},
            on_invoke=handler,
        )
        agent = _make_agent([tool])
        config = RunConfig(max_parallel_tools=0)
        with pytest.raises(ValueError, match="max_parallel_tools"):
            await execute_tool_calls(
                agent=agent,
                tool_calls=[_make_tool_call("c0", "tool_0")],
                ctx_wrapper=_make_ctx(),
                hooks=_make_hooks(),
                config=config,
                model="gpt-4o-mini",
                parallel=True,
            )

"""Unit tests for FunctionTool rate-limiting (ToolRateLimit)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.tools.tool_rate_limit import ToolRateLimit

MINIMAL_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _make_tool(**overrides: Any) -> FunctionTool:
    defaults: dict[str, Any] = {"name": "t", "schema": MINIMAL_SCHEMA}
    defaults.update(overrides)
    return FunctionTool(**defaults)


# ── ToolRateLimit dataclass ─────────────────────────────────────────


class TestToolRateLimit:
    def test_defaults(self) -> None:
        cfg = ToolRateLimit(rpm=10)
        assert cfg.rpm == 10
        assert cfg.behavior == "wait"

    def test_zero_rpm_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ToolRateLimit(rpm=0)

    def test_negative_rpm_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ToolRateLimit(rpm=-1)

    def test_frozen(self) -> None:
        cfg = ToolRateLimit(rpm=10)
        with pytest.raises(Exception):
            cfg.rpm = 5  # type: ignore[misc]


# ── FunctionTool integration ────────────────────────────────────────


class TestFunctionToolRateLimitField:
    def test_default_is_none(self) -> None:
        tool = _make_tool()
        assert tool.rate_limit is None

    def test_invalid_rpm_caught_at_config_construction(self) -> None:
        # ``ToolRateLimit`` is frozen and validates rpm > 0 in its own
        # ``__post_init__``. A negative-rpm config can't be built, so
        # FunctionTool never receives one.
        with pytest.raises(ValueError, match="positive"):
            ToolRateLimit(rpm=0)
        with pytest.raises(ValueError, match="positive"):
            ToolRateLimit(rpm=-1)

    def test_invalid_max_wait_caught_at_config_construction(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ToolRateLimit(rpm=10, max_wait_seconds=0)
        with pytest.raises(ValueError, match="positive"):
            ToolRateLimit(rpm=10, max_wait_seconds=-5)


# ── acquire_rate_slot — wait behavior ───────────────────────────────


class TestAcquireRateSlotWait:
    @pytest.mark.asyncio
    async def test_no_rate_limit_admits_all(self) -> None:
        tool = _make_tool()
        for _ in range(50):
            assert await tool.acquire_rate_slot() is True

    @pytest.mark.asyncio
    async def test_under_limit_admits_immediately(self) -> None:
        tool = _make_tool(rate_limit=ToolRateLimit(rpm=3))
        with patch("time.monotonic", return_value=100.0):
            assert await tool.acquire_rate_slot() is True
            assert await tool.acquire_rate_slot() is True
            assert await tool.acquire_rate_slot() is True

    @pytest.mark.asyncio
    async def test_over_limit_sleeps_until_window_expires(self) -> None:
        tool = _make_tool(rate_limit=ToolRateLimit(rpm=3, behavior="wait"))

        # 1) Fill the window at t=100
        with patch("time.monotonic", return_value=100.0):
            for _ in range(3):
                assert await tool.acquire_rate_slot() is True

        # 2) At t=130 (still within window), the 4th call must sleep.
        # Patch sleep to advance the simulated clock past the window
        # so the retry loop succeeds without real wall-clock waiting.
        clock = {"now": 130.0}

        def _now() -> float:
            return clock["now"]

        async def _fake_sleep(d: float) -> None:
            clock["now"] += d

        with patch("time.monotonic", side_effect=_now), patch("asyncio.sleep", side_effect=_fake_sleep):
            assert await tool.acquire_rate_slot() is True
        # The slot opened only after the simulated sleep advanced past
        # the 60-second window from the first stamped call (100 -> 160).
        assert clock["now"] >= 160.0


# ── acquire_rate_slot — error behavior ──────────────────────────────


class TestAcquireRateSlotError:
    @pytest.mark.asyncio
    async def test_under_limit_admits(self) -> None:
        tool = _make_tool(rate_limit=ToolRateLimit(rpm=2, behavior="error"))
        with patch("time.monotonic", return_value=100.0):
            assert await tool.acquire_rate_slot() is True
            assert await tool.acquire_rate_slot() is True

    @pytest.mark.asyncio
    async def test_over_limit_returns_false(self) -> None:
        tool = _make_tool(rate_limit=ToolRateLimit(rpm=2, behavior="error"))
        with patch("time.monotonic", return_value=100.0):
            assert await tool.acquire_rate_slot() is True
            assert await tool.acquire_rate_slot() is True
            assert await tool.acquire_rate_slot() is False
            # The third saturated call doesn't consume a slot, so a
            # follow-up acquire (still in the same window) also fails.
            assert await tool.acquire_rate_slot() is False


# ── Window slide ─────────────────────────────────────────────────────


class TestWindowSlide:
    @pytest.mark.asyncio
    async def test_old_timestamps_drop_out(self) -> None:
        tool = _make_tool(rate_limit=ToolRateLimit(rpm=2, behavior="error"))
        with patch("time.monotonic", return_value=100.0):
            assert await tool.acquire_rate_slot() is True
            assert await tool.acquire_rate_slot() is True
            # Window is full at t=100 — third call rejected
            assert await tool.acquire_rate_slot() is False
        # 61s later the original two timestamps are out of window;
        # two fresh slots are admitted.
        with patch("time.monotonic", return_value=161.0):
            assert await tool.acquire_rate_slot() is True
            assert await tool.acquire_rate_slot() is True
            # Window saturated again with the two new timestamps.
            assert await tool.acquire_rate_slot() is False


# ── Concurrency under shared lock ───────────────────────────────────


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_lock_serialises_concurrent_acquires(self) -> None:
        tool = _make_tool(rate_limit=ToolRateLimit(rpm=5, behavior="error"))
        # Hold time constant so the window doesn't slide between calls.
        with patch("time.monotonic", return_value=200.0):
            results = await asyncio.gather(*(tool.acquire_rate_slot() for _ in range(7)))
        assert results.count(True) == 5
        assert results.count(False) == 2

    @pytest.mark.asyncio
    async def test_three_concurrent_waiters_all_admitted_eventually(self) -> None:
        # With rpm=1 and 3 concurrent waiters under behavior="wait", each
        # waiter sleeps independently (FIFO Lock, lock released before
        # sleep). Verify all three eventually admit by simulating a
        # clock that advances on every sleep.
        tool = _make_tool(rate_limit=ToolRateLimit(rpm=1, behavior="wait"))
        clock = {"now": 1000.0}

        def _now() -> float:
            return clock["now"]

        async def _fake_sleep(d: float) -> None:
            clock["now"] += d

        with patch("time.monotonic", side_effect=_now), patch("asyncio.sleep", side_effect=_fake_sleep):
            results = await asyncio.gather(*(tool.acquire_rate_slot() for _ in range(3)))
        assert results == [True, True, True]
        # Three calls at rpm=1 means at minimum two full window slides.
        assert clock["now"] >= 1000.0 + 60.0


# ── max_wait_seconds cap ─────────────────────────────────────────────


class TestMaxWaitSeconds:
    @pytest.mark.asyncio
    async def test_wait_cap_falls_back_to_error(self) -> None:
        # Cap the cumulative wait at 10s. A saturated window (rpm=1,
        # next slot in 60s) requires sleeping 60s, which exceeds the
        # cap, so the second call falls back to error semantics.
        tool = _make_tool(
            rate_limit=ToolRateLimit(rpm=1, behavior="wait", max_wait_seconds=10.0),
        )
        with patch("time.monotonic", return_value=500.0):
            assert await tool.acquire_rate_slot() is True
            # Second call would need to sleep ~60s; cap is 10s.
            assert await tool.acquire_rate_slot() is False

    @pytest.mark.asyncio
    async def test_wait_cap_admits_when_under_cap(self) -> None:
        # rpm=2 and a long max_wait_seconds: the second call admits
        # immediately, no sleep needed.
        tool = _make_tool(
            rate_limit=ToolRateLimit(rpm=2, behavior="wait", max_wait_seconds=5.0),
        )
        with patch("time.monotonic", return_value=500.0):
            assert await tool.acquire_rate_slot() is True
            assert await tool.acquire_rate_slot() is True


# ── @function_tool decorator passthrough ────────────────────────────


class TestDecoratorPassthrough:
    def test_decorator_max_calls_per_minute_shorthand(self) -> None:
        from troopai.adk.tools.function_tool import function_tool

        @function_tool(name="search", max_calls_per_minute=30)
        def search(query: str) -> str:
            return query

        assert search.rate_limit is not None
        assert search.rate_limit.rpm == 30
        assert search.rate_limit.behavior == "wait"

    def test_decorator_explicit_rate_limit(self) -> None:
        from troopai.adk.tools.function_tool import function_tool

        cfg = ToolRateLimit(rpm=10, behavior="error")

        @function_tool(name="api", rate_limit=cfg)
        def api(payload: str) -> str:
            return payload

        assert api.rate_limit is cfg

    def test_decorator_both_raises(self) -> None:
        from troopai.adk.tools.function_tool import function_tool

        # Validation happens inside function_tool() before it returns
        # the decorator, so we can call it directly without applying.
        with pytest.raises(ValueError, match="not both"):
            function_tool(
                name="x",
                rate_limit=ToolRateLimit(rpm=5),
                max_calls_per_minute=10,
            )


# ── Executor integration ────────────────────────────────────────────


def _make_agent(tools: list[Any]) -> Any:
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


class TestExecutorIntegration:
    @pytest.mark.asyncio
    async def test_error_behavior_returns_rate_limit_message(self) -> None:
        from troopai.adk.run.context import RunContext
        from troopai.adk.run.tools_executor import execute_tool_calls
        from troopai.adk.types.responses.llm_response import (
            LLMResponseFunctionToolCall,
        )

        invocations: list[int] = []

        async def _handler(_ctx: Any, _raw: str) -> str:
            invocations.append(1)
            return "ok"

        tool = FunctionTool(
            name="api",
            description="api",
            schema=MINIMAL_SCHEMA,
            on_invoke=_handler,
            rate_limit=ToolRateLimit(rpm=2, behavior="error"),
        )
        agent = _make_agent([tool])

        from troopai.adk.hooks.hooks import RunHooks
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG

        with patch("time.monotonic", return_value=500.0):
            for i in range(3):
                tc = LLMResponseFunctionToolCall(
                    call_id=f"c{i}",
                    name="api",
                    arguments="{}",
                )
                results, _ = await execute_tool_calls(
                    agent=agent,
                    tool_calls=[tc],
                    ctx_wrapper=RunContext(context=None),
                    hooks=RunHooks(),
                    config=DEFAULT_RUN_CONFIG,
                    model="gpt-4o-mini",
                )
                if i < 2:
                    assert results[0].output == "ok"
                else:
                    assert "rate-limited" in str(results[0].output)
        # Underlying handler ran exactly twice — the third invocation
        # short-circuited at the rate-limit gate.
        assert len(invocations) == 2


# ── Eager lock initialisation ───────────────────────────────────────


class TestRateLockEagerInit:
    """Verify that _rate_lock is initialised at construction time.

    Prior to the fix, _rate_lock was lazily created on first
    acquire_rate_slot() call. If clone() was called on a tool that had
    never been used, both the original and the clone had _rate_lock=None.
    Concurrent coroutines on two different clones would each create
    independent Lock objects, breaking mutual exclusion over the shared
    _rate_state deque.
    """

    def test_rate_lock_not_none_when_rate_limit_set(self) -> None:
        """_rate_lock is allocated in __post_init__ for rate-limited tools."""
        tool = _make_tool(rate_limit=ToolRateLimit(rpm=10))
        assert tool._rate_lock is not None, "_rate_lock must be non-None after construction when rate_limit is set"

    def test_rate_lock_is_none_when_no_rate_limit(self) -> None:
        """_rate_lock stays None for tools without a rate limit."""
        tool = _make_tool()
        assert tool._rate_lock is None

    def test_clone_shares_same_lock_object(self) -> None:
        """clone() must share the same Lock instance as the original.

        If each clone received its own Lock, concurrent calls on two
        different clones would not be mutually exclusive.
        """
        original = _make_tool(rate_limit=ToolRateLimit(rpm=5))
        clone = original.clone()
        assert clone._rate_lock is original._rate_lock, (
            "clone() must share the same Lock instance as the original — "
            "independent locks defeat rate-limit mutual exclusion"
        )

    @pytest.mark.asyncio
    async def test_clone_lock_is_valid_asyncio_lock(self) -> None:
        """The shared lock on a clone is a real asyncio.Lock (usable)."""
        import asyncio

        tool = _make_tool(rate_limit=ToolRateLimit(rpm=5))
        clone = tool.clone()
        assert isinstance(clone._rate_lock, asyncio.Lock)
        # Acquire and release to confirm it is a working Lock.
        async with clone._rate_lock:
            pass  # No exception == functional lock

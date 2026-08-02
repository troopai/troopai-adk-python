"""Tests for ``troopai.adk.sandbox.session.concurrency``."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from troopai.adk.sandbox.session.concurrency import gather_in_order


def _make_factory(result: int, delay: float = 0.0) -> Callable[[], Awaitable[int]]:
    async def factory() -> int:
        if delay > 0:
            await asyncio.sleep(delay)
        return result

    return factory


@pytest.mark.asyncio
async def test_empty_input_returns_empty() -> None:
    assert await gather_in_order([]) == []


@pytest.mark.asyncio
async def test_preserves_input_order_regardless_of_completion_order() -> None:
    factories: list[Callable[[], Awaitable[int]]] = [
        _make_factory(0, delay=0.05),  # slowest
        _make_factory(1, delay=0.02),
        _make_factory(2, delay=0.01),  # fastest
    ]
    assert await gather_in_order(factories) == [0, 1, 2]


@pytest.mark.asyncio
async def test_unbounded_concurrency_is_default() -> None:
    started: list[int] = []

    def make(i: int) -> Callable[[], Awaitable[int]]:
        async def factory() -> int:
            started.append(i)
            await asyncio.sleep(0.02)
            return i

        return factory

    factories = [make(i) for i in range(5)]
    await gather_in_order(factories)
    # All 5 should have started before the first one finishes.
    assert len(started) == 5


@pytest.mark.asyncio
async def test_max_concurrency_limits_parallelism() -> None:
    in_flight = {"n": 0}
    peak = {"n": 0}

    def make() -> Callable[[], Awaitable[int]]:
        async def factory() -> int:
            in_flight["n"] += 1
            peak["n"] = max(peak["n"], in_flight["n"])
            await asyncio.sleep(0.01)
            in_flight["n"] -= 1
            return 0

        return factory

    factories = [make() for _ in range(10)]
    await gather_in_order(factories, max_concurrency=3)
    assert peak["n"] <= 3


@pytest.mark.asyncio
async def test_first_exception_propagates_and_cancels_pending() -> None:
    async def bad() -> int:
        raise RuntimeError("nope")

    async def slow() -> int:
        await asyncio.sleep(0.5)
        return 1

    factories = [bad, slow, slow]
    with pytest.raises(RuntimeError, match="nope"):
        await gather_in_order(factories, max_concurrency=3)


def test_rejects_invalid_max_concurrency() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        asyncio.run(gather_in_order([_make_factory(0)], max_concurrency=0))

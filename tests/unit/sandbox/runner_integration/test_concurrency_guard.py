"""Tests for ``SandboxConcurrencyGuard`` (P30)."""

from __future__ import annotations

import asyncio

import pytest

from troopai.adk.exceptions.exceptions import SandboxConcurrencyError
from troopai.adk.sandbox.runner_integration.concurrency_guard import (
    SandboxConcurrencyGuard,
)


class TestSingleAcquire:
    @pytest.mark.asyncio
    async def test_acquire_then_release(self) -> None:
        guard = SandboxConcurrencyGuard()
        await guard.acquire()
        guard.release()

    @pytest.mark.asyncio
    async def test_release_when_not_held_is_idempotent(self) -> None:
        guard = SandboxConcurrencyGuard()
        # No raise.
        guard.release()
        # And again.
        guard.release()


class TestConcurrentAcquireRaises:
    @pytest.mark.asyncio
    async def test_second_acquire_raises_immediately(self) -> None:
        guard = SandboxConcurrencyGuard()
        await guard.acquire()
        with pytest.raises(SandboxConcurrencyError, match="concurrently"):
            await guard.acquire()
        guard.release()

    @pytest.mark.asyncio
    async def test_after_release_can_reacquire(self) -> None:
        guard = SandboxConcurrencyGuard()
        await guard.acquire()
        guard.release()
        # Now should succeed.
        await guard.acquire()
        guard.release()


class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_context_manager_acquires_and_releases(self) -> None:
        guard = SandboxConcurrencyGuard()
        async with guard:
            with pytest.raises(SandboxConcurrencyError):
                await guard.acquire()
        # After exit, can reacquire.
        async with guard:
            pass

    @pytest.mark.asyncio
    async def test_concurrent_tasks_one_wins(self) -> None:
        guard = SandboxConcurrencyGuard()
        results: list[str] = []

        async def runner(tag: str) -> None:
            try:
                async with guard:
                    results.append(f"{tag}-acquired")
                    await asyncio.sleep(0.05)
                    results.append(f"{tag}-released")
            except SandboxConcurrencyError:
                results.append(f"{tag}-rejected")

        await asyncio.gather(runner("a"), runner("b"))
        # Exactly one acquired, the other was rejected.
        acquired = [r for r in results if "-acquired" in r]
        rejected = [r for r in results if "-rejected" in r]
        assert len(acquired) == 1
        assert len(rejected) == 1


class TestTOCTOUFix:
    """Regression: locked()-then-acquire() had a TOCTOU window.

    Two coroutines could both pass the locked()==False check before either
    called acquire(). The fix uses asyncio.wait_for(acquire(), timeout=0) to
    make the check+acquire atomic.
    """

    @pytest.mark.asyncio
    async def test_many_concurrent_acquires_only_one_succeeds(self) -> None:
        """Fire N concurrent acquires; only exactly one must succeed."""
        guard = SandboxConcurrencyGuard()
        successes: list[str] = []
        errors: list[str] = []

        async def try_acquire(tag: str) -> None:
            try:
                await guard.acquire()
                successes.append(tag)
            except SandboxConcurrencyError:
                errors.append(tag)

        tasks = [try_acquire(str(i)) for i in range(20)]
        await asyncio.gather(*tasks)

        # Release the one holder so the guard is left clean.
        if len(successes) == 1:
            guard.release()

        assert len(successes) == 1, f"Expected 1 success, got {len(successes)}: {successes}"
        assert len(errors) == 19

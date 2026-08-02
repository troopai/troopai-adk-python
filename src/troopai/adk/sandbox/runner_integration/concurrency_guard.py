"""``SandboxConcurrencyGuard`` — per-agent exclusivity primitive.

A ``SandboxAgent`` carries one guard lazily attached on first
acquisition. The Runner calls ``acquire`` at the top of ``arun`` and
``release`` in the ``finally`` block. A second concurrent ``arun``
on the same agent raises ``SandboxConcurrencyError`` immediately
rather than racing against the first run for the session.

Cross-process usage (multiprocessing swarms) is NOT supported —
``asyncio.Lock`` does not pickle. Frame the constraint loudly so
the failure mode is visible at construction time.
"""

from __future__ import annotations

import asyncio
import threading

from troopai.adk.exceptions.exceptions import SandboxConcurrencyError

__all__ = ["SandboxConcurrencyGuard"]


class SandboxConcurrencyGuard:
    """Single-acquirer guard backed by ``asyncio.Lock``.

    Use ``async with guard:`` for the common path; the context
    manager raises ``SandboxConcurrencyError`` if a second concurrent
    enter races a still-held first.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._held_by_run = False
        # Threading-side lock guards the bool flag for the rare
        # multi-thread asyncio case (rare but possible with
        # ``asyncio.run`` per thread).
        self._flag_lock = threading.Lock()

    async def acquire(self) -> None:
        """Acquire the guard or raise ``SandboxConcurrencyError``.

        Raises immediately if the lock is already held — does NOT
        wait. The caller's ``Runner.arun`` is the authoritative entry
        point; a second concurrent run is a programming error, not
        a queue-up situation.

        The ``locked()`` check and the ``raise`` are not separated by
        any ``await``, so no other coroutine can interleave between
        them within a single event-loop thread.  The ``_flag_lock``
        threading.Lock covers the multi-thread-asyncio edge case for
        the ``_held_by_run`` boolean.
        """
        with self._flag_lock:
            if self._held_by_run:
                raise SandboxConcurrencyError(
                    "SandboxAgent cannot be reused concurrently — "
                    "a previous Runner.arun is still in progress on this agent"
                )
        if self._lock.locked():
            raise SandboxConcurrencyError(
                "SandboxAgent cannot be reused concurrently — the per-agent guard is held by another run"
            )
        await self._lock.acquire()
        with self._flag_lock:
            self._held_by_run = True

    def release(self) -> None:
        """Release the guard. Safe to call when not held (idempotent)."""
        with self._flag_lock:
            if not self._held_by_run:
                return
            self._held_by_run = False
        if self._lock.locked():
            self._lock.release()

    async def __aenter__(self) -> SandboxConcurrencyGuard:
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.release()

"""Bounded-concurrency task gathering with stable result ordering.

Materializing a manifest may involve dozens of independent
async operations (downloading files, cloning repos, hashing
local sources). ``gather_in_order`` fans them out across a
worker pool capped at ``max_concurrency``, returning the
results in **input order** so the caller can correlate them
with the originating factories without juggling indices.

Designed to drop in where ``asyncio.gather`` would over-saturate
the event loop (or a remote API's rate limit).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import cast

logger = logging.getLogger(__name__)

__all__ = ["gather_in_order"]

_MISSING: object = object()


async def gather_in_order[T](
    task_factories: Sequence[Callable[[], Awaitable[T]]],
    *,
    max_concurrency: int | None = None,
) -> list[T]:
    """Run ``task_factories`` with bounded parallelism; preserve input order.

    Args:
        task_factories: Each entry is a zero-arg async callable
            producing one result. The factories are invoked in
            input order, but multiple may run concurrently up to
            ``max_concurrency``.
        max_concurrency: Worker-pool size. ``None`` (default)
            uses ``len(task_factories)`` workers — same as
            ``asyncio.gather``. Pass an integer ``>= 1`` to cap.

    Returns:
        Results in input order — ``result[i]`` belongs to
        ``task_factories[i]``.

    Raises:
        ValueError: ``max_concurrency < 1``.
        BaseException: The first exception any factory raised
            (or ``CancelledError`` if the caller is cancelled).
            In-flight tasks are cancelled in a ``finally`` block
            before the exception propagates so the caller doesn't
            have to drain stragglers.
    """
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    if len(task_factories) == 0:
        return []

    results: list[T | object] = [_MISSING] * len(task_factories)
    worker_count = len(task_factories)
    if max_concurrency is not None:
        worker_count = min(worker_count, max_concurrency)

    # Seeding indices into a Queue makes the "one index per claim"
    # contract structural — a future refactor adding an ``await``
    # to the worker body cannot accidentally let two workers grab
    # the same slot.
    indices: asyncio.Queue[int] = asyncio.Queue()
    for i in range(len(task_factories)):
        indices.put_nowait(i)

    async def _worker() -> None:
        while True:
            try:
                index = indices.get_nowait()
            except asyncio.QueueEmpty:
                return
            results[index] = await task_factories[index]()

    tasks = [asyncio.create_task(_worker()) for _ in range(worker_count)]
    try:
        # ``asyncio.gather`` re-raises the first exception any worker
        # raises (Exception OR BaseException) — same behaviour we want.
        # The ``finally`` below handles fail-fast cancellation of any
        # workers still running when the exception fires.
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        # Drain so cancellations don't leak ``Task was destroyed but
        # is pending!`` warnings; ``return_exceptions=True`` swallows
        # the consequential CancelledError per task.
        await asyncio.gather(*tasks, return_exceptions=True)

    # Every position is filled by ``_worker``: the queue.get_nowait/
    # results[index] = await ... pair is atomic from the worker's
    # perspective, and ``asyncio.gather`` above re-raised before this
    # point if any worker failed before writing its slot.
    return [cast(T, result) for result in results]

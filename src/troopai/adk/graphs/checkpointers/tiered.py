"""``TieredCheckpointer`` — hot/cold composite with read-through fallback and archive.

Wraps two :class:`~troopai.adk.graphs.checkpointer.Checkpointer` backends:

- **Hot tier**: low-latency store (Redis, in-memory) for active runs.
- **Cold tier**: archival store (S3, Postgres, SQLite) for long-term retention.

Write path: all saves go to the hot tier. Read path: hot is consulted first;
on a miss the cold tier is read and the result is re-warmed into hot.
Age-based archival: :meth:`archive` migrates hot entries older than
``archive_after_seconds`` to the cold tier (measured in-process from the
last save or re-warm through this composite).

Hook path: :meth:`register` installs the composite itself as the hook target,
so hook-driven auto-saves (``on_node_end`` / ``on_graph_end``) call the
composite's :meth:`save`, which writes to the hot tier AND records the
timestamp for :meth:`archive`.

Concurrency semantics are inherited from the hot store. The age used by
:meth:`archive` is tracked in-memory by this composite (reset on process
restart), not read from the backends — so :meth:`archive` only considers
threads saved or loaded through this instance since it was created.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, override

from troopai.adk.graphs.checkpointer import (
    Checkpointer,
    GraphCheckpoint,
)
from troopai.adk.graphs.checkpointers.hooks import CheckpointerHooks
from troopai.adk.graphs.hooks import HookRegistry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from troopai.adk.graphs.graph import Graph
    from troopai.adk.graphs.state import GraphState


logger = logging.getLogger(__name__)


class TieredCheckpointer(Checkpointer):
    """Hot+cold composite: writes go to hot; reads fall through hot→cold and
    re-warm hot; :meth:`archive` migrates aged entries hot→cold.

    Hook-driven auto-saves (via :meth:`register`) go through the composite's
    own :meth:`save`, so both the hot tier and the archive-eligibility table
    (``_saved_at``) are updated on every hook-triggered write. An entry in
    that table persists until the thread is removed via :meth:`delete` or
    migrated by a successful :meth:`archive` — completing a run does not by
    itself drop it. Callers handling many distinct, short-lived ``thread_id``
    s should call :meth:`delete` on completion or run :meth:`archive`
    periodically so the table does not grow with the total number of runs.

    Concurrency semantics are inherited from the hot store. The age used by
    :meth:`archive` is tracked in-memory by this composite (reset on process
    restart), not read from the backends — so :meth:`archive` only considers
    threads saved or loaded through this instance since it was created.
    """

    def __init__(
        self,
        *,
        hot: Checkpointer,
        cold: Checkpointer,
        archive_after_seconds: float,
    ) -> None:
        """Initialise the composite with the two storage tiers.

        Args:
            hot: Low-latency checkpointer used for all writes and
                primary reads (e.g.
                :class:`~troopai.adk.graphs.checkpointers.redis.RedisCheckpointer`,
                :class:`~troopai.adk.graphs.checkpointers.in_memory.InMemoryCheckpointer`).
            cold: Archival checkpointer consulted on hot misses and used
                as the archive destination (e.g.
                :class:`~troopai.adk.graphs.checkpointers.s3.S3Checkpointer`,
                :class:`~troopai.adk.graphs.checkpointers.postgres.PostgresCheckpointer`).
            archive_after_seconds: Minimum age in seconds before a hot
                entry is eligible for :meth:`archive`. The age is
                measured from the last save or re-warm recorded in-memory
                by this composite.

        Raises:
            ValueError: When ``archive_after_seconds`` is negative.
        """
        if archive_after_seconds < 0.0:
            raise ValueError(f"archive_after_seconds must be >= 0.0, got {archive_after_seconds!r}.")
        self._hot = hot
        self._cold = cold
        self._archive_after = archive_after_seconds
        self._saved_at: dict[str, float] = {}
        # Per-thread guard serialising hot-tier writes against migration:
        # without it a save() landing between _migrate_one's hot.load and
        # hot.delete is silently discarded by the delete. Each lock is
        # reference-counted by the number of in-flight callers (holders
        # plus pending waiters); the entry is dropped only when that count
        # reaches zero, so growth is bounded by the number of concurrently
        # in-flight threads and no waiter is ever left holding an orphaned
        # lock that a fresh caller could bypass.
        self._thread_locks: dict[str, asyncio.Lock] = {}
        self._lock_refs: dict[str, int] = {}
        logger.debug(
            "TieredCheckpointer initialised with archive_after_seconds=%.3f.",
            archive_after_seconds,
        )

    @asynccontextmanager
    async def _lock_for(self, thread_id: str) -> AsyncIterator[None]:
        """Acquire the per-thread write guard, creating it on first use.

        The lock is reference-counted: the refcount is bumped *before*
        awaiting acquisition so a pending waiter keeps the entry alive,
        and decremented on exit. The map entry is dropped only when the
        count returns to zero. This prevents the orphaned-lock race where
        dropping a just-released lock with a scheduled-but-not-yet-resumed
        waiter would let that waiter and a fresh caller serialise on two
        different lock objects for the same ``thread_id``. The guard's job
        is solely the migration load→delete window; ordering between plain
        concurrent saves stays inherited from the hot store.
        """
        lock = self._thread_locks.setdefault(thread_id, asyncio.Lock())
        self._lock_refs[thread_id] = self._lock_refs.get(thread_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            count = self._lock_refs[thread_id] - 1
            if count == 0:
                del self._lock_refs[thread_id]
                del self._thread_locks[thread_id]
            else:
                self._lock_refs[thread_id] = count

    # -- HookProvider surface -------------------------------------------

    @override
    def register(self, registry: HookRegistry) -> None:
        """Route hook-driven auto-saves through this composite (so they hit the
        hot tier AND populate the archive-eligibility table)."""
        registry.add(CheckpointerHooks(self))
        logger.debug("TieredCheckpointer: registered composite as hook target.")

    # -- CRUD surface ---------------------------------------------------

    @override
    async def save(self, checkpoint: GraphCheckpoint) -> None:
        """Persist ``checkpoint`` to the hot tier and record its save time.

        Serialised per thread against :meth:`archive` migration so the
        write cannot land inside a migration's load→delete window and be
        discarded by the trailing hot delete.
        """
        async with self._lock_for(checkpoint.thread_id):
            await self._hot.save(checkpoint)
            self._saved_at[checkpoint.thread_id] = time.time()
        logger.debug(
            "TieredCheckpointer.save: thread_id=%s superstep=%s → hot.",
            checkpoint.thread_id,
            checkpoint.superstep,
        )

    @override
    async def load(
        self,
        thread_id: str,
        graph: Graph[Any],
    ) -> GraphState[Any] | None:
        """Rehydrate the latest checkpoint for ``thread_id``.

        Reads the hot tier first. On a miss, falls through to the cold
        tier. When the cold tier has a match, the checkpoint is re-warmed
        into the hot tier so subsequent reads avoid the cold path. If the
        re-warm fails it is logged as a warning and the cold state is
        returned uncached — the next load will attempt cold again.

        Args:
            thread_id: The logical run key.
            graph: The :class:`Graph` the checkpoint belongs to. Passed
                through to both backends for ``graph_id`` validation.

        Returns:
            A rehydrated :class:`GraphState`, or ``None`` when neither
            tier has a checkpoint for ``thread_id``.
        """
        state = await self._hot.load(thread_id, graph)
        if state is not None:
            logger.debug("TieredCheckpointer.load: hot hit for thread_id=%s.", thread_id)
            return state

        logger.debug("TieredCheckpointer.load: hot miss for thread_id=%s; consulting cold.", thread_id)
        state = await self._cold.load(thread_id, graph)
        if state is None:
            return None

        # Re-warm the hot tier from the cold copy (best-effort; failure is
        # non-fatal). Takes the per-thread guard: it is a hot-tier write
        # like save(), with the same migration-window hazard.
        try:
            async with self._lock_for(thread_id):
                await self._hot.save(
                    GraphCheckpoint(
                        thread_id=thread_id,
                        graph_id=graph.id,
                        state=state.to_dict(),
                        superstep=state.superstep,
                    )
                )
                self._saved_at[thread_id] = time.time()
        except Exception:
            logger.warning(
                "TieredCheckpointer.load: re-warm of hot tier failed for thread_id=%s; returning cold state uncached.",
                thread_id,
                exc_info=True,
            )

        logger.debug("TieredCheckpointer.load: cold hit for thread_id=%s; re-warmed hot.", thread_id)
        return state

    @override
    async def list_checkpoints(self) -> list[str]:
        """Return the union of thread ids across both tiers, sorted."""
        hot = await self._hot.list_checkpoints()
        cold = await self._cold.list_checkpoints()
        return sorted(set(hot) | set(cold))

    @override
    async def delete(self, thread_id: str) -> None:
        """Delete the checkpoint for ``thread_id`` from both tiers.

        Attempts deletion from both hot and cold even if one tier raises.
        ``_saved_at`` is always cleared. Re-raises the first error if any
        tier failed.
        """
        errors: list[Exception] = []
        async with self._lock_for(thread_id):
            for tier in (self._hot, self._cold):
                try:
                    await tier.delete(thread_id)
                except Exception as exc:  # accumulate, re-raise after both attempted
                    errors.append(exc)
            self._saved_at.pop(thread_id, None)
        logger.debug("TieredCheckpointer.delete: thread_id=%s removed from both tiers.", thread_id)
        if len(errors) > 0:
            raise errors[0]

    # -- Archival -------------------------------------------------------

    async def archive(self, graph: Graph[Any]) -> int:
        """Move hot entries older than ``archive_after_seconds`` to cold.

        Only threads whose save timestamp is recorded by this composite are
        considered. One failure to migrate a single thread is logged and
        skipped — it stays in hot and is retried on the next call. When
        the hot entry is gone before archival (e.g. evicted), a warning is
        logged and the tracking entry is dropped.

        Args:
            graph: The :class:`Graph` whose checkpoints should be archived.
                Used to rehydrate the hot state before writing to cold.

        Returns:
            The number of thread ids moved from hot to cold.
        """
        cutoff = time.time() - self._archive_after
        moved = 0
        for thread_id, saved in list(self._saved_at.items()):
            if saved > cutoff:
                continue
            moved += await self._migrate_one(thread_id, graph)
        logger.info("TieredCheckpointer.archive: moved %d entries hot→cold.", moved)
        return moved

    async def _migrate_one(self, thread_id: str, graph: Graph[Any]) -> int:
        """Attempt to move one hot entry to cold. Returns 1 on success, 0 on skip/error.

        Holds the per-thread guard across load→cold-save→hot-delete, so a
        concurrent :meth:`save` cannot land inside the window and be
        discarded by the trailing delete — the save waits and re-creates
        the hot entry after the migration completes.
        """
        try:
            async with self._lock_for(thread_id):
                state = await self._hot.load(thread_id, graph)
                if state is None:
                    logger.warning(
                        "TieredCheckpointer.archive: hot entry gone for thread_id=%s before archival; "
                        "dropping tracking.",
                        thread_id,
                    )
                    self._saved_at.pop(thread_id, None)
                    return 0
                await self._cold.save(
                    GraphCheckpoint(
                        thread_id=thread_id,
                        graph_id=graph.id,
                        state=state.to_dict(),
                        superstep=state.superstep,
                    )
                )
                await self._hot.delete(thread_id)
                self._saved_at.pop(thread_id, None)
            logger.debug("TieredCheckpointer.archive: moved thread_id=%s hot→cold.", thread_id)
            return 1
        except Exception:
            logger.exception(
                "TieredCheckpointer.archive: failed to migrate thread_id=%s; leaving it in hot.",
                thread_id,
            )
            return 0


__all__ = ["TieredCheckpointer"]

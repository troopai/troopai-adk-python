"""``TieredSwarmCheckpointer`` — hot/cold composite for swarm runs.

Wraps two :class:`~troopai.adk.swarms.checkpointer.SwarmCheckpointer` backends
in a hot/cold tiering pattern:

- **Hot tier**: low-latency store (Redis, in-memory) for active runs.
- **Cold tier**: archival store (S3, Postgres) for long-term retention.

Write path: all saves go to the hot tier. Read path: hot is consulted first;
on a miss the cold tier is read and the result is re-warmed into hot.
Age-based archival: :meth:`archive` migrates hot entries older than
``archive_after_seconds`` to the cold tier (measured in-process from the
last save or re-warm through this composite).

Hook path: :meth:`register` installs
:class:`~troopai.adk.swarms.checkpointers.hooks.SwarmCheckpointerHooks` with
the composite as the owner, so hook-driven auto-saves
(``on_swarm_turn_end`` / ``on_swarm_turn_interrupt``) call the composite's
:meth:`save`, which writes to the hot tier AND records the timestamp for
:meth:`archive`.

Concurrency semantics are inherited from the hot store. The age used by
:meth:`archive` is tracked in-memory by this composite (reset on process
restart), not read from the backends — so :meth:`archive` only considers
threads saved or loaded through this instance since it was created.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from troopai.adk.swarms.checkpointer import SwarmCheckpoint

if TYPE_CHECKING:
    from troopai.adk.swarms.checkpointer import SwarmCheckpointer, SwarmHookRegistry
    from troopai.adk.swarms.swarm import Swarm


logger = logging.getLogger(__name__)


class TieredSwarmCheckpointer:
    """Hot+cold composite: writes go to hot; reads fall through hot→cold and
    re-warm hot; :meth:`archive` migrates aged entries hot→cold.

    Hook-driven auto-saves (via :meth:`register`) go through the composite's
    own :meth:`save`, so both the hot tier and the archive-eligibility table
    (``_saved_at``) are updated on every hook-triggered write.

    Concurrency semantics are inherited from the hot store. The age used by
    :meth:`archive` is tracked in-memory by this composite (reset on process
    restart), not read from the backends — so :meth:`archive` only considers
    threads saved or loaded through this instance since it was created.

    """

    def __init__(
        self,
        *,
        hot: SwarmCheckpointer,
        cold: SwarmCheckpointer,
        archive_after_seconds: float,
        thread_id: str = "default",
    ) -> None:
        """Initialise the composite with two backing checkpointers.

        Args:
            hot: Low-latency checkpointer used for all writes and
                primary reads.
            cold: Archival checkpointer consulted on hot misses and as
                the archive destination.
            archive_after_seconds: Minimum age (seconds) before a hot
                entry is eligible for :meth:`archive`. The age is
                measured from the last save or re-warm recorded by this
                composite.
            thread_id: Identifier used by :meth:`register`'s auto-save
                hook. Defaults to ``"default"`` when the caller does
                not supply an explicit id.

        Raises:
            ValueError: When ``archive_after_seconds`` is negative.
        """
        if archive_after_seconds < 0.0:
            raise ValueError(f"archive_after_seconds must be >= 0.0, got {archive_after_seconds!r}.")
        self._hot = hot
        self._cold = cold
        self._archive_after = archive_after_seconds
        self._thread_id = thread_id
        self._saved_at: dict[str, float] = {}
        logger.debug(
            "TieredSwarmCheckpointer initialised with archive_after_seconds=%.3f.",
            archive_after_seconds,
        )

    # -- Hook surface ---------------------------------------------------

    def register(self, registry: SwarmHookRegistry) -> None:
        """Route hook-driven auto-saves through this composite (so they hit the
        hot tier AND populate the archive-eligibility table)."""
        from troopai.adk.swarms.checkpointers.hooks import SwarmCheckpointerHooks

        registry.add(SwarmCheckpointerHooks(self, self._thread_id))
        logger.debug("TieredSwarmCheckpointer: registered composite as hook target.")

    # -- CRUD surface ---------------------------------------------------

    async def save(self, checkpoint: SwarmCheckpoint) -> None:
        """Persist ``checkpoint`` to the hot tier and record its save time."""
        await self._hot.save(checkpoint)
        self._saved_at[checkpoint.thread_id] = time.time()
        logger.debug(
            "TieredSwarmCheckpointer.save: thread_id=%s turn=%d → hot.",
            checkpoint.thread_id,
            checkpoint.turn,
        )

    async def load(
        self,
        thread_id: str,
        swarm: Swarm[Any],
    ) -> SwarmCheckpoint | None:
        """Return the latest checkpoint for ``thread_id`` or ``None``.

        Reads the hot tier first. On a miss, falls through to the cold tier.
        When the cold tier has a match, the raw :class:`SwarmCheckpoint` is
        re-warmed into hot so subsequent reads avoid the cold path. If the
        re-warm fails it is logged as a warning and the cold state is returned
        uncached — the next load will attempt cold again.

        Args:
            thread_id: The logical run key.
            swarm: The :class:`Swarm` the checkpoint belongs to. Passed
                through to both backends for structural parity.

        Returns:
            The latest :class:`SwarmCheckpoint`, or ``None`` when neither
            tier holds a checkpoint for ``thread_id``.
        """
        checkpoint = await self._hot.load(thread_id, swarm)
        if checkpoint is not None:
            logger.debug("TieredSwarmCheckpointer.load: hot hit for thread_id=%s.", thread_id)
            return checkpoint

        logger.debug("TieredSwarmCheckpointer.load: hot miss for thread_id=%s; consulting cold.", thread_id)
        checkpoint = await self._cold.load(thread_id, swarm)
        if checkpoint is None:
            return None

        # Re-warm the hot tier from the cold copy (best-effort; failure is non-fatal).
        try:
            await self._hot.save(checkpoint)
            self._saved_at[thread_id] = time.time()
        except Exception:
            logger.warning(
                "TieredSwarmCheckpointer.load: re-warm of hot tier failed for thread_id=%s; returning cold state uncached.",
                thread_id,
                exc_info=True,
            )

        logger.debug("TieredSwarmCheckpointer.load: cold hit for thread_id=%s; re-warmed hot.", thread_id)
        return checkpoint

    async def list_checkpoints(self) -> list[str]:
        """Return the union of thread ids across both tiers, sorted."""
        hot = await self._hot.list_checkpoints()
        cold = await self._cold.list_checkpoints()
        return sorted(set(hot) | set(cold))

    async def delete(self, thread_id: str) -> None:
        """Delete the checkpoint for ``thread_id`` from both tiers.

        Attempts deletion from both hot and cold even if one tier raises.
        ``_saved_at`` is always cleared. Re-raises the first error if any
        tier failed.
        """
        errors: list[Exception] = []
        for tier in (self._hot, self._cold):
            try:
                await tier.delete(thread_id)
            except Exception as exc:  # accumulate, re-raise after both attempted
                errors.append(exc)
        self._saved_at.pop(thread_id, None)
        if len(errors) == 0:
            logger.debug("TieredSwarmCheckpointer.delete: thread_id=%s removed from both tiers.", thread_id)
        else:
            logger.warning(
                "TieredSwarmCheckpointer.delete: thread_id=%s — one or more tiers failed: %s",
                thread_id,
                errors,
            )
            if len(errors) > 1:
                raise errors[0] from errors[1]
            raise errors[0]

    # -- Archival -------------------------------------------------------

    async def archive(self, swarm: Swarm[Any]) -> int:
        """Move hot entries older than ``archive_after_seconds`` to cold.

        Only threads whose save timestamp is recorded by this composite are
        considered. One failure to migrate a single thread is logged and
        skipped — it stays in hot and is retried on the next call. When
        the hot entry is gone before archival (e.g. evicted), a warning is
        logged and the tracking entry is dropped.

        The raw :class:`SwarmCheckpoint` is moved directly to cold with no
        rehydration — the swarm payload is already a plain dict and
        round-trips faithfully without deserialisation.

        Args:
            swarm: The :class:`Swarm` used to consult the hot backend on
                load (parity with the :class:`SwarmCheckpointer` Protocol).

        Returns:
            The number of thread ids moved from hot to cold.
        """
        cutoff = time.time() - self._archive_after
        moved = 0
        for thread_id, saved in list(self._saved_at.items()):
            if saved > cutoff:
                continue
            moved += await self._migrate_one(thread_id, swarm)
        logger.info("TieredSwarmCheckpointer.archive: moved %d entries hot→cold.", moved)
        return moved

    async def _migrate_one(self, thread_id: str, swarm: Swarm[Any]) -> int:
        """Attempt to move one hot entry to cold. Returns 1 on success, 0 on skip/error."""
        try:
            checkpoint = await self._hot.load(thread_id, swarm)
            if checkpoint is None:
                logger.warning(
                    "TieredSwarmCheckpointer.archive: hot entry gone for thread_id=%s before archival; dropping tracking.",
                    thread_id,
                )
                self._saved_at.pop(thread_id, None)
                return 0
            await self._cold.save(checkpoint)
            await self._hot.delete(thread_id)
            self._saved_at.pop(thread_id, None)
            logger.debug("TieredSwarmCheckpointer.archive: moved thread_id=%s hot→cold.", thread_id)
            return 1
        except Exception:
            logger.exception(
                "TieredSwarmCheckpointer.archive: failed to migrate thread_id=%s; leaving it in hot.",
                thread_id,
            )
            return 0


__all__ = ["TieredSwarmCheckpointer"]

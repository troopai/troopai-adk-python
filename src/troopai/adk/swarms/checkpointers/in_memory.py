"""In-memory :class:`SwarmCheckpointer` — reference implementation.

Suitable for single-process workflows + tests. Cross-process resume
requires a durable backend.

Auto-save: on registration via :meth:`InMemorySwarmCheckpointer.register`,
attaches a :class:`~troopai.adk.swarms.checkpointers.hooks.SwarmCheckpointerHooks`
instance to the supplied registry. The hooks override both
``on_swarm_turn_end`` and ``on_swarm_turn_interrupt`` to call
:meth:`save` with the current :class:`SwarmState.to_dict()` snapshot.
The interrupt override makes parked HITL state durable: the swarm loop
returns before ``on_swarm_turn_end`` fires when a turn suspends, so
without the interrupt hook the parked ``pending_interrupts`` and
``nested_agent_snapshots`` would never reach the checkpoint store.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from troopai.adk.swarms.checkpointer import SwarmCheckpoint

if TYPE_CHECKING:
    from troopai.adk.swarms.checkpointer import SwarmHookRegistry
    from troopai.adk.swarms.swarm import Swarm

logger = logging.getLogger(__name__)


class InMemorySwarmCheckpointer:
    """Stores :class:`SwarmCheckpoint` instances in a process-local dict.

    Keyed by ``thread_id``. Each save overwrites any prior checkpoint
    for the same thread; load returns the latest.
    """

    def __init__(self, thread_id: str = "default") -> None:
        """Initialise the in-memory store.

        Args:
            thread_id: Identifier used by :meth:`register`'s auto-save
                hook. Defaults to ``"default"`` when the caller does not
                supply an explicit id.
        """
        self._store: dict[str, SwarmCheckpoint] = {}
        self._thread_id = thread_id

    async def save(self, checkpoint: SwarmCheckpoint) -> None:
        """Persist ``checkpoint`` under its ``thread_id``."""
        self._store[checkpoint.thread_id] = checkpoint
        logger.debug(
            "InMemorySwarmCheckpointer: saved thread_id=%s turn=%d",
            checkpoint.thread_id,
            checkpoint.turn,
        )

    async def load(
        self,
        thread_id: str,
        swarm: Swarm[Any],
    ) -> SwarmCheckpoint | None:
        """Return the latest checkpoint for ``thread_id`` or ``None``.

        ``swarm`` is accepted for protocol parity with the graphs
        :class:`Checkpointer.load` shape. Cross-validation of the
        persisted state against the live ``swarm`` is intentionally
        deferred — member-name resolution in :meth:`SwarmState.from_dict`
        provides the de-facto integrity check at rehydration time.
        """
        del swarm
        return self._store.get(thread_id)

    async def list_checkpoints(self) -> list[str]:
        """Return all stored thread_ids, sorted lexicographically."""
        return sorted(self._store.keys())

    async def delete(self, thread_id: str) -> None:
        """Remove the checkpoint for ``thread_id``; no-op if absent."""
        removed = self._store.pop(thread_id, None)
        if removed is not None:
            logger.debug("InMemorySwarmCheckpointer: deleted thread_id=%s", thread_id)
        else:
            logger.debug("InMemorySwarmCheckpointer: delete no-op, unknown thread_id=%s", thread_id)

    def register(self, registry: SwarmHookRegistry) -> None:
        """Subscribe a :class:`SwarmCheckpointerHooks` to ``registry``."""
        from troopai.adk.swarms.checkpointers.hooks import SwarmCheckpointerHooks

        registry.add(SwarmCheckpointerHooks(self, self._thread_id))


__all__ = ["InMemorySwarmCheckpointer"]

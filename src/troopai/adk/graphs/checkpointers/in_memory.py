"""``InMemoryCheckpointer`` — dict-backed checkpointer for tests and demos.

Non-persistent; every checkpoint lives in a process-local dict keyed
by ``thread_id``. Useful for:

- Unit and integration tests (no I/O, no cleanup).
- Notebooks (cheap, zero-setup).
- Single-process demos where a crash recovery story isn't needed.

For durable, multi-process or crash-recoverable runs use
:class:`~troopai.adk.graphs.checkpointers.sqlite.SQLiteCheckpointer`.

Design mirrors :class:`~troopai.adk.session.sqlite_session.SQLiteSession`
in structure (``save`` / ``load`` / ``list`` / ``delete``) so users who
graduate to the SQLite implementation swap one import and nothing
else.

The checkpointer subscribes to :meth:`GraphHooks.on_node_end` and
:meth:`GraphHooks.on_graph_end` via the hook-provider pattern (through
:class:`~troopai.adk.graphs.checkpointers.hooks.CheckpointerHooks`).
Every node completion persists the current :class:`GraphState`. The
graph loop itself never calls :meth:`save` — it just fires hooks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, override

from troopai.adk.graphs.checkpointer import (
    Checkpointer,
    GraphCheckpoint,
)
from troopai.adk.graphs.checkpointers.hooks import CheckpointerHooks
from troopai.adk.graphs.hooks import HookRegistry

if TYPE_CHECKING:
    from troopai.adk.graphs.graph import Graph
    from troopai.adk.graphs.state import GraphState


logger = logging.getLogger(__name__)


class InMemoryCheckpointer(Checkpointer):
    """In-process, dict-backed :class:`Checkpointer` implementation.

    Every checkpoint lives in :attr:`_store` keyed by ``thread_id``.
    Thread-safe for async callers inside one event loop because all
    ops are effectively synchronous; multi-thread use requires the
    caller to add its own lock.

    Attributes:
        _store: Per-``thread_id`` latest :class:`GraphCheckpoint`.
            Only one checkpoint is retained per thread — time-travel
            (replay from any superstep) is not supported and would
            require a list-per-thread shape.
    """

    def __init__(self) -> None:
        self._store: dict[str, GraphCheckpoint] = {}
        logger.debug("InMemoryCheckpointer initialised.")

    # -- HookProvider surface ---------------------------------------

    @override
    def register(self, registry: HookRegistry) -> None:
        """Subscribe to ``on_node_end`` / ``on_graph_end``."""
        registry.add(CheckpointerHooks(self))
        logger.debug("InMemoryCheckpointer registered on HookRegistry.")

    # -- CRUD surface -----------------------------------------------

    @override
    async def save(self, checkpoint: GraphCheckpoint) -> None:
        """Persist ``checkpoint``. Later saves overwrite earlier ones."""
        self._store[checkpoint.thread_id] = checkpoint
        logger.debug(
            "InMemoryCheckpointer.save: thread_id=%s superstep=%s",
            checkpoint.thread_id,
            checkpoint.superstep,
        )

    @override
    async def load(
        self,
        thread_id: str,
        graph: Graph[Any],
    ) -> GraphState[Any] | None:
        """Return the rehydrated :class:`GraphState` for ``thread_id``.

        Returns ``None`` when no checkpoint exists. Raises ``ValueError``
        when the checkpoint's ``graph_id`` doesn't match ``graph.id``.
        """
        from troopai.adk.graphs.state import GraphState

        checkpoint = self._store.get(thread_id)
        if checkpoint is None:
            logger.debug(
                "InMemoryCheckpointer.load: no checkpoint for thread_id=%s",
                thread_id,
            )
            return None
        if checkpoint.graph_id != graph.id:
            raise ValueError(
                f"Checkpoint graph_id={checkpoint.graph_id!r} does not match "
                f"supplied graph.id={graph.id!r}. Refusing to load."
            )
        return GraphState.from_dict(checkpoint.state, graph)

    @override
    async def list_checkpoints(self) -> list[str]:
        """Return a sorted list of thread ids currently in the store."""
        return sorted(self._store.keys())

    @override
    async def delete(self, thread_id: str) -> None:
        """Remove the checkpoint for ``thread_id`` (no-op if absent)."""
        self._store.pop(thread_id, None)
        logger.debug(
            "InMemoryCheckpointer.delete: thread_id=%s",
            thread_id,
        )


__all__ = ["InMemoryCheckpointer"]

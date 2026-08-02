"""Checkpointer — pluggable persistence for graph runs.

The Strands insight this module adopts: a checkpointer is NOT a
function the driver calls at fixed boundaries. It is a
:class:`HookProvider` that subscribes to the graph lifecycle hooks it
cares about. The graph loop fires hooks; the checkpointer listens.
The loop contains ZERO persistence code — swap the checkpointer and
persistence behaviour changes without touching the driver.

Concurrency contract: network hot-store backends (Postgres, Redis) MUST
detect concurrent modification of a single ``thread_id`` and raise
:class:`~troopai.adk.exceptions.CheckpointConflictError` on the losing
writer. Single-process in-memory backends and archival, last-write-wins
object stores (S3) need not implement this check.

Tolerant-loader contract: state payloads carry no version field. On
load, unknown keys must be ignored and absent keys must take their
defaults. Persisted formats evolve via field additions, never via a
version discriminator.

Concretely:

- :class:`Checkpointer` Protocol extends
  :class:`~troopai.adk.graphs.hooks.HookProvider`. Implementations
  register themselves on :meth:`GraphHooks.on_node_end` (persist
  after each node) and :meth:`GraphHooks.on_graph_end` (finalise).
- :class:`GraphCheckpoint` is the serialisable snapshot. It wraps a
  :class:`GraphState` plus metadata (thread id, graph id, timestamp,
  superstep). ``pending_sends`` carries unexecuted fan-out packets;
  ``versions_seen`` lives on the state.

The shipped implementations are
:class:`~troopai.adk.graphs.checkpointers.in_memory.InMemoryCheckpointer`
— a dict-backed store adequate for tests, notebooks, and single-process
demos — and
:class:`~troopai.adk.graphs.checkpointers.sqlite.SQLiteCheckpointer`
— a durable ``aiosqlite``-backed store for multi-process or
crash-recoverable runs.

Why this matters beyond "it's pluggable". The alternative — the
driver calling ``checkpointer.save()`` inline — bakes persistence
semantics into the loop:

- You can't attach TWO checkpointers (e.g. a local SQLite for debug
  plus a Redis one for shared state). The loop is coded for exactly
  one.
- You can't easily compose checkpointing with other observability
  (a metrics exporter that ALSO wants ``on_node_end``) without
  interleaving concerns.
- Testing the loop requires a checkpointer stub; testing a
  checkpointer requires a loop stub.

The hook-provider pattern solves all three: the registry fans out to
any number of providers, the loop doesn't know which are
"checkpointers" and which are "metrics", and each can be unit-tested
in isolation by firing synthetic hooks at a ``HookRegistry``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from troopai.adk.graphs.graph import Graph
    from troopai.adk.graphs.hooks import HookRegistry
    from troopai.adk.graphs.state import GraphState


logger = logging.getLogger(__name__)


@dataclass
class GraphCheckpoint:
    """Serialisable snapshot of a graph run at one superstep boundary.

    Produced by :meth:`Checkpointer.save` and consumed by
    :meth:`Checkpointer.load` to resume an interrupted or crashed
    run. The payload is the :class:`GraphState` serialised through
    :meth:`GraphState.to_dict`; the outer envelope carries metadata
    needed to reconstruct a run independently of the in-memory graph.

    Attributes:
        thread_id: Caller-supplied identifier for the logical run.
            The key under which this checkpoint is stored.
        graph_id: Id of the :class:`Graph` this checkpoint belongs to.
            Mismatch between the stored id and the graph supplied at
            :meth:`Checkpointer.load` time raises a ``ValueError``.
        state: Serialised :class:`GraphState.to_dict` payload. Rehydrate
            via :meth:`GraphState.from_dict`.
        created_at: Unix timestamp (seconds) of when the checkpoint
            was produced. Useful for TTL eviction and debugging.
        superstep: Superstep number at which this checkpoint was
            taken. Mirrors ``state.superstep``; hoisted to the top
            level for cheap listing without deserialising the full
            state.
        pending_sends: Dynamic-fan-out packets. Unexecuted ``Send``
            packets that MUST survive a crash so the engine can
            replay them on resume. Always empty until dynamic
            fan-out is implemented.
    """

    thread_id: str
    """Checkpoint key."""

    graph_id: str
    """Owning graph id."""

    state: dict[str, Any]
    """Serialised :meth:`GraphState.to_dict` payload."""

    created_at: float = field(default_factory=time.time)
    """Unix timestamp when the checkpoint was produced."""

    superstep: int = 0
    """Superstep at which the checkpoint was taken."""

    pending_sends: list[Any] = field(default_factory=list)
    """Dynamic-fan-out packets. Always empty until dynamic fan-out is implemented."""


@runtime_checkable
class Checkpointer(Protocol):
    """Protocol for pluggable graph-run persistence.

    Extends :class:`HookProvider` so checkpointers attach themselves
    to a :class:`HookRegistry` via :meth:`register`. The Protocol
    contract is:

    1. :meth:`register` attaches the checkpointer's hook callbacks
       (typically ``on_node_end`` and ``on_graph_end``).
    2. :meth:`save` / :meth:`load` / :meth:`list_checkpoints` /
       :meth:`delete` provide the explicit CRUD surface for callers
       that want to inspect or manage checkpoints directly.

    The explicit CRUD methods exist alongside the hook-registered
    callbacks because some use cases (resumption, time-travel debug,
    admin tools) need to read / mutate checkpoints outside a running
    graph. The hooks handle the happy path during a live run;
    CRUD handles everything else.

    An implementation SHOULD:

    - Persist atomically — a partial write on crash is worse than
      no write.
    - Accept a missing ``thread_id`` on :meth:`load` and return
      ``None`` (first-time run, not an error).

    ``@runtime_checkable`` means user code can pass raw objects to
    :meth:`Runner.arun_graph(..., hooks=[checkpointer])` without
    inheriting from a concrete base.
    """

    def register(self, registry: HookRegistry) -> None:
        """Attach this checkpointer's hook callbacks to ``registry``.

        Called once by :func:`run_graph_loop` when the checkpointer is
        passed as a hook provider. Implementations typically construct
        an internal ``GraphHooks`` subclass and call ``registry.add``.
        """
        ...

    async def save(self, checkpoint: GraphCheckpoint) -> None:
        """Persist a :class:`GraphCheckpoint`.

        Implementations SHOULD be idempotent on the same
        ``(thread_id, superstep)`` pair. ``thread_id`` is the upsert
        key; later saves at higher supersteps overwrite earlier ones.
        """
        ...

    async def load(
        self,
        thread_id: str,
        graph: Graph[Any],
    ) -> GraphState[Any] | None:
        """Rehydrate the latest checkpoint for ``thread_id``.

        Args:
            thread_id: The logical run key.
            graph: The :class:`Graph` the checkpoint belongs to. Needed
                to deserialise the :class:`GraphState`; mismatch
                between ``graph.id`` and the stored ``graph_id`` MUST
                raise ``ValueError``.

        Returns:
            A rehydrated :class:`GraphState`, or ``None`` when no
            checkpoint exists for ``thread_id``.
        """
        ...

    async def list_checkpoints(self) -> list[str]:
        """List all known ``thread_id``\\ s this checkpointer holds."""
        ...

    async def delete(self, thread_id: str) -> None:
        """Delete the checkpoint for ``thread_id``.

        Missing keys MUST be a no-op (not an error).
        """
        ...


__all__ = [
    "Checkpointer",
    "GraphCheckpoint",
]

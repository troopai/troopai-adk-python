"""Swarm-run persistence types + Checkpointer protocol.

Mirrors the graphs :mod:`troopai.adk.graphs.checkpointer` module shape:
a frozen :class:`SwarmCheckpoint` payload + a runtime-checkable
:class:`SwarmCheckpointer` Protocol that exposes ``register``, ``save``,
``load``, ``list_checkpoints``, and ``delete``.

The graphs and swarms checkpointer protocols are deliberately
separate-but-parallel: same shape, different payload type. A generic
unification can come later as a refactor without breaking either
subsystem.

Concurrency contract: network hot-store backends (Postgres, Redis) MUST
detect concurrent modification of a single ``thread_id`` and raise
:class:`~troopai.adk.exceptions.CheckpointConflictError` on the losing
writer. Single-process in-memory backends and archival, last-write-wins
object stores (S3) need not implement this check.

Tolerant-loader contract: state payloads carry no version field. On
load, unknown keys must be ignored and absent keys must take their
defaults. Persisted formats evolve via field additions, never via a
version discriminator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from troopai.adk.swarms.swarm import Swarm


@runtime_checkable
class SwarmHookRegistry(Protocol):
    """Structural protocol for the swarm hook registry.

    Any object exposing ``add(hooks)`` satisfies it — notably
    ``troopai.adk.swarms.hooks.HookRegistry``, which the swarm loop
    builds and through which a checkpointer's
    :class:`~troopai.adk.swarms.checkpointers.hooks.SwarmCheckpointerHooks`
    are registered for auto-save.
    """

    def add(self, hooks: Any) -> None:
        """Attach a :class:`SwarmHooks` instance to the registry.

        Args:
            hooks: The :class:`~troopai.adk.swarms.hooks.SwarmHooks`
                instance to attach.
        """
        ...


@dataclass(frozen=True)
class SwarmCheckpoint:
    """A persisted snapshot of a swarm run at a turn boundary.

    Attributes:
        thread_id: Logical run identifier; the key under which the
            checkpoint is stored.
        state: Output of :meth:`SwarmState.to_dict`. Loader uses
            :meth:`SwarmState.from_dict` with the caller-supplied
            :class:`Swarm` to rehydrate. Member-name resolution in
            :meth:`SwarmState.from_dict` is the de-facto integrity
            check against a swarm mismatch.
        turn: The 1-indexed turn count at the time of save.
    """

    thread_id: str
    """Logical run identifier; the key under which the checkpoint is stored."""

    state: dict[str, Any]
    """Output of :meth:`SwarmState.to_dict`. JSON-safe."""

    turn: int
    """1-indexed turn count at the time of save."""


@runtime_checkable
class SwarmCheckpointer(Protocol):
    """Persistence backend for swarm runs.

    Exposes a ``register`` hook so a caller-supplied registry can wire
    automatic save calls on ``on_swarm_turn_end`` and
    ``on_swarm_turn_interrupt``.

    Implementations:
        - :class:`InMemorySwarmCheckpointer` — reference impl.
        - ``PostgresSwarmCheckpointer`` / ``RedisSwarmCheckpointer`` /
          ``S3SwarmCheckpointer`` — network backends for cross-process resume.
        - ``TieredSwarmCheckpointer`` — hot/cold composite.
    """

    async def save(self, checkpoint: SwarmCheckpoint) -> None:
        """Persist ``checkpoint`` under its ``thread_id``.

        Args:
            checkpoint: The snapshot to persist.
        """
        ...

    async def load(
        self,
        thread_id: str,
        swarm: Swarm[Any],
    ) -> SwarmCheckpoint | None:
        """Return the latest checkpoint for ``thread_id`` or ``None``.

        ``swarm`` is supplied for parity with the graphs
        ``Checkpointer.load`` shape and to allow implementations to
        cross-validate against the persisted state's member names if
        they choose. The in-memory reference implementation does not
        validate the swarm against persisted member names;
        :meth:`SwarmState.from_dict` provides the integrity check at
        rehydration time.

        Args:
            thread_id: Logical run identifier to look up.
            swarm: The :class:`~troopai.adk.swarms.swarm.Swarm` the
                checkpoint belongs to.

        Returns:
            The stored :class:`SwarmCheckpoint`, or ``None`` when no
            checkpoint exists for ``thread_id``.
        """
        ...

    async def list_checkpoints(self) -> list[str]:
        """Return all known thread_ids, sorted.

        Returns:
            Sorted list of ``thread_id`` strings currently stored.
        """
        ...

    async def delete(self, thread_id: str) -> None:
        """Remove the checkpoint for ``thread_id``; no-op if absent.

        Args:
            thread_id: Logical run identifier to delete.
        """
        ...

    def register(self, registry: SwarmHookRegistry) -> None:
        """Subscribe a :class:`SwarmHooks` instance to ``registry`` for auto-save.

        Args:
            registry: The hook registry to attach auto-save callbacks to.
        """
        ...


__all__ = ["SwarmCheckpoint", "SwarmCheckpointer", "SwarmHookRegistry"]

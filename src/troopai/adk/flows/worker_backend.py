"""Worker-backend Protocol for distributed Flow execution.

A :class:`FlowWorkerBackend` is the coordination surface between
multiple :class:`FlowExecutor` workers running across processes /
hosts. The Flow primitive's BSP (Bulk Synchronous Parallel) loop
distributes at the **batch boundary**: one worker claims an entire
batch via :meth:`claim_batch`, runs every step in that batch (in
parallel within its process via :func:`asyncio.gather`), and
writes the resulting :class:`FlowCheckpoint` back via
:meth:`release_batch`. This preserves the executor's BSP
invariants (AND-gate resolution, sequential successor dispatch)
without fragmenting them across workers.

Per-executor optimisation state (rate-limit buckets, step caches,
``pending_triggers``) stays local to each worker — only the
:class:`FlowCheckpoint` is the canonical shared state. This
deliberately matches the LangGraph superstep-checkpointer pattern:
in-memory state is ephemeral, the checkpoint is authoritative.

**Rate-limit caveat**: :class:`FlowStepRateLimit` enforcement is
per-executor (per-batch claim). When the same step fires across
batches claimed by different workers, the rate-limit bucket
resets between claims — the documented ``rpm`` cap therefore
applies *per batch window*, not globally across the distributed
deployment. Set ``rate_limit=None`` on Flow steps when a globally
coordinated limit is required, and enforce the limit at the
worker pool's boundary instead.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from troopai.adk.flows.checkpoint import FlowCheckpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class FlowBatchClaim:
    """Audit record of one worker's claim on a flow batch.

    Frozen snapshot: produced when a worker successfully calls
    :meth:`FlowWorkerBackend.claim_batch` and returned by
    :meth:`FlowWorkerBackend.list_claims` for observability.

    Attributes:
        flow_id: The flow instance whose batch was claimed.
        batch_id: Monotonic batch identifier (the executor's
            ``step_count`` at batch start).
        worker_id: Opaque identifier of the claiming worker
            (typically ``f"{hostname}-{pid}-{uuid4}"``).
        claimed_at: Backend-clock timestamp at claim time. The
            :class:`InMemoryFlowWorkerBackend` uses ``time.monotonic``
            (single-process); the :class:`SqliteFlowWorkerBackend`
            uses ``time.time`` (Unix epoch, cross-process
            comparable).
        heartbeat_at: Backend-clock timestamp of the most recent
            heartbeat. Equal to :attr:`claimed_at` immediately
            after a successful claim.
    """

    flow_id: str
    """Identifier of the flow instance being claimed."""

    batch_id: int
    """Monotonic batch identifier."""

    worker_id: str
    """Opaque identifier of the claiming worker."""

    claimed_at: float
    """Backend-clock timestamp at claim time (``time.monotonic`` for the
    in-memory backend, ``time.time`` epoch seconds for the SQLite backend)."""

    heartbeat_at: float
    """Backend-clock timestamp of the most recent heartbeat — same clock as
    :attr:`claimed_at`; equal to it immediately after a successful claim."""


class FlowWorkerBackend(Protocol):
    """Coordination surface for distributed Flow execution.

    Implementations MUST be safe for the concurrency level they
    document:
    :class:`InMemoryFlowWorkerBackend` is single-process only
    (asyncio :class:`asyncio.Lock`-guarded);
    :class:`SqliteFlowWorkerBackend` is cross-process on one host
    (``BEGIN IMMEDIATE`` transactions).

    Every method is async to keep the surface uniform across
    in-memory and I/O-bound backends; the in-memory implementation
    completes instantly.
    """

    async def claim_batch(
        self,
        flow_id: str,
        batch_id: int,
        worker_id: str,
        *,
        ttl_seconds: float = 60.0,
    ) -> bool:
        """Attempt to claim ``(flow_id, batch_id)`` for ``worker_id``.

        Returns ``True`` when the claim succeeds (no other worker
        holds the batch, or the prior claim has expired past
        ``ttl_seconds``). Returns ``False`` when another worker
        already owns the batch within its TTL window — the caller
        backs off and tries the next batch or another flow.

        The TTL serves as a liveness check: a crashed worker's
        claim becomes claimable by another worker once
        ``time.monotonic() - heartbeat_at > ttl_seconds``.

        Args:
            flow_id: Identifier of the flow instance.
            batch_id: Monotonic batch identifier.
            worker_id: Opaque identifier of the claiming worker.
            ttl_seconds: Seconds after the last heartbeat before an
                existing claim may be superseded.

        Returns:
            ``True`` when the claim is granted; ``False`` when blocked
            by a live existing claim.
        """
        ...

    async def heartbeat(
        self,
        flow_id: str,
        batch_id: int,
        worker_id: str,
    ) -> bool:
        """Refresh the claim's ``heartbeat_at`` timestamp.

        Returns ``True`` when the heartbeat lands on a claim still
        owned by ``worker_id``; ``False`` when the claim was lost
        (expired and re-claimed by another worker, or never
        existed). A ``False`` return means the worker should abort
        its current batch.

        Args:
            flow_id: Identifier of the flow instance.
            batch_id: Monotonic batch identifier.
            worker_id: The worker that owns the claim.

        Returns:
            ``True`` when the heartbeat succeeded; ``False`` when the
            claim was already lost.
        """
        ...

    async def release_batch(
        self,
        flow_id: str,
        batch_id: int,
        worker_id: str,
        checkpoint: FlowCheckpoint,
    ) -> None:
        """Atomically release the claim and persist the post-batch checkpoint.

        Implementations MUST treat the release as a single
        atomic operation — the checkpoint write and the claim
        release happen together so a worker that crashes between
        the two never leaves a partial state. SQLite implements
        this with a single transaction; the in-memory backend
        with one asyncio Lock acquire.

        Args:
            flow_id: Identifier of the flow instance.
            batch_id: Monotonic batch identifier.
            worker_id: The worker releasing the claim.
            checkpoint: Post-batch :class:`FlowCheckpoint` to persist.
        """
        ...

    async def load_checkpoint(self, flow_id: str) -> FlowCheckpoint | None:
        """Return the most recent persisted checkpoint for ``flow_id``, or ``None``.

        Cold-start flows produce ``None``; resumed flows pick up
        from the last :meth:`release_batch` write.

        Args:
            flow_id: Identifier of the flow instance.

        Returns:
            The most recent :class:`FlowCheckpoint`, or ``None`` for a
            cold-start flow.
        """
        ...

    async def save_checkpoint(self, checkpoint: FlowCheckpoint) -> None:
        """Persist ``checkpoint`` outside the claim/release cycle.

        Used by callers that want to seed a backend before
        starting any workers (e.g. resume from an externally
        produced checkpoint). Implementations MAY overwrite the
        prior checkpoint atomically.

        Args:
            checkpoint: The :class:`FlowCheckpoint` to persist.
        """
        ...

    async def load_checkpoint_by_id(self, checkpoint_id: str) -> FlowCheckpoint | None:
        """Return the persisted checkpoint whose ``flow_id`` equals ``checkpoint_id``.

        Convenience alias over :meth:`load_checkpoint` that makes the
        intent explicit when the caller holds only the string id (not
        the full :class:`FlowCheckpoint` object).  The two methods are
        semantically equivalent — both key on ``flow_id``.

        The default implementation delegates to
        :meth:`load_checkpoint` so any existing backend that already
        implements the prior Protocol members automatically satisfies
        this widened surface without a code change.

        Returns ``None`` when no checkpoint exists for
        ``checkpoint_id``.

        Args:
            checkpoint_id: The :attr:`FlowCheckpoint.flow_id` to look
                up.

        Returns:
            The stored :class:`FlowCheckpoint`, or ``None`` when not
            found.
        """
        return await self.load_checkpoint(checkpoint_id)

    async def list_claims(self, flow_id: str) -> tuple[FlowBatchClaim, ...]:
        """Return a snapshot of every live claim against ``flow_id``.

        Useful for operational tooling (which worker owns the
        next batch, how long has it been heartbeating). Excludes
        expired claims by definition — a TTL-expired claim is
        returned only if ``list_claims`` is called within the same
        tick as the expiry.

        Args:
            flow_id: Identifier of the flow instance.

        Returns:
            Tuple of :class:`FlowBatchClaim` audit records for every
            active claim. Empty when no live claims exist.
        """
        ...


@dataclass
class _InMemoryClaim:
    """Mutable claim record used by :class:`InMemoryFlowWorkerBackend`.

    Attributes:
        worker_id: Opaque identifier of the worker that holds this claim.
        claimed_at: Monotonic timestamp at the moment the claim was
            first granted.
        heartbeat_at: Monotonic timestamp of the most recent successful
            heartbeat; updated in-place by :meth:`InMemoryFlowWorkerBackend.heartbeat`.
    """

    worker_id: str
    claimed_at: float
    heartbeat_at: float


@dataclass
class InMemoryFlowWorkerBackend:
    """Single-process :class:`FlowWorkerBackend` implementation.

    Suitable for tests, single-host single-process runs, and as
    the default backend when no distribution is configured.
    Stores claims + checkpoints in plain dicts guarded by per-loop
    :class:`asyncio.Lock` instances (see :attr:`_locks`).

    NOT safe across processes — use :class:`SqliteFlowWorkerBackend`
    for cross-process work on one host.

    Attributes:
        clock: Override-able monotonic clock; primarily for tests.
            Defaults to :func:`time.monotonic`.
    """

    clock: Callable[[], float] = field(default=time.monotonic)
    """Override-able monotonic clock; primarily for tests."""

    _claims: dict[tuple[str, int], _InMemoryClaim] = field(default_factory=dict)
    """``(flow_id, batch_id) → claim record``."""

    _checkpoints: dict[str, FlowCheckpoint] = field(default_factory=dict)
    """``flow_id → most recent checkpoint``."""

    _locks: dict[asyncio.AbstractEventLoop, asyncio.Lock] = field(default_factory=dict)
    """Per-event-loop locks (one :class:`asyncio.Lock` per running loop).

    A single shared lock binds itself to the first loop that uses it, so
    reusing one backend from a second loop (``Runner.run_flow`` called
    from different threads, sequential ``asyncio.run`` calls) raises
    "lock is bound to a different event loop" on contention. Locks are
    therefore allocated lazily per running loop.
    """

    async def claim_batch(
        self,
        flow_id: str,
        batch_id: int,
        worker_id: str,
        *,
        ttl_seconds: float = 60.0,
    ) -> bool:
        """Return ``True`` on a fresh claim or after TTL expiry; ``False`` otherwise."""
        lock = self._ensure_lock()
        async with lock:
            now = self.clock()
            existing = self._claims.get((flow_id, batch_id))
            if existing is not None and now - existing.heartbeat_at <= ttl_seconds:
                return False
            self._claims[(flow_id, batch_id)] = _InMemoryClaim(
                worker_id=worker_id,
                claimed_at=now,
                heartbeat_at=now,
            )
            return True

    async def heartbeat(
        self,
        flow_id: str,
        batch_id: int,
        worker_id: str,
    ) -> bool:
        """Refresh ``heartbeat_at`` when the claim is still owned by ``worker_id``."""
        lock = self._ensure_lock()
        async with lock:
            existing = self._claims.get((flow_id, batch_id))
            if existing is None or existing.worker_id != worker_id:
                return False
            existing.heartbeat_at = self.clock()
            return True

    async def release_batch(
        self,
        flow_id: str,
        batch_id: int,
        worker_id: str,
        checkpoint: FlowCheckpoint,
    ) -> None:
        """Release the claim and persist ``checkpoint`` atomically."""
        lock = self._ensure_lock()
        async with lock:
            existing = self._claims.get((flow_id, batch_id))
            if existing is not None and existing.worker_id == worker_id:
                del self._claims[(flow_id, batch_id)]
                self._checkpoints[flow_id] = checkpoint
            else:
                logger.warning(
                    "InMemoryFlowWorkerBackend: release_batch for flow=%s batch=%d "
                    "worker=%s lost its claim (TTL takeover); dropping checkpoint write.",
                    flow_id,
                    batch_id,
                    worker_id,
                )

    async def load_checkpoint(self, flow_id: str) -> FlowCheckpoint | None:
        """Return the most recent checkpoint for ``flow_id``, or ``None``."""
        lock = self._ensure_lock()
        async with lock:
            return self._checkpoints.get(flow_id)

    async def save_checkpoint(self, checkpoint: FlowCheckpoint) -> None:
        """Overwrite the persisted checkpoint for ``checkpoint.flow_id``."""
        lock = self._ensure_lock()
        async with lock:
            self._checkpoints[checkpoint.flow_id] = checkpoint

    async def load_checkpoint_by_id(self, checkpoint_id: str) -> FlowCheckpoint | None:
        """Return the checkpoint whose ``flow_id`` equals ``checkpoint_id``, or ``None``."""
        return await self.load_checkpoint(checkpoint_id)

    async def list_claims(self, flow_id: str) -> tuple[FlowBatchClaim, ...]:
        """Return the live claims for ``flow_id`` as frozen audit records."""
        lock = self._ensure_lock()
        async with lock:
            return tuple(
                FlowBatchClaim(
                    flow_id=flow_id,
                    batch_id=batch_id,
                    worker_id=claim.worker_id,
                    claimed_at=claim.claimed_at,
                    heartbeat_at=claim.heartbeat_at,
                )
                for (fid, batch_id), claim in self._claims.items()
                if fid == flow_id
            )

    def _ensure_lock(self) -> asyncio.Lock:
        """Return the :class:`asyncio.Lock` bound to the running event loop.

        Allocation is lazy and per-loop so the backend can be
        constructed outside an event loop (e.g. at module import time)
        and reused across sequential loops without tripping asyncio's
        loop-binding check. Only called from async methods, so a loop
        is always running here.
        """
        loop = asyncio.get_running_loop()
        lock = self._locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[loop] = lock
        return lock


_PROTOCOL_CHECK: FlowWorkerBackend = InMemoryFlowWorkerBackend()
"""Module-level assertion that :class:`InMemoryFlowWorkerBackend` satisfies the Protocol.

Static-checker bait — mypy / pyright verify Protocol conformance
at this assignment site without requiring a runtime call.
"""

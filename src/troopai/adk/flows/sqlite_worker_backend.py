"""SQLite-backed :class:`FlowWorkerBackend` for cross-process distribution.

A file-backed backend suitable for distributing flow execution
across multiple processes on a single host. Uses ``BEGIN IMMEDIATE``
transactions for claim contention and stores the full
:class:`FlowCheckpoint` JSON payload in one row per flow.

Tables:

- ``flow_claims(flow_id TEXT, batch_id INTEGER, worker_id TEXT,
  claimed_at REAL, heartbeat_at REAL, PRIMARY KEY (flow_id, batch_id))``
- ``flow_checkpoints(flow_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
  updated_at REAL NOT NULL)``

Schema evolution: additive columns only, with tolerant loads.
There is intentionally no ``schema_version`` column on either
table — adding fields with safe defaults is the supported
forward-compatibility path.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from troopai.adk.flows.worker_backend import FlowBatchClaim

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from troopai.adk.flows.checkpoint import FlowCheckpoint


_SCHEMA = """
CREATE TABLE IF NOT EXISTS flow_claims (
    flow_id TEXT NOT NULL,
    batch_id INTEGER NOT NULL,
    worker_id TEXT NOT NULL,
    claimed_at REAL NOT NULL,
    heartbeat_at REAL NOT NULL,
    PRIMARY KEY (flow_id, batch_id)
);

CREATE TABLE IF NOT EXISTS flow_checkpoints (
    flow_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


@dataclass
class SqliteFlowWorkerBackend:
    """File-backed :class:`FlowWorkerBackend` using SQLite for cross-process coordination.

    Suitable for multiple worker processes on one host sharing
    one SQLite file. ``BEGIN IMMEDIATE`` acquires a reserved
    lock on every claim attempt so contention is resolved by
    SQLite itself; the asyncio dimension is handled by running
    DB calls in the default executor via
    :meth:`asyncio.to_thread`.

    The clock defaults to :func:`time.time` (Unix epoch seconds)
    — NOT :func:`time.monotonic` — so claim timestamps are
    comparable across processes. The :class:`InMemoryFlowWorkerBackend`
    uses ``time.monotonic`` because its claims never cross a
    process boundary.

    Attributes:
        path: Path to the SQLite file. Created on first access
            (with ``WAL`` journal mode enabled for concurrent
            reader / writer access). Schema is initialised
            eagerly in ``__post_init__`` so concurrent writers
            never race on ``CREATE TABLE IF NOT EXISTS``.
        clock: Override-able wall-clock (epoch seconds); primarily for
            tests. Defaults to :func:`time.time`.
    """

    path: str | Path
    """SQLite file path."""

    clock: Callable[[], float] = field(default=time.time)
    """Override-able wall-clock (epoch seconds); cross-process comparable."""

    def __post_init__(self) -> None:
        """Eagerly create the schema so threadpool workers never race on init."""
        if str(self.path) == ":memory:":
            raise ValueError(
                "SqliteFlowWorkerBackend requires a file path; ':memory:' is not "
                "supported because each connection opens an independent empty "
                "database, so the schema created here would not survive. Use a "
                "temporary file path in tests."
            )
        conn = sqlite3.connect(str(self.path), isolation_level=None, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    async def claim_batch(
        self,
        flow_id: str,
        batch_id: int,
        worker_id: str,
        *,
        ttl_seconds: float = 60.0,
    ) -> bool:
        """Attempt to claim ``(flow_id, batch_id)`` for ``worker_id``.

        Delegates to :meth:`_claim_batch_sync` in
        :func:`asyncio.to_thread`. Uses ``BEGIN IMMEDIATE`` to serialize
        contention. An existing claim is superseded only when its
        heartbeat has lapsed beyond ``ttl_seconds``.

        Args:
            flow_id: Identifier of the flow instance.
            batch_id: Monotonic batch identifier.
            worker_id: Opaque identifier of the claiming worker.
            ttl_seconds: TTL for the existing claim; a live claim
                within this window blocks the new attempt.

        Returns:
            ``True`` when the claim is granted; ``False`` when another
            worker holds a live claim.
        """
        return await asyncio.to_thread(
            self._claim_batch_sync,
            flow_id,
            batch_id,
            worker_id,
            ttl_seconds,
        )

    async def heartbeat(
        self,
        flow_id: str,
        batch_id: int,
        worker_id: str,
    ) -> bool:
        """Refresh ``heartbeat_at`` when the row is still owned by ``worker_id``.

        Args:
            flow_id: Identifier of the flow instance.
            batch_id: Monotonic batch identifier.
            worker_id: The worker that owns the claim.

        Returns:
            ``True`` when the heartbeat landed on a claim owned by
            ``worker_id``; ``False`` when the claim was lost or never
            existed.
        """
        return await asyncio.to_thread(self._heartbeat_sync, flow_id, batch_id, worker_id)

    async def release_batch(
        self,
        flow_id: str,
        batch_id: int,
        worker_id: str,
        checkpoint: FlowCheckpoint,
    ) -> None:
        """Release the claim and persist ``checkpoint`` in one atomic transaction.

        When the DELETE matches zero rows (claim was taken over on TTL
        expiry), the checkpoint write is skipped and a WARNING is logged.

        Args:
            flow_id: Identifier of the flow instance.
            batch_id: Monotonic batch identifier.
            worker_id: The worker releasing the claim.
            checkpoint: Post-batch :class:`FlowCheckpoint` to persist.
        """
        await asyncio.to_thread(
            self._release_batch_sync,
            flow_id,
            batch_id,
            worker_id,
            checkpoint,
        )

    async def load_checkpoint(self, flow_id: str) -> FlowCheckpoint | None:
        """Return the persisted checkpoint for ``flow_id``, or ``None``.

        Args:
            flow_id: Identifier of the flow instance.

        Returns:
            The most recent :class:`FlowCheckpoint` written by
            :meth:`release_batch` or :meth:`save_checkpoint`, or
            ``None`` when no checkpoint exists for ``flow_id``.
        """
        return await asyncio.to_thread(self._load_checkpoint_sync, flow_id)

    async def save_checkpoint(self, checkpoint: FlowCheckpoint) -> None:
        """Persist ``checkpoint`` outside the claim/release cycle.

        Args:
            checkpoint: The :class:`FlowCheckpoint` to persist.
                Overwrites any prior checkpoint for the same
                ``checkpoint.flow_id``.
        """
        await asyncio.to_thread(self._save_checkpoint_sync, checkpoint)

    async def load_checkpoint_by_id(self, checkpoint_id: str) -> FlowCheckpoint | None:
        """Return the checkpoint whose ``flow_id`` equals ``checkpoint_id``, or ``None``.

        Convenience alias over :meth:`load_checkpoint` for callers that
        hold only the string id. Semantically identical to
        ``load_checkpoint(checkpoint_id)``.

        Args:
            checkpoint_id: The :attr:`FlowCheckpoint.flow_id` to look
                up.

        Returns:
            The stored :class:`FlowCheckpoint`, or ``None`` when not
            found.
        """
        return await self.load_checkpoint(checkpoint_id)

    async def list_claims(self, flow_id: str) -> tuple[FlowBatchClaim, ...]:
        """Return the live claims for ``flow_id`` as frozen audit records.

        Args:
            flow_id: Identifier of the flow instance.

        Returns:
            Tuple of :class:`FlowBatchClaim` audit records for every
            active claim against ``flow_id``. Empty when no claims
            exist.
        """
        return await asyncio.to_thread(self._list_claims_sync, flow_id)

    def _claim_batch_sync(
        self,
        flow_id: str,
        batch_id: int,
        worker_id: str,
        ttl_seconds: float,
    ) -> bool:
        """Synchronous claim implementation; runs in ``asyncio.to_thread``."""
        now = self.clock()
        with contextlib.closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT worker_id, heartbeat_at FROM flow_claims WHERE flow_id=? AND batch_id=?",
                    (flow_id, batch_id),
                ).fetchone()
                if row is not None and (now - row["heartbeat_at"]) <= ttl_seconds:
                    conn.execute("ROLLBACK")
                    return False
                conn.execute(
                    "INSERT OR REPLACE INTO flow_claims "
                    "(flow_id, batch_id, worker_id, claimed_at, heartbeat_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (flow_id, batch_id, worker_id, now, now),
                )
                conn.execute("COMMIT")
                return True
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def _heartbeat_sync(self, flow_id: str, batch_id: int, worker_id: str) -> bool:
        """Atomic heartbeat — UPDATE-then-rowcount, no TOCTOU window."""
        now = self.clock()
        with contextlib.closing(self._connect()) as conn:
            cursor = conn.execute(
                "UPDATE flow_claims SET heartbeat_at=? WHERE flow_id=? AND batch_id=? AND worker_id=?",
                (now, flow_id, batch_id, worker_id),
            )
            return cursor.rowcount > 0

    def _release_batch_sync(
        self,
        flow_id: str,
        batch_id: int,
        worker_id: str,
        checkpoint: FlowCheckpoint,
    ) -> None:
        """Atomic claim-delete + checkpoint-write, gated on ownership.

        Skips the checkpoint write when the DELETE matched zero rows
        (the claim was already taken over by another worker on TTL
        expiry). Logs at WARNING when this happens so operators can
        diagnose silent ownership loss.
        """
        now = self.clock()
        with contextlib.closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = conn.execute(
                    "DELETE FROM flow_claims WHERE flow_id=? AND batch_id=? AND worker_id=?",
                    (flow_id, batch_id, worker_id),
                )
                if cursor.rowcount == 0:
                    conn.execute("ROLLBACK")
                    logger.warning(
                        "SqliteFlowWorkerBackend: release_batch for flow=%s batch=%d "
                        "worker=%s lost its claim (TTL takeover); dropping checkpoint write.",
                        flow_id,
                        batch_id,
                        worker_id,
                    )
                    return
                conn.execute(
                    "INSERT OR REPLACE INTO flow_checkpoints (flow_id, payload, updated_at) VALUES (?, ?, ?)",
                    (checkpoint.flow_id, checkpoint.to_json(), now),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def _load_checkpoint_sync(self, flow_id: str) -> FlowCheckpoint | None:
        """Synchronous load implementation."""
        from troopai.adk.flows.checkpoint import FlowCheckpoint

        with contextlib.closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT payload FROM flow_checkpoints WHERE flow_id=?",
                (flow_id,),
            ).fetchone()
            if row is None:
                return None
            return FlowCheckpoint.from_json(row["payload"])

    def _save_checkpoint_sync(self, checkpoint: FlowCheckpoint) -> None:
        """Synchronous save implementation.

        Uses ``BEGIN IMMEDIATE`` to acquire a write lock upfront, matching
        the pattern in :meth:`_release_batch_sync`. Without it, the INSERT
        runs in autocommit mode under a shared lock and a concurrent reader
        can observe a partial write on a page boundary.
        """
        now = self.clock()
        with contextlib.closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO flow_checkpoints (flow_id, payload, updated_at) VALUES (?, ?, ?)",
                    (checkpoint.flow_id, checkpoint.to_json(), now),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def _list_claims_sync(self, flow_id: str) -> tuple[FlowBatchClaim, ...]:
        """Synchronous list-claims implementation."""
        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT batch_id, worker_id, claimed_at, heartbeat_at FROM flow_claims WHERE flow_id=?",
                (flow_id,),
            ).fetchall()
            return tuple(
                FlowBatchClaim(
                    flow_id=flow_id,
                    batch_id=row["batch_id"],
                    worker_id=row["worker_id"],
                    claimed_at=row["claimed_at"],
                    heartbeat_at=row["heartbeat_at"],
                )
                for row in rows
            )

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with the required pragmas.

        Schema is initialised eagerly in :meth:`__post_init__`; this
        method only opens connections for queries / transactions.
        """
        conn = sqlite3.connect(str(self.path), isolation_level=None, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

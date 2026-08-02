"""``PostgresSwarmCheckpointer`` — ACID swarm-run persistence via PostgreSQL + JSONB.

Uses ``psycopg`` 3 (async) with a connection pool and optimistic locking:
a ``lock_token`` UUID column acts as a fencing token. On ``save`` the token
is verified; a stale token (concurrent writer) raises
:class:`~troopai.adk.exceptions.CheckpointConflictError`.

One row per ``thread_id`` — the latest checkpoint overwrites the prior
one (upsert-on-insert, conditional-UPDATE thereafter).

The table is created automatically on first connection if absent.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

try:
    from psycopg.types.json import Jsonb
    from psycopg_pool import AsyncConnectionPool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PostgresSwarmCheckpointer requires psycopg[binary,pool]>=3.2: pip install 'troopai-adk-python[checkpointer-postgres]'"
    ) from exc

from troopai.adk.exceptions import CheckpointConflictError
from troopai.adk.swarms.checkpointer import SwarmCheckpoint, SwarmHookRegistry

if TYPE_CHECKING:
    from troopai.adk.swarms.swarm import Swarm


logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS swarm_checkpoints (
  thread_id  TEXT PRIMARY KEY,
  turn       INTEGER NOT NULL,
  created_at DOUBLE PRECISION NOT NULL,
  updated_at DOUBLE PRECISION NOT NULL,
  state      JSONB NOT NULL,
  lock_token UUID NOT NULL DEFAULT gen_random_uuid()
)
"""

_INSERT_RETURNING = (
    "INSERT INTO swarm_checkpoints "
    "(thread_id, turn, created_at, updated_at, state) "
    "VALUES (%s, %s, %s, %s, %s) "
    "ON CONFLICT (thread_id) DO NOTHING RETURNING lock_token"
)

_UPDATE_RETURNING = (
    "UPDATE swarm_checkpoints SET "
    "turn=%s, updated_at=%s, state=%s, lock_token=gen_random_uuid() "
    "WHERE thread_id=%s AND lock_token=%s::uuid RETURNING lock_token"
)


class PostgresSwarmCheckpointer:
    """ACID swarm checkpointer backed by PostgreSQL JSONB + optimistic locking.

    Each logical ``thread_id`` maps to one row. ``save`` is an upsert on
    first write; subsequent saves use a conditional ``UPDATE`` guarded by
    the fencing token cached from the prior ``load`` or ``save``. A
    concurrent writer that rotates the token causes the losing ``save``
    to raise :class:`~troopai.adk.exceptions.CheckpointConflictError`.

    The caller owns the lifecycle — call :meth:`close` at application
    shutdown or after a test.

    Requires PostgreSQL 13+ (``gen_random_uuid()`` built-in; no
    extension needed).

    Attributes:
        conninfo: libpq connection string used to open the pool.
    """

    def __init__(self, conninfo: str, thread_id: str = "default") -> None:
        """Initialise the checkpointer and lazily open the connection pool.

        Args:
            conninfo: libpq connection string used to open the pool.
            thread_id: Identifier used by :meth:`register`'s auto-save
                hook. Defaults to ``"default"`` when the caller does not
                supply an explicit id.
        """
        self.conninfo = conninfo
        self._thread_id = thread_id
        self._pool: AsyncConnectionPool | None = None
        self._tokens: dict[str, str] = {}
        self._init_lock = asyncio.Lock()
        logger.debug("PostgresSwarmCheckpointer initialised with conninfo=%r", conninfo)

    async def _get_pool(self) -> AsyncConnectionPool:
        """Open the pool and create the schema on first call.

        The init lock serializes concurrent first callers so exactly one
        pool is opened. If schema creation fails the half-opened pool is
        closed before the error propagates, so no connections leak.
        """
        async with self._init_lock:
            if self._pool is None:
                pool: AsyncConnectionPool = AsyncConnectionPool(self.conninfo, open=False)
                await pool.open()
                try:
                    async with pool.connection() as conn:
                        # DDL: the returned cursor is unused.
                        await conn.execute(_CREATE_TABLE)
                except Exception:
                    await pool.close()
                    raise
                self._pool = pool
                logger.debug("PostgresSwarmCheckpointer: pool opened and schema ensured.")
            return self._pool

    async def close(self) -> None:
        """Close the connection pool. Idempotent and safe in a ``finally``."""
        async with self._init_lock:
            if self._pool is not None:
                pool, self._pool = self._pool, None
                await pool.close()
                logger.debug("PostgresSwarmCheckpointer: pool closed.")

    def register(self, registry: SwarmHookRegistry) -> None:
        """Subscribe a :class:`SwarmCheckpointerHooks` to ``registry``."""
        from troopai.adk.swarms.checkpointers.hooks import SwarmCheckpointerHooks

        registry.add(SwarmCheckpointerHooks(self, self._thread_id))
        logger.debug("PostgresSwarmCheckpointer registered on SwarmHookRegistry.")

    async def save(self, checkpoint: SwarmCheckpoint) -> None:
        """Upsert ``checkpoint`` with optimistic locking.

        On the first save for a ``thread_id`` this executes an ``INSERT …
        ON CONFLICT DO NOTHING RETURNING lock_token``; subsequent saves
        perform a conditional ``UPDATE … WHERE lock_token = <cached>
        RETURNING lock_token``. If either returns no row — meaning a
        concurrent writer won the race — :class:`CheckpointConflictError`
        is raised.

        To resume an existing ``thread_id`` on a fresh instance, call
        :meth:`load` first so the fencing token is cached; otherwise the
        first ``save`` takes the insert path and a pre-existing row raises
        :class:`CheckpointConflictError`.

        Args:
            checkpoint: The snapshot to persist.

        Raises:
            CheckpointConflictError: When a concurrent writer has rotated
                the fencing token since this instance last observed it.
        """
        pool = await self._get_pool()
        token = self._tokens.get(checkpoint.thread_id)
        now = time.time()

        async with pool.connection() as conn:
            if token is None:
                cur = await conn.execute(
                    _INSERT_RETURNING,
                    (
                        checkpoint.thread_id,
                        checkpoint.turn,
                        now,
                        now,
                        Jsonb(checkpoint.state),
                    ),
                )
            else:
                cur = await conn.execute(
                    _UPDATE_RETURNING,
                    (
                        checkpoint.turn,
                        now,
                        Jsonb(checkpoint.state),
                        checkpoint.thread_id,
                        token,
                    ),
                )
            row = await cur.fetchone()

        if row is None:
            raise CheckpointConflictError(checkpoint.thread_id)

        self._tokens[checkpoint.thread_id] = str(row[0])
        logger.debug(
            "PostgresSwarmCheckpointer.save: thread_id=%s turn=%s path=%s",
            checkpoint.thread_id,
            checkpoint.turn,
            "insert" if token is None else "update",
        )

    async def load(
        self,
        thread_id: str,
        swarm: Swarm[Any],
    ) -> SwarmCheckpoint | None:
        """Rehydrate the checkpoint for ``thread_id`` (``None`` if absent).

        Caches the observed ``lock_token`` so a subsequent :meth:`save`
        can verify it has not been rotated by a concurrent writer.

        The ``swarm`` parameter is accepted for protocol parity with the
        graphs ``Checkpointer.load`` shape. Member-name resolution in
        :meth:`SwarmState.from_dict` provides the de-facto integrity check
        at rehydration time.

        Args:
            thread_id: The logical run key.
            swarm: The :class:`Swarm` the checkpoint belongs to. Accepted
                for protocol parity; member validation happens at
                :meth:`SwarmState.from_dict` call time.

        Returns:
            A :class:`SwarmCheckpoint`, or ``None`` when no checkpoint
            exists for ``thread_id``.
        """
        del swarm
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT turn, state, lock_token FROM swarm_checkpoints WHERE thread_id=%s",
                (thread_id,),
            )
            row = await cur.fetchone()

        if row is None:
            logger.debug("PostgresSwarmCheckpointer.load: no checkpoint for thread_id=%s", thread_id)
            return None

        turn, state, lock_token = row
        self._tokens[thread_id] = str(lock_token)
        logger.debug("PostgresSwarmCheckpointer.load: thread_id=%s turn=%s", thread_id, turn)
        return SwarmCheckpoint(thread_id=thread_id, state=state, turn=turn)

    async def list_checkpoints(self) -> list[str]:
        """Return a sorted list of thread ids currently stored."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT thread_id FROM swarm_checkpoints ORDER BY thread_id")
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    async def delete(self, thread_id: str) -> None:
        """Delete the checkpoint for ``thread_id`` (no-op if absent)."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            # DML: the returned cursor is unused (delete is a no-op if absent).
            await conn.execute(
                "DELETE FROM swarm_checkpoints WHERE thread_id=%s",
                (thread_id,),
            )
        self._tokens.pop(thread_id, None)
        logger.debug("PostgresSwarmCheckpointer.delete: thread_id=%s", thread_id)


__all__ = ["PostgresSwarmCheckpointer"]

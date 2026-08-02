"""``PostgresCheckpointer`` — ACID graph-run persistence via PostgreSQL + JSONB.

Uses ``psycopg`` 3 (async) with a connection pool and optimistic locking:
a ``lock_token`` UUID column acts as a fencing token. On ``save`` the token
is verified; a stale token (concurrent writer) raises
:class:`~troopai.adk.exceptions.CheckpointConflictError`.

One row per ``thread_id`` — the latest checkpoint overwrites the prior
one (upsert-on-insert, conditional-UPDATE thereafter). Time-travel /
replay-from-any-superstep is out of scope; that would require a
row-per-superstep schema.

The table is created automatically on first connection if absent.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, override

try:
    from psycopg.types.json import Jsonb
    from psycopg_pool import AsyncConnectionPool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PostgresCheckpointer requires psycopg[binary,pool]>=3.2: pip install 'troopai-adk-python[checkpointer-postgres]'"
    ) from exc

from troopai.adk.exceptions import CheckpointConflictError
from troopai.adk.graphs.checkpointer import Checkpointer, GraphCheckpoint
from troopai.adk.graphs.checkpointers.hooks import CheckpointerHooks
from troopai.adk.graphs.hooks import HookRegistry

if TYPE_CHECKING:
    from troopai.adk.graphs.graph import Graph
    from troopai.adk.graphs.state import GraphState


logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS graph_checkpoints (
  thread_id  TEXT PRIMARY KEY,
  graph_id   TEXT NOT NULL,
  superstep  INTEGER NOT NULL,
  created_at DOUBLE PRECISION NOT NULL,
  updated_at DOUBLE PRECISION NOT NULL,
  state      JSONB NOT NULL,
  lock_token UUID NOT NULL DEFAULT gen_random_uuid()
)
"""

_INSERT_RETURNING = (
    "INSERT INTO graph_checkpoints "
    "(thread_id, graph_id, superstep, created_at, updated_at, state) "
    "VALUES (%s, %s, %s, %s, %s, %s) "
    "ON CONFLICT (thread_id) DO NOTHING RETURNING lock_token"
)

_UPDATE_RETURNING = (
    "UPDATE graph_checkpoints SET "
    "graph_id=%s, superstep=%s, updated_at=%s, state=%s, lock_token=gen_random_uuid() "
    "WHERE thread_id=%s AND lock_token=%s::uuid RETURNING lock_token"
)


class PostgresCheckpointer(Checkpointer):
    """ACID graphs checkpointer backed by PostgreSQL JSONB + optimistic locking.

    Each logical ``thread_id`` maps to one row. ``save`` is an upsert on
    first write; subsequent saves use a conditional ``UPDATE`` guarded by
    the fencing token cached from the prior ``load`` or ``save``. A
    concurrent writer that rotates the token causes the losing ``save``
    to raise :class:`~troopai.adk.exceptions.CheckpointConflictError`.

    The caller owns the lifecycle — call :meth:`close` at application
    shutdown or after a test. :class:`~troopai.adk.run.runner.Runner` does
    not close a caller-supplied checkpointer (same contract as
    :class:`~troopai.adk.graphs.checkpointers.sqlite.SQLiteCheckpointer`).

    Requires PostgreSQL 13+ (``gen_random_uuid()`` built-in; no
    extension needed).

    Attributes:
        conninfo: libpq connection string used to open the pool.
    """

    def __init__(self, conninfo: str) -> None:
        self.conninfo = conninfo
        self._pool: AsyncConnectionPool | None = None
        self._tokens: dict[str, str] = {}
        self._init_lock = asyncio.Lock()
        logger.debug("PostgresCheckpointer initialised with conninfo=%r", conninfo)

    async def _get_pool(self) -> AsyncConnectionPool:
        """Open the pool and create the schema on first call.

        The init lock serializes concurrent first callers (the BSP loop
        fans out ``on_node_end`` hooks) so exactly one pool is opened. If
        schema creation fails the half-opened pool is closed before the
        error propagates, so no connections leak.
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
                logger.debug("PostgresCheckpointer: pool opened and schema ensured.")
            return self._pool

    async def close(self) -> None:
        """Close the connection pool. Idempotent and safe in a ``finally``."""
        async with self._init_lock:
            if self._pool is not None:
                pool, self._pool = self._pool, None
                await pool.close()
                logger.debug("PostgresCheckpointer: pool closed.")

    @override
    def register(self, registry: HookRegistry) -> None:
        """Subscribe to ``on_node_end`` / ``on_graph_end``."""
        registry.add(CheckpointerHooks(self))
        logger.debug("PostgresCheckpointer registered on HookRegistry.")

    @override
    async def save(self, checkpoint: GraphCheckpoint) -> None:
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
                        checkpoint.graph_id,
                        checkpoint.superstep,
                        checkpoint.created_at,
                        now,
                        Jsonb(checkpoint.state),
                    ),
                )
            else:
                cur = await conn.execute(
                    _UPDATE_RETURNING,
                    (
                        checkpoint.graph_id,
                        checkpoint.superstep,
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
            "PostgresCheckpointer.save: thread_id=%s superstep=%s path=%s",
            checkpoint.thread_id,
            checkpoint.superstep,
            "insert" if token is None else "update",
        )

    @override
    async def load(
        self,
        thread_id: str,
        graph: Graph[Any],
    ) -> GraphState[Any] | None:
        """Rehydrate the checkpoint for ``thread_id`` (``None`` if absent).

        Caches the observed ``lock_token`` so a subsequent :meth:`save`
        can verify it has not been rotated by a concurrent writer.

        Args:
            thread_id: The logical run key.
            graph: The :class:`Graph` the checkpoint belongs to. Mismatch
                between ``graph.id`` and the stored ``graph_id`` raises
                ``ValueError``.

        Returns:
            A rehydrated :class:`GraphState`, or ``None`` when no
            checkpoint exists for ``thread_id``.

        Raises:
            ValueError: When the stored ``graph_id`` does not match
                ``graph.id``.
        """
        from troopai.adk.graphs.state import GraphState

        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT graph_id, state, lock_token FROM graph_checkpoints WHERE thread_id=%s",
                (thread_id,),
            )
            row = await cur.fetchone()

        if row is None:
            logger.debug("PostgresCheckpointer.load: no checkpoint for thread_id=%s", thread_id)
            return None

        db_graph_id, state, lock_token = row
        if db_graph_id != graph.id:
            raise ValueError(
                f"Checkpoint graph_id={db_graph_id!r} does not match supplied graph.id={graph.id!r}. Refusing to load."
            )

        self._tokens[thread_id] = str(lock_token)
        logger.debug(
            "PostgresCheckpointer.load: thread_id=%s graph_id=%s",
            thread_id,
            graph.id,
        )
        return GraphState.from_dict(state, graph)

    @override
    async def list_checkpoints(self) -> list[str]:
        """Return a sorted list of thread ids currently stored."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute("SELECT thread_id FROM graph_checkpoints ORDER BY thread_id")
            rows = await cur.fetchall()
        return [r[0] for r in rows]

    @override
    async def delete(self, thread_id: str) -> None:
        """Delete the checkpoint for ``thread_id`` (no-op if absent)."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            # DML: the returned cursor is unused (delete is a no-op if absent).
            await conn.execute(
                "DELETE FROM graph_checkpoints WHERE thread_id=%s",
                (thread_id,),
            )
        self._tokens.pop(thread_id, None)
        logger.debug("PostgresCheckpointer.delete: thread_id=%s", thread_id)

    async def __aenter__(self) -> PostgresCheckpointer:
        """Support ``async with PostgresCheckpointer(...) as cp:`` usage."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the connection pool on context-manager exit."""
        await self.close()


__all__ = ["PostgresCheckpointer"]

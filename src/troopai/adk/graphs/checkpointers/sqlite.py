"""``SQLiteCheckpointer`` — durable, single-file graph-run persistence.

Mirrors :class:`~troopai.adk.session.sqlite_session.SQLiteSession` in
structure (``aiosqlite``, lazy-open connection, JSON columns). One row
per ``thread_id`` — the latest checkpoint overwrites the previous
(``save`` is an upsert). Time-travel / replay-from-any-superstep is out
of scope; that would require a row-per-superstep shape.

The checkpointer subscribes to ``on_node_end`` and ``on_graph_end``
via :class:`~troopai.adk.graphs.checkpointers.hooks.CheckpointerHooks`,
so the graph loop contains zero persistence code.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, override

import aiosqlite

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
  created_at REAL NOT NULL,
  state_json TEXT NOT NULL
)
"""


class SQLiteCheckpointer(Checkpointer):
    """Durable :class:`Checkpointer` backed by a single SQLite file.

    Uses ``aiosqlite`` for truly async database access. The connection
    is opened lazily on first use and held for the lifetime of the
    instance. Call :meth:`close` when done (e.g. after a test or at
    application shutdown).

    **Connection lifecycle and ownership.** The caller owns the
    checkpointer instance and its connection lifecycle; pass the same
    instance for the initial run and any later resume so both share
    the same process-lifetime store.  The connection opens lazily on
    first use and is reused for the instance's lifetime.
    :class:`~troopai.adk.run.runner.Runner` does not close a
    caller-supplied checkpointer — the runner does not own it, the same
    as :class:`~troopai.adk.graphs.checkpointers.in_memory.InMemoryCheckpointer`.
    Call :meth:`close` at application shutdown, or per-instance if you
    construct one per request.

    Attributes:
        path: Filesystem path to the SQLite database. Created on first
            connection if absent.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._conn: aiosqlite.Connection | None = None
        self._init_lock = asyncio.Lock()
        logger.debug("SQLiteCheckpointer initialised at path=%s", path)

    async def _db(self) -> aiosqlite.Connection:
        """Open (once) and return the connection, ensuring the schema."""
        async with self._init_lock:
            if self._conn is None:
                self._conn = await aiosqlite.connect(self.path)
                self._conn.row_factory = aiosqlite.Row
                await self._conn.execute(_CREATE_TABLE)
                await self._conn.commit()
        return self._conn

    @override
    def register(self, registry: HookRegistry) -> None:
        """Subscribe to ``on_node_end`` / ``on_graph_end``."""
        registry.add(CheckpointerHooks(self))
        logger.debug("SQLiteCheckpointer registered on HookRegistry.")

    @override
    async def save(self, checkpoint: GraphCheckpoint) -> None:
        """Upsert ``checkpoint`` by ``thread_id`` (latest wins)."""
        db = await self._db()
        await db.execute(
            "INSERT INTO graph_checkpoints "
            "(thread_id, graph_id, superstep, created_at, state_json) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(thread_id) DO UPDATE SET "
            "graph_id=excluded.graph_id, superstep=excluded.superstep, "
            "created_at=excluded.created_at, state_json=excluded.state_json",
            (
                checkpoint.thread_id,
                checkpoint.graph_id,
                checkpoint.superstep,
                checkpoint.created_at,
                json.dumps(checkpoint.state, separators=(",", ":")),
            ),
        )
        await db.commit()
        logger.debug(
            "SQLiteCheckpointer.save: thread_id=%s superstep=%s",
            checkpoint.thread_id,
            checkpoint.superstep,
        )

    @override
    async def load(
        self,
        thread_id: str,
        graph: Graph[Any],
    ) -> GraphState[Any] | None:
        """Rehydrate the checkpoint for ``thread_id`` (``None`` if absent).

        Args:
            thread_id: The logical run key.
            graph: The :class:`Graph` the checkpoint belongs to. Mismatch
                between ``graph.id`` and the stored ``graph_id`` raises
                ``ValueError``.

        Returns:
            A rehydrated :class:`GraphState`, or ``None`` when no checkpoint
            exists for ``thread_id``.

        Raises:
            ValueError: When the stored ``graph_id`` does not match
                ``graph.id``.
        """
        from troopai.adk.graphs.state import GraphState

        db = await self._db()
        async with db.execute(
            "SELECT graph_id, state_json FROM graph_checkpoints WHERE thread_id = ?",
            (thread_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            logger.debug(
                "SQLiteCheckpointer.load: no checkpoint for thread_id=%s",
                thread_id,
            )
            return None
        if row["graph_id"] != graph.id:
            raise ValueError(
                f"Checkpoint graph_id={row['graph_id']!r} does not match "
                f"supplied graph.id={graph.id!r}. Refusing to load."
            )
        return GraphState.from_dict(json.loads(row["state_json"]), graph)

    @override
    async def list_checkpoints(self) -> list[str]:
        """Return a sorted list of thread ids currently stored."""
        db = await self._db()
        async with db.execute("SELECT thread_id FROM graph_checkpoints ORDER BY thread_id") as cur:
            rows = await cur.fetchall()
        return [r["thread_id"] for r in rows]

    @override
    async def delete(self, thread_id: str) -> None:
        """Delete the checkpoint for ``thread_id`` (no-op if absent)."""
        db = await self._db()
        await db.execute(
            "DELETE FROM graph_checkpoints WHERE thread_id = ?",
            (thread_id,),
        )
        await db.commit()
        logger.debug("SQLiteCheckpointer.delete: thread_id=%s", thread_id)

    async def close(self) -> None:
        """Close the underlying connection. Idempotent."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> SQLiteCheckpointer:
        """Support ``async with SQLiteCheckpointer(...) as cp:`` usage."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the connection on context-manager exit."""
        await self.close()


__all__ = ["SQLiteCheckpointer"]

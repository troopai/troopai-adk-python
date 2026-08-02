"""``PostgresTaskStore`` — durable, shared A2A task store on PostgreSQL.

The Postgres counterpart of
:class:`~troopai.adk.a2a.task_store.SQLiteTaskStore`: a single
``a2a_tasks`` table reached through a ``psycopg`` async connection pool.
Because every replica talks to the same database, A2A background tasks and
``A2AContinuationToken`` resumption survive across a horizontally-scaled
deployment — which a per-pod SQLite file cannot.

Satisfies the :class:`~troopai.adk.a2a.task_store.TaskStore` protocol
structurally. Wire types (``a2a.types.Task``) are confined to this module
and serialized as JSON, exactly as the SQLite store does.

Optional extras: the ``a2a`` extra (protocol types) plus ``a2a-postgres``
(psycopg). Install with ``pip install 'troopai-adk-python[a2a,a2a-postgres]'``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

try:
    from a2a.types import Task, TaskState, TaskStatus
    from google.protobuf.json_format import MessageToDict, ParseDict
except ImportError as ie:
    if ie.name is not None and ie.name != "a2a" and not ie.name.startswith("a2a."):
        raise
    raise ImportError(
        "Please install the 'a2a' extra to use A2A protocol support. Run: pip install 'troopai-adk-python[a2a]'",
        name="a2a",
    ) from ie

try:
    from psycopg_pool import AsyncConnectionPool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PostgresTaskStore requires psycopg[binary,pool]>=3.2: pip install 'troopai-adk-python[a2a-postgres]'"
    ) from exc

logger = logging.getLogger(__name__)

# Terminal states mirror SQLiteTaskStore — kept local so this module is the
# single boundary that imports a2a.types for the Postgres path.
_TERMINAL_STATES: frozenset[int] = frozenset(
    [
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    ]
)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS a2a_tasks (
    task_id      TEXT PRIMARY KEY,
    status_state INTEGER NOT NULL,
    task_json    TEXT NOT NULL,
    updated_at   DOUBLE PRECISION NOT NULL
)
"""

_CREATE_INDEX = "CREATE INDEX IF NOT EXISTS idx_a2a_tasks_state ON a2a_tasks (status_state)"

_UPSERT = (
    "INSERT INTO a2a_tasks (task_id, status_state, task_json, updated_at) VALUES (%s, %s, %s, %s) "
    "ON CONFLICT (task_id) DO UPDATE SET "
    "status_state = EXCLUDED.status_state, task_json = EXCLUDED.task_json, updated_at = EXCLUDED.updated_at"
)

_DEFAULT_TTL_SECONDS = 86_400
_DEFAULT_MAX_TERMINAL_ROWS = 1_000


class PostgresTaskStore:
    """Durable A2A ``TaskStore`` backed by PostgreSQL via psycopg.

    The caller owns the lifecycle — call :meth:`close` at shutdown. Before
    accepting requests, call :meth:`recover_on_startup` once to mark tasks
    a prior process left non-terminal as FAILED.

    Attributes:
        conninfo: libpq connection string used to open the pool.
    """

    def __init__(
        self,
        conninfo: str,
        *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        max_terminal_rows: int = _DEFAULT_MAX_TERMINAL_ROWS,
    ) -> None:
        """Initialise the store and lazily open the pool on first use.

        Args:
            conninfo: libpq connection string (e.g.
                ``"postgresql://user:pass@host/db"``).
            ttl_seconds: Delete terminal tasks older than this many seconds
                on the next write; pass ``0`` to disable. Default 86 400.
            max_terminal_rows: After the TTL sweep, delete the oldest
                terminal rows beyond this cap; pass ``0`` to disable.
                Default 1 000.

        Raises:
            ValueError: If ``conninfo`` is empty.
        """
        if len(conninfo) == 0:
            raise ValueError("conninfo must be a non-empty libpq connection string.")
        self.conninfo = conninfo
        self._ttl_seconds = ttl_seconds
        self._max_terminal_rows = max_terminal_rows
        self._pool: AsyncConnectionPool | None = None
        self._init_lock = asyncio.Lock()

    async def _get_pool(self) -> AsyncConnectionPool:
        """Open the pool and create the schema on first call.

        The init lock serializes concurrent first callers; a failed schema
        creation closes the half-opened pool before propagating.

        Returns:
            The open connection pool.
        """
        async with self._init_lock:
            if self._pool is None:
                pool: AsyncConnectionPool = AsyncConnectionPool(self.conninfo, open=False)
                await pool.open()
                try:
                    async with pool.connection() as conn:
                        await conn.execute(_CREATE_TABLE)
                        await conn.execute(_CREATE_INDEX)
                except Exception:
                    await pool.close()
                    raise
                self._pool = pool
            return self._pool

    async def close(self) -> None:
        """Close the connection pool. Idempotent and safe in a ``finally``."""
        async with self._init_lock:
            if self._pool is not None:
                pool, self._pool = self._pool, None
                await pool.close()

    async def get(self, task_id: str) -> Task | None:
        """Return the stored task by id, or ``None``.

        Args:
            task_id: The A2A task identifier.

        Returns:
            The deserialized :class:`~a2a.types.Task`, or ``None``.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cursor = await conn.execute("SELECT task_json FROM a2a_tasks WHERE task_id = %s", (task_id,))
            row = await cursor.fetchone()
        if row is None:
            return None
        return _deserialize(row[0])

    async def save(self, task: Task) -> None:
        """Upsert a task by its ``id`` and run the retention sweep.

        The sweep (TTL + max-rows cap on terminal tasks) runs in the same
        transaction to amortize cost across writes without a background task.

        Args:
            task: The task to persist.
        """
        pool = await self._get_pool()
        now = time.time()
        states = list(_TERMINAL_STATES)
        async with pool.connection() as conn:
            await conn.execute(_UPSERT, (task.id, task.status.state, _serialize(task), now))
            if self._ttl_seconds > 0:
                await conn.execute(
                    "DELETE FROM a2a_tasks WHERE status_state = ANY(%s) AND updated_at < %s",
                    (states, now - self._ttl_seconds),
                )
            if self._max_terminal_rows > 0:
                cursor = await conn.execute("SELECT COUNT(*) FROM a2a_tasks WHERE status_state = ANY(%s)", (states,))
                row = await cursor.fetchone()
                count: int = row[0] if row is not None else 0
                excess = count - self._max_terminal_rows
                if excess > 0:
                    await conn.execute(
                        "DELETE FROM a2a_tasks WHERE task_id IN ("
                        "SELECT task_id FROM a2a_tasks WHERE status_state = ANY(%s) ORDER BY updated_at ASC LIMIT %s)",
                        (states, excess),
                    )

    async def delete(self, task_id: str) -> None:
        """Remove a task. No-op if not found.

        Args:
            task_id: The A2A task identifier to remove.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM a2a_tasks WHERE task_id = %s", (task_id,))

    async def list_by_status(self, *, terminal: bool) -> list[Task]:
        """Return tasks filtered by terminal / non-terminal status.

        Args:
            terminal: When ``True`` return COMPLETED/FAILED/CANCELED/REJECTED;
                when ``False`` return the rest.

        Returns:
            The matching tasks in unspecified order.
        """
        states = list(_TERMINAL_STATES)
        clause = "status_state = ANY(%s)" if terminal else "status_state <> ALL(%s)"
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cursor = await conn.execute(f"SELECT task_json FROM a2a_tasks WHERE {clause}", (states,))
            rows = await cursor.fetchall()
        return [_deserialize(row[0]) for row in rows]

    async def recover_on_startup(self) -> int:
        """Mark non-terminal tasks as FAILED before accepting requests.

        A previous process cannot continue tasks it left running, so the
        honest outcome is FAILED; clients may resubmit.

        Returns:
            The number of rows updated.
        """
        non_terminal = await self.list_by_status(terminal=False)
        if len(non_terminal) == 0:
            return 0
        updated = 0
        for task in non_terminal:
            task.status.CopyFrom(TaskStatus(state=TaskState.TASK_STATE_FAILED))
            await self.save(task)
            updated += 1
        logger.warning(
            "Startup recovery: marked %d non-terminal A2A task(s) as FAILED (previous executor did not complete them).",
            updated,
        )
        return updated


def _serialize(task: Task) -> str:
    """Serialize a protobuf Task to a compact JSON string (camelCase)."""
    return json.dumps(MessageToDict(task), separators=(",", ":"))


def _deserialize(task_json: str) -> Task:
    """Deserialize a JSON string back to a protobuf Task."""
    return ParseDict(json.loads(task_json), Task())


__all__ = ["PostgresTaskStore"]

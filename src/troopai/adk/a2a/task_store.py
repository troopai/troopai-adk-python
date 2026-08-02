"""Persistent and in-memory task stores for the A2A executor.

:class:`TaskStore` is the narrow protocol all stores satisfy. Two
implementations ship out of the box:

* :class:`InMemoryTaskStore` — identical to the original in-memory dict
  behavior of ``A2AExecutor``. Zero new dependencies; chosen automatically
  when no ``task_store`` argument is supplied.
* :class:`SQLiteTaskStore` — durable store backed by
  :class:`~troopai.adk.databases.connections.sqlite.SQLiteDatabaseConnection`.
  Enables restart recovery and bounded retention (TTL and/or max-rows).

Wire types (``a2a.types.Task``) are serialized via
``google.protobuf.json_format`` and stored as JSON TEXT. This boundary is
intentional: keeping protobuf inside this module means the rest of the ADK
never imports ``a2a.types``.

Startup recovery
----------------
On construction, :class:`SQLiteTaskStore` (and any caller that uses it)
should call :meth:`SQLiteTaskStore.recover_on_startup` once.  It scans rows
whose status is not a terminal state (COMPLETED / FAILED / CANCELED /
REJECTED) and rewrites them to ``TASK_STATE_FAILED``.

Rationale: the executor that was running those tasks is gone — it cannot
continue them, cannot respond to cancels, and cannot publish further status
events.  Marking them ``FAILED`` (rather than a hypothetical "resumable")
is the semantically honest choice: the *outcome* of the prior execution
attempt is unknown; the only fact we know is that it did not complete.
Clients that want idempotent retry can resubmit. A "resumable" status would
imply the executor can continue from where it left off, which requires
explicit checkpoint/replay logic that this executor does not implement.

Bounded retention (R3)
-----------------------
:class:`SQLiteTaskStore` accepts two optional retention bounds (both
default to conservative values):

* ``ttl_seconds`` — terminal tasks older than this many seconds are swept
  on the next write.  Default: 86 400 s (24 hours).
* ``max_terminal_rows`` — if the terminal-task count exceeds this value
  after a sweep, the oldest rows beyond the cap are deleted.
  Default: 1 000.

Both values can be disabled by passing ``0`` (or ``None``).  The sweep
runs on :meth:`save` to amortize cost across writes without a background
thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Protocol, runtime_checkable

import aiosqlite

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

from troopai.adk.databases.connections.sqlite import SQLiteDatabaseConnection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Terminal-state set — kept in one place so recover_on_startup and
# list_by_status share the same definition.
# ---------------------------------------------------------------------------

_TERMINAL_STATES: frozenset[int] = frozenset(
    [
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    ]
)


def _is_terminal(state: int) -> bool:
    return state in _TERMINAL_STATES


# ---------------------------------------------------------------------------
# TaskStore protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TaskStore(Protocol):
    """Narrow protocol every task-store implementation satisfies.

    All methods are async; implementations decide whether they need I/O.
    Wire type: ``a2a.types.Task`` (protobuf).  Callers never see raw
    JSON or SQL — serialization is an implementation detail.
    """

    async def get(self, task_id: str) -> Task | None:
        """Return the stored :class:`~a2a.types.Task`, or ``None``.

        Args:
            task_id: The A2A task identifier.

        Returns:
            The :class:`~a2a.types.Task` if found, otherwise ``None``.
        """
        ...

    async def save(self, task: Task) -> None:
        """Upsert a :class:`~a2a.types.Task` by its ``id`` field.

        Args:
            task: The task to persist.  The ``id`` field acts as the
                primary key; an existing row with the same id is
                overwritten.
        """
        ...

    async def delete(self, task_id: str) -> None:
        """Remove the task with *task_id*.  No-op if not found.

        Args:
            task_id: The A2A task identifier to remove.
        """
        ...

    async def list_by_status(self, *, terminal: bool) -> list[Task]:
        """Return tasks filtered by terminal / non-terminal status.

        Args:
            terminal: When ``True`` return only tasks whose state is one
                of COMPLETED / FAILED / CANCELED / REJECTED.  When
                ``False`` return the rest (SUBMITTED / WORKING /
                INPUT_REQUIRED / AUTH_REQUIRED / UNSPECIFIED).

        Returns:
            A list of matching :class:`~a2a.types.Task` objects in
            unspecified order.
        """
        ...


# ---------------------------------------------------------------------------
# InMemoryTaskStore — zero-dependency default (mirrors original dict)
# ---------------------------------------------------------------------------


_DEFAULT_MAX_TASKS = 1_000
"""Cost-conservative default cap on the in-memory task dict (R3)."""


class InMemoryTaskStore:
    """In-memory :class:`TaskStore` backed by a plain dict.

    This is the default when no ``task_store`` is passed to
    :class:`~troopai.adk.a2a.executor.A2AExecutor`.  A typed dict wrapper
    with no dependencies beyond the ``a2a`` extra and no I/O cost.

    Bounded retention (R3)
    ----------------------
    The dict is capped at ``max_tasks`` entries (default 1 000).  When a
    save pushes the count over the cap, the least-recently-saved tasks are
    evicted first, so a long-lived process cannot grow the store without
    bound.  Pass ``max_tasks=0`` to disable the bound (unbounded growth —
    only for callers that manage task lifetime themselves).

    Tasks do not survive process restarts with this implementation.
    """

    def __init__(self, *, max_tasks: int = _DEFAULT_MAX_TASKS) -> None:
        """Construct an in-memory task store.

        Args:
            max_tasks: Maximum number of tasks to retain.  When a save
                exceeds this, the least-recently-saved tasks are evicted.
                Default 1 000; pass ``0`` to disable the bound.

        Raises:
            ValueError: If ``max_tasks`` is negative.
        """
        if max_tasks < 0:
            raise ValueError("InMemoryTaskStore.max_tasks MUST be >= 0 (0 disables the bound).")
        self._tasks: dict[str, Task] = {}
        self._max_tasks = max_tasks

    async def get(self, task_id: str) -> Task | None:
        """Return the stored task by id, or ``None``.

        Args:
            task_id: The A2A task identifier.

        Returns:
            The stored :class:`~a2a.types.Task`, or ``None`` if not found.
        """
        return self._tasks.get(task_id)

    async def save(self, task: Task) -> None:
        """Upsert a task by its ``id`` field, evicting the oldest past the cap.

        Args:
            task: The task to store.
        """
        # Re-insert so an updated task moves to the most-recently-saved end
        # of the dict; eviction then removes the genuinely stale
        # (least-recently-saved) entries first.
        self._tasks.pop(task.id, None)
        self._tasks[task.id] = task
        self._evict()

    def _evict(self) -> None:
        """Drop the least-recently-saved tasks beyond ``max_tasks`` (R3)."""
        if self._max_tasks <= 0:
            return
        while len(self._tasks) > self._max_tasks:
            oldest_id = next(iter(self._tasks))
            del self._tasks[oldest_id]

    async def delete(self, task_id: str) -> None:
        """Remove a task.  No-op if not found.

        Args:
            task_id: The A2A task identifier to remove.
        """
        self._tasks.pop(task_id, None)

    async def list_by_status(self, *, terminal: bool) -> list[Task]:
        """Return tasks filtered by terminal / non-terminal status.

        Args:
            terminal: When ``True`` return only terminal tasks.

        Returns:
            A list of matching tasks.
        """
        return [t for t in self._tasks.values() if _is_terminal(t.status.state) == terminal]


# ---------------------------------------------------------------------------
# SQLiteTaskStore — durable, restart-recoverable store
# ---------------------------------------------------------------------------

_DDL = """\
CREATE TABLE IF NOT EXISTS a2a_tasks (
    task_id     TEXT PRIMARY KEY,
    status_state INTEGER NOT NULL,
    task_json   TEXT NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_a2a_tasks_state
    ON a2a_tasks (status_state);
"""

# Default retention bounds — conservative: keep 24 h of terminal tasks,
# cap at 1 000 rows.  Pass 0 to disable either bound.
_DEFAULT_TTL_SECONDS = 86_400
_DEFAULT_MAX_TERMINAL_ROWS = 1_000


class SQLiteTaskStore:
    """Durable :class:`TaskStore` backed by a SQLite database.

    Uses :class:`~troopai.adk.databases.connections.sqlite.SQLiteDatabaseConnection`
    (including its ``0o600`` file-hardening).  Each method opens a
    short-lived connection via the context manager, matching the
    session/memory module pattern.

    Parameterized SQL is used throughout; no string interpolation of
    user-supplied values.

    Serialization: ``a2a.types.Task`` → JSON via
    ``google.protobuf.json_format.MessageToDict`` (camelCase field names,
    which round-trip cleanly through ``ParseDict``).

    Retention
    ---------
    ``ttl_seconds`` and ``max_terminal_rows`` bound how many terminal-task
    rows accumulate.  The sweep runs on every :meth:`save` call.

    Args:
        db: An already-constructed
            :class:`~troopai.adk.databases.connections.sqlite.SQLiteDatabaseConnection`.
            The caller owns its lifetime.
        ttl_seconds: Delete terminal tasks older than this many seconds.
            Pass ``0`` to disable TTL sweeps.  Default: 86 400 (24 h).
        max_terminal_rows: After TTL sweep, if more than this many
            terminal rows remain the oldest are deleted.  Pass ``0`` to
            disable.  Default: 1 000.
    """

    def __init__(
        self,
        db: SQLiteDatabaseConnection,
        *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        max_terminal_rows: int = _DEFAULT_MAX_TERMINAL_ROWS,
    ) -> None:
        self._db = db
        self._ttl_seconds = ttl_seconds
        self._max_terminal_rows = max_terminal_rows
        self._tables_ready = False
        self._init_lock = asyncio.Lock()

    async def _ensure_ready(self) -> None:
        """Create tables if not yet done (lazy, one-shot).

        Double-checked lock: the flag is tested outside the lock first for
        the fast path, then re-tested inside to guard against two coroutines
        racing to initialize at the same time.
        """
        if not self._tables_ready:
            async with self._init_lock:
                if not self._tables_ready:
                    await self._ensure_schema()
                    self._tables_ready = True

    async def _ensure_schema(self) -> None:
        """Create the ``a2a_tasks`` table and index if absent."""
        async with self._db.connect() as conn:
            for stmt in _DDL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    await conn.execute(stmt)
            await conn.commit()

    async def get(self, task_id: str) -> Task | None:
        """Return the stored task by id, or ``None``.

        Args:
            task_id: The A2A task identifier.

        Returns:
            The deserialized :class:`~a2a.types.Task`, or ``None`` if not
            found.
        """
        await self._ensure_ready()
        async with self._db.connect() as conn:
            cursor = await conn.execute(
                "SELECT task_json FROM a2a_tasks WHERE task_id = ?",
                (task_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return _deserialize(row["task_json"])

    async def save(self, task: Task) -> None:
        """Upsert a task and run the retention sweep.

        Args:
            task: The task to persist.  Overwrites any existing row with
                the same ``task_id``.
        """
        await self._ensure_ready()
        now = time.time()
        task_json = _serialize(task)
        async with self._db.connect() as conn:
            await conn.execute(
                """
                INSERT INTO a2a_tasks (task_id, status_state, task_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status_state = excluded.status_state,
                    task_json    = excluded.task_json,
                    updated_at   = excluded.updated_at
                """,
                (task.id, task.status.state, task_json, now),
            )
            await conn.commit()
            await self._sweep(conn, now)
            await conn.commit()

    async def delete(self, task_id: str) -> None:
        """Remove a task.  No-op if not found.

        Args:
            task_id: The A2A task identifier to remove.
        """
        await self._ensure_ready()
        async with self._db.connect() as conn:
            await conn.execute(
                "DELETE FROM a2a_tasks WHERE task_id = ?",
                (task_id,),
            )
            await conn.commit()

    async def list_by_status(self, *, terminal: bool) -> list[Task]:
        """Return tasks filtered by terminal / non-terminal status.

        Args:
            terminal: When ``True`` return only terminal tasks; when
                ``False`` return non-terminal tasks.

        Returns:
            A list of :class:`~a2a.types.Task` objects.
        """
        await self._ensure_ready()
        terminal_states = tuple(_TERMINAL_STATES)
        placeholder = ",".join("?" * len(terminal_states))
        if terminal:
            sql = f"SELECT task_json FROM a2a_tasks WHERE status_state IN ({placeholder})"
            params: tuple[int, ...] = terminal_states
        else:
            sql = f"SELECT task_json FROM a2a_tasks WHERE status_state NOT IN ({placeholder})"
            params = terminal_states
        async with self._db.connect() as conn:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
        return [_deserialize(row["task_json"]) for row in rows]

    async def recover_on_startup(self) -> int:
        """Mark non-terminal tasks as FAILED.

        Must be called once **before** the server begins accepting requests.
        Calling it while ``execute()`` calls are in flight is unsafe: a task
        that completes concurrently may be overwritten as ``FAILED``.

        Any task left in a non-terminal state by a previous process is marked
        ``TASK_STATE_FAILED`` — the previous executor cannot continue it, so
        the honest outcome is failure.  Returns the number of rows updated.

        Returns:
            The count of rows updated.
        """
        await self._ensure_ready()
        non_terminal_tasks = await self.list_by_status(terminal=False)
        if not non_terminal_tasks:
            return 0
        updated = 0
        for task in non_terminal_tasks:
            task.status.CopyFrom(TaskStatus(state=TaskState.TASK_STATE_FAILED))
            await self.save(task)
            updated += 1
        logger.warning(
            "Startup recovery: marked %d non-terminal A2A task(s) as FAILED (previous executor did not complete them).",
            updated,
        )
        return updated

    async def _sweep(self, conn: aiosqlite.Connection, now: float) -> None:
        """Delete terminal tasks exceeding TTL or max-rows cap.

        Runs inside an already-open connection (caller commits).

        Args:
            conn: The open ``aiosqlite.Connection`` to reuse.
            now: Current Unix timestamp (``time.time()``).
        """
        terminal_in = tuple(_TERMINAL_STATES)
        placeholder = ",".join("?" * len(terminal_in))

        if self._ttl_seconds > 0:
            cutoff = now - self._ttl_seconds
            await conn.execute(
                f"DELETE FROM a2a_tasks WHERE status_state IN ({placeholder}) AND updated_at < ?",
                (*terminal_in, cutoff),
            )

        if self._max_terminal_rows > 0:
            cursor = await conn.execute(
                f"SELECT COUNT(*) FROM a2a_tasks WHERE status_state IN ({placeholder})",
                terminal_in,
            )
            row = await cursor.fetchone()
            count: int = row[0] if row else 0
            excess = count - self._max_terminal_rows
            if excess > 0:
                await conn.execute(
                    f"""
                    DELETE FROM a2a_tasks WHERE task_id IN (
                        SELECT task_id FROM a2a_tasks
                        WHERE status_state IN ({placeholder})
                        ORDER BY updated_at ASC
                        LIMIT ?
                    )
                    """,
                    (*terminal_in, excess),
                )


# ---------------------------------------------------------------------------
# Serialization helpers — confined to this module
# ---------------------------------------------------------------------------


def _serialize(task: Task) -> str:
    """Serialize a protobuf Task to a JSON string.

    Uses camelCase field names (MessageToDict default) so the round-trip
    via ParseDict works without extra configuration.

    Args:
        task: The :class:`~a2a.types.Task` to serialize.

    Returns:
        A JSON string representation of the task.
    """
    return json.dumps(MessageToDict(task), separators=(",", ":"))


def _deserialize(task_json: str) -> Task:
    """Deserialize a JSON string back to a protobuf Task.

    Args:
        task_json: JSON string previously produced by :func:`_serialize`.

    Returns:
        A :class:`~a2a.types.Task` with fields populated from the JSON.
    """
    return ParseDict(json.loads(task_json), Task())


__all__ = [
    "InMemoryTaskStore",
    "SQLiteTaskStore",
    "TaskStore",
]

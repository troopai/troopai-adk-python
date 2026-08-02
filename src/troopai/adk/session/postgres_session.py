"""PostgreSQL-backed session implementation — single bound session.

``PostgresSession`` implements the :class:`Session` ABC for one
conversation, backed by an :class:`AsyncConnectionPool` owned by
:class:`~troopai.adk.session.postgres_multi_sessions.PostgresMultiSessions`.
It mirrors :class:`~troopai.adk.session.sqlite_session.SQLiteSession`
so the same agent code runs against a shared Postgres backend for
multi-replica deployments.

This module (and ``postgres_multi_sessions``) are the only place psycopg
is imported; the rest of the session layer stays database-agnostic.
Install with ``pip install 'troopai-adk-python[session-postgres]'``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, override

try:
    from psycopg_pool import AsyncConnectionPool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PostgresSession requires psycopg: pip install 'troopai-adk-python[session-postgres]'",
        name="psycopg",
    ) from exc

from troopai.adk.context.context_editing import ContextEditor
from troopai.adk.exceptions import SessionAppendConflictError
from troopai.adk.session.session import Session
from troopai.adk.session.session_event import SessionEvent
from troopai.adk.session.state import _DELETED, APP_PREFIX

if TYPE_CHECKING:
    from psycopg import AsyncConnection

    from troopai.adk.session.session_settings import SessionSettings
    from troopai.adk.session.state import State

logger = logging.getLogger(__name__)

# Schema — the Postgres port of the SQLite session tables. agent_messages
# carries a BIGSERIAL ``seq`` (Postgres has no rowid) for insertion order;
# app-state is JSONB so per-key merges stay race-free across sessions.
_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS agent_sessions (
        session_id TEXT NOT NULL,
        app_name   TEXT NOT NULL DEFAULT '',
        user_id    TEXT NOT NULL DEFAULT '',
        state      TEXT NOT NULL DEFAULT '{}',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (app_name, user_id, session_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_app_state (
        app_name TEXT PRIMARY KEY,
        state    JSONB NOT NULL DEFAULT '{}'::jsonb
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_messages (
        seq         BIGSERIAL PRIMARY KEY,
        app_name    TEXT NOT NULL DEFAULT '',
        user_id     TEXT NOT NULL DEFAULT '',
        session_id  TEXT NOT NULL,
        event_id    TEXT NOT NULL,
        author      TEXT NOT NULL DEFAULT '',
        timestamp   DOUBLE PRECISION NOT NULL,
        data        TEXT NOT NULL,
        state_delta TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_agent_messages_session
        ON agent_messages (app_name, user_id, session_id, seq)
    """,
)

_SELECT_UPDATED_AT = "SELECT updated_at FROM agent_sessions WHERE app_name=%s AND user_id=%s AND session_id=%s"
_BUMP_UPDATED_AT = "UPDATE agent_sessions SET updated_at=now() WHERE app_name=%s AND user_id=%s AND session_id=%s"
_INSERT_MESSAGE = (
    "INSERT INTO agent_messages "
    "(app_name, user_id, session_id, event_id, author, timestamp, data, state_delta) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
)
_SELECT_EVENTS = (
    "SELECT event_id, author, timestamp, data, state_delta FROM agent_messages "
    "WHERE app_name=%s AND user_id=%s AND session_id=%s"
)


async def ensure_schema(pool: AsyncConnectionPool) -> None:
    """Create the session tables and index if they do not exist.

    Args:
        pool: An open connection pool to run the DDL on.
    """
    async with pool.connection() as conn:
        for statement in _SCHEMA_STATEMENTS:
            await conn.execute(statement)


async def get_app_state(conn: AsyncConnection, app_name: str) -> dict[str, Any]:
    """Load app-scoped state (JSONB) for *app_name*.

    Args:
        conn: An open connection.
        app_name: Application name whose state to load.

    Returns:
        The parsed state dict, or an empty dict when no row exists.
    """
    cursor = await conn.execute("SELECT state FROM agent_app_state WHERE app_name=%s", (app_name,))
    row = await cursor.fetchone()
    if row is None:
        return {}
    value = row[0]
    return value if isinstance(value, dict) else json.loads(value)


async def merge_app_state_delta(conn: AsyncConnection, app_name: str, delta: dict[str, object]) -> None:
    """Merge per-key app-state changes against the live JSONB row.

    Each key is applied individually with ``jsonb_set`` (or the ``-``
    operator for deletions) so two sessions of the same app writing
    disjoint keys cannot clobber each other.

    Args:
        conn: An open connection inside the caller's transaction.
        app_name: Application name whose state to update.
        delta: Full app-scoped keys mapped to new values, or the
            ``_DELETED`` sentinel to drop a key.
    """
    for key, value in delta.items():
        if value is _DELETED:
            await conn.execute("UPDATE agent_app_state SET state = state - %s WHERE app_name=%s", (key, app_name))
        else:
            value_json = json.dumps(value, separators=(",", ":"))
            await conn.execute(
                "INSERT INTO agent_app_state (app_name, state) "
                "VALUES (%s, jsonb_set('{}'::jsonb, ARRAY[%s], %s::jsonb)) "
                "ON CONFLICT (app_name) DO UPDATE "
                "SET state = jsonb_set(agent_app_state.state, ARRAY[%s], %s::jsonb)",
                (app_name, key, value_json, key, value_json),
            )


def _row_to_event(row: tuple[Any, ...]) -> SessionEvent:
    """Convert an ``agent_messages`` row tuple to a :class:`SessionEvent`.

    Args:
        row: ``(event_id, author, timestamp, data, state_delta)``.

    Returns:
        The reconstructed :class:`SessionEvent`.
    """
    event_id, author, timestamp, data, state_delta = row
    return SessionEvent(
        id=event_id,
        author=author,
        content=json.loads(data),
        timestamp=float(timestamp),
        state_delta=json.loads(state_delta) if len(state_delta) > 0 else {},
    )


def _drop_orphaned_tool_result_events(events: list[SessionEvent]) -> list[SessionEvent]:
    """Drop events whose content is a tool result with no in-window tool call.

    A limit slices the history to its most-recent suffix, which can begin in
    the middle of a tool-call / tool-result exchange.  The leading (or any
    parallel-call) tool result then has no matching ``function_call`` in the
    returned window, and Anthropic / Gemini reject such orphans with a 400.

    :meth:`ContextEditor.remove_orphaned_tool_results` operates on Layer-1
    content items and returns the very objects it keeps, so surviving events
    are recovered by content identity.

    Args:
        events: The windowed events, oldest first.

    Returns:
        The events whose content survives orphan removal, order preserved.
    """
    kept = ContextEditor.remove_orphaned_tool_results([event.content for event in events])
    kept_ids = {id(content) for content in kept}
    return [event for event in events if id(event.content) in kept_ids]


class PostgresSession(Session):
    """Bound Postgres session — implements the :class:`Session` ABC.

    Obtained via :class:`PostgresMultiSessions`; shares the manager's
    connection pool. Methods open a short-lived pooled connection (which
    commits on clean exit), mirroring :class:`SQLiteSession`.
    """

    def __init__(
        self,
        session_id: str,
        app_name: str,
        user_id: str,
        pool: AsyncConnectionPool,
        state: State,
        settings: SessionSettings | None = None,
        *,
        strict_concurrency: bool = False,
        updated_at_watermark: str | None = None,
    ) -> None:
        """Initialise a bound session. Use :class:`PostgresMultiSessions` to obtain instances.

        Args:
            session_id: Unique identifier for this session.
            app_name: Application name for multi-tenant scoping.
            user_id: User identifier for multi-tenant scoping.
            pool: Shared connection pool owned by the manager.
            state: Pre-loaded :class:`State` (session data merged with app data).
            settings: Per-session settings, or ``None`` to use defaults.
            strict_concurrency: When ``True``, ``add`` / ``save_state``
                check the row's ``updated_at`` against a watermark and
                raise :exc:`SessionAppendConflictError` on a concurrent
                advance. Best-effort, not atomic. Default ``False``.
            updated_at_watermark: The ``updated_at`` recorded when this
                handle loaded; only consulted when ``strict_concurrency``.
        """
        self._session_id = session_id
        self._app_name = app_name
        self._user_id = user_id
        self._pool = pool
        self._state = state
        self._settings = settings
        self._strict_concurrency = strict_concurrency
        self._updated_at_watermark = updated_at_watermark

    @property
    @override
    def id(self) -> str:
        return self._session_id

    @property
    @override
    def app_name(self) -> str:
        return self._app_name

    @property
    @override
    def user_id(self) -> str:
        return self._user_id

    @property
    @override
    def settings(self) -> SessionSettings | None:
        return self._settings

    @property
    @override
    def state(self) -> State:
        return self._state

    @override
    async def get(self, limit: int | None = None) -> list[SessionEvent]:
        """Retrieve conversation events in chronological order (oldest first).

        Args:
            limit: Maximum number of events; falls back to the session's
                :attr:`SessionSettings.limit`, then to all events.

        Returns:
            List of :class:`SessionEvent` oldest-first.
        """
        effective_limit = limit
        if effective_limit is None and self._settings is not None:
            effective_limit = self._settings.limit
        async with self._pool.connection() as conn:
            if effective_limit is not None:
                cursor = await conn.execute(
                    f"{_SELECT_EVENTS} ORDER BY seq DESC LIMIT %s",
                    (self._app_name, self._user_id, self._session_id, effective_limit),
                )
                rows = list(reversed(await cursor.fetchall()))
            else:
                cursor = await conn.execute(
                    f"{_SELECT_EVENTS} ORDER BY seq ASC",
                    (self._app_name, self._user_id, self._session_id),
                )
                rows = list(await cursor.fetchall())
        events = [_row_to_event(row) for row in rows]
        if effective_limit is not None:
            events = _drop_orphaned_tool_result_events(events)
        return events

    @override
    async def add(self, events: list[SessionEvent]) -> None:
        """Append events to the session, bumping ``updated_at``.

        Args:
            events: Events to insert; an empty list is a no-op.

        Raises:
            SessionAppendConflictError: When ``strict_concurrency`` and a
                concurrent writer advanced ``updated_at``.
        """
        if len(events) == 0:
            return
        params = [
            (
                self._app_name,
                self._user_id,
                self._session_id,
                event.id,
                event.author,
                event.timestamp,
                json.dumps(event.content, separators=(",", ":")),
                json.dumps(event.state_delta, separators=(",", ":")),
            )
            for event in events
        ]
        async with self._pool.connection() as conn:
            await self._check_watermark(conn)
            async with conn.cursor() as cursor:
                await cursor.executemany(_INSERT_MESSAGE, params)
            await conn.execute(_BUMP_UPDATED_AT, (self._app_name, self._user_id, self._session_id))
            await self._refresh_watermark(conn)

    @override
    async def pop_last(self) -> SessionEvent | None:
        """Remove and return the most recent event, or ``None`` if empty."""
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM agent_messages WHERE seq = ("
                "  SELECT seq FROM agent_messages "
                "  WHERE app_name=%s AND user_id=%s AND session_id=%s ORDER BY seq DESC LIMIT 1"
                ") RETURNING event_id, author, timestamp, data, state_delta",
                (self._app_name, self._user_id, self._session_id),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            await self._touch_updated_at(conn)
        return _row_to_event(row)

    @override
    async def clear(self) -> None:
        """Delete all events for this session (bumps ``updated_at`` if any removed)."""
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM agent_messages WHERE app_name=%s AND user_id=%s AND session_id=%s",
                (self._app_name, self._user_id, self._session_id),
            )
            if cursor.rowcount > 0:
                await self._touch_updated_at(conn)

    @override
    async def save_state(self) -> None:
        """Persist pending session-scoped state and merge app-scoped changes.

        Raises:
            SessionAppendConflictError: When ``strict_concurrency`` and a
                concurrent writer advanced ``updated_at``.
            RuntimeError: When the session row is missing (delete race), so
                the in-memory delta is preserved rather than silently lost.
        """
        if not self._state.has_changes():
            return
        app_delta = {k: v for k, v in self._state.pending_delta().items() if k.startswith(APP_PREFIX)}
        state_json = json.dumps(self._state.to_persist(), separators=(",", ":"))
        async with self._pool.connection() as conn:
            await self._check_watermark(conn)
            cursor = await conn.execute(
                "UPDATE agent_sessions SET state=%s, updated_at=now() "
                "WHERE app_name=%s AND user_id=%s AND session_id=%s",
                (state_json, self._app_name, self._user_id, self._session_id),
            )
            if cursor.rowcount == 0:
                raise RuntimeError(
                    f"save_state: no session row for (app={self._app_name!r}, user={self._user_id!r}, "
                    f"session={self._session_id!r}); state was NOT persisted (the session may have been deleted)."
                )
            await self._refresh_watermark(conn)
            if len(app_delta) > 0:
                await merge_app_state_delta(conn, self._app_name, app_delta)
        self._state.commit()
        logger.debug("Saved state for session %s", self._session_id)

    async def _check_watermark(self, conn: AsyncConnection) -> None:
        """Raise if a strict handle's watermark is stale (no-op when not strict)."""
        if not self._strict_concurrency:
            return
        cursor = await conn.execute(_SELECT_UPDATED_AT, (self._app_name, self._user_id, self._session_id))
        row = await cursor.fetchone()
        current = str(row[0]) if row is not None else None
        if self._updated_at_watermark is not None and current is not None and current != self._updated_at_watermark:
            raise SessionAppendConflictError(self._session_id)

    async def _refresh_watermark(self, conn: AsyncConnection) -> None:
        """Refresh a strict handle's watermark to the row's current ``updated_at``."""
        if not self._strict_concurrency:
            return
        cursor = await conn.execute(_SELECT_UPDATED_AT, (self._app_name, self._user_id, self._session_id))
        row = await cursor.fetchone()
        if row is not None:
            self._updated_at_watermark = str(row[0])

    async def _touch_updated_at(self, conn: AsyncConnection) -> None:
        """Bump ``updated_at`` after a destructive change and refresh the watermark."""
        await conn.execute(_BUMP_UPDATED_AT, (self._app_name, self._user_id, self._session_id))
        await self._refresh_watermark(conn)


__all__ = ["PostgresSession", "ensure_schema", "get_app_state", "merge_app_state_delta"]

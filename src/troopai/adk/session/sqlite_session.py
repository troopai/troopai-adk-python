"""SQLite-backed session implementation — single bound session.

``SQLiteSession`` implements the :class:`Session` ABC for one conversation.
Uses ``aiosqlite`` for truly async database access.  Not constructed
directly — obtained via :class:`SQLiteMultiSessions`.

Also defines the table schemas and connection helpers shared by both
``SQLiteSession`` and ``SQLiteMultiSessions``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, override

import aiosqlite

from troopai.adk.context.context_editing import ContextEditor
from troopai.adk.databases import SQLiteDatabaseConnection
from troopai.adk.exceptions import SessionAppendConflictError
from troopai.adk.session.session import Session
from troopai.adk.session.session_event import SessionEvent
from troopai.adk.session.session_settings import SessionSettings
from troopai.adk.session.state import _DELETED, APP_PREFIX, State

logger = logging.getLogger(__name__)

# =====================================================================
# Table names and schemas — single source of truth for the storage layer
# =====================================================================

DEFAULT_SESSIONS_TABLE = "agent_sessions"
DEFAULT_MESSAGES_TABLE = "agent_messages"
DEFAULT_APP_STATE_TABLE = "agent_app_state"

SESSIONS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS {sessions_table} (
    session_id TEXT NOT NULL,
    app_name TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT '{{}}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (app_name, user_id, session_id)
);
"""

APP_STATE_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS {app_state_table} (
    app_name TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT '{{}}'
);
"""

MESSAGES_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS {messages_table} (
    app_name TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT '',
    timestamp REAL NOT NULL,
    data TEXT NOT NULL,
    state_delta TEXT NOT NULL DEFAULT '{{}}'
);
CREATE INDEX IF NOT EXISTS idx_{messages_table}_session
    ON {messages_table}(app_name, user_id, session_id);
"""


def build_create_schema_sql() -> str:
    """Build the combined DDL for all session tables.

    Returns:
        A single SQL string containing ``CREATE TABLE IF NOT EXISTS``
        statements for the sessions, app-state, and messages tables.
    """
    return "".join(
        [
            SESSIONS_TABLE_SCHEMA.format(sessions_table=DEFAULT_SESSIONS_TABLE),
            APP_STATE_TABLE_SCHEMA.format(app_state_table=DEFAULT_APP_STATE_TABLE),
            MESSAGES_TABLE_SCHEMA.format(
                messages_table=DEFAULT_MESSAGES_TABLE,
                sessions_table=DEFAULT_SESSIONS_TABLE,
            ),
        ]
    )


# =====================================================================
# Async DB connection helpers
# =====================================================================


async def ensure_tables(db_conn: SQLiteDatabaseConnection) -> None:
    """Create session tables if they don't exist.

    Args:
        db_conn: Open database connection used to execute the DDL.
    """
    async with db_conn.connect() as db:
        await db.executescript(build_create_schema_sql())
        await db.commit()


# =====================================================================
# App state helpers (used by SQLiteSession and SQLiteMultiSessions)
# =====================================================================


async def get_session_updated_at(
    db: aiosqlite.Connection,
    app_name: str,
    user_id: str,
    session_id: str,
) -> str | None:
    """Return the ``updated_at`` value for a session row, or ``None`` if absent.

    Args:
        db: Active ``aiosqlite`` connection.
        app_name: Application name for scoping the lookup.
        user_id: User identifier for scoping the lookup.
        session_id: Session identifier whose timestamp to fetch.

    Returns:
        The ``updated_at`` string from the sessions table, or ``None``
        when the session row does not exist.
    """
    cursor = await db.execute(
        f"SELECT updated_at FROM {DEFAULT_SESSIONS_TABLE} "
        f"WHERE app_name = :app AND user_id = :uid AND session_id = :sid",
        {"app": app_name, "uid": user_id, "sid": session_id},
    )
    row = await cursor.fetchone()
    return str(row["updated_at"]) if row is not None else None


async def get_app_state(db: aiosqlite.Connection, app_name: str) -> dict[str, Any]:
    """Load app-scoped state from the app_state table.

    Args:
        db: Active ``aiosqlite`` connection.
        app_name: Application name whose state to load.

    Returns:
        Parsed state dict, or an empty dict if no row exists for
        *app_name*.
    """
    cursor = await db.execute(
        f"SELECT state FROM {DEFAULT_APP_STATE_TABLE} WHERE app_name = :app",
        {"app": app_name},
    )
    row = await cursor.fetchone()
    return json.loads(row["state"]) if row is not None else {}


async def merge_app_state_delta(
    db: aiosqlite.Connection,
    app_name: str,
    delta: dict[str, object],
) -> None:
    """Merge per-key changes into the shared app-state row at the SQL level.

    Each key in *delta* is applied with ``json_set`` (or ``json_remove``
    for the ``_DELETED`` sentinel) directly against the stored row, so the
    merge never round-trips the full dict through Python.  This closes the
    read-modify-write window where two sessions of the same app could read
    the same snapshot and clobber each other's disjoint keys: every change
    reads the live row value at execution time inside the caller's
    transaction rather than overwriting a stale in-memory copy.

    Args:
        db: Active ``aiosqlite`` connection.  The caller owns the
            surrounding transaction and is responsible for committing.
        app_name: Application name whose state to update.
        delta: Mapping of full app-scoped keys (e.g. ``"app:theme"``) to
            their new values, or the ``_DELETED`` sentinel to drop a key.
    """
    for key, value in delta.items():
        if value is _DELETED:
            await db.execute(
                f"UPDATE {DEFAULT_APP_STATE_TABLE} "
                f"SET state = json_remove(state, '$.' || json_quote(:key)) "
                f"WHERE app_name = :app",
                {"app": app_name, "key": key},
            )
        else:
            value_json = json.dumps(value, separators=(",", ":"))
            await db.execute(
                f"INSERT INTO {DEFAULT_APP_STATE_TABLE} (app_name, state) "
                f"VALUES (:app, json_set('{{}}', '$.' || json_quote(:key), json(:val))) "
                f"ON CONFLICT(app_name) DO UPDATE SET "
                f"state = json_set(state, '$.' || json_quote(:key), json(:val))",
                {"app": app_name, "key": key, "val": value_json},
            )


# =====================================================================
# SQLiteSession — bound session implementation
# =====================================================================


class SQLiteSession(Session):
    """Bound session — implements the ``Session`` ABC.

    Obtained via :meth:`SQLiteMultiSessions.create`,
    :meth:`SQLiteMultiSessions.get`, or
    :meth:`SQLiteMultiSessions.get_or_create`.

    Uses ``aiosqlite`` for truly async database access.
    """

    def __init__(
        self,
        session_id: str,
        app_name: str,
        user_id: str,
        db: SQLiteDatabaseConnection,
        state: State,
        settings: SessionSettings | None = None,
        *,
        strict_concurrency: bool = False,
        updated_at_watermark: str | None = None,
    ) -> None:
        """Initialise a bound session.  Use :class:`SQLiteMultiSessions` to obtain instances.

        Args:
            session_id: Unique identifier for this session.
            app_name: Application name for multi-tenant scoping.
            user_id: User identifier for multi-tenant scoping.
            db: Shared database connection managed by the
                :class:`SQLiteMultiSessions` that created this session.
            state: Pre-loaded :class:`State` for this session
                (session-scoped data merged with app-scoped data).
            settings: Per-session settings, or ``None`` to use defaults.
            strict_concurrency: When ``True``, :meth:`add` and
                :meth:`save_state` check the
                session row's ``updated_at`` timestamp before inserting.
                If another writer has advanced it since this handle last
                loaded or appended, :exc:`SessionAppendConflictError` is
                raised.  Default ``False`` preserves the unconditional
                append behaviour.  This guard is best-effort, not atomic:
                the check and insert are separated by asyncio yield points
                so a concurrent writer can still slip through.  Use an
                application-level mutex for a hard guarantee.
            updated_at_watermark: The ``updated_at`` value recorded when
                this handle was loaded or last successfully appended.
                Only consulted when ``strict_concurrency=True``; ignored
                otherwise.
        """
        self._session_id = session_id
        self._app_name = app_name
        self._user_id = user_id
        self._db = db
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

        When a limit applies (argument or session settings), returns the N
        most-recent events, ordered oldest-first within that slice.

        Args:
            limit: Maximum number of events to retrieve.  If ``None``,
                falls back to the session's :attr:`SessionSettings.limit`,
                then retrieves all events.

        Returns:
            List of :class:`SessionEvent` in chronological order
            (oldest first).
        """
        effective_limit = limit
        if effective_limit is None and self._settings is not None:
            effective_limit = self._settings.limit

        async with self._db.connect() as db:
            if effective_limit is not None:
                cursor = await db.execute(
                    f"SELECT event_id, author, timestamp, data, state_delta "
                    f"FROM {DEFAULT_MESSAGES_TABLE} "
                    f"WHERE app_name = :app AND user_id = :uid AND session_id = :sid "
                    f"ORDER BY rowid DESC LIMIT :lim",
                    {"app": self._app_name, "uid": self._user_id, "sid": self._session_id, "lim": effective_limit},
                )
                fetched = await cursor.fetchall()
                rows = list(reversed(list(fetched)))
            else:
                cursor = await db.execute(
                    f"SELECT event_id, author, timestamp, data, state_delta "
                    f"FROM {DEFAULT_MESSAGES_TABLE} "
                    f"WHERE app_name = :app AND user_id = :uid AND session_id = :sid "
                    f"ORDER BY rowid ASC",
                    {"app": self._app_name, "uid": self._user_id, "sid": self._session_id},
                )
                rows = list(await cursor.fetchall())

        events = [_row_to_event(row) for row in rows]
        if effective_limit is not None:
            events = _drop_orphaned_tool_result_events(events)
        return events

    @override
    async def add(self, events: list[SessionEvent]) -> None:
        """Append events to the session.

        When ``strict_concurrency=True`` (opt-in), the session row's
        ``updated_at`` timestamp is checked before inserting.  If
        another writer has advanced it since this handle last loaded or
        appended, :exc:`~troopai.adk.exceptions.SessionAppendConflictError`
        is raised.  On success the watermark is refreshed so subsequent
        :meth:`add` calls on the same handle continue to work.

        .. note:: **Best-effort, not atomic.**  The timestamp check (SELECT)
            and the row insert (INSERT) are separated by asyncio yield points.
            A concurrent writer can interleave between them, so this guard
            narrows the conflict window significantly but does not close it
            entirely.  For a hard serialisation guarantee use an
            application-level mutex or a single-writer pattern.

        Args:
            events: Events to insert into the messages table.  An
                empty list is a no-op.

        Raises:
            SessionAppendConflictError: When ``strict_concurrency=True``
                and a concurrent writer advanced ``updated_at`` since
                this handle last loaded or appended.
        """
        if len(events) == 0:
            return

        async with self._db.connect() as db:
            if self._strict_concurrency:
                cursor = await db.execute(
                    f"SELECT updated_at FROM {DEFAULT_SESSIONS_TABLE} "
                    f"WHERE app_name = :app AND user_id = :uid AND session_id = :sid",
                    {"app": self._app_name, "uid": self._user_id, "sid": self._session_id},
                )
                row = await cursor.fetchone()
                current_updated_at = str(row["updated_at"]) if row is not None else None
                if (
                    self._updated_at_watermark is not None
                    and current_updated_at is not None
                    and current_updated_at != self._updated_at_watermark
                ):
                    raise SessionAppendConflictError(self._session_id)

            await db.executemany(
                f"INSERT INTO {DEFAULT_MESSAGES_TABLE} "
                f"(app_name, user_id, session_id, event_id, author, timestamp, data, state_delta) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
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
                ],
            )
            # Use sub-second precision so strict-concurrency handles can
            # reliably detect appends that happen within the same wall-clock
            # second.  CURRENT_TIMESTAMP has 1-second granularity; the
            # strftime variant carries milliseconds and is still a valid
            # ISO-8601 TEXT value.
            await db.execute(
                f"UPDATE {DEFAULT_SESSIONS_TABLE} "
                f"SET updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now') "
                f"WHERE app_name = :app AND user_id = :uid AND session_id = :sid",
                {"app": self._app_name, "uid": self._user_id, "sid": self._session_id},
            )
            await db.commit()

            if self._strict_concurrency:
                # Refresh watermark so this handle can append again.
                cursor2 = await db.execute(
                    f"SELECT updated_at FROM {DEFAULT_SESSIONS_TABLE} "
                    f"WHERE app_name = :app AND user_id = :uid AND session_id = :sid",
                    {"app": self._app_name, "uid": self._user_id, "sid": self._session_id},
                )
                refresh_row = await cursor2.fetchone()
                if refresh_row is not None:
                    self._updated_at_watermark = str(refresh_row["updated_at"])

    @override
    async def pop_last(self) -> SessionEvent | None:
        """Remove and return the most recent event, or ``None`` if empty.

        Uses a single ``DELETE … RETURNING`` statement (SQLite ≥ 3.35) so
        the select and delete are atomic — two concurrent callers cannot
        both receive the same event.
        """
        async with self._db.connect() as db:
            cursor = await db.execute(
                f"DELETE FROM {DEFAULT_MESSAGES_TABLE} "
                f"WHERE rowid = ("
                f"  SELECT rowid FROM {DEFAULT_MESSAGES_TABLE} "
                f"  WHERE app_name = :app AND user_id = :uid AND session_id = :sid "
                f"  ORDER BY rowid DESC LIMIT 1"
                f") RETURNING event_id, author, timestamp, data, state_delta",
                {"app": self._app_name, "uid": self._user_id, "sid": self._session_id},
            )
            row = await cursor.fetchone()
            if row is None:
                return None

            # A pop mutates the session, so bump updated_at to keep
            # SessionInfo.updated_at honest as a "last modified" marker
            # (add()/save_state() do the same). Refresh a strict handle's
            # watermark to its own write so a later add() on this handle
            # does not false-positive on the bump it just caused.
            await self._touch_updated_at(db)
            await db.commit()

        return SessionEvent(
            id=row["event_id"],
            author=row["author"],
            content=json.loads(row["data"]),
            timestamp=float(row["timestamp"]),
            state_delta=json.loads(row["state_delta"]) if row["state_delta"] else {},
        )

    @override
    async def clear(self) -> None:
        """Delete all events for this session."""
        async with self._db.connect() as db:
            cursor = await db.execute(
                f"DELETE FROM {DEFAULT_MESSAGES_TABLE} WHERE app_name = :app AND user_id = :uid AND session_id = :sid",
                {"app": self._app_name, "uid": self._user_id, "sid": self._session_id},
            )
            # Only bump updated_at when events were actually removed, so a
            # clear() on an already-empty session does not lie about a
            # "last modified" time it did not change.
            if cursor.rowcount > 0:
                await self._touch_updated_at(db)
            await db.commit()

    async def _touch_updated_at(self, db: aiosqlite.Connection) -> None:
        """Bump the session row's ``updated_at`` after a destructive change.

        Uses the same sub-second ``strftime`` format as :meth:`add` and
        :meth:`save_state` so a strict-concurrency handle's watermark stays
        comparable across operation types.  When ``strict_concurrency`` is
        enabled the handle's watermark is refreshed to the value this write
        produced, so a subsequent :meth:`add` on the same handle does not
        false-positive on the bump it just caused.

        Args:
            db: Active ``aiosqlite`` connection inside the caller's
                transaction.  The caller is responsible for committing.
        """
        await db.execute(
            f"UPDATE {DEFAULT_SESSIONS_TABLE} "
            f"SET updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now') "
            f"WHERE app_name = :app AND user_id = :uid AND session_id = :sid",
            {"app": self._app_name, "uid": self._user_id, "sid": self._session_id},
        )
        if self._strict_concurrency:
            cursor = await db.execute(
                f"SELECT updated_at FROM {DEFAULT_SESSIONS_TABLE} "
                f"WHERE app_name = :app AND user_id = :uid AND session_id = :sid",
                {"app": self._app_name, "uid": self._user_id, "sid": self._session_id},
            )
            row = await cursor.fetchone()
            if row is not None:
                self._updated_at_watermark = str(row["updated_at"])

    @override
    async def save_state(self) -> None:
        """Persist pending state changes to the database."""
        if not self._state.has_changes():
            return

        # pending_delta() preserves the _DELETED sentinel (unlike delta(),
        # which converts it to None) so "delete this key" stays distinct from
        # "store None under this key" when persisting app-scoped changes.
        app_delta: dict[str, object] = {
            k: v for k, v in self._state.pending_delta().items() if k.startswith(APP_PREFIX)
        }

        async with self._db.connect() as db:
            # A strict handle guards every row write, not just appends —
            # save_state() also bumps updated_at, so it must honour and
            # then refresh the watermark or the next add() on this handle
            # would false-positive on its own state write.
            if self._strict_concurrency:
                check_cursor = await db.execute(
                    f"SELECT updated_at FROM {DEFAULT_SESSIONS_TABLE} "
                    f"WHERE app_name = :app AND user_id = :uid AND session_id = :sid",
                    {"app": self._app_name, "uid": self._user_id, "sid": self._session_id},
                )
                check_row = await check_cursor.fetchone()
                current_updated_at = str(check_row["updated_at"]) if check_row is not None else None
                if (
                    self._updated_at_watermark is not None
                    and current_updated_at is not None
                    and current_updated_at != self._updated_at_watermark
                ):
                    raise SessionAppendConflictError(self._session_id)

            state_json = json.dumps(self._state.to_persist(), separators=(",", ":"))
            cursor = await db.execute(
                f"UPDATE {DEFAULT_SESSIONS_TABLE} "
                f"SET state = :state, updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now') "
                f"WHERE app_name = :app AND user_id = :uid AND session_id = :sid",
                {"state": state_json, "app": self._app_name, "uid": self._user_id, "sid": self._session_id},
            )
            # A missing session row makes the UPDATE a 0-row no-op; without
            # this guard save_state would still reach _state.commit() below
            # and mark the delta persisted, silently losing it. Raise so the
            # loss surfaces and the in-memory delta is preserved.
            if cursor.rowcount == 0:
                raise RuntimeError(
                    f"save_state: no session row for (app={self._app_name!r}, "
                    f"user={self._user_id!r}, session={self._session_id!r}); "
                    f"state was NOT persisted (the session may have been deleted)."
                )

            if self._strict_concurrency:
                # Refresh the watermark to the value this write produced so
                # the handle's next write does not trip over its own bump.
                refresh_cursor = await db.execute(
                    f"SELECT updated_at FROM {DEFAULT_SESSIONS_TABLE} "
                    f"WHERE app_name = :app AND user_id = :uid AND session_id = :sid",
                    {"app": self._app_name, "uid": self._user_id, "sid": self._session_id},
                )
                refresh_row = await refresh_cursor.fetchone()
                if refresh_row is not None:
                    self._updated_at_watermark = str(refresh_row["updated_at"])

            if len(app_delta) > 0:
                # Per-key SQL merge instead of read-modify-write: each key is
                # applied against the live row inside this transaction, so two
                # sessions of the same app writing disjoint keys cannot clobber
                # each other (the prior full-dict upsert lost the slower
                # writer's keys when both read the same snapshot).
                await merge_app_state_delta(db, self._app_name, app_delta)

            await db.commit()

        self._state.commit()
        logger.debug("Saved state for session %s", self._session_id)

    @override
    async def close(self) -> None:
        """No-op — the manager owns the connection lifecycle."""


# =====================================================================
# Helpers
# =====================================================================


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


def _row_to_event(row: aiosqlite.Row) -> SessionEvent:
    """Convert a database row to a SessionEvent.

    Args:
        row: A row fetched from the messages table, with columns
            ``event_id``, ``author``, ``timestamp``, ``data``, and
            ``state_delta``.

    Returns:
        A :class:`SessionEvent` populated from the row's columns.
    """
    return SessionEvent(
        id=row["event_id"],
        author=row["author"],
        content=json.loads(row["data"]),
        timestamp=float(row["timestamp"]),
        state_delta=json.loads(row["state_delta"]) if row["state_delta"] else {},
    )

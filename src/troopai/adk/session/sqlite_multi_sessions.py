"""SQLite-backed session manager — manages multiple sessions.

``SQLiteMultiSessions`` is the manager/store.  It opens a database and
provides methods to create, retrieve, list, and delete sessions.  Each
returned session is a :class:`SQLiteSession` that the Runner can use.

Uses ``aiosqlite`` for truly async database access.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import aiosqlite

from troopai.adk.databases import SQLiteDatabaseConnection
from troopai.adk.session.multi_sessions import MultiSessions
from troopai.adk.session.session import Session
from troopai.adk.session.session_settings import SessionSettings
from troopai.adk.session.sqlite_session import (
    DEFAULT_MESSAGES_TABLE,
    DEFAULT_SESSIONS_TABLE,
    SQLiteSession,
    ensure_tables,
    get_app_state,
    merge_app_state_delta,
)
from troopai.adk.session.state import APP_PREFIX, State

logger = logging.getLogger(__name__)


# =====================================================================
# SessionInfo — lightweight metadata returned by list()
# =====================================================================


@dataclass(frozen=True)
class SessionInfo:
    """Metadata for a session in the database.

    Returned by :meth:`SQLiteMultiSessions.list`.

    Attributes:
        session_id: The unique session identifier.
        app_name: Application name.
        user_id: User identifier.
        created_at: When the session was created.
        updated_at: When the session was last modified.
    """

    session_id: str
    """The unique session identifier."""

    app_name: str
    """Application name."""

    user_id: str
    """User identifier."""

    created_at: str
    """When the session was created."""

    updated_at: str
    """When the session was last modified."""


# =====================================================================
# Private helpers
# =====================================================================


def _split_app_initial_state(
    initial_state: dict[str, Any],
) -> tuple[dict[str, object], dict[str, Any]]:
    """Split an initial-state dict into its app-scoped and session-scoped tiers.

    Args:
        initial_state: The developer-supplied initial state, which may mix
            ``app:``-prefixed (shared) keys with plain session keys.

    Returns:
        A ``(app_initial, session_initial)`` pair: the app-scoped keys destined
        for the app-state table, and the remaining keys for the session column.
    """
    app_initial: dict[str, object] = {k: v for k, v in initial_state.items() if k.startswith(APP_PREFIX)}
    session_initial: dict[str, Any] = {k: v for k, v in initial_state.items() if not k.startswith(APP_PREFIX)}
    return app_initial, session_initial


async def _session_exists(
    db: aiosqlite.Connection,
    app_name: str,
    user_id: str,
    session_id: str,
) -> bool:
    """Check whether a session exists in the database.

    Args:
        db: Active ``aiosqlite`` connection.
        app_name: Application name for scoping the lookup.
        user_id: User identifier for scoping the lookup.
        session_id: Session identifier to check.

    Returns:
        ``True`` if the session exists, ``False`` otherwise.
    """
    cursor = await db.execute(
        f"SELECT 1 FROM {DEFAULT_SESSIONS_TABLE} WHERE app_name = :app AND user_id = :uid AND session_id = :sid",
        {"app": app_name, "uid": user_id, "sid": session_id},
    )
    return await cursor.fetchone() is not None


async def _get_session_state(
    db: aiosqlite.Connection,
    app_name: str,
    user_id: str,
    session_id: str,
) -> dict[str, Any]:
    """Load session-scoped state from the sessions table.

    Args:
        db: Active ``aiosqlite`` connection.
        app_name: Application name for scoping the lookup.
        user_id: User identifier for scoping the lookup.
        session_id: Session whose state to load.

    Returns:
        Parsed state dict, or an empty dict if the session row is
        not found.
    """
    cursor = await db.execute(
        f"SELECT state FROM {DEFAULT_SESSIONS_TABLE} WHERE app_name = :app AND user_id = :uid AND session_id = :sid",
        {"app": app_name, "uid": user_id, "sid": session_id},
    )
    row = await cursor.fetchone()
    return json.loads(row["state"]) if row is not None else {}


def _is_migration_needed(db_path: str) -> bool:
    """Check whether the sessions table is missing required columns.

    Returns ``True`` when the ``sessions`` table exists but lacks the
    ``app_name`` column (and thus the multi-tenant columns this manager
    requires). ``False`` for a fresh or absent database.

    Args:
        db_path: Filesystem path to the SQLite database file, or
            ``":memory:"``.

    Returns:
        ``True`` if the table lacks the required columns, ``False``
        otherwise (including if the database does not exist yet).
    """
    path = Path(db_path)
    if db_path == ":memory:" or not path.exists():
        return False

    try:
        # ``sqlite3.Connection.__exit__`` only commits/rolls back — it does
        # NOT close the connection.  Wrap in ``contextlib.closing`` so the
        # file handle is released immediately after inspection rather than
        # lingering until GC (which can block file deletion on Windows).
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (DEFAULT_SESSIONS_TABLE,),
            )
            if cursor.fetchone() is None:
                return False  # No tables yet — fresh DB

            cursor = conn.execute(f"PRAGMA table_info({DEFAULT_SESSIONS_TABLE})")
            columns = {row[1] for row in cursor.fetchall()}

        # The multi-tenant columns (app_name, user_id, state, updated_at) are
        # required; their absence is detected via the app_name column.
        return "app_name" not in columns

    except sqlite3.Error:
        logger.warning("_is_migration_needed: failed to inspect '%s'; assuming no migration needed", db_path)
        return False


# =====================================================================
# SQLiteMultiSessions — manager for sessions in one SQLite database
# =====================================================================


class SQLiteMultiSessions(MultiSessions):
    """Manager for SQLite-backed sessions.

    Opens a database and provides methods to create, retrieve, list,
    and delete sessions.  Each session is a :class:`Session` instance
    that the Runner can use directly.

    Uses ``aiosqlite`` for truly async database access — no locks,
    no blocking the event loop.

    Args:
        path: Path to the SQLite database file, or ``":memory:"`` for
            in-memory storage.
        app_name: Application name for multi-tenant scoping.
        settings: Default :class:`SessionSettings` applied when
            ``create()`` or ``get_or_create()`` are called without
            explicit settings.

    Raises:
        RuntimeError: If the database exists but its ``sessions`` table is
            missing the required columns (``app_name``, ``user_id``,
            ``state``, ``updated_at``).

    Example::

        # Manager — create, get, list, delete sessions
        sessions = SQLiteMultiSessions(path="sessions.db", app_name="myapp")
        session = await sessions.create("conv-001", user_id="user-1")

        # Session — read/write conversation history (Session ABC)
        await session.add([...])
        history = await session.get()

        # Back to manager — collection operations
        all_sessions = await sessions.list(user_id="user-1")
        await sessions.delete("conv-old", user_id="user-1")
        await sessions.close()
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        app_name: str = "",
        settings: SessionSettings | None = None,
    ) -> None:
        self._db_path = str(path)
        self._app_name = app_name
        self._settings = settings

        if _is_migration_needed(self._db_path):
            raise RuntimeError(
                f"Database '{self._db_path}' is missing required columns in the "
                f"'{DEFAULT_SESSIONS_TABLE}' table ('app_name', 'user_id', 'state', "
                f"'updated_at'). Use a fresh database path, or drop and recreate the table."
            )

        self._db = SQLiteDatabaseConnection(path)
        self._tables_ready = False
        self._init_lock = asyncio.Lock()

    @property
    def app_name(self) -> str:
        """The application name for this manager."""
        return self._app_name

    async def _ensure_ready(self) -> None:
        """Create tables if not yet done (lazy, one-shot)."""
        if not self._tables_ready:
            async with self._init_lock:
                if not self._tables_ready:
                    await ensure_tables(self._db)
                    self._tables_ready = True

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    @override
    async def create(
        self,
        session_id: str,
        user_id: str = "",
        state: dict[str, Any] | None = None,
        settings: SessionSettings | None = None,
    ) -> Session:
        """Create a new session.

        Args:
            session_id: Unique identifier for the session.
            user_id: User identifier for multi-tenant scoping.
            state: Optional initial state dict.
            settings: Per-session settings.  Falls back to the manager's
                default settings if ``None``.

        Returns:
            A :class:`Session` instance bound to the new session.

        Raises:
            ValueError: If a session with this ID already exists
                for the given app_name + user_id.
        """
        await self._ensure_ready()
        initial_state = state or {}
        # App-scoped keys are owned by the app-state table, not the session
        # column: split them out so create() routes each tier where it lives,
        # mirroring save_state(). Otherwise app: keys sit only in the session
        # column, are dropped by to_persist() on the first save, and are never
        # shared with sibling sessions.
        app_initial, session_initial = _split_app_initial_state(initial_state)
        state_json = json.dumps(session_initial, separators=(",", ":"))

        async with self._db.connect() as db:
            try:
                cursor = await db.execute(
                    f"INSERT INTO {DEFAULT_SESSIONS_TABLE} "
                    f"(session_id, app_name, user_id, state) "
                    f"VALUES (:sid, :app, :uid, :state)",
                    {"sid": session_id, "app": self._app_name, "uid": user_id, "state": state_json},
                )
            except aiosqlite.IntegrityError:
                raise ValueError(
                    f"Session '{session_id}' already exists (app_name='{self._app_name}', user_id='{user_id}')"
                )
            if cursor.rowcount == 0:
                raise ValueError(
                    f"Session '{session_id}' already exists (app_name='{self._app_name}', user_id='{user_id}')"
                )
            if len(app_initial) > 0:
                await merge_app_state_delta(db, self._app_name, app_initial)
            await db.commit()

        logger.info("Created session %s (app=%s, user=%s)", session_id, self._app_name, user_id)

        session_state = await self._build_state(initial_state)
        return SQLiteSession(
            session_id=session_id,
            app_name=self._app_name,
            user_id=user_id,
            db=self._db,
            state=session_state,
            settings=settings or self._settings,
        )

    @override
    async def get(
        self,
        session_id: str,
        user_id: str = "",
    ) -> Session | None:
        """Retrieve an existing session.

        Args:
            session_id: The session to retrieve.
            user_id: User identifier.

        Returns:
            A :class:`Session` instance, or ``None`` if the session
            does not exist.
        """
        await self._ensure_ready()

        async with self._db.connect() as db:
            if not await _session_exists(db, self._app_name, user_id, session_id):
                return None
            session_data = await _get_session_state(
                db,
                self._app_name,
                user_id,
                session_id,
            )

        session_state = await self._build_state(session_data)
        return SQLiteSession(
            session_id=session_id,
            app_name=self._app_name,
            user_id=user_id,
            db=self._db,
            state=session_state,
            settings=self._settings,
        )

    @override
    async def get_or_create(
        self,
        session_id: str,
        user_id: str = "",
        state: dict[str, Any] | None = None,
        settings: SessionSettings | None = None,
    ) -> Session:
        """Retrieve a session, creating it if it doesn't exist.

        Args:
            session_id: The session to retrieve or create.
            user_id: User identifier.
            state: Initial state for new sessions.
            settings: Per-session settings for new sessions.

        Returns:
            A :class:`Session` instance.
        """
        await self._ensure_ready()
        initial_state = state or {}
        app_initial, session_initial = _split_app_initial_state(initial_state)
        state_json = json.dumps(session_initial, separators=(",", ":"))

        async with self._db.connect() as db:
            cursor = await db.execute(
                f"INSERT OR IGNORE INTO {DEFAULT_SESSIONS_TABLE} "
                f"(session_id, app_name, user_id, state) "
                f"VALUES (:sid, :app, :uid, :state)",
                {"sid": session_id, "app": self._app_name, "uid": user_id, "state": state_json},
            )
            # Seed app-scoped initial state only when this call actually
            # created the row; an existing session keeps its stored (shared)
            # app state rather than being overwritten by these defaults.
            if cursor.rowcount > 0 and len(app_initial) > 0:
                await merge_app_state_delta(db, self._app_name, app_initial)
            await db.commit()
            session_data = await _get_session_state(
                db,
                self._app_name,
                user_id,
                session_id,
            )

        session_state = await self._build_state(session_data)
        return SQLiteSession(
            session_id=session_id,
            app_name=self._app_name,
            user_id=user_id,
            db=self._db,
            state=session_state,
            settings=settings or self._settings,
        )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    @override
    async def list(
        self,
        user_id: str | None = None,
    ) -> list[SessionInfo]:
        """List sessions in the database.

        Args:
            user_id: If provided, filter by user.  If ``None``, list
                all sessions for this app.

        Returns:
            List of :class:`SessionInfo` ordered by creation time
            (oldest first).
        """
        await self._ensure_ready()

        async with self._db.connect() as db:
            if user_id is not None:
                cursor = await db.execute(
                    f"SELECT session_id, app_name, user_id, created_at, updated_at "
                    f"FROM {DEFAULT_SESSIONS_TABLE} "
                    f"WHERE app_name = :app AND user_id = :uid "
                    f"ORDER BY created_at ASC",
                    {"app": self._app_name, "uid": user_id},
                )
            else:
                cursor = await db.execute(
                    f"SELECT session_id, app_name, user_id, created_at, updated_at "
                    f"FROM {DEFAULT_SESSIONS_TABLE} "
                    f"WHERE app_name = :app "
                    f"ORDER BY created_at ASC",
                    {"app": self._app_name},
                )
            rows = await cursor.fetchall()

        return [
            SessionInfo(
                session_id=r["session_id"],
                app_name=r["app_name"],
                user_id=r["user_id"],
                created_at=str(r["created_at"]),
                updated_at=str(r["updated_at"]),
            )
            for r in rows
        ]

    @override
    async def delete(
        self,
        session_id: str,
        user_id: str = "",
    ) -> bool:
        """Delete a session and all its messages.

        Args:
            session_id: The session to delete.
            user_id: User identifier.

        Returns:
            ``True`` if the session existed and was deleted,
            ``False`` otherwise.
        """
        await self._ensure_ready()

        async with self._db.connect() as db:
            await db.execute(
                f"DELETE FROM {DEFAULT_MESSAGES_TABLE} WHERE app_name = :app AND user_id = :uid AND session_id = :sid",
                {"app": self._app_name, "uid": user_id, "sid": session_id},
            )
            cursor = await db.execute(
                f"DELETE FROM {DEFAULT_SESSIONS_TABLE} WHERE app_name = :app AND user_id = :uid AND session_id = :sid",
                {"app": self._app_name, "uid": user_id, "sid": session_id},
            )
            await db.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info("Deleted session %s (app=%s, user=%s)", session_id, self._app_name, user_id)
        return deleted

    @override
    async def count(
        self,
        user_id: str | None = None,
    ) -> int:
        """Return the number of sessions.

        Args:
            user_id: If provided, count only this user's sessions.
                If ``None``, count all sessions for this app.

        Returns:
            Total number of matching sessions.
        """
        await self._ensure_ready()

        async with self._db.connect() as db:
            if user_id is not None:
                cursor = await db.execute(
                    f"SELECT COUNT(*) as cnt FROM {DEFAULT_SESSIONS_TABLE} WHERE app_name = :app AND user_id = :uid",
                    {"app": self._app_name, "uid": user_id},
                )
            else:
                cursor = await db.execute(
                    f"SELECT COUNT(*) as cnt FROM {DEFAULT_SESSIONS_TABLE} WHERE app_name = :app",
                    {"app": self._app_name},
                )
            row = await cursor.fetchone()

        return row["cnt"] if row is not None else 0

    async def get_app_state(self) -> dict[str, Any]:
        """Return a snapshot of app-scoped state without constructing a Session.

        Reads directly from the app-state table.  The returned dict
        uses the raw ``app:`` keys as stored (e.g. ``"app:theme"``).
        Returns an empty dict when no app-scoped state has been written
        yet.

        This is a read-only helper — it does not load session history or
        construct a :class:`Session` object.

        Returns:
            Mapping of app-scoped keys to their current values.
        """
        await self._ensure_ready()
        async with self._db.connect() as db:
            return await get_app_state(db, self._app_name)

    @override
    async def close(self) -> None:
        """Close the database connection."""
        await self._db.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _build_state(self, session_data: dict[str, Any]) -> State:
        """Build a State by merging app-scoped state with session data.

        Session-scoped keys override app-scoped keys of the same name.

        Args:
            session_data: Session-scoped key/value pairs loaded from
                the sessions table.

        Returns:
            A :class:`State` with app-scoped data as the base and
            session-scoped data applied on top.
        """
        async with self._db.connect() as db:
            app_state = await get_app_state(db, self._app_name)

        merged = dict(app_state)
        merged.update(session_data)
        return State.from_dict(merged)

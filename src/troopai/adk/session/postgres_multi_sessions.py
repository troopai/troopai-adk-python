"""PostgreSQL-backed session manager — shared sessions across replicas.

``PostgresMultiSessions`` is the manager/store: it owns an
:class:`AsyncConnectionPool` and provides CRUD over a collection of
sessions, each returned as a :class:`PostgresSession` the Runner can use.
It mirrors :class:`~troopai.adk.session.sqlite_multi_sessions.SQLiteMultiSessions`
but is backed by Postgres so every replica reads and writes the same
conversation state.

Install with ``pip install 'troopai-adk-python[session-postgres]'``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, override

try:
    from psycopg_pool import AsyncConnectionPool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PostgresMultiSessions requires psycopg: pip install 'troopai-adk-python[session-postgres]'",
        name="psycopg",
    ) from exc

from troopai.adk.session.multi_sessions import MultiSessions
from troopai.adk.session.postgres_session import PostgresSession, ensure_schema, get_app_state, merge_app_state_delta
from troopai.adk.session.sqlite_multi_sessions import SessionInfo
from troopai.adk.session.state import APP_PREFIX, State

if TYPE_CHECKING:
    from troopai.adk.session.session import Session
    from troopai.adk.session.session_settings import SessionSettings

logger = logging.getLogger(__name__)

_INSERT_SESSION = (
    "INSERT INTO agent_sessions (session_id, app_name, user_id, state) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING"
)
_SELECT_STATE = "SELECT state FROM agent_sessions WHERE app_name=%s AND user_id=%s AND session_id=%s"


def _split_app_initial_state(
    initial_state: dict[str, Any],
) -> tuple[dict[str, object], dict[str, Any]]:
    """Split an initial-state dict into its app-scoped and session-scoped tiers.

    App-scoped (``app:``) keys are owned by the app-state table; the rest live
    in the session column. Splitting on creation mirrors ``save_state`` so an
    app-scoped default is shared and survives the first save instead of being
    dropped by ``to_persist``.

    Args:
        initial_state: The developer-supplied initial state.

    Returns:
        A ``(app_initial, session_initial)`` pair.
    """
    app_initial: dict[str, object] = {k: v for k, v in initial_state.items() if k.startswith(APP_PREFIX)}
    session_initial: dict[str, Any] = {k: v for k, v in initial_state.items() if not k.startswith(APP_PREFIX)}
    return app_initial, session_initial


class PostgresMultiSessions(MultiSessions):
    """Manager for Postgres-backed sessions, shared across replicas.

    The pool opens lazily on first use and the schema is created then,
    serialized by an init lock. The caller owns the lifecycle — call
    :meth:`close` at shutdown.

    Args:
        conninfo: libpq connection string (e.g.
            ``"postgresql://user:pass@host/db"`` or
            ``"host=... dbname=..."``).
        app_name: Application name for multi-tenant scoping.
        settings: Default :class:`SessionSettings` applied when
            ``create`` / ``get_or_create`` are called without explicit
            settings.

    Example::

        sessions = PostgresMultiSessions("postgresql://localhost/agents", app_name="support")
        session = await sessions.get_or_create("conv-1", user_id="u1")
        result = await Runner.arun(agent, "hi", session=session)
        await sessions.close()
    """

    def __init__(
        self,
        conninfo: str,
        app_name: str = "",
        settings: SessionSettings | None = None,
    ) -> None:
        if len(conninfo) == 0:
            raise ValueError("conninfo must be a non-empty libpq connection string.")
        self._conninfo = conninfo
        self._app_name = app_name
        self._settings = settings
        self._pool: AsyncConnectionPool | None = None
        self._init_lock = asyncio.Lock()

    @property
    def app_name(self) -> str:
        """The application name for this manager."""
        return self._app_name

    async def _get_pool(self) -> AsyncConnectionPool:
        """Open the pool and create the schema on first call.

        The init lock serializes concurrent first callers; a failed schema
        creation closes the half-opened pool so no connections leak.

        Returns:
            The ready connection pool.
        """
        async with self._init_lock:
            if self._pool is None:
                pool: AsyncConnectionPool = AsyncConnectionPool(self._conninfo, open=False)
                await pool.open()
                try:
                    await ensure_schema(pool)
                except Exception:
                    await pool.close()
                    raise
                self._pool = pool
                logger.debug("PostgresMultiSessions: pool opened and schema ensured.")
            return self._pool

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
            settings: Per-session settings; falls back to the manager default.

        Returns:
            A :class:`PostgresSession` bound to the new session.

        Raises:
            ValueError: If a session with this id already exists for the
                ``(app_name, user_id)`` scope.
        """
        pool = await self._get_pool()
        initial_state = state if state is not None else {}
        app_initial, session_initial = _split_app_initial_state(initial_state)
        state_json = json.dumps(session_initial, separators=(",", ":"))
        async with pool.connection() as conn:
            cursor = await conn.execute(_INSERT_SESSION, (session_id, self._app_name, user_id, state_json))
            if cursor.rowcount == 0:
                raise ValueError(
                    f"Session '{session_id}' already exists (app_name='{self._app_name}', user_id='{user_id}')"
                )
            if len(app_initial) > 0:
                await merge_app_state_delta(conn, self._app_name, app_initial)
        logger.info("Created session %s (app=%s, user=%s)", session_id, self._app_name, user_id)
        session_state = await self._build_state(pool, initial_state)
        return PostgresSession(session_id, self._app_name, user_id, pool, session_state, settings or self._settings)

    @override
    async def get(self, session_id: str, user_id: str = "") -> Session | None:
        """Retrieve an existing session, or ``None`` if not found.

        Args:
            session_id: The session to retrieve.
            user_id: User identifier.

        Returns:
            A :class:`PostgresSession`, or ``None``.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cursor = await conn.execute(_SELECT_STATE, (self._app_name, user_id, session_id))
            row = await cursor.fetchone()
            if row is None:
                return None
            session_data = json.loads(row[0])
        session_state = await self._build_state(pool, session_data)
        return PostgresSession(session_id, self._app_name, user_id, pool, session_state, self._settings)

    @override
    async def get_or_create(
        self,
        session_id: str,
        user_id: str = "",
        state: dict[str, Any] | None = None,
        settings: SessionSettings | None = None,
    ) -> Session:
        """Retrieve a session, creating it if absent.

        Args:
            session_id: The session to retrieve or create.
            user_id: User identifier.
            state: Initial state for a new session.
            settings: Per-session settings for a new session.

        Returns:
            A :class:`PostgresSession`.
        """
        pool = await self._get_pool()
        initial_state = state if state is not None else {}
        app_initial, session_initial = _split_app_initial_state(initial_state)
        state_json = json.dumps(session_initial, separators=(",", ":"))
        async with pool.connection() as conn:
            insert_cursor = await conn.execute(_INSERT_SESSION, (session_id, self._app_name, user_id, state_json))
            # Seed app-scoped defaults only when this call created the row.
            if insert_cursor.rowcount > 0 and len(app_initial) > 0:
                await merge_app_state_delta(conn, self._app_name, app_initial)
            cursor = await conn.execute(_SELECT_STATE, (self._app_name, user_id, session_id))
            row = await cursor.fetchone()
            session_data = json.loads(row[0]) if row is not None else {}
        session_state = await self._build_state(pool, session_data)
        return PostgresSession(session_id, self._app_name, user_id, pool, session_state, settings or self._settings)

    @override
    async def list(self, user_id: str | None = None) -> list[SessionInfo]:
        """List sessions (optionally filtered by user), oldest first.

        Args:
            user_id: If provided, filter by user; otherwise list all for the app.

        Returns:
            List of :class:`SessionInfo`.
        """
        pool = await self._get_pool()
        select = "SELECT session_id, app_name, user_id, created_at, updated_at FROM agent_sessions WHERE app_name=%s"
        async with pool.connection() as conn:
            if user_id is not None:
                cursor = await conn.execute(
                    f"{select} AND user_id=%s ORDER BY created_at ASC", (self._app_name, user_id)
                )
            else:
                cursor = await conn.execute(f"{select} ORDER BY created_at ASC", (self._app_name,))
            rows = await cursor.fetchall()
        return [
            SessionInfo(
                session_id=row[0],
                app_name=row[1],
                user_id=row[2],
                created_at=str(row[3]),
                updated_at=str(row[4]),
            )
            for row in rows
        ]

    @override
    async def delete(self, session_id: str, user_id: str = "") -> bool:
        """Delete a session and its messages.

        Args:
            session_id: The session to delete.
            user_id: User identifier.

        Returns:
            ``True`` if the session existed and was deleted.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "DELETE FROM agent_messages WHERE app_name=%s AND user_id=%s AND session_id=%s",
                (self._app_name, user_id, session_id),
            )
            cursor = await conn.execute(
                "DELETE FROM agent_sessions WHERE app_name=%s AND user_id=%s AND session_id=%s",
                (self._app_name, user_id, session_id),
            )
            deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted session %s (app=%s, user=%s)", session_id, self._app_name, user_id)
        return deleted

    @override
    async def count(self, user_id: str | None = None) -> int:
        """Return the number of sessions (optionally filtered by user).

        Args:
            user_id: If provided, count only this user's sessions.

        Returns:
            The matching session count.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            if user_id is not None:
                cursor = await conn.execute(
                    "SELECT COUNT(*) FROM agent_sessions WHERE app_name=%s AND user_id=%s",
                    (self._app_name, user_id),
                )
            else:
                cursor = await conn.execute("SELECT COUNT(*) FROM agent_sessions WHERE app_name=%s", (self._app_name,))
            row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    async def get_app_state(self) -> dict[str, Any]:
        """Return a snapshot of app-scoped state without constructing a Session.

        Returns:
            Mapping of app-scoped keys (raw ``app:`` form) to values.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            return await get_app_state(conn, self._app_name)

    @override
    async def close(self) -> None:
        """Close the connection pool. Idempotent and safe in a ``finally``."""
        async with self._init_lock:
            if self._pool is not None:
                pool, self._pool = self._pool, None
                await pool.close()
                logger.debug("PostgresMultiSessions: pool closed.")

    async def _build_state(self, pool: AsyncConnectionPool, session_data: dict[str, Any]) -> State:
        """Merge app-scoped state (base) with session data (override).

        Args:
            pool: The connection pool.
            session_data: Session-scoped key/value pairs.

        Returns:
            A :class:`State` with app data as base and session data applied.
        """
        async with pool.connection() as conn:
            app_state = await get_app_state(conn, self._app_name)
        merged = dict(app_state)
        merged.update(session_data)
        return State.from_dict(merged)


__all__ = ["PostgresMultiSessions", "SessionInfo"]

"""Postgres-backed AuditSink. Append-only INSERT per event.

Postgres docs: https://www.postgresql.org/docs/current/sql-insert.html
"""

from __future__ import annotations

import asyncio
import logging

try:
    from psycopg_pool import AsyncConnectionPool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PostgresAuditSink requires psycopg[binary,pool]>=3.2: pip install 'troopai-adk-python[audit-postgres]'"
    ) from exc

from troopai.adk.audit.event import AuditEvent

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_events (
  id          BIGSERIAL PRIMARY KEY,
  tenant_id   TEXT,
  agent_name  TEXT NOT NULL,
  tool_name   TEXT NOT NULL,
  call_id     TEXT NOT NULL,
  args_hash   TEXT NOT NULL,
  result_hash TEXT,
  outcome     TEXT NOT NULL,
  ts          TIMESTAMPTZ NOT NULL
)
"""

_INSERT = """
INSERT INTO audit_events
  (tenant_id, agent_name, tool_name, call_id, args_hash, result_hash, outcome, ts)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


class PostgresAuditSink:
    """ACID append-only audit sink. Pool opened lazily on first use.

    The caller owns the lifecycle — call :meth:`close` at shutdown.

    Attributes:
        conninfo: libpq connection string used to open the pool.
    """

    def __init__(self, conninfo: str) -> None:
        """Create a Postgres audit sink.

        Args:
            conninfo: libpq connection string (e.g.
                ``"postgresql://user:pass@host/dbname"``). The pool is
                opened lazily on the first :meth:`record` call.
        """
        self.conninfo = conninfo
        self._pool: AsyncConnectionPool | None = None
        self._init_lock = asyncio.Lock()
        logger.debug("PostgresAuditSink initialised.")

    async def _get_pool(self) -> AsyncConnectionPool:
        """Open the pool and create the schema on first call.

        The init lock serializes concurrent first callers so exactly one
        pool is opened. If pool open or schema creation fails, the pool is
        closed before the error propagates, so no connections leak.

        Returns:
            The open and schema-initialised connection pool.
        """
        async with self._init_lock:
            if self._pool is None:
                pool: AsyncConnectionPool = AsyncConnectionPool(self.conninfo, open=False)
                try:
                    await pool.open()
                    async with pool.connection() as conn:
                        await conn.execute(_CREATE_TABLE)
                except Exception:
                    await pool.close()
                    raise
                self._pool = pool
                logger.debug("PostgresAuditSink: pool opened and schema ensured.")
            return self._pool

    async def close(self) -> None:
        """Close the pool. Idempotent and safe in a ``finally``.

        Holds ``_init_lock`` while reading and nulling ``self._pool`` to
        serialize against concurrent ``_get_pool()`` calls. Without the lock
        a concurrent ``_get_pool()`` could assign ``self._pool`` after this
        method checks ``None`` and exits, leaving an unclosed pool.
        """
        async with self._init_lock:
            if self._pool is not None:
                pool, self._pool = self._pool, None
                await pool.close()
                logger.debug("PostgresAuditSink: pool closed.")

    async def record(self, event: AuditEvent) -> None:
        """INSERT ``event`` into the ``audit_events`` table.

        Args:
            event: The audit event to persist.
        """
        pool = await self._get_pool()
        try:
            async with pool.connection() as conn:
                await conn.execute(
                    _INSERT,
                    (
                        event.tenant_id,
                        event.agent_name,
                        event.tool_name,
                        event.tool_call_id,
                        event.args_hash,
                        event.result_hash,
                        event.outcome,
                        event.timestamp,
                    ),
                )
        except Exception:
            # Governance's best-effort handler logs only a generic warning;
            # name the failed table + tenant here so a compliance team can
            # locate the missing audit row.
            logger.error("audit pg insert FAILED table=audit_events tenant=%s", event.tenant_id)
            raise
        logger.debug("audit pg insert tenant=%s tool=%s", event.tenant_id, event.tool_name)


__all__ = ["PostgresAuditSink"]

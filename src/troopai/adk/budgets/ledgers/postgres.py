"""Postgres-backed CostLedger. Atomic per-window increment.

Postgres docs: https://www.postgresql.org/docs/current/sql-insert.html#SQL-ON-CONFLICT
"""

from __future__ import annotations

import asyncio
import logging
import time

try:
    from psycopg_pool import AsyncConnectionPool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PostgresCostLedger requires psycopg[binary,pool]>=3.2: pip install 'troopai-adk-python[cost-ledger-postgres]'"
    ) from exc

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cost_ledger (
  tenant_id  TEXT NOT NULL,
  period_key TEXT NOT NULL,
  spend      DOUBLE PRECISION NOT NULL DEFAULT 0,
  updated_at DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (tenant_id, period_key)
)
"""

_UPSERT = """
INSERT INTO cost_ledger (tenant_id, period_key, spend, updated_at)
VALUES (%s, %s, %s, %s)
ON CONFLICT (tenant_id, period_key)
DO UPDATE SET spend = cost_ledger.spend + EXCLUDED.spend,
              updated_at = EXCLUDED.updated_at
"""

_SELECT = "SELECT spend FROM cost_ledger WHERE tenant_id = %s AND period_key = %s"


class PostgresCostLedger:
    """ACID cost ledger. Connection pool opened lazily on first use.

    Each ``(tenant_id, period_key)`` pair maps to one row. ``record`` is an
    atomic UPSERT that adds ``cost_usd`` to the running total; no optimistic
    locking token is needed because the increment is naturally atomic.

    The caller owns the lifecycle — call :meth:`close` at application
    shutdown or after a test.

    Requires PostgreSQL 9.5+ (``ON CONFLICT`` upsert support).

    Attributes:
        conninfo: libpq connection string used to open the pool.
    """

    def __init__(self, conninfo: str) -> None:
        """Create a ledger bound to the given Postgres connection string.

        Args:
            conninfo: libpq connection string used to open the connection pool
                on first use (e.g. ``"host=localhost dbname=mydb"`` or a
                ``postgresql://`` URL).
        """
        self.conninfo = conninfo
        self._pool: AsyncConnectionPool | None = None
        self._init_lock = asyncio.Lock()
        logger.debug("PostgresCostLedger initialised.")

    async def _get_pool(self) -> AsyncConnectionPool:
        """Open the pool and create the schema on first call.

        The init lock serializes concurrent first callers so exactly one pool
        is opened. If pool open or schema creation fails, the pool is closed
        before the error propagates, so no connections leak.
        """
        async with self._init_lock:
            if self._pool is None:
                pool: AsyncConnectionPool = AsyncConnectionPool(self.conninfo, open=False)
                try:
                    await pool.open()
                    async with pool.connection() as conn:
                        # DDL: the returned cursor is unused.
                        await conn.execute(_CREATE_TABLE)
                except Exception:
                    await pool.close()
                    raise
                self._pool = pool
                logger.debug("PostgresCostLedger: pool opened and schema ensured.")
            return self._pool

    async def close(self) -> None:
        """Close the connection pool. Idempotent and safe in a ``finally``.

        Holds ``_init_lock`` while reading and nulling ``self._pool`` to
        serialize against concurrent ``_get_pool()`` calls. Without the lock
        a concurrent ``_get_pool()`` could assign ``self._pool`` after this
        method checks ``None`` and exits, leaving an unclosed pool.
        """
        async with self._init_lock:
            if self._pool is not None:
                pool, self._pool = self._pool, None
                await pool.close()
                logger.debug("PostgresCostLedger: pool closed.")

    async def spend(self, tenant_id: str, period_key: str) -> float:
        """Return accumulated USD for ``tenant_id`` in ``period_key`` (0 if absent).

        Args:
            tenant_id: Tenant identifier.
            period_key: Time-window key (e.g. ``"2026-05-01"`` for a DAY bucket).

        Returns:
            Accumulated spend in USD, or ``0.0`` when no record exists.
        """
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(_SELECT, (tenant_id, period_key))
            row = await cur.fetchone()
        return float(row[0]) if row is not None else 0.0

    async def record(self, tenant_id: str, period_key: str, cost_usd: float) -> None:
        """Atomically add ``cost_usd`` to ``tenant_id``'s window total.

        Args:
            tenant_id: Tenant identifier.
            period_key: Time-window key (e.g. ``"2026-05-01"`` for a DAY bucket).
            cost_usd: Non-negative USD amount to add.

        Raises:
            ValueError: If ``cost_usd`` is negative.
        """
        if cost_usd < 0:
            raise ValueError(f"cost_usd must be non-negative; got {cost_usd}")
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(_UPSERT, (tenant_id, period_key, cost_usd, time.time()))
        logger.debug(
            "pg ledger record tenant=%s key=%s +%.6f",
            tenant_id,
            period_key,
            cost_usd,
        )


__all__ = ["PostgresCostLedger"]

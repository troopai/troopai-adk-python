"""SQLite-backed persistence for agent run records and quota enforcement.

:class:`AgentStatusStore` manages a standalone SQLite database for
recording per-run metadata and querying cumulative statistics.
Independent of the Session module — uses its own database file.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from troopai.adk.databases.connections.sqlite import SQLiteDatabaseConnection
from troopai.adk.exceptions import QuotaExceeded
from troopai.adk.status.types import AgentQuota, AgentRunRecord, AgentStatus

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS agent_run_records (
    id TEXT PRIMARY KEY,
    agent_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'success',
    started_at REAL NOT NULL,
    ended_at REAL NOT NULL,
    duration_ms REAL NOT NULL,
    requests INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    tenant_id TEXT,
    cost_usd REAL
)
"""

_ALTER_ADD_TENANT_ID = "ALTER TABLE agent_run_records ADD COLUMN tenant_id TEXT"
_ALTER_ADD_COST_USD = "ALTER TABLE agent_run_records ADD COLUMN cost_usd REAL"

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_agent_started
    ON agent_run_records (agent_name, started_at)
"""

_CREATE_TENANT_INDEX = """
CREATE INDEX IF NOT EXISTS idx_tenant_agent_started
    ON agent_run_records (tenant_id, agent_name, started_at)
"""

_INSERT_RECORD = """
INSERT INTO agent_run_records
    (id, agent_name, status, started_at, ended_at, duration_ms,
     requests, input_tokens, output_tokens, total_tokens, error,
     tenant_id, cost_usd)
VALUES
    (:id, :agent_name, :status, :started_at, :ended_at, :duration_ms,
     :requests, :input_tokens, :output_tokens, :total_tokens, :error,
     :tenant_id, :cost_usd)
"""

_AGGREGATE_STATUS = """
SELECT
    COUNT(*) AS total_runs,
    COALESCE(SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END), 0)
        AS successful_runs,
    COALESCE(SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END), 0)
        AS failed_runs,
    COALESCE(SUM(requests), 0) AS total_requests,
    COALESCE(SUM(input_tokens), 0) AS total_input_tokens,
    COALESCE(SUM(output_tokens), 0) AS total_output_tokens,
    COALESCE(SUM(total_tokens), 0) AS total_tokens,
    COALESCE(SUM(duration_ms), 0.0) AS total_duration_ms,
    MIN(started_at) AS first_run_at,
    MAX(started_at) AS last_run_at,
    COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd
FROM agent_run_records
WHERE agent_name = :agent_name
  AND (:since IS NULL OR started_at >= :since)
  AND (:tenant_id IS NULL OR tenant_id = :tenant_id)
"""

_QUOTA_QUERY = """
SELECT
    COUNT(*) AS run_count,
    COALESCE(SUM(requests), 0) AS total_requests,
    COALESCE(SUM(total_tokens), 0) AS total_tokens
FROM agent_run_records
WHERE (:agent_name IS NULL OR agent_name = :agent_name)
  AND started_at >= :cutoff
  AND (:tenant_id IS NULL OR tenant_id = :tenant_id)
"""

_SELECT_RECORDS = """
SELECT id, agent_name, status, started_at, ended_at, duration_ms,
       requests, input_tokens, output_tokens, total_tokens, error,
       tenant_id, cost_usd
FROM agent_run_records
WHERE agent_name = :agent_name
  AND (:since IS NULL OR started_at >= :since)
  AND (:tenant_id IS NULL OR tenant_id = :tenant_id)
ORDER BY started_at DESC
"""

_PRUNE = """
DELETE FROM agent_run_records
WHERE started_at < :cutoff
"""


class AgentStatusStore:
    """Persistent store for agent run records and quota enforcement.

    Manages a standalone SQLite database for recording agent run
    metadata and querying cumulative statistics.  Independent of the
    Session module — uses its own ``db_path``.

    Args:
        path: Path to the SQLite database file, or ``":memory:"``
            for in-memory storage.

    Example::

        store = AgentStatusStore(path="agent_status.db")
        await store.record(record)
        status = await store.get_status("my-agent")
        await store.close()
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._db = SQLiteDatabaseConnection(path=path)
        self._ready = False

    async def _ensure_ready(self) -> None:
        """Create the table and indexes on first use; self-heal old schemas."""
        if self._ready:
            return
        async with self._db.connect() as conn:
            await conn.execute(_CREATE_TABLE)
            # Self-heal: idempotently add columns that may be absent in
            # databases created before tenant_id/cost_usd were introduced.
            # Must run before index creation — the tenant index depends on
            # the tenant_id column existing.  SQLite raises
            # OperationalError("duplicate column name: ...") when the column
            # already exists; treat that as a no-op.
            for alter_sql, col in (
                (_ALTER_ADD_TENANT_ID, "tenant_id"),
                (_ALTER_ADD_COST_USD, "cost_usd"),
            ):
                try:
                    await conn.execute(alter_sql)
                except sqlite3.OperationalError as exc:
                    if "duplicate column name" in str(exc).lower():
                        logger.debug("Column %s already present, skipping ALTER", col)
                    else:
                        raise
            await conn.execute(_CREATE_INDEX)
            await conn.execute(_CREATE_TENANT_INDEX)
            await conn.commit()
        self._ready = True
        logger.debug("AgentStatusStore ready (tables created)")

    async def record(self, run_record: AgentRunRecord) -> None:
        """Insert a run record into the database.

        Args:
            run_record: The completed run record to persist.
        """
        await self._ensure_ready()
        async with self._db.connect() as conn:
            await conn.execute(
                _INSERT_RECORD,
                {
                    "id": run_record.id,
                    "agent_name": run_record.agent_name,
                    "status": run_record.status,
                    "started_at": run_record.started_at,
                    "ended_at": run_record.ended_at,
                    "duration_ms": run_record.duration_ms,
                    "requests": run_record.requests,
                    "input_tokens": run_record.input_tokens,
                    "output_tokens": run_record.output_tokens,
                    "total_tokens": run_record.total_tokens,
                    "error": run_record.error,
                    "tenant_id": run_record.tenant_id,
                    "cost_usd": run_record.cost_usd,
                },
            )
            await conn.commit()
        logger.info(
            "Recorded run '%s' for agent '%s': %s (%d tokens)",
            run_record.id[:8],
            run_record.agent_name,
            run_record.status,
            run_record.total_tokens,
        )

    async def get_status(
        self,
        agent_name: str,
        since: float | None = None,
        *,
        tenant_id: str | None = None,
    ) -> AgentStatus:
        """Get aggregated status for an agent.

        Args:
            agent_name: The agent to query.
            since: Optional UTC timestamp cutoff.  Only runs started
                at or after this time are included.  If ``None``, all
                runs are included.
            tenant_id: If provided, restrict results to runs belonging
                to this tenant.

        Returns:
            Aggregated :class:`AgentStatus` with cumulative stats.
        """
        await self._ensure_ready()
        async with self._db.connect() as conn:
            cursor = await conn.execute(
                _AGGREGATE_STATUS,
                {"agent_name": agent_name, "since": since, "tenant_id": tenant_id},
            )
            row = await cursor.fetchone()

        if row is None or row[0] == 0:
            return AgentStatus(
                agent_name=agent_name,
                total_runs=0,
                successful_runs=0,
                failed_runs=0,
                total_requests=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_tokens=0,
                total_duration_ms=0.0,
                error_rate=0.0,
                avg_duration_ms=0.0,
                first_run_at=None,
                last_run_at=None,
                total_cost_usd=0.0,
            )

        # SELECT column order (0-based):
        # 0  total_runs
        # 1  successful_runs
        # 2  failed_runs
        # 3  total_requests
        # 4  total_input_tokens
        # 5  total_output_tokens
        # 6  total_tokens
        # 7  total_duration_ms
        # 8  first_run_at
        # 9  last_run_at
        # 10 total_cost_usd
        total_runs = row[0]
        failed_runs = row[2]
        total_duration_ms = float(row[7])

        return AgentStatus(
            agent_name=agent_name,
            total_runs=total_runs,
            successful_runs=row[1],
            failed_runs=failed_runs,
            total_requests=row[3],
            total_input_tokens=row[4],
            total_output_tokens=row[5],
            total_tokens=row[6],
            total_duration_ms=total_duration_ms,
            error_rate=failed_runs / total_runs if total_runs > 0 else 0.0,
            avg_duration_ms=(total_duration_ms / total_runs if total_runs > 0 else 0.0),
            first_run_at=row[8],
            last_run_at=row[9],
            total_cost_usd=float(row[10]),
        )

    async def get_records(
        self,
        agent_name: str,
        since: float | None = None,
        limit: int | None = None,
        *,
        tenant_id: str | None = None,
    ) -> list[AgentRunRecord]:
        """Retrieve individual run records for an agent.

        Args:
            agent_name: The agent to query.
            since: Optional UTC timestamp cutoff.
            limit: Maximum number of records (most recent first).
            tenant_id: If provided, restrict results to runs belonging
                to this tenant.

        Returns:
            List of :class:`AgentRunRecord` ordered by most recent first.
        """
        await self._ensure_ready()
        query = _SELECT_RECORDS
        if limit is not None:
            query += f" LIMIT {limit}"

        async with self._db.connect() as conn:
            cursor = await conn.execute(
                query,
                {"agent_name": agent_name, "since": since, "tenant_id": tenant_id},
            )
            rows = await cursor.fetchall()

        # SELECT column order (0-based):
        # 0  id           6  requests
        # 1  agent_name   7  input_tokens
        # 2  status       8  output_tokens
        # 3  started_at   9  total_tokens
        # 4  ended_at    10  error
        # 5  duration_ms 11  tenant_id  12  cost_usd
        return [
            AgentRunRecord(
                id=row[0],
                agent_name=row[1],
                status=row[2],
                started_at=row[3],
                ended_at=row[4],
                duration_ms=row[5],
                requests=row[6],
                input_tokens=row[7],
                output_tokens=row[8],
                total_tokens=row[9],
                error=row[10],
                tenant_id=row[11],
                cost_usd=row[12],
            )
            for row in rows
        ]

    async def check_quota(
        self,
        agent_name: str,
        quota: AgentQuota,
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Check if an agent is within its quota.

        Args:
            agent_name: The agent to check.
            quota: The quota to enforce.
            tenant_id: If provided, restrict the usage count to this
                tenant only.

        Raises:
            QuotaExceeded: If any limit in the quota is exceeded.
        """
        await self._ensure_ready()
        cutoff = time.time() - quota.window_seconds

        # Wildcard quotas query all agents (agent_name IS NULL in SQL)
        query_agent = None if quota.agent_name == "*" else agent_name

        async with self._db.connect() as conn:
            cursor = await conn.execute(
                _QUOTA_QUERY,
                {"agent_name": query_agent, "cutoff": cutoff, "tenant_id": tenant_id},
            )
            row = await cursor.fetchone()

        if row is None:
            return

        run_count, total_requests, total_tokens = row[0], row[1], row[2]

        if quota.max_runs is not None and run_count >= quota.max_runs:
            raise QuotaExceeded(
                agent_name=agent_name,
                quota=quota,
                current_value=run_count,
                limit=quota.max_runs,
                resource="runs",
            )
        if quota.max_requests is not None and total_requests >= quota.max_requests:
            raise QuotaExceeded(
                agent_name=agent_name,
                quota=quota,
                current_value=total_requests,
                limit=quota.max_requests,
                resource="requests",
            )
        if quota.max_total_tokens is not None and total_tokens >= quota.max_total_tokens:
            raise QuotaExceeded(
                agent_name=agent_name,
                quota=quota,
                current_value=total_tokens,
                limit=quota.max_total_tokens,
                resource="total_tokens",
            )

    async def check_quotas(
        self,
        agent_name: str,
        quotas: list[AgentQuota],
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Check multiple quotas for an agent.

        Matches quotas where ``quota.agent_name == agent_name``
        or ``quota.agent_name == "*"``.  Raises on first violation.

        Args:
            agent_name: The agent to check.
            quotas: List of quotas to enforce.
            tenant_id: If provided, restrict the usage count to this
                tenant only.

        Raises:
            QuotaExceeded: On the first quota violation found.
        """
        for quota in quotas:
            if quota.agent_name == "*" or quota.agent_name == agent_name:
                await self.check_quota(agent_name, quota, tenant_id=tenant_id)

    async def prune(self, older_than: float) -> int:
        """Delete run records older than the given UTC timestamp.

        Args:
            older_than: UTC timestamp cutoff.  Records with
                ``started_at < older_than`` are deleted.

        Returns:
            Number of records deleted.
        """
        await self._ensure_ready()
        async with self._db.connect() as conn:
            cursor = await conn.execute(
                _PRUNE,
                {"cutoff": older_than},
            )
            await conn.commit()
            deleted = cursor.rowcount

        logger.info("Pruned %d old run records", deleted)
        return deleted

    async def close(self) -> None:
        """Close the database connection."""
        await self._db.close()

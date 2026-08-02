from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("psycopg_pool")

from troopai.adk.audit.event import AuditEvent
from troopai.adk.audit.sink import AuditSink
from troopai.adk.audit.sinks.postgres import PostgresAuditSink


def test_postgres_is_an_audit_sink() -> None:
    sink = PostgresAuditSink("postgresql://localhost/test")
    assert isinstance(sink, AuditSink)


async def test_close_holds_init_lock_preventing_pool_leak() -> None:
    """close() must hold _init_lock so it cannot race _get_pool().

    Regression: close() read self._pool without the lock. A concurrent
    _get_pool() could set self._pool after close() checked None and exited,
    leaving an unclosed pool.

    We verify the fix by asserting that close() actually acquires _init_lock:
    when a task already holds _init_lock, close() must block until released.
    """
    sink = PostgresAuditSink("postgresql://localhost/test")

    mock_pool = AsyncMock()
    mock_pool.close = AsyncMock()
    sink._pool = mock_pool  # type: ignore[assignment]  # test introspection

    close_started = asyncio.Event()
    close_completed = asyncio.Event()

    async def run_close() -> None:
        close_started.set()
        await sink.close()
        close_completed.set()

    # Acquire the init lock externally — simulates _get_pool() holding it.
    async with sink._init_lock:
        task = asyncio.ensure_future(run_close())
        await close_started.wait()
        # Give the event loop a chance to advance the close task.
        await asyncio.sleep(0)
        # close() must be blocked inside the lock — pool.close() must NOT be called yet.
        mock_pool.close.assert_not_awaited()
        # Release the lock — close() can now proceed.

    await asyncio.wait_for(task, timeout=2.0)
    # After the lock is released, close() must have closed and nulled the pool.
    mock_pool.close.assert_awaited_once()
    assert sink._pool is None


async def test_close_is_idempotent_when_pool_is_none() -> None:
    """close() on a never-opened sink (pool is None) must not raise."""
    sink = PostgresAuditSink("postgresql://localhost/test")
    assert sink._pool is None
    await sink.close()  # must not raise
    await sink.close()  # must not raise


def test_schema_targets_append_only_table() -> None:
    from troopai.adk.audit.sinks import postgres

    assert "audit_events" in postgres._CREATE_TABLE
    assert "INSERT INTO audit_events" in postgres._INSERT


async def test_insert_failure_logs_table_before_reraising(caplog) -> None:
    """A failed INSERT logs the table + tenant at ERROR before propagating.

    Governance's best-effort handler logs only a generic warning; the sink
    must name its target so a compliance team can locate the missing row.
    """

    class _FailingPool:
        def connection(self) -> _FailingPool:
            return self

        async def __aenter__(self) -> None:
            raise RuntimeError("pg down")

        async def __aexit__(self, *args: object) -> bool:
            return False

    sink = PostgresAuditSink("postgresql://localhost/test")
    event = AuditEvent(
        tenant_id="t1",
        agent_name="a",
        tool_name="t",
        tool_call_id="c1",
        args_hash="h",
        result_hash=None,
        outcome="ok",
        timestamp=datetime.now(UTC),
    )
    with (
        patch.object(sink, "_get_pool", AsyncMock(return_value=_FailingPool())),
        caplog.at_level(logging.ERROR, logger="troopai.adk.audit.sinks.postgres"),
        pytest.raises(RuntimeError, match="pg down"),
    ):
        await sink.record(event)
    assert "FAILED" in caplog.text
    assert "audit_events" in caplog.text

from __future__ import annotations

import asyncio
import os
import uuid
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("psycopg_pool")
PG_DSN = os.environ.get("TROOPAI_TEST_PG_DSN")

# ---------------------------------------------------------------------------
# Unit tests that do NOT require a live Postgres server
# ---------------------------------------------------------------------------


async def test_close_holds_init_lock_preventing_pool_leak() -> None:
    """close() must hold _init_lock so it cannot race _get_pool().

    Regression: close() read self._pool without the lock. A concurrent
    _get_pool() could set self._pool after close() checked None and exited,
    leaving an unclosed pool.

    We verify the fix by asserting that close() actually acquires _init_lock:
    when a task already holds _init_lock, close() must block until released.
    """
    from troopai.adk.budgets.ledgers.postgres import PostgresCostLedger

    ledger = PostgresCostLedger("host=localhost dbname=test")

    mock_pool = AsyncMock()
    mock_pool.close = AsyncMock()
    ledger._pool = mock_pool  # type: ignore[assignment]  # test introspection

    close_started = asyncio.Event()
    close_completed = asyncio.Event()

    async def run_close() -> None:
        close_started.set()
        await ledger.close()
        close_completed.set()

    # Acquire the init lock externally — simulates _get_pool() holding it.
    async with ledger._init_lock:
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
    assert ledger._pool is None


async def test_close_is_idempotent_when_pool_is_none() -> None:
    """close() on a never-opened ledger (pool is None) must not raise."""
    from troopai.adk.budgets.ledgers.postgres import PostgresCostLedger

    ledger = PostgresCostLedger("host=localhost dbname=test")
    assert ledger._pool is None
    await ledger.close()  # must not raise
    await ledger.close()  # must not raise


async def test_record_rejects_negative_cost_usd() -> None:
    """record() must raise ValueError for negative cost_usd.

    Regression: Postgres UPSERT accepts negative values without error,
    silently corrupting the running total. The guard must be explicit and
    must fire before any DB round-trip (no pool needed for this test).
    """
    from troopai.adk.budgets.ledgers.postgres import PostgresCostLedger

    ledger = PostgresCostLedger("host=localhost dbname=test")
    with pytest.raises(ValueError, match="non-negative"):
        await ledger.record("t1", "2026-05-daily", -0.01)


async def test_record_accepts_zero_cost_usd_without_pool() -> None:
    """record() with cost_usd == 0.0 must not raise before touching the pool.

    We expect it to eventually fail when it tries to open the pool (no live
    DB), but the non-negative guard must NOT have raised first.
    """
    from troopai.adk.budgets.ledgers.postgres import PostgresCostLedger

    ledger = PostgresCostLedger("host=localhost dbname=test")
    # Should NOT raise ValueError("non-negative") — any other error (e.g.
    # connection refused) is acceptable here since no DB is available.
    with pytest.raises(Exception) as exc_info:
        await ledger.record("t1", "2026-05-daily", 0.0)
    assert "non-negative" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Live integration tests (require TROOPAI_TEST_PG_DSN)
# ---------------------------------------------------------------------------

pytestmark_live = pytest.mark.skipif(PG_DSN is None, reason="set TROOPAI_TEST_PG_DSN to run")


@pytest.mark.postgres
@pytestmark_live  # type: ignore[misc]  # dynamic mark application
async def test_postgres_record_and_spend() -> None:
    from troopai.adk.budgets.ledgers.postgres import PostgresCostLedger

    assert PG_DSN is not None
    ledger = PostgresCostLedger(PG_DSN)
    try:
        key = f"test-{uuid.uuid4()}"  # unique per run — UPSERT accumulates across runs
        await ledger.record("tenant-A", key, 0.20)
        await ledger.record("tenant-A", key, 0.05)
        assert await ledger.spend("tenant-A", key) == pytest.approx(0.25)
        assert await ledger.spend("tenant-B", key) == 0.0
    finally:
        await ledger.close()

"""RedisCostLedger unit tests — run unconditionally via fakeredis (in-process).

``fakeredis[lua]>=2.21`` is a dev dependency; no Redis server is needed.
An optional live-server smoke test runs only when ``TROOPAI_TEST_REDIS_URL``
is set in the environment.
"""

from __future__ import annotations

import os

import pytest
from fakeredis.aioredis import FakeRedis

from troopai.adk.budgets import CostLedger
from troopai.adk.budgets.ledgers.redis import RedisCostLedger

# ---------------------------------------------------------------------------
# Core fakeredis-backed tests — always run
# ---------------------------------------------------------------------------


async def test_record_and_spend() -> None:
    """Two records for the same (tenant, key) accumulate; other tenants/keys are isolated."""
    ledger = RedisCostLedger(client=FakeRedis())
    await ledger.record("tenant-A", "2026-05-daily", 0.20)
    await ledger.record("tenant-A", "2026-05-daily", 0.05)
    assert await ledger.spend("tenant-A", "2026-05-daily") == pytest.approx(0.25)
    # Different tenant — same key — must be 0.
    assert await ledger.spend("tenant-B", "2026-05-daily") == 0.0
    # Same tenant — different key — must be 0.
    assert await ledger.spend("tenant-A", "2026-04-daily") == 0.0


async def test_spend_absent_key_is_zero() -> None:
    """spend() on a never-recorded key returns 0.0 without error."""
    ledger = RedisCostLedger(client=FakeRedis())
    assert await ledger.spend("no-tenant", "no-key") == 0.0


async def test_colon_in_tenant_id_does_not_collide_across_period_separator() -> None:
    """A ':' in tenant_id must not collide with the period separator.

    Security regression: the key was f"...{tenant_id}:{period_key}", so
    tenant 'a:b' + period 'c' shared a bucket with tenant 'a' + period 'b:c'
    — cross-tenant budget bleed. The tenant_id segment is now encoded so the
    two resolve to distinct buckets.
    """
    ledger = RedisCostLedger(client=FakeRedis())
    await ledger.record("a:b", "c", 1.00)
    # tenant 'a' + period 'b:c' must NOT see tenant 'a:b' + period 'c' spend.
    assert await ledger.spend("a", "b:c") == 0.0
    assert await ledger.spend("a:b", "c") == pytest.approx(1.00)


def test_requires_exactly_one_of_client_url() -> None:
    """Passing neither or both of client= / url= raises ValueError."""
    with pytest.raises(ValueError, match="requires either"):
        RedisCostLedger()
    with pytest.raises(ValueError, match="not both"):
        RedisCostLedger(client=FakeRedis(), url="redis://localhost:6379/0")


def test_negative_ttl_rejected() -> None:
    """ttl_seconds <= 0 is rejected at construction time."""
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        RedisCostLedger(client=FakeRedis(), ttl_seconds=0)
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        RedisCostLedger(client=FakeRedis(), ttl_seconds=-1)


async def test_close_only_closes_owned_client() -> None:
    """close() leaves a caller-supplied client open; the client remains usable."""
    client = FakeRedis()
    ledger = RedisCostLedger(client=client)
    await ledger.close()
    # After close(), the caller-supplied client must still accept commands.
    await client.set("__probe__", "1")
    probe = await client.get("__probe__")
    assert probe == b"1"


def test_is_a_cost_ledger() -> None:
    """RedisCostLedger is a concrete CostLedger (protocol / ABC check)."""
    assert isinstance(RedisCostLedger(client=FakeRedis()), CostLedger)


async def test_record_rejects_negative_cost_usd() -> None:
    """record() must raise ValueError for negative cost_usd.

    Regression: Redis INCRBYFLOAT accepts negative values without error,
    silently corrupting the running total. The guard must be explicit.
    """
    ledger = RedisCostLedger(client=FakeRedis())
    with pytest.raises(ValueError, match="non-negative"):
        await ledger.record("t1", "2026-05-daily", -0.01)


async def test_record_accepts_zero_cost_usd() -> None:
    """record() must accept cost_usd == 0.0 (a no-op increment is valid)."""
    ledger = RedisCostLedger(client=FakeRedis())
    await ledger.record("t1", "2026-05-daily", 0.0)
    assert await ledger.spend("t1", "2026-05-daily") == 0.0


async def test_ttl_is_set_when_configured() -> None:
    """When ttl_seconds is set, the underlying Redis key gains a positive TTL after the first record."""
    client = FakeRedis()
    ledger = RedisCostLedger(client=client, ttl_seconds=100)
    tenant, period = "tenant-ttl", "2026-05-daily"
    await ledger.record(tenant, period, 0.10)
    key = f"cost:ledger:{tenant}:{period}"
    ttl = await client.ttl(key)
    # TTL must be positive and within the configured ceiling.
    assert 0 < ttl <= 100


async def test_record_uses_pipeline_for_incr_and_expire() -> None:
    """record() must issue INCRBYFLOAT + EXPIRE NX inside a single pipeline.

    Regression: the two commands were sequential (non-atomic). A connection
    drop between them left the key without a TTL, causing it to persist
    forever. Using a pipeline makes EXPIRE NX a single atomic round-trip.

    We verify the pipeline path by checking that the TTL is set AND the
    accumulated value is correct — if the pipeline weren't used the test
    would still pass under fakeredis, so we also assert that the pipeline
    is actually entered by patching the client's pipeline method.
    """
    import unittest.mock

    client = FakeRedis()
    real_pipeline = client.pipeline

    pipeline_entered = []

    def tracking_pipeline(transaction: bool = True) -> object:
        pipe = real_pipeline(transaction=transaction)
        pipeline_entered.append(True)
        return pipe

    with unittest.mock.patch.object(client, "pipeline", side_effect=tracking_pipeline):
        ledger = RedisCostLedger(client=client, ttl_seconds=300)
        await ledger.record("t1", "2026-05-daily", 0.10)

    assert len(pipeline_entered) == 1, "record() must enter a pipeline exactly once"
    assert await ledger.spend("t1", "2026-05-daily") == pytest.approx(0.10)
    # TTL must have been set via the pipeline's EXPIRE NX.
    key = "cost:ledger:t1:2026-05-daily"
    ttl = await client.ttl(key)
    assert 0 < ttl <= 300


async def test_pipeline_expire_nx_does_not_slide_forward() -> None:
    """Subsequent record() calls must not reset (slide forward) the TTL.

    EXPIRE … NX is only applied on bucket creation; repeated writes within the
    same window must not refresh the expiry. Verifies the NX semantics are
    preserved when using the pipeline path.
    """
    client = FakeRedis()
    ledger = RedisCostLedger(client=client, ttl_seconds=60)
    await ledger.record("tx", "2026-05-daily", 0.10)
    key = "cost:ledger:tx:2026-05-daily"
    ttl_first = await client.ttl(key)
    # A second record must NOT extend the TTL (NX flag).
    await ledger.record("tx", "2026-05-daily", 0.05)
    ttl_second = await client.ttl(key)
    # TTL should not have increased from the second write.
    assert ttl_second <= ttl_first


# ---------------------------------------------------------------------------
# Optional live-server smoke test (skipped when TROOPAI_TEST_REDIS_URL is unset)
# ---------------------------------------------------------------------------

_REDIS_URL = os.environ.get("TROOPAI_TEST_REDIS_URL")


@pytest.mark.skipif(_REDIS_URL is None, reason="set TROOPAI_TEST_REDIS_URL to run live smoke test")
async def test_live_record_and_spend() -> None:
    """Live Redis smoke test: record + spend round-trip on a real server."""
    import uuid

    assert _REDIS_URL is not None
    ledger = RedisCostLedger(url=_REDIS_URL)
    try:
        key = f"test-{uuid.uuid4()}"
        await ledger.record("tenant-A", key, 0.20)
        await ledger.record("tenant-A", key, 0.05)
        assert await ledger.spend("tenant-A", key) == pytest.approx(0.25)
        assert await ledger.spend("tenant-B", key) == 0.0
    finally:
        await ledger.close()

from __future__ import annotations

import asyncio

import pytest

from troopai.adk.budgets import CostLedger, InMemoryCostLedger


async def test_record_and_spend() -> None:
    ledger = InMemoryCostLedger()
    assert await ledger.spend("t1", "2026-05-24") == 0.0
    await ledger.record("t1", "2026-05-24", 0.10)
    await ledger.record("t1", "2026-05-24", 0.05)
    assert await ledger.spend("t1", "2026-05-24") == pytest.approx(0.15)
    assert await ledger.spend("t1", "2026-05-25") == 0.0
    assert await ledger.spend("t2", "2026-05-24") == 0.0


async def test_concurrent_record_sums() -> None:
    ledger = InMemoryCostLedger()
    await asyncio.gather(*[ledger.record("t1", "k", 0.01) for _ in range(100)])
    assert await ledger.spend("t1", "k") == pytest.approx(1.0)


def test_in_memory_is_a_cost_ledger() -> None:
    assert isinstance(InMemoryCostLedger(), CostLedger)


async def test_record_rejects_negative_cost_usd() -> None:
    """record() must raise ValueError for negative cost_usd.

    Regression: negative spend was silently accepted and would corrupt the
    running total by subtracting from it. The Protocol contract says
    cost_usd must be non-negative.
    """
    ledger = InMemoryCostLedger()
    with pytest.raises(ValueError, match="non-negative"):
        await ledger.record("t1", "2026-05-24", -0.01)


async def test_record_accepts_zero_cost_usd() -> None:
    """record() must accept cost_usd == 0.0 (a no-op increment is valid)."""
    ledger = InMemoryCostLedger()
    await ledger.record("t1", "2026-05-24", 0.0)
    assert await ledger.spend("t1", "2026-05-24") == 0.0

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from troopai.adk.budgets import BudgetPeriod, TenantBudget, period_key


def test_period_key_calendar_buckets() -> None:
    ts = datetime(2026, 5, 24, 13, 7, tzinfo=UTC)
    assert period_key(BudgetPeriod.DAY, ts) == "2026-05-24"
    assert period_key(BudgetPeriod.HOUR, ts) == "2026-05-24T13"
    assert period_key(BudgetPeriod.MONTH, ts) == "2026-05"


def test_period_key_rejects_naive_datetime() -> None:
    """period_key() must raise ValueError when passed a naive (no tzinfo) datetime.

    Regression: a naive datetime silently produced wrong calendar buckets when
    the caller passed local time instead of UTC. The ValueError is the correct
    signal for an accidental naive datetime.now() call.
    """
    naive = datetime(2026, 5, 24, 13, 7)  # no tzinfo — naive
    assert naive.tzinfo is None
    with pytest.raises(ValueError, match="timezone-aware"):
        period_key(BudgetPeriod.DAY, naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        period_key(BudgetPeriod.HOUR, naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        period_key(BudgetPeriod.MONTH, naive)


def test_budget_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="dollars_per_run"):
        TenantBudget(dollars_per_run=0.0)
    with pytest.raises(ValueError, match="dollars_per_period"):
        TenantBudget(dollars_per_period=-1.0)


def test_budget_rejects_inf_and_nan() -> None:
    """A non-finite dollar cap silently disables enforcement → reject it.

    ``<= 0`` is False for both ``inf`` and ``nan``, so they slipped past the
    guard: ``inf`` is never exceeded, and every ``cost > nan`` is False. A
    non-finite cap is a silent no-cap, so ``__post_init__`` must reject it.
    """
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="dollars_per_run"):
            TenantBudget(dollars_per_run=bad)
        with pytest.raises(ValueError, match="dollars_per_period"):
            TenantBudget(dollars_per_period=bad)


def test_budget_defaults() -> None:
    b = TenantBudget(dollars_per_run=1.0)
    assert b.period is BudgetPeriod.DAY
    assert b.kill_on_exceed is True
    assert b.dollars_per_period is None


def test_budget_both_none_is_valid() -> None:
    b = TenantBudget()
    assert b.dollars_per_run is None
    assert b.dollars_per_period is None

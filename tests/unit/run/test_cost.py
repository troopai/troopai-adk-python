"""Regression tests for ``run/cost.py`` budget checks.

Focus: the fail-closed period sentinel. When the cost ledger is unreachable
and ``ledger_fail_open=False``, the caller passes ``period_spend=inf`` to
signal "period treated as fully spent". The gate must trip even when the
provider has no cost table (``estimate is None``); otherwise the LLM call
would proceed unchecked and the (unreachable) ledger could not record it.
"""

from __future__ import annotations

import math

import pytest

from troopai.adk.budgets import TenantBudget
from troopai.adk.exceptions import TenantBudgetExceeded
from troopai.adk.run.cost import check_tenant_budget


def test_infinite_period_spend_trips_with_none_estimate() -> None:
    # Fail-closed sentinel: ledger down, provider has no cost table.
    # The period cap must still be enforced — the early estimate-None
    # return must NOT short-circuit the infinite-spend breach.
    budget = TenantBudget(dollars_per_period=2.0)
    with pytest.raises(TenantBudgetExceeded) as ei:
        check_tenant_budget(
            budget,
            "t1",
            run_cost=0.0,
            period_spend=float("inf"),
            estimate=None,
        )
    assert ei.value.scope == "period"
    assert ei.value.tenant_id == "t1"
    assert math.isinf(ei.value.spend)


def test_infinite_period_spend_trips_with_estimate() -> None:
    # Same sentinel, but the provider does report a cost estimate.
    budget = TenantBudget(dollars_per_period=2.0)
    with pytest.raises(TenantBudgetExceeded) as ei:
        check_tenant_budget(
            budget,
            "t1",
            run_cost=0.0,
            period_spend=float("inf"),
            estimate=0.10,
        )
    assert ei.value.scope == "period"
    assert ei.value.estimated_cost == 0.10


def test_infinite_period_spend_ignored_without_period_cap() -> None:
    # No per-period cap configured: the sentinel is irrelevant and a
    # missing estimate still skips the (run-only) gate.
    budget = TenantBudget(dollars_per_run=1.0)
    check_tenant_budget(
        budget,
        "t1",
        run_cost=0.10,
        period_spend=float("inf"),
        estimate=None,
    )


def test_finite_period_spend_with_none_estimate_skips_gate() -> None:
    # Healthy ledger (finite spend) + no cost table: gate is skipped and
    # enforcement falls back to post-call recording.
    budget = TenantBudget(dollars_per_period=2.0)
    check_tenant_budget(
        budget,
        "t1",
        run_cost=0.0,
        period_spend=1.95,
        estimate=None,
    )

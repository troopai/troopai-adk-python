from __future__ import annotations

import pytest

from troopai.adk.budgets import TenantBudget
from troopai.adk.exceptions import TenantBudgetExceeded
from troopai.adk.run.cost import check_tenant_budget


def test_run_scope_breach() -> None:
    budget = TenantBudget(dollars_per_run=1.0)
    with pytest.raises(TenantBudgetExceeded) as ei:
        check_tenant_budget(budget, "t1", run_cost=0.95, period_spend=0.0, estimate=0.10)
    assert ei.value.scope == "run"
    assert ei.value.tenant_id == "t1"


def test_period_scope_breach() -> None:
    budget = TenantBudget(dollars_per_period=2.0)
    with pytest.raises(TenantBudgetExceeded) as ei:
        check_tenant_budget(budget, "t1", run_cost=0.0, period_spend=1.95, estimate=0.10)
    assert ei.value.scope == "period"


def test_within_budget_no_raise() -> None:
    budget = TenantBudget(dollars_per_run=1.0, dollars_per_period=10.0)
    check_tenant_budget(budget, "t1", run_cost=0.10, period_spend=0.10, estimate=0.10)


def test_none_estimate_skips_gate() -> None:
    budget = TenantBudget(dollars_per_run=0.01)
    check_tenant_budget(budget, "t1", run_cost=999.0, period_spend=0.0, estimate=None)


def test_run_checked_before_period() -> None:
    # When both would breach, run scope is reported first.
    budget = TenantBudget(dollars_per_run=1.0, dollars_per_period=2.0)
    with pytest.raises(TenantBudgetExceeded) as ei:
        check_tenant_budget(budget, "t1", run_cost=0.95, period_spend=1.95, estimate=0.10)
    assert ei.value.scope == "run"


def test_exact_cap_allowed() -> None:
    # Strict > means a projected total that exactly equals the cap is allowed.
    budget = TenantBudget(dollars_per_run=1.0, dollars_per_period=2.0)
    check_tenant_budget(budget, "t1", run_cost=0.90, period_spend=1.90, estimate=0.10)

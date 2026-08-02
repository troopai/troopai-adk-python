from __future__ import annotations

from troopai.adk.budgets import TenantBudget
from troopai.adk.context.context_config import CompactionConfig
from troopai.adk.context.context_manager import effective_compaction_config


def test_pressure_tightens_trigger_and_preserve() -> None:
    base = CompactionConfig(enabled=True, trigger_tokens=100_000, preserve_recent_items=6, cost_aware=True)
    budget = TenantBudget(dollars_per_run=1.0)
    tightened = effective_compaction_config(base, budget, run_cost=0.90, threshold=0.8)
    assert tightened.trigger_tokens == 50_000
    assert tightened.preserve_recent_items == 1


def test_below_threshold_unchanged() -> None:
    base = CompactionConfig(enabled=True, trigger_tokens=100_000, preserve_recent_items=6, cost_aware=True)
    budget = TenantBudget(dollars_per_run=1.0)
    same = effective_compaction_config(base, budget, run_cost=0.10, threshold=0.8)
    assert same is base


def test_cost_aware_off_is_noop() -> None:
    base = CompactionConfig(enabled=True, trigger_tokens=100_000, preserve_recent_items=6, cost_aware=False)
    budget = TenantBudget(dollars_per_run=1.0)
    same = effective_compaction_config(base, budget, run_cost=0.99, threshold=0.8)
    assert same is base


def test_no_budget_is_noop() -> None:
    base = CompactionConfig(enabled=True, trigger_tokens=100_000, preserve_recent_items=6, cost_aware=True)
    same = effective_compaction_config(base, None, run_cost=0.99, threshold=0.8)
    assert same is base


def test_no_per_run_cap_is_noop() -> None:
    base = CompactionConfig(enabled=True, trigger_tokens=100_000, preserve_recent_items=6, cost_aware=True)
    budget = TenantBudget(dollars_per_period=5.0)  # no dollars_per_run
    same = effective_compaction_config(base, budget, run_cost=0.99, threshold=0.8)
    assert same is base


def test_at_threshold_tightens() -> None:
    # >= semantics: utilization exactly at the threshold tightens.
    base = CompactionConfig(enabled=True, trigger_tokens=100_000, preserve_recent_items=6, cost_aware=True)
    budget = TenantBudget(dollars_per_run=1.0)
    tightened = effective_compaction_config(base, budget, run_cost=0.80, threshold=0.8)
    assert tightened is not base
    assert tightened.trigger_tokens == 50_000


def test_small_trigger_tokens_floored_at_one() -> None:
    # max(1, 1 // 2) == 1, not 0 — guards against a compaction spiral.
    base = CompactionConfig(enabled=True, trigger_tokens=1, preserve_recent_items=2, cost_aware=True)
    budget = TenantBudget(dollars_per_run=1.0)
    tightened = effective_compaction_config(base, budget, run_cost=0.99, threshold=0.8)
    assert tightened.trigger_tokens == 1

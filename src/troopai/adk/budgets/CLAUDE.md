# Budgets Module

Per-tenant dollar-cap enforcement with pluggable cross-run accounting backends.

## Files

| File | Purpose |
|---|---|
| `budget.py` | `TenantBudget` frozen dataclass, `BudgetPeriod` enum, `period_key()` helper |
| `ledger.py` | `CostLedger` Protocol (`@runtime_checkable`) + `InMemoryCostLedger` |
| `ledgers/postgres.py` | `PostgresCostLedger` — ACID upsert, pool opened lazily on first use |
| `ledgers/redis.py` | `RedisCostLedger` — `INCRBYFLOAT` atomic increment, optional TTL |

## Architecture Decisions

| Decision | What | Why |
|----------|------|-----|
| **Calendar-period buckets** | `period_key()` maps UTC datetime → string key per `BudgetPeriod` (`HOUR`/`DAY`/`MONTH`) | Deterministic bucketing without stored timestamps; one counter row per `(tenant_id, key)` |
| **Atomic increment, no lock token** | `record()` increments in-place; no `lock_token` | The increment itself is the concurrency-safe operation — UPSERT (Postgres) or `INCRBYFLOAT` (Redis) make locking unnecessary |
| **Best-effort soft cap** | Budget gate runs pre-call on an estimate | TOCTOU window: concurrent runs may overshoot by one in-flight call each; strict enforcement requires serializing calls per tenant outside the framework |
| **Protocol mirrors checkpointer idiom** | `CostLedger` is `@runtime_checkable`, async I/O | Same pluggable-backend pattern as graph/swarm checkpointers; custom ledgers drop in without base-class inheritance |
| **`dollars_per_period` fail-fast** | `validate_budget_config()` raises `UserError` at run start when `dollars_per_period` is set without a `cost_ledger` | Misconfiguration surfaces immediately, not silently mid-run |
| **Extras-gated backends** | Postgres and Redis backends raise `ImportError` with pip install hint if their SDK is absent | No optional dep is forced on consumers who don't need it |

## Flow

```
RunConfig.validate_budget_config() → fail-fast on misconfiguration
  ↓ (per LLM call)
loop: check_tenant_budget(estimate) → raises TenantBudgetExceeded or returns
  ↓ (post call)
loop: ledger.record(tenant_id, period_key, actual_cost)
```

Enforcement logic lives in `run/cost.py:check_tenant_budget` and
`validate_budget_config`. The loop in `run/loop.py` calls both.

## Pointers

- Usage guide and snippets: `docs/cost/cost.md`
- Runnable examples: `examples/cost/tenant_budget.py`
- Runner integration fields: `run/config.py` (`tenant_budget`, `cost_ledger`)
- Profile helpers: `run/profile.py` (`tenant_budget`, `cost_ledger`)
- Extras: `pyproject.toml` (`cost-ledger-postgres`, `cost-ledger-redis`)

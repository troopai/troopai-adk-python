(guides/cost)=

# 💰 Cost

Every LLM call costs money. The ADK gives you four separable levers to
measure, control, and route around that cost — and they compose freely.

## Four mechanisms at a glance

| Mechanism | When it runs | What it does |
|---|---|---|
| **Estimator** (`LLM.estimate_cost`) | Before each LLM call | Predicts token cost as a `CostEstimate` |
| **Ledger** (`CostLedger` Protocol) | After each LLM call | Records actual spend; multiple backends |
| **Router** (`LLMRouter` ABC) | Before each LLM call | Picks the cheapest or fastest candidate |
| **Budget** (`TenantBudget`) | Throughout the run | Hard ceiling; raises or warns on breach |

Rule: estimator predicts, ledger records, router selects, budget enforces.
None of these is active unless you configure it — the framework never adds
cost behavior you did not opt into.

See {doc}`/concepts/index` for the architectural picture.

---

## `LLM.estimate_cost` — pre-call prediction

Before each LLM call the runner can predict its cost. Every `LLM`
subclass exposes `estimate_cost`:

```python
from troopai.adk.llms import LiteLLM

llm = LiteLLM(model="gpt-4o-mini")
estimate = llm.estimate_cost(messages, model="gpt-4o-mini", max_output_tokens=512)

print(estimate.input_tokens)            # precisely counted via TokenCounter
print(estimate.estimated_output_tokens) # equals max_output_tokens if set, else 0
print(estimate.estimated_cost_usd)      # float or None (no cost table for this model)
print(estimate.output_bounded)          # True only when max_output_tokens was supplied
```

The return type is a frozen `CostEstimate` dataclass (`llms/cost.py`):

| Field | Type | Semantics |
|---|---|---|
| `model` | `str` | The model the estimate is for |
| `input_tokens` | `int` | Precisely counted input tokens |
| `estimated_output_tokens` | `int` | `max_output_tokens` when set; 0 otherwise |
| `estimated_cost_usd` | `float \| None` | Dollar estimate, or `None` when the provider has no cost table |
| `output_bounded` | `bool` | `True` when `max_output_tokens` bounded the estimate |

**Key guarantees.** Input tokens are counted precisely using the same
`TokenCounter` the runner uses internally. Output is bounded by
`max_output_tokens` when you supply it; when you do not, no token
count is invented — the estimate is an *input-only floor* and
`estimated_output_tokens` is 0. Callers must treat
`estimated_cost_usd is None` as "cost unavailable" and must never
block a run solely on a missing estimate.

`LiteLLM` overrides `LLM.cost()` with litellm's pricing tables so
`estimate_cost` returns a real dollar figure for hundreds of supported
models. The base `LLM.cost()` returns `None`; custom providers that
can price calls should override it.

The framework calls `estimate_cost` automatically when a
`TenantBudget` or a `CheapestFirstRouter` is active. You can also
call it directly to surface cost information in your own tooling.

---

## `CostLedger` Protocol — post-call recording

Every completed LLM call appends its actual cost to a *ledger*. The
ledger is a `@runtime_checkable` Protocol (`budgets/ledger.py`):

```python
from troopai.adk.budgets import CostLedger  # the Protocol

class CostLedger(Protocol):
    async def spend(self, tenant_id: str, period_key: str) -> float: ...
    async def record(self, tenant_id: str, period_key: str, cost_usd: float) -> None: ...
```

`record` is a pure atomic increment — each call adds `cost_usd` to
the running total for `(tenant_id, period_key)`. The period key is a
UTC calendar string (`"2026-05-27"` for a DAY bucket, `"2026-05T10"`
for HOUR, `"2026-05"` for MONTH) produced by `period_key()` from
`budgets/budget.py`.

### Backend implementations

Three backends ship out of the box. All satisfy the `CostLedger` Protocol
without inheriting from it — structural typing, not nominal.

**`InMemoryCostLedger`** — process-local, no external dependency. Default
for single-process deployments and tests:

```python
from troopai.adk.budgets import InMemoryCostLedger

ledger = InMemoryCostLedger()
```

**`PostgresCostLedger`** — ACID-durable. Uses `INSERT … ON CONFLICT … DO
UPDATE` so concurrent runners for the same tenant are safe. Requires
PostgreSQL 9.5+ and the `cost-ledger-postgres` extra:

```python
from troopai.adk.budgets.ledgers.postgres import PostgresCostLedger

ledger = PostgresCostLedger(conninfo="postgresql://user:pass@host/db")
# Call ledger.close() at shutdown.
```

```
pip install 'troopai-adk-python[cost-ledger-postgres]'
```

**`RedisCostLedger`** — uses `INCRBYFLOAT` for lock-free atomic
increments. Supports an optional TTL so stale period buckets
self-evict (requires Redis 7.0+ for `EXPIRE … NX`). Requires the
`cost-ledger-redis` extra:

```python
from troopai.adk.budgets.ledgers.redis import RedisCostLedger

ledger = RedisCostLedger(url="redis://localhost:6379/0", ttl_seconds=90000)
# Or from a pre-configured client (caller owns lifecycle):
ledger = RedisCostLedger(client=my_redis_client)
```

```
pip install 'troopai-adk-python[cost-ledger-redis]'
```

The Protocol pattern mirrors the checkpointer idiom used by graph and
swarm checkpointers. Custom ledgers — for example a DynamoDB backend —
implement `spend` and `record` as async methods; no base class is
required.

---

## `LLMRouter` ABC — model selection per call

A router returns an *ordered candidate list*; the runner tries each
in turn, escalating to the next on failure. The `LLMRouter` ABC lives
in `llms/routing/router.py`:

```python
from troopai.adk.llms.routing import LLMRouter, RoutedModel, RoutingContext

class LLMRouter(ABC):
    def candidates(self, ctx: RoutingContext) -> Sequence[RoutedModel]: ...
    def should_escalate(self, response: LLMResponse | None) -> bool: ...
```

`RoutingContext` carries `messages` (Layer-1 input), `tenant_id`, and
`run_cost` (accumulated USD this run), so routers can make
context-aware decisions. `RoutedModel` bundles an `LLM` instance
with its model name string.

### `CheapestFirstRouter`

Orders candidates by estimated input cost ascending. Candidates whose
cost table is absent sort last, so priced models are tried before
unpriced ones:

```python
from troopai.adk.llms.routing import CheapestFirstRouter, RoutedModel
from troopai.adk.llms import LiteLLM

router = CheapestFirstRouter(models=[
    RoutedModel(llm=LiteLLM(model="gpt-4o-mini"), model="gpt-4o-mini"),
    RoutedModel(llm=LiteLLM(model="gpt-4o"),      model="gpt-4o"),
])
```

`CheapestFirstRouter` calls `LLM.estimate_cost` once per candidate on
every invocation (token counting is O(candidates × input tokens)), so
keep the candidate list small on hot paths.

### `LatencyFirstRouter`

Orders candidates by a developer-supplied latency map
(`model_name → observed_latency_ms`). Models absent from the map sort
last. The `troopai.agent.turn.duration_ms` OTel histogram is a natural
data source:

```python
from troopai.adk.llms.routing import LatencyFirstRouter

router = LatencyFirstRouter(
    models=[...],
    latencies={"gpt-4o-mini": 450.0, "gpt-4o": 1200.0},
)
```

### Escalation triggers

The runner escalates to the next candidate on:

1. A provider exception from the current candidate's LLM call.
2. An output-schema validation failure.
3. `should_escalate(response)` returning `True` (custom content check).

`TroopAIError` subclasses — including `TenantBudgetExceeded` and
guardrail rejections — propagate directly to the caller and do **not**
trigger escalation. When all candidates are exhausted,
`NoRoutingCandidateError` is raised.

In streaming mode, escalation is only possible before the first token
is yielded; once token streaming begins the runner commits to the
current candidate.

Wire the router via `RunConfig` or a runner profile:

```python
from troopai.adk.run import RunConfig

config = RunConfig(router=router)
# or
result = await Runner.configure().router(router).agent(agent).arun(prompt)
```

---

## `TenantBudget` — hard per-run ceiling

`TenantBudget` is a frozen dataclass that enforces a dollar cap on a
run's accumulated spend and on cross-run period spend.

```python
from troopai.adk.budgets import TenantBudget, BudgetPeriod
from troopai.adk.run import RunConfig
from troopai.adk.run.context import RunContext

budget = TenantBudget(
    dollars_per_run=0.10,        # cap on a single run
    dollars_per_period=5.00,     # cross-run cap in one calendar window
    period=BudgetPeriod.DAY,     # HOUR / DAY / MONTH (UTC)
    kill_on_exceed=True,         # True = raise; False = warn and continue
)

config = RunConfig(tenant_budget=budget, cost_ledger=ledger)
ctx = RunContext(context=None, tenant_id="tenant-abc")

result = await Runner.arun(agent, prompt, context=ctx, run_config=config)
```

| Field | Default | Description |
|---|---|---|
| `dollars_per_run` | `None` | Dollar cap on a single run's accumulated cost |
| `dollars_per_period` | `None` | Cap on cross-run spend in one calendar window; requires a `cost_ledger` |
| `period` | `BudgetPeriod.DAY` | Calendar granularity for `dollars_per_period` |
| `kill_on_exceed` | `True` | `True` raises `TenantBudgetExceeded`; `False` warns and continues |

**What happens on exhaustion.** The gate runs pre-call using
`estimate_cost`. When the projected total exceeds the cap and
`kill_on_exceed=True`, the runner raises `TenantBudgetExceeded` (a
subclass of `TroopAIError`). This exception propagates unchanged — the
router does not escalate on it, and the runner short-circuits without
further LLM calls.

**`dollars_per_period` requires a `cost_ledger`.** The framework
validates this at run start (`validate_budget_config`) and raises
`UserError` before the first LLM call if the ledger is missing. A
`dollars_per_run` cap has no such requirement: accumulation is tracked
in-memory on `RunContext.cost_usd`.

**Best-effort semantics.** The check is pre-call. Two concurrent runs
for the same tenant can both pass the gate before either records its
actual spend, so the true accumulated cost may overshoot the cap by at
most one in-flight call per concurrent request. Applications that need
strict enforcement should serialize LLM calls per tenant outside the
framework.

---

## Cost-aware compaction

When conversation history grows beyond a token threshold, the context
manager can compact (summarize) older turns to shed input tokens —
the dominant cost driver. Setting `cost_aware=True` on
`CompactionConfig` tightens compaction automatically as the run
approaches its per-run budget:

```python
from troopai.adk.context import ContextManagementConfig, CompactionConfig
from troopai.adk.budgets import TenantBudget
from troopai.adk.run import RunConfig

config = RunConfig(
    tenant_budget=TenantBudget(dollars_per_run=0.10),
    context_management=ContextManagementConfig(
        compaction=CompactionConfig(
            enabled=True,
            cost_aware=True,  # tighten under budget pressure
        ),
    ),
)
```

Under budget pressure, compaction triggers earlier and preserves fewer
recent items. The pressure threshold reuses
`token_budget_warning_threshold` (default 80 %). Without a budget,
`cost_aware` has no effect.

Compaction is off by default (`CompactionConfig.enabled` defaults to
`False`). Every compaction behavior is opt-in; no tokens are
summarized without explicit configuration. This follows the
cost-conservative defaults principle: default values affecting token
cost must be off or bounded, and developers never opt out of cost they
did not choose.

---

## Per-tenant cost attribution

Ledger entries are keyed by `tenant_id`. Set `RunContext.tenant_id`
to wire attribution into the ledger, budget gate, OTel spans, and
metric dimensions in one step:

```python
ctx = RunContext(context=my_app_context, tenant_id="tenant-abc")
result = await Runner.arun(agent, prompt, context=ctx, run_config=config)
```

The runner threads `tenant_id` to:

- `CostLedger.record` and `CostLedger.spend` (keyed per tenant)
- `TenantBudget` gate checks
- OTel span attribute `troopai.tenant.id`
- The `tenant` metric dimension on histograms and counters

When `tenant_id` is `None`, all cost features still operate on the
run-level `RunContext.cost_usd` accumulator; ledger entries and
period caps are skipped because there is no tenant to attribute them
to.

For governance and billing pipelines, treat the `CostLedger` backend
as the source of truth for cross-run spend. Query it with
`await ledger.spend(tenant_id, period_key(BudgetPeriod.MONTH, now))`.

---

## Inspecting cost at run time

`RunContext.cost_usd` accumulates the run's actual spend in real time.
`RunResult.context` carries the final `RunContext`:

```python
result = await Runner.arun(agent, "Summarise this document", run_config=config)

# Total USD for this run (0.0 when the provider has no cost table)
print(f"Run cost: ${result.context.cost_usd:.6f}")

# Token breakdown (input, output, cached, reasoning)
usage = result.context.usage
print(f"Input tokens:  {usage.input_tokens}")
print(f"Output tokens: {usage.output_tokens}")
print(f"Requests:      {usage.requests}")
```

`LLMUsage` accumulates via `__add__` across every LLM call in the run,
including multi-agent swarms and handoffs. Access the per-request
breakdown via `usage.usage` (a `list[LLMSingleRequestUsage]`) when
you need call-level detail.

For cross-run period spend, query the ledger directly:

```python
from datetime import datetime, timezone
from troopai.adk.budgets import BudgetPeriod
from troopai.adk.budgets.budget import period_key

month_key = period_key(BudgetPeriod.MONTH, datetime.now(timezone.utc))
month_spend = await ledger.spend("tenant-abc", month_key)
print(f"Month-to-date: ${month_spend:.4f}")
```

---

## Common patterns

### Cap per-user spend

Assign each user a `tenant_id` and configure a `TenantBudget` with a
per-day or per-month cap backed by a shared `PostgresCostLedger` or
`RedisCostLedger`. The framework enforces the cap pre-call; your
billing pipeline reads the same ledger for invoicing.

```python
budget = TenantBudget(dollars_per_period=1.00, period=BudgetPeriod.MONTH)
config = RunConfig(tenant_budget=budget, cost_ledger=shared_ledger)
```

### Route to a cheaper model for the first N turns

Use `RoutingContext.run_cost` to escalate from a cheap model to an
expensive one only when the response warrants it:

```python
from troopai.adk.llms.routing import LLMRouter, RoutedModel, RoutingContext
from troopai.adk.types.responses.llm_response import LLMResponse

class BudgetThenPremiumRouter(LLMRouter):
    def __init__(self, cheap: RoutedModel, premium: RoutedModel) -> None:
        self._cheap = cheap
        self._premium = premium

    def candidates(self, ctx: RoutingContext) -> list[RoutedModel]:
        # Use the premium model once the run has already spent 5 cents
        if ctx.run_cost > 0.05:
            return [self._premium, self._cheap]
        return [self._cheap, self._premium]
```

### Ledger as billing source

The `CostLedger` Protocol is intentionally minimal so it maps cleanly
onto a billing database row. Implement the two methods against your
existing storage layer — no base class, no framework coupling — and
pass your instance as `RunConfig.cost_ledger`.

---

## See also

- {doc}`/concepts/index` — Concepts: Cost mechanisms table
- {doc}`/context/context_management` — full compaction configuration reference
- `examples/cost/` — runnable demos (pre-call estimation, tenant budgets, smart routing, cost-aware compaction)

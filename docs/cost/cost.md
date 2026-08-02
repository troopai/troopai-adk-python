# Cost Governance and Smart Routing

This document covers the cost-control features shipped in this release: pre-call
cost estimation, per-tenant budget caps, pluggable cost ledger backends,
model routing with automatic fallback, and cost-aware compaction.

All features are **opt-in and purely additive**. Existing agent configurations
are unaffected; no cost-related behavior is injected unless explicitly configured.

---

## Pre-call cost estimation

Before each LLM call the framework can estimate the dollar cost of that call.
The estimate is available on any `LLM` subclass via `estimate_cost`:

```python
estimate = llm.estimate_cost(messages, model, max_output_tokens=512)

print(estimate.input_tokens)          # precisely counted
print(estimate.estimated_output_tokens)  # bounded by max_output_tokens if set
print(estimate.estimated_cost_usd)    # float or None
print(estimate.output_bounded)        # True when max_output_tokens was supplied
```

**Semantics:**

- `input_tokens` — counted precisely via the same token-counter used by the
  runner.
- `estimated_output_tokens` — equals `max_output_tokens` when provided, else 0.
  No default output token count is invented; the estimate is an **input-only
  floor** when output length is unbounded.
- `estimated_cost_usd` — `None` when the provider has no cost table. Callers
  must treat `None` as "cost unavailable" and never block a run solely because
  an estimate returned `None`.
- `output_bounded` — `True` only when `max_output_tokens` was supplied.

The returned `CostEstimate` is a frozen dataclass; the `LLM` ABC provides the
default implementation and `LiteLLM` overrides `cost()` with litellm's pricing
tables so `estimate_cost` returns a real dollar figure for supported models.

---

## Per-tenant budgets

A `TenantBudget` attaches dollar limits to a run's tenant. The tenant identity
comes from `RunContext.tenant_id` — set this field when starting a run for a
specific customer.

```python
from troopai.adk.budgets import TenantBudget, BudgetPeriod
from troopai.adk.run import RunConfig
from troopai.adk.context import RunContext

budget = TenantBudget(
    dollars_per_run=0.10,       # hard cap per single run
    dollars_per_period=5.00,    # cross-run cap per calendar period
    period=BudgetPeriod.DAY,    # HOUR / DAY / MONTH (calendar UTC)
    kill_on_exceed=True,        # True = raise; False = warn and continue
)

config = RunConfig(tenant_budget=budget, cost_ledger=my_ledger)
ctx = RunContext(tenant_id="tenant-abc")

result = await Runner.arun(agent, prompt, context=ctx, run_config=config)
```

Or via a reusable runner profile:

```python
result = await (
    Runner.configure(context=ctx)
    .tenant_budget(budget)
    .cost_ledger(ledger)
    .agent(agent)
    .arun(prompt)
)
```

**Fields:**

| Field | Default | Description |
|---|---|---|
| `dollars_per_run` | `None` | Dollar cap on a single run's accumulated cost. |
| `dollars_per_period` | `None` | Cap on cross-run spend in one calendar window. Requires a `cost_ledger`. |
| `period` | `BudgetPeriod.DAY` | Calendar granularity for `dollars_per_period`. |
| `kill_on_exceed` | `True` | `True` raises `TenantBudgetExceeded`; `False` logs a warning and emits the breach event, then continues. |

**Budget enforcement is best-effort (soft cap).** The gate runs pre-call
using an estimated cost. Because two concurrent runs for the same tenant can
both pass the gate before either records its actual spend, the true accumulated
cost may exceed the configured cap by up to one in-flight call's worth per
concurrent request. This is a TOCTOU window inherent to distributed pre-call
checks; applications that need strict enforcement should serialize LLM calls
per tenant outside the framework.

When `kill_on_exceed=True` and the gate fires, the runner raises
`TenantBudgetExceeded` (a subclass of `TroopAIError`). This exception propagates
unchanged — the router does **not** escalate on it.

**`dollars_per_period` requires a `cost_ledger`** (fail-fast at run start via
`UserError` before any LLM call). A `dollars_per_run` cap does not need a
ledger; accumulation is tracked in-memory on `RunContext.cost_usd`.

---

## Cost ledger (per-period accounting)

A cost ledger provides the cross-run spend counter that `dollars_per_period`
reads and writes. The interface is a `@runtime_checkable` Protocol:

```python
class CostLedger(Protocol):
    async def spend(self, tenant_id: str, period_key: str) -> float: ...
    async def record(self, tenant_id: str, period_key: str, cost_usd: float) -> None: ...
```

`record` is an atomic increment — no optimistic-locking token is needed.

Three backends ship out of the box:

### InMemoryCostLedger

Process-local; suitable for single-process deployments and tests. No extras
required.

```python
from troopai.adk.budgets import InMemoryCostLedger

ledger = InMemoryCostLedger()
```

### PostgresCostLedger

ACID-durable ledger. Each `(tenant_id, period_key)` pair maps to one row;
`record` uses an atomic `INSERT … ON CONFLICT … DO UPDATE` so concurrent
runners for the same tenant are safe. Requires PostgreSQL 9.5+.

```python
from troopai.adk.budgets.ledgers.postgres import PostgresCostLedger

ledger = PostgresCostLedger(conninfo="postgresql://user:pass@host/db")
# Call ledger.close() at application shutdown.
```

Install the extra:

```
pip install 'troopai-adk-python[cost-ledger-postgres]'
```

### RedisCostLedger

Fast ephemeral ledger. Uses `INCRBYFLOAT` for atomic increments. Supports an
optional TTL for self-evicting stale windows.

```python
from troopai.adk.budgets.ledgers.redis import RedisCostLedger

# From a URL (this instance owns the client):
ledger = RedisCostLedger(url="redis://localhost:6379/0", ttl_seconds=90000)

# From a pre-configured client (caller owns the lifecycle):
ledger = RedisCostLedger(client=my_redis_client)
```

**TTL semantics:** set `ttl_seconds` to at least as long as the budget period
so the key outlives the window it guards. The TTL is stamped once on bucket
creation (using the Redis `EXPIRE … NX` flag, requiring Redis server 7.0+) and
never extended on subsequent writes. `None` (the default) keeps keys
indefinitely — the cost-conservative default; TTL eviction is opt-in.

Install the extra:

```
pip install 'troopai-adk-python[cost-ledger-redis]'
```

---

## Smart routing

A `LLMRouter` returns an ordered list of candidates; the runner tries them in
order, escalating to the next on failure.

```python
from troopai.adk.llms.routing import CheapestFirstRouter, RoutedModel
from troopai.adk.llms import LiteLLM

router = CheapestFirstRouter(models=[
    RoutedModel(llm=LiteLLM(model="gpt-4o-mini"), model="gpt-4o-mini"),
    RoutedModel(llm=LiteLLM(model="gpt-4o"), model="gpt-4o"),
])

config = RunConfig(router=router)
```

Or via a runner profile:

```python
result = await Runner.configure().router(router).agent(agent).arun(prompt)
```

### Built-in routers

**`CheapestFirstRouter`** — orders candidates by estimated input cost
(ascending). Candidates with no cost table (`estimated_cost_usd is None`) sort
last so priced models are tried before unpriced ones. Because each call to
`candidates()` re-estimates per-candidate, keep the candidate list small on
hot paths.

**`LatencyFirstRouter`** — orders candidates by a developer-supplied latency
map (`model_name -> observed_latency_ms`). Models absent from the map sort
last. The `troopai.agent.turn.duration_ms` histogram (emitted by the metrics
subsystem) is a natural source for this map.

### Custom routers

Subclass `LLMRouter` and implement `candidates()`. Override `should_escalate()`
to drive escalation from response content:

```python
from troopai.adk.llms.routing import LLMRouter, RoutedModel, RoutingContext
from troopai.adk.types.responses.llm_response import LLMResponse

class MyRouter(LLMRouter):
    def candidates(self, ctx: RoutingContext) -> list[RoutedModel]:
        ...

    def should_escalate(self, response: LLMResponse | None) -> bool:
        # Return True to move to the next candidate.
        return False
```

`RoutingContext` exposes `messages` (the Layer-1 input), `tenant_id`, and
`run_cost` (accumulated USD this run) so routers can make context-aware
decisions.

### Escalation triggers

The runner escalates to the next candidate on:

1. **Provider exception** from the current candidate's LLM call.
2. **Output-schema validation failure** — the response did not conform to the
   agent's `output_schema`.
3. **`should_escalate(response)` returns `True`** — custom content-based check.

The runner does **not** escalate on framework exceptions (`TroopAIError`
subclasses, including `TenantBudgetExceeded` and guardrail rejections). These
propagate directly to the caller.

When all candidates are exhausted without a successful response, the runner
raises `NoRoutingCandidateError`.

### Streaming and routing

In streaming mode, escalation is only possible **before the first token** is
yielded. Once token streaming begins, the runner commits to the current
candidate. `should_escalate` is never called mid-stream.

### Budget composition with routing

When a `tenant_budget` and a `router` are both active, the budget gate runs
per-candidate before each attempt. A candidate that would exceed the budget
triggers the budget exception rather than escalation — budget kills are not
routing failures.

### Future work

`QualityFirstRouter` (routes based on evaluation scores from the eval
framework) is deferred pending the eval framework completion.

---

## Cost-aware compaction

When the context manager is configured with `cost_aware=True` and a
per-run `TenantBudget` is active, compaction tightens as the run approaches
its budget:

```python
from troopai.adk.context import ContextManagementConfig, CompactionConfig

config = RunConfig(
    tenant_budget=TenantBudget(dollars_per_run=0.10),
    context_management=ContextManagementConfig(
        compaction=CompactionConfig(
            enabled=True,
            cost_aware=True,      # tighten under budget pressure
        ),
    ),
)
```

When budget pressure is detected, compaction triggers earlier and preserves
fewer recent items, shedding input tokens (the dominant cost driver). The
pressure threshold reuses `token_budget_warning_threshold` (default 80%).

`cost_aware` defaults to `False` — compaction behavior is opt-in and unchanged
when no budget is set.

---

## See also

- `examples/cost/` — runnable demos for each feature (pre-call estimation,
  tenant budgets, smart routing, cost-aware compaction).
- `examples/observability/tenant_and_cost.py` — combining tenant tracing with
  cost governance.
- `docs/context/context_management.md` — full compaction configuration
  reference.

# Sandbox Cost and Billing

The sandbox module models cost at three levels: a static rate card per backend
(`SandboxCostDescriptor`), a per-command computed estimate accumulated into
`SandboxUsage`, and an optional provider-reported session total (`billed_cost_usd`)
retrieved when live billing is enabled.

## `SandboxCostDescriptor`

Each backend client declares a `cost` class attribute of type
`SandboxCostDescriptor`. The descriptor is the backend's static rate card.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `usd_per_minute` | `float` | `0.0` | Dollar cost per wall-clock minute of session life |
| `usd_per_cpu_second` | `float \| None` | `None` | Optional CPU-second rate (live-billing reconciliation) |
| `usd_per_gb_second` | `float \| None` | `None` | Optional GiB-second memory rate (live-billing reconciliation) |
| `free` | `bool` | `False` | When `True` the backend costs nothing (self-hosted / local) |

Two methods:

- `rate_key() -> float` — scalar used to rank backends cheapest-first.
  Returns `0.0` when `free=True`, otherwise returns `usd_per_minute`.
- `cost_for_ms(duration_ms: int) -> float` — computed dollar cost for a
  command of `duration_ms` wall-clock milliseconds. Returns `0.0` when
  `free=True`, otherwise `usd_per_minute * (duration_ms / 60000.0)`.

## Per-backend rate table

Rates are approximate starting points. Every backend exposes `cost` as a
class attribute, so you can subclass a client and override `cost` to reflect
your actual contracted rate.

| Backend | Class | `usd_per_minute` | `free` | `network` | `persistent` |
|---|---|---|---|---|---|
| Local subprocess | `LocalSubprocessSandboxClient` | — | `True` | `True` | `False` |
| Docker container | `DockerSandboxClient` | — | `True` | `True` | `True` |
| Kubernetes pod | `K8sPodSandboxClient` | — | `True` | `True` | `True` |
| E2B | `E2bSandboxClient` | `0.06` | `False` | `True` | `True` |
| Cloudflare | `CloudflareSandboxClient` | `0.05` | `False` | `True` | `False` |
| Daytona | `DaytonaSandboxClient` | `0.08` | `False` | `True` | `True` |
| Blaxel | `BlaxelSandboxClient` | `0.09` | `False` | `True` | `True` |
| Modal | `ModalSandboxClient` | `0.10` | `False` | `True` | `True` |
| Runloop | `RunloopSandboxClient` | `0.10` | `False` | `True` | `True` |
| Vercel | `VercelSandboxClient` | `0.12` | `False` | `True` | `False` |

All priced rates are in USD per minute. Check each provider's pricing page
for the current contracted rate; these values are used to rank backends
cheapest-first and to compute per-command estimates.

## `SandboxBackendCapabilities`

Each backend client declares a `capabilities` class attribute of type
`SandboxBackendCapabilities`. The selector matches these against a run's
`SandboxRequirements`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `network` | `bool` | `False` | Backend grants outbound network access |
| `persistent` | `bool` | `False` | Backend offers a persistent (non-ephemeral) workspace |
| `max_cpu` | `int \| None` | `None` | Maximum CPU count available (`None` = unknown) |
| `max_memory_mb` | `int \| None` | `None` | Maximum memory in MiB (`None` = unknown) |
| `regions` | `tuple[str, ...]` | `()` | Regions the backend can run in |

`capabilities.satisfies(requirements)` returns `True` when the backend meets
every field the `SandboxRequirements` states.

## `SandboxUsage` cost fields

`SandboxUsage` accumulates resource counters for one or more sandbox sessions.
Two cost fields live on the accumulator (a third, `cost_usd`, is the per-command
figure on `SandboxSingleExecUsage`):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `computed_cost_usd` | `float` | `0.0` | Sum of per-command `cost_usd` values (rate-card estimate) |
| `billed_cost_usd` | `float \| None` | `None` | Provider-reported session cost; set only when live billing ran |

`computed_cost_usd` grows with every command: after each `run_command` call
the framework computes `cost.cost_for_ms(duration_ms)` and folds the result
into the accumulator. A free backend (`free=True`) yields
`cost_for_ms == 0.0`, so its per-command `cost_usd` is `0.0` and
`computed_cost_usd` stays `0.0`. A per-command `cost_usd` is `None` only when
`client.cost` is `None` — a custom backend that declares no rate card (every
shipped backend declares one); a `None` per-command cost is skipped in the
fold, so `computed_cost_usd` stays `0.0`.

`billed_cost_usd` is `None` until live billing is enabled and
`fetch_billing` returns a `SandboxBillingRecord`. If both values are set,
`computed_cost_usd` is the rate-card estimate and `billed_cost_usd` is the
provider-reported figure; treat them as complementary signals.

`SandboxUsage` is exposed on `RunResult.sandbox_usage` after the run
completes. Access it to log cost, enforce per-run budgets, or route usage
records to a billing system.

```python
result = await Runner.arun(agent, prompt, run_config=run_config)
if result.sandbox_usage is not None:
    usage = result.sandbox_usage
    print(f"commands run: {usage.exec_count}")
    print(f"rate-card estimate: ${usage.computed_cost_usd:.6f}")
    if usage.billed_cost_usd is not None:
        print(f"provider-reported: ${usage.billed_cost_usd:.6f}")
```

The sandbox session is bracketed once per `Runner.arun`: a single
`SandboxUsage` accumulates every command for the run (handoffs between
`SandboxAgent`s execute inside that one bracket) and is attached directly to
`RunResult.sandbox_usage` — there is no cross-session merge step.
`SandboxUsage` also supports `__add__` for explicitly aggregating usage
across separate runs: counters sum elementwise, `memory_peak_mb` takes the
max, `executions` are concatenated, and `billed_cost_usd` sums only when at
least one operand has a non-`None` value.

## Live billing (opt-in)

Setting `SandboxRunConfig(capture_live_cost=True)` tells the Runner to call
`client.fetch_billing(session)` during teardown, after the session stops.
The result, if non-`None`, is stored in `SandboxUsage.billed_cost_usd`.

`capture_live_cost` defaults to `False`. This is the cost-conservative
default: the billing API call is a network round-trip that the developer
must explicitly opt into.

`fetch_billing` is best-effort: any exception from the provider call is
suppressed (and logged at `DEBUG`) so a billing-endpoint failure never fails
the run. A `None` return means no provider-reported figure is available for
this session — `computed_cost_usd` is the estimate to use. The `DEBUG` log
is what distinguishes a thrown billing error from a backend that simply
reports no per-sandbox cost.

### E2B

E2B meters compute usage at the account level, not per sandbox. The
`E2bSandboxClient.fetch_billing` override therefore returns `None` by
design. `computed_cost_usd` (the rate-card estimate) is the per-run cost
approximation for E2B runs. A per-sandbox cost endpoint would be wired
through this method if E2B exposes one in the future.

### Other backends

`BaseSandboxClient.fetch_billing` returns `None` by default. Hosted-bridge
subclasses can override it to call the provider's billing API and return a
`SandboxBillingRecord(cost_usd=..., currency="USD", unit=..., raw=...)`.

## `SandboxBillingRecord`

`SandboxBillingRecord` is a Pydantic `BaseModel` (received + validated):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `cost_usd` | `float` | — | Dollar cost the provider reported for the session |
| `currency` | `str` | `"USD"` | ISO currency code |
| `unit` | `str \| None` | `None` | Provider billing unit label (e.g. `"compute-seconds"`) |
| `raw` | `dict \| None` | `None` | Untouched provider payload for audit / debugging |

See [selection.md](selection.md) for how the rate card feeds backend selection.
See [observability.md](observability.md) for how usage is accumulated and surfaced via hooks.

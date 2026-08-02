# Observability

The observability stack layers span tracing, OTel metric instruments, and
structured logging. All three are **opt-in**: by default the framework runs
with zero observability overhead.

## Span-attribute convention

`TracingConvention` selects the attribute vocabulary the OTel bridge emits
when it serializes a framework span to an OpenTelemetry span:

| Convention | What it emits | Best for |
|---|---|---|
| `TracingConvention.DEFAULT` | GenAI semconv (`gen_ai.*`) + framework (`troopai.*`) attributes | Jaeger, Honeycomb, Datadog, Grafana Tempo, generic OTLP collectors |
| `TracingConvention.OPENINFERENCE` | OpenInference keys (`openinference.span.kind`, `input.value`, `output.value`, `llm.token_count.*`) | Phoenix / Arize — read natively by their LLM dashboards without an adapter |

The convention is a **setup-time choice** passed to `setup_otel(convention=...)`:

```python
from troopai.adk.tracing import TracingConvention, set_tracer
from troopai.adk.tracing.otel import setup_otel

tracer = setup_otel(
    service_name="my-agent",
    convention=TracingConvention.OPENINFERENCE,
)
set_tracer(tracer)
```

`TracingConvention` is unconditionally importable from `troopai.adk.tracing`
with no optional extras required.

**Privacy note.** Selecting `TracingConvention.OPENINFERENCE` causes span
`input.value` and `output.value` to carry LLM prompt/completion content and
tool I/O to the configured backend — this is the design intent of the
OpenInference convention (capturing prompts and responses for the LLM
dashboard). Tool I/O is credential-redacted and truncated by default using the
same gate as `TracingConvention.DEFAULT` (Bearer tokens, API-key shapes, PEM
private keys, and common secret JSON fields are replaced with redaction markers
before export). Pass `record_tool_io_full=True` to `setup_otel` or
`OTelTracer` to opt in to raw tool I/O. Treat the configured backend as having
visibility into prompt and response content at all times.

### Span-kind → OpenInference-kind mapping

| Framework factory | OpenInference `openinference.span.kind` |
|---|---|
| `agent_span` | `AGENT` |
| `generation_span` | `LLM` |
| `function_span` (tool) | `TOOL` |
| `function_span` (MCP) | `TOOL` |
| `handoff_span` | `AGENT` |
| `guardrail_span` | `GUARDRAIL` |
| `response_span` | `LLM` |
| `custom_span` | `CHAIN` (inner `data["type"]`: `graph`/`graph_superstep`/`graph_node` → `CHAIN`; `swarm`/`swarm_turn` → `AGENT`; `sandbox` → `TOOL`; unrecognised or absent → `CHAIN`) |

## Metric instruments

`MetricsTracer` records OTel metric instruments from typed `SpanData`
payloads at `Span.finish()`. It integrates with no extra protocol — compose
it alongside any span tracer in a `MultiTracer`:

```python
from troopai.adk.tracing import MetricsTracer, MultiTracer, set_tracer
from troopai.adk.tracing.metrics import setup_metrics
from troopai.adk.tracing.otel import setup_otel

otel = setup_otel(service_name="my-agent")
metrics: MetricsTracer = setup_metrics(service_name="my-agent")
set_tracer(MultiTracer([otel, metrics]))
```

### Enabling metric recording per run

`RunConfig.metrics_enabled` (default `False`) gates instrument recording.
It is **independent** of `RunConfig.tracing_enabled`: both flags can be
active at the same time, or individually:

```python
from troopai.adk.run.config import RunConfig

config = RunConfig(tracing_enabled=True, metrics_enabled=True)
```

When `metrics_enabled=False`, the `MetricsTracer` span objects are still
created (they are cheap) but `MetricSpan.finish()` skips the instrument
recording.

### Instrument table

| Instrument name | Type | Unit | Recorded from |
|---|---|---|---|
| `troopai.agent.turn.duration_ms` | Histogram | `ms` | `AgentSpanData` at finish |
| `troopai.llm.tokens.prompt` | Histogram | `{token}` | `GenerationSpanData.usage` — `prompt_tokens` or `input_tokens` |
| `troopai.llm.tokens.completion` | Histogram | `{token}` | `GenerationSpanData.usage` — `completion_tokens` or `output_tokens` |
| `troopai.llm.requests` | Counter | `1` | `GenerationSpanData` at finish; label `status=success\|error` |
| `troopai.agent.tool.calls` | Counter | `1` | `FunctionSpanData` at finish; label `status=success\|error` |
| `troopai.graph.node.duration_ms` | Histogram | `ms` | `custom_span` with `data["type"]="graph_node"` |
| `troopai.swarm.turn.duration_ms` | Histogram | `ms` | `custom_span` with `data["type"]="swarm_turn"` |

`setup_metrics` wires a `MeterProvider` with a
`PeriodicExportingMetricReader` (default export interval: 60 s) and returns a
`MetricsTracer` bound to the installed provider.

## Exporter helpers

Four thin setup helpers cover the most common backends. All require the
`otel` extra (`pip install 'troopai-adk-python[otel]'`).

### Phoenix (Arize)

Phoenix ingests OTLP and reads OpenInference attributes natively. The
`setup_phoenix` helper is `setup_otel` pre-configured with
`TracingConvention.OPENINFERENCE`:

```python
from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.exporters import setup_phoenix

set_tracer(setup_phoenix(service_name="my-agent"))
```

Set `endpoint` to your Phoenix OTLP collector URL, or leave it as `None` to
read `OTEL_EXPORTER_OTLP_ENDPOINT` from the environment.

### Logfire (Pydantic)

```python
from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.exporters import setup_logfire

set_tracer(setup_logfire(token="<write-token>", service_name="my-agent"))
```

The Logfire OTLP ingestion endpoint accepts the write token as the raw
`Authorization` header value (no `Bearer` prefix).

### LangSmith

```python
from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.exporters import setup_langsmith

set_tracer(setup_langsmith(api_key="<ls-api-key>", project="my-project"))
```

### Helicone (gateway, not a span exporter)

Helicone is an LLM proxy gateway — it intercepts LLM traffic rather than
consuming OTLP spans. Wire its values into your LLM config instead of
calling `set_tracer`:

```python
import os
from troopai.adk.tracing.exporters import setup_helicone

cfg = setup_helicone(api_key=os.environ["HELICONE_API_KEY"])
# cfg.base_url  →  set as provider base_url
# cfg.headers   →  merge into provider extra_headers
```

## Structured logging

`log_event` is a thin helper that emits a `logging.Logger.log` call with an
`event` key and arbitrary structured fields attached via `extra`. Structured
handlers (JSON formatters, OTel log bridges) can key off the fields:

```python
from troopai.adk.tracing import log_event
from troopai.adk.tracing.logging import EVENT_AGENT_TURN_START, EVENT_LLM_REQUEST

log_event(logger, EVENT_AGENT_TURN_START, agent_name="classifier", turn=1)
log_event(logger, EVENT_LLM_REQUEST, model="gpt-4o-mini", level=logging.DEBUG)
```

`log_event` and all `EVENT_*` constants are unconditionally available — no
optional extra required.

### Canonical event names

| Constant | Value |
|---|---|
| `EVENT_AGENT_TURN_START` | `agent.turn.start` |
| `EVENT_AGENT_TURN_END` | `agent.turn.end` |
| `EVENT_LLM_REQUEST` | `llm.request` |
| `EVENT_TOOL_CALL` | `tool.call` |
| `EVENT_HANDOFF` | `handoff` |
| `EVENT_GRAPH_NODE` | `graph.node` |
| `EVENT_SWARM_TURN` | `swarm.turn` |

## Fan-out pattern

Combine a span tracer, a metrics tracer, and an in-memory recorder for
local testing:

```python
from troopai.adk.tracing import MultiTracer, set_tracer
from troopai.adk.tracing.exporters import setup_phoenix
from troopai.adk.tracing.metrics import setup_metrics

phoenix = setup_phoenix(service_name="my-agent")
metrics = setup_metrics(service_name="my-agent")
set_tracer(MultiTracer([phoenix, metrics]))
```

`MultiTracer` returns a `CompositeSpan` whose `.data` rebinds (the runner
does `span.data = dataclasses.replace(...)` to attach final results)
propagate to every child span before `finish()` fires.

## Tenant + cost

### Tagging a run with a tenant

Pass `tenant_id` to `RunConfig` to attach an opaque tenant identifier to
every observability signal emitted during that run:

```python
from troopai.adk.run.config import RunConfig

config = RunConfig(
    tenant_id="acme",
    tracing_enabled=True,
    metrics_enabled=True,
)
result = await Runner.arun(agent, "Hello!", run_config=config)
```

The identifier threads to:

- **Spans** — the `troopai.tenant.id` attribute is set on the root agent
  span **and on every generation span**, under both
  `TracingConvention.DEFAULT` and `TracingConvention.OPENINFERENCE`, so
  it is visible in every OTLP-compatible backend (Jaeger, Phoenix,
  Honeycomb, Datadog, Logfire, LangSmith).
- **Metric dimensions** — a `tenant` label is added to the LLM
  instruments recorded by `record_generation` (token counts,
  request counter, cost histogram: `troopai.llm.tokens.prompt`,
  `troopai.llm.tokens.completion`, `troopai.llm.requests`,
  `troopai.llm.cost.usd`) when `tenant_id` is non-None.
  When no tenant is set the dimension is absent, keeping the cardinality
  of untenanted runs under control.
- **Status records** — `AgentRunRecord.tenant_id` is persisted in the
  `AgentStatusStore` database and used for per-tenant quota enforcement
  and cost aggregation.
- **Structured logs** — `log_event(logger, event, ..., tenant_id="acme")`
  attaches `tenant_id` to the `LogRecord` via `extra`, so JSON formatters
  or OTel log bridges can filter per tenant. Use it from your own hooks and
  tools; framework-internal call sites are wired in a follow-on pass.

### Reading run cost

`RunContext.cost_usd` holds the running total USD cost accumulated by the
agent loop from the LLM's built-in cost lookup. It is a best-effort
figure — it is `0.0` when the resolved model has no price entry in the
underlying provider library. Access it via the `RunResult.context` field
after the run:

```python
result = await Runner.arun(agent, "Hello!", run_config=config)
logger.info("run cost: $%.6f", result.context.cost_usd)  # 0.0 if unavailable
```

`RunContext.tenant_id` mirrors the `RunConfig.tenant_id` value so that
downstream hooks and tools can read it from the context without carrying
the config object separately.

### Per-tenant cost aggregation

`AgentStatusStore` (backed by SQLite) records an `AgentRunRecord` for
every completed run, including `tenant_id` and `cost_usd`. Pass
`StatusTrackingHooks` to `Runner.arun` to enable recording:

```python
from troopai.adk.status import AgentStatusStore, StatusTrackingHooks

store = AgentStatusStore(path="agent_status.db")
hooks = StatusTrackingHooks(store=store)

result = await Runner.arun(
    agent,
    "Hello!",
    run_config=RunConfig(tenant_id="acme", tracing_enabled=True, metrics_enabled=True),
    hooks=hooks,
)

# Aggregate all runs tagged "acme":
status = await store.get_status(agent.name, tenant_id="acme")
logger.info("acme total_cost_usd=%.6f", status.total_cost_usd)
```

`get_status` accepts an optional `since` timestamp to restrict the window
(e.g. last 24 hours). `AgentQuota` can be composed with
`StatusTrackingHooks` to enforce per-tenant token or request caps — pass
`quotas=[...]` to the constructor.

### Cost availability

`LLM.cost()` returns `None` for models that have no price entry in the
provider library. The agent loop accumulates `None` as `0.0`. Applications
that need reliable cost figures should either use a model known to be
priced or implement a custom `LLM` subclass that returns a deterministic
cost.

### Runnable example

`examples/observability/tenant_and_cost.py` ties all of the above
together: a `RunConfig` with `tenant_id="acme"`, an `AgentStatusStore`,
`StatusTrackingHooks`, and a `MultiTracer` composing an OTel
(OpenInference) tracer with a `MetricsTracer`. After the run it logs
`result.context.cost_usd` and `status.total_cost_usd`.

## See also

- `docs/tracing/tracing.md` — core tracing layer, opt-in model, span table.
- `docs/tracing/otel.md` — OTel bridge, vendor walkthroughs, attribute
  mapping.
- `docs/tracing/custom_span.md` — authoring application-level spans.
- `examples/observability/metrics_and_openinference.py` — OTel + metrics
  MultiTracer example.
- `examples/observability/tenant_and_cost.py` — tenant tagging, per-tenant
  cost readback, status aggregation.
- `tests/unit/tracing/` — unit tests for every tracing subsystem.

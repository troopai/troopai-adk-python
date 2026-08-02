(guides/tracing)=

# 🔭 Tracing

The TroopAI ADK ships a provider-agnostic observability layer built on
[OpenTelemetry](https://opentelemetry.io/) (OTel) with
[OpenInference](https://github.com/Arize-ai/openinference) semantic
conventions. Every LLM call, tool execution, handoff, guardrail evaluation,
and graph transition is wrapped in a typed span. All observability is
**opt-in**: the default tracer is `NoOpTracer`, so un-configured deployments
run with zero overhead.

---

## What tracing covers

The framework instruments the entire agent execution path. For each run the
following spans are emitted automatically:

| Operation | Factory | OTel span name |
|---|---|---|
| Agent turn | `agent_span` | `agent.<agent_name>` |
| LLM generation | `generation_span` | `llm.generation` |
| Provider-level response | `response_span` | `llm.response` |
| Tool execution | `function_span` | `tool.<tool_name>` |
| MCP tool execution | `function_span` (with `mcp_data`) | `mcp.<tool_name>` |
| Agent-to-Agent call | `function_span` (with `a2a_data`) | `a2a.<task_id>` |
| Agent handoff | `handoff_span` | `agent.handoff` |
| Guardrail evaluation | `guardrail_span` | `guardrail.<name>` |
| Graph execution | `graph_span` / `graph_superstep_span` / `graph_node_span` | `graph.*` |
| Swarm turn | `swarm_turn_span` | `swarm.turn` |
| Sandbox execution | `sandbox_span` | `sandbox.<backend_id>` |
| Developer-authored | `custom_span` | caller-provided name |

Application code can add its own spans with `custom_span` (see
[Custom spans](#custom-spans) below). The built-in span factories are the only
entry points the ADK exposes — application code never calls `get_tracer()`
directly.

---

## Span hierarchy

Each `Runner.arun()` invocation roots a single agent span. Child spans nest
inside it as the agent loop progresses:

```
agent.my_agent                         ← AgentSpanData
  llm.generation                       ← GenerationSpanData (model, tokens, config)
    llm.response                       ← ResponseSpanData  (response_id)
  tool.lookup_crm                      ← FunctionSpanData  (input, output)
  tool.send_email                      ← FunctionSpanData
  agent.handoff                        ← HandoffSpanData   (from_agent, to_agent)
  agent.billing_agent                  ← AgentSpanData (sub-agent)
    llm.generation
    guardrail.pii_check                ← GuardrailSpanData (name, triggered)
```

For graph and swarm workloads the hierarchy gains additional levels:

```
graph.execution
  graph.superstep
    graph.node.classify
      agent.classifier
        llm.generation
    graph.node.enrich
      agent.enricher
        llm.generation
        tool.web_search
```

### OpenInference span attributes

When `TracingConvention.OPENINFERENCE` is selected the OTel bridge emits the
following attribute keys (read natively by Phoenix / Arize without an adapter):

| Span kind | Attribute | Source field |
|---|---|---|
| All | `openinference.span.kind` | span factory type |
| LLM | `llm.model_name` | `GenerationSpanData.model` |
| LLM | `llm.token_count.prompt` | `usage["prompt_tokens"]` or `usage["input_tokens"]` |
| LLM | `llm.token_count.completion` | `usage["completion_tokens"]` or `usage["output_tokens"]` |
| LLM | `llm.token_count.total` | `usage["total_tokens"]` |
| LLM | `llm.invocation_parameters` | `GenerationSpanData.model_config` (JSON) |
| LLM | `input.value` / `output.value` | prompt messages / response messages |
| Tool | `tool.name` | `FunctionSpanData.name` |
| Tool | `input.value` / `output.value` | `FunctionSpanData.input` / `output` (redacted by default) |
| Agent | `troopai.agent.name` | `AgentSpanData.name` |
| Agent | `troopai.tenant.id` | `AgentSpanData.tenant_id` |
| Handoff | `troopai.handoff.from` / `troopai.handoff.to` | `HandoffSpanData.from_agent` / `to_agent` |
| Guardrail | `troopai.guardrail.triggered` | `GuardrailSpanData.triggered` |

Under `TracingConvention.DEFAULT` (GenAI semconv), generation spans use
`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, and
`gen_ai.usage.output_tokens`. Framework-specific fields use the `troopai.*`
prefix in both conventions.

The `openinference.span.kind` values by factory are: `AGENT` (agent, handoff),
`LLM` (generation, response), `TOOL` (function/MCP/sandbox), `GUARDRAIL`
(guardrail), and `CHAIN` for most custom spans (with `AGENT` for
`graph_node`-typed custom spans routed through swarm paths, and `TOOL` for
sandbox-typed spans).

---

## Enabling tracing

### Install the optional extra

OpenTelemetry is an optional dependency. Install it with the `otel` extra:

```bash
pip install 'troopai-adk-python[otel]'
```

The core ADK has zero runtime dependency on `opentelemetry`. Importing
`OTelTracer` or calling `setup_otel` without the extra installed raises
`TracingDependencyError` with the install command. `TracingConvention`,
`log_event`, and all `EVENT_*` constants are always importable — they
require no extra.

### Configure the OTel SDK

Wire the tracer at application startup before running any agents:

```python
import os
from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.otel import setup_otel

tracer = setup_otel(
    service_name="my-agent-service",
    endpoint="http://localhost:4317",   # OTLP gRPC endpoint
    console=True,                       # also print spans to stdout
)
set_tracer(tracer)
```

`setup_otel` installs a `TracerProvider` with a `BatchSpanProcessor` and
returns an `OTelTracer`. When `endpoint` is `None`, OTel reads
`OTEL_EXPORTER_OTLP_ENDPOINT` from the environment, falling back to
`http://localhost:4317`.

### Enable per-run tracing

Tracing is gated on `RunConfig.tracing_enabled` (default `False`). Enable it
per run:

```python
from troopai.adk.run.config import RunConfig

config = RunConfig(tracing_enabled=True)
result = await runner.arun(agent, "Hello", run_config=config)
```

You can attach arbitrary metadata to all spans emitted during a run via
`tracing_metadata`:

```python
config = RunConfig(
    tracing_enabled=True,
    tracing_metadata={"env": "production", "pipeline": "onboarding"},
)
```

---

## Exporters

Four thin setup helpers cover the most common managed backends. All require
the `otel` extra.

### Arize Phoenix

Phoenix ingests OTLP and reads OpenInference attributes natively.
`setup_phoenix` is `setup_otel` pre-configured with
`TracingConvention.OPENINFERENCE`:

```python
import os
from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.exporters import setup_phoenix

set_tracer(setup_phoenix(
    endpoint=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"),  # None → env default
    service_name="my-agent",
))
```

Docs: <https://docs.arize.com/phoenix/tracing/how-to-tracing/setup-tracing>

### Pydantic Logfire

```python
import os
from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.exporters import setup_logfire

set_tracer(setup_logfire(
    token=os.environ["LOGFIRE_TOKEN"],
    service_name="my-agent",
))
```

Logfire's OTLP ingestion takes the write token as the raw `Authorization`
header value — without a `Bearer ` prefix.

Docs: <https://logfire.pydantic.dev/docs/how-to-guides/alternative-clients/>

### LangSmith

```python
import os
from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.exporters import setup_langsmith

set_tracer(setup_langsmith(
    api_key=os.environ["LANGSMITH_API_KEY"],
    project="my-project",        # optional; groups runs in the LangSmith UI
    service_name="my-agent",
))
```

Docs: <https://docs.smith.langchain.com/observability/how_to_guides/trace_with_opentelemetry>

### Generic OTLP

For any OTLP-compatible collector (Jaeger, Honeycomb, Datadog, Grafana Tempo):

```python
import os
from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.otel import setup_otel

set_tracer(setup_otel(
    endpoint=os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"],
    service_name="my-agent",
    headers={"x-honeycomb-team": os.environ["HONEYCOMB_API_KEY"]},
))
```

Never hard-code API keys in source — always load from the environment.

---

## MetricsTracer

`MetricsTracer` records OTel metric instruments from typed `SpanData`
payloads at `Span.finish()`. It operates independently of `tracing_enabled`:
the gate flag is `RunConfig.metrics_enabled` (default `False`).

`setup_metrics` wires a `MeterProvider` with a `PeriodicExportingMetricReader`
(default export interval: 60 s) and returns a `MetricsTracer` bound to the
provider:

```python
from troopai.adk.tracing import MultiTracer, set_tracer
from troopai.adk.tracing.exporters import setup_phoenix
from troopai.adk.tracing.metrics import setup_metrics

otel = setup_phoenix(service_name="my-agent")
metrics = setup_metrics(service_name="my-agent")
set_tracer(MultiTracer([otel, metrics]))
```

`MultiTracer` fans out every `*_span()` call to all wrapped tracers. The
returned `CompositeSpan` propagates `.data` rebinds (the runner updates span
data after each LLM call with `span.data = dataclasses.replace(...)`) to
every child before `finish()` fires, ensuring the metrics tracer always
observes the final token counts.

### Instruments recorded

| Instrument | Type | Unit | Trigger |
|---|---|---|---|
| `troopai.agent.turn.duration_ms` | Histogram | `ms` | `AgentSpanData` at finish |
| `troopai.llm.tokens.prompt` | Histogram | `{token}` | `GenerationSpanData.usage` |
| `troopai.llm.tokens.completion` | Histogram | `{token}` | `GenerationSpanData.usage` |
| `troopai.llm.requests` | Counter | `1` | `GenerationSpanData` at finish; label `status=success\|error` |
| `troopai.llm.cost.usd` | Histogram | `{usd}` | `GenerationSpanData.cost_usd` when present |
| `troopai.agent.tool.calls` | Counter | `1` | `FunctionSpanData` at finish; label `status=success\|error` |
| `troopai.graph.node.duration_ms` | Histogram | `ms` | `custom_span` with `data["type"]="graph_node"` |
| `troopai.swarm.turn.duration_ms` | Histogram | `ms` | `custom_span` with `data["type"]="swarm_turn"` |

When `RunConfig.tenant_id` is set, a `tenant` dimension is added to all LLM
instruments, enabling per-tenant cost and usage reporting.

---

## Redaction

Tool inputs and outputs frequently contain credentials, PII, and large
payloads. `OTelTracer` applies credential-shape redaction and truncation
before writing tool I/O to span attributes.

**Redaction is on by default.** The following patterns are replaced with
redaction markers before export:

- Bearer tokens (`Bearer <token>` → `Bearer ***`)
- OpenAI keys (`sk-***`), Anthropic keys (`sk-ant-***`)
- Google API keys (`AIza***`)
- AWS access key IDs (`AKIA***`, `ASIA***`)
- GitHub tokens (`gh_***`)
- Slack tokens (`xox-***`)
- PEM-encoded private keys
- JSON/assignment-shaped secret fields (`api_key`, `password`, `secret`,
  `token`, `authorization`, `access_token`, `client_secret`, `private_key`,
  `refresh_token`, `aws_secret_access_key` — case-insensitive)

In addition, tool I/O values are **truncated** to 2 048 characters by default
(`_DEFAULT_TOOL_IO_MAX_CHARS`). Redaction runs on the full string first so
that a secret near the tail of a long value is not missed by the truncation
boundary.

To opt in to raw, unredacted tool I/O (trusted internal environments only),
pass `record_tool_io_full=True` to `OTelTracer` or `setup_otel`:

```python
from troopai.adk.tracing.otel import setup_otel

# WARNING: emits raw tool inputs and outputs, including PII and credentials.
tracer = setup_otel(
    service_name="dev-debug",
    console=True,
    record_tool_io_full=True,
)
```

The `tool_io_max_chars` parameter controls the per-attribute char cap when
`record_tool_io_full` is `False`.

```{admonition} Privacy note
:class: warning

Both `TracingConvention.DEFAULT` and `TracingConvention.OPENINFERENCE` apply
the same redaction gate to tool I/O. Redaction is last-resort defence — it
does not replace proper secret hygiene in tool code. Treat your configured
observability backend as having visibility into prompt and response content.
```

---

## Structured logging

The `log_event` helper emits a `logging.Logger.log` call with an `event` key
and arbitrary structured fields attached via `extra`. Structured handlers
(JSON formatters, OTel log bridge) can key off the fields:

```python
import logging
from troopai.adk.tracing import log_event
from troopai.adk.tracing.logging import (
    EVENT_AGENT_TURN_START,
    EVENT_AGENT_TURN_END,
    EVENT_LLM_REQUEST,
    EVENT_TOOL_CALL,
)

logger = logging.getLogger(__name__)

log_event(logger, EVENT_AGENT_TURN_START, agent_name="classifier", turn=1)
log_event(logger, EVENT_LLM_REQUEST, model="gpt-4o-mini", level=logging.DEBUG)
log_event(logger, EVENT_TOOL_CALL, tool="lookup_crm")
log_event(logger, EVENT_AGENT_TURN_END, agent_name="classifier", turn=1)
```

`log_event` and all `EVENT_*` constants are unconditionally available — no
optional extra is required. The canonical event constants are:

| Constant | Value |
|---|---|
| `EVENT_AGENT_TURN_START` | `agent.turn.start` |
| `EVENT_AGENT_TURN_END` | `agent.turn.end` |
| `EVENT_LLM_REQUEST` | `llm.request` |
| `EVENT_TOOL_CALL` | `tool.call` |
| `EVENT_HANDOFF` | `handoff` |
| `EVENT_GRAPH_NODE` | `graph.node` |
| `EVENT_SWARM_TURN` | `swarm.turn` |

### Correlation with spans

The OTel log bridge (when configured) propagates the active trace context to
log records, correlating structured log entries with the span that emitted
them. A JSON formatter seeing a `LogRecord` can extract `event`, `agent_name`,
`model`, and `tenant_id` fields alongside the OTel `trace_id` and `span_id`
injected by the bridge:

```python
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

log_provider = LoggerProvider()
log_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
set_logger_provider(log_provider)
```

With the OTel log bridge installed, every `log_event` call made during an
active span automatically carries that span's trace and span IDs, enabling
trace-to-log correlation in backends such as Grafana, Honeycomb, and Datadog.

---

## Custom spans

`custom_span` is the only tracing factory ADK application code should call
directly. Use it to instrument business logic that sits outside the
framework's built-in span kinds:

```python
from troopai.adk.tracing.spans import custom_span

async def rank_candidates(results: list[dict]) -> list[dict]:
    with custom_span("rank_candidates", data={"n": len(results)}) as span:
        ranked = await run_ranking_model(results)
        span.data.data["top_score"] = ranked[0]["score"] if ranked else 0.0
    return ranked
```

Custom spans nest inside the current active span automatically via a
`contextvars.ContextVar` stack. Inside a `function_span` (tool body) the
custom span becomes a child of the tool span, which is itself a child of the
enclosing agent span.

To record an error on a span:

```python
with custom_span("payment_check", data={"amount": amount}) as span:
    try:
        result = check_payment(amount)
    except PaymentError as exc:
        span.set_error(str(exc), data={"code": exc.code})
        raise
```

The `disabled` flag lets you bypass tracing in test environments without
removing the instrumentation call:

```python
with custom_span("expensive_step", data={...}, disabled=not tracing_enabled):
    ...
```

---

## Common patterns

### Production export to a managed service

A production setup typically combines a span tracer with the metrics tracer
so both spans and OTel instruments land in the same backend:

```python
import os
from troopai.adk.tracing import MultiTracer, set_tracer
from troopai.adk.tracing.exporters import setup_phoenix
from troopai.adk.tracing.metrics import setup_metrics
from troopai.adk.run.config import RunConfig

phoenix = setup_phoenix(
    endpoint=os.environ["PHOENIX_ENDPOINT"],
    service_name="prod-agent",
)
metrics = setup_metrics(
    endpoint=os.environ["PHOENIX_ENDPOINT"],
    service_name="prod-agent",
)
set_tracer(MultiTracer([phoenix, metrics]))

config = RunConfig(tracing_enabled=True, metrics_enabled=True)
```

### Local debugging with Phoenix

For local development, Phoenix can run as a Docker container and serve a
browser UI at `http://localhost:6006`. The OTLP collector listens on port
`4317`:

```bash
docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix
```

```python
from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.exporters import setup_phoenix

set_tracer(setup_phoenix(
    endpoint="http://localhost:4317",
    service_name="dev-agent",
))
```

Add `console=True` to `setup_otel` (or call `setup_otel` directly) to also
print spans to stdout during development:

```python
from troopai.adk.tracing.otel import setup_otel
from troopai.adk.types.tracing.convention import TracingConvention

tracer = setup_otel(
    endpoint="http://localhost:4317",
    service_name="dev-agent",
    console=True,
    convention=TracingConvention.OPENINFERENCE,
)
```

### A/B comparison of agent versions via spans

Tag spans with metadata that identifies the variant, then filter by
`troopai.metadata.*` attributes in your observability backend:

```python
from troopai.adk.run.config import RunConfig

config_a = RunConfig(
    tracing_enabled=True,
    tracing_metadata={"variant": "A", "prompt_version": "v_stable"},
    tenant_id="experiment_001",
)
config_b = RunConfig(
    tracing_enabled=True,
    tracing_metadata={"variant": "B", "prompt_version": "v_experimental"},
    tenant_id="experiment_001",
)
```

Both variants share a `tenant_id` so their token-count and latency metrics
land on the same `tenant` dimension. The `variant` field in
`troopai.metadata.variant` lets you split traces by agent version in the
observability backend.

```{admonition} Thread safety
:class: note

`set_tracer` installs a module-level tracer. For concurrent `Runner.arun()`
calls sharing a process, install the tracer once at startup rather than
swapping it per request. Per-run behaviour is controlled via `RunConfig`
flags, not by swapping the tracer.
```

---

## See also

- {doc}`../architecture/governance` — cross-cutting invariants including the
  opt-in philosophy and cost-conservative defaults.
- {doc}`../observability/observability` — full observability reference:
  `TracingConvention`, metric instruments, exporter helpers, structured
  logging, and tenant tagging.
- {doc}`../tracing/custom_span` — authoring developer-facing spans via
  `custom_span`.
- {doc}`../tracing/otel` — OpenTelemetry bridge detail, Jaeger / Honeycomb /
  Datadog setup, and the full attribute-mapping table.

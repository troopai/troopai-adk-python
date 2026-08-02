# OpenTelemetry Bridge

The OTel bridge emits every framework span through the OpenTelemetry
API, so any collector that speaks OTLP — Jaeger, Honeycomb, Datadog,
Phoenix, Langwatch, Grafana Tempo, New Relic — ingests TroopAI agent
traces without a vendor-specific SDK.

## Installation

OTel is an **optional** extra. The core framework has zero runtime
dependency on `opentelemetry`.

```bash
pip install 'troopai-adk-python[otel]'
```

If the extra is missing and application code tries to construct an
`OTelTracer` or call `setup_otel(...)`, the framework raises
`TracingDependencyError` with the install command — not a confusing
low-level `ImportError`.

```python
from troopai.adk.exceptions import TracingDependencyError
from troopai.adk.tracing.otel import OTelTracer

try:
    OTelTracer()
except TracingDependencyError as e:
    # e.missing == "opentelemetry"
    # str(e) contains the "pip install 'troopai-adk-python[otel]'" command
    ...
```

## `setup_otel` — fluent installer

```python
from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.otel import setup_otel

tracer = setup_otel(
    endpoint="http://localhost:4317",   # optional; reads OTEL_EXPORTER_OTLP_ENDPOINT when None
    service_name="my-agent",             # shows up in collector UIs
    console=True,                        # also print spans to stdout
    headers={"x-honeycomb-team": "..."}, # vendor API keys
)
set_tracer(tracer)
```

What it installs:

1. A `TracerProvider` with `service.name=<service_name>` as the
   resource attribute.
2. A `BatchSpanProcessor(OTLPSpanExporter)` for background shipping.
3. When `console=True`, a `SimpleSpanProcessor(ConsoleSpanExporter)`
   so spans are also printed to stdout for inspection.
4. Any extra `SpanProcessor` instances supplied via
   `additional_processors=[...]` — used to coexist with
   OpenInference / Phoenix processors, or to tap spans into an
   in-memory recorder for test assertions.

## Vendor walkthroughs

### Jaeger (local dev, Docker)

```bash
docker run -d --name jaeger -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one
```

```python
tracer = setup_otel(endpoint="http://localhost:4317", service_name="my-agent")
set_tracer(tracer)
```

Open `http://localhost:16686`, pick the `my-agent` service, and
verify the trace tree shape: `agent.<name>` → `llm.generation` (per
turn) → `tool.<name>` / `mcp.<name>` (per tool call) → `agent.handoff`
/ `guardrail.<name>` where applicable.

### Honeycomb

```python
tracer = setup_otel(
    endpoint="https://api.honeycomb.io:443",
    service_name="my-agent",
    headers={"x-honeycomb-team": os.environ["HONEYCOMB_API_KEY"]},
)
set_tracer(tracer)
```

### Datadog

Deploy the Datadog OTel collector as a sidecar or DaemonSet, then:

```python
tracer = setup_otel(
    endpoint="http://datadog-agent:4317",
    service_name="my-agent",
)
set_tracer(tracer)
```

### Phoenix / Arize / Langwatch

These platforms ingest GenAI semconv attributes directly. Point
`endpoint` at their OTLP receiver and the spans will be correlated
into LLM-dashboard views automatically — no adapter required.

## Span name and attribute mapping

| Span kind | OTel name | Key attributes |
|-----------|-----------|----------------|
| `agent_span` | `agent.<agent_name>` | `troopai.agent.name`, `troopai.agent.handoffs`, `troopai.agent.tools`, `troopai.agent.output_type`, `troopai.metadata.<key>` |
| `function_span` (tool) | `tool.<tool_name>` | `troopai.tool.name`, `troopai.tool.input`, `troopai.tool.output` |
| `function_span` (MCP) | `mcp.<tool_name>` | `troopai.mcp.server_name`, `troopai.mcp.tool_name`, plus the `tool.*` set above |
| `generation_span` | `llm.generation` | `gen_ai.system="troopai"`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.request.<k>` (from `model_config`) |
| `response_span` | `llm.response` | `gen_ai.system="troopai"`, `gen_ai.response.id` |
| `handoff_span` | `agent.handoff` | `troopai.handoff.from`, `troopai.handoff.to` |
| `guardrail_span` | `guardrail.<name>` | `troopai.guardrail.name`, `troopai.guardrail.triggered` |
| `custom_span` | caller-provided name | `troopai.span.name`, `troopai.custom.<k>` (from `data`) |

**GenAI semconv** keys follow the [OpenTelemetry GenAI
semantic-convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
draft, so spans are ingestable by Phoenix / Langwatch / Honeycomb LLM
dashboards without an adapter. Framework-specific fields use the
`troopai.*` namespace.

### Nested values

Nested dicts flatten with dotted prefixes:
`model_config={"temperature": 0.7, "top_p": 0.9}` becomes
`gen_ai.request.temperature=0.7`, `gen_ai.request.top_p=0.9`. Values
that cannot be expressed as OTel scalars (unknown types, heterogeneous
lists) are JSON-encoded into a string attribute.

### Errors

`Span.set_error(message, data={"type": ...})` maps to:

- `Status(StatusCode.ERROR, message)` on the OTel span, and
- an `exception` event with `exception.message` /
  `exception.type` attributes, matching the OTel exception semconv.

## Parent-child semantics

The bridge relies on OTel's own context propagation
(`opentelemetry.context`) to auto-parent children. A child span started
while an outer span is the current span is automatically attached as
its child — no framework bookkeeping. This means `OTelSpan` does **not**
register itself on the framework `_current_span` ContextVar (see
`src/troopai/adk/tracing/otel/otel_span.py` for the rationale).

## Using an existing `TracerProvider`

If the host application already configures its own `TracerProvider`
(typical in larger services), pass it explicitly instead of calling
`setup_otel`:

```python
from opentelemetry import trace as otel_trace
from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.otel import OTelTracer

provider = otel_trace.get_tracer_provider()  # installed elsewhere
set_tracer(OTelTracer(provider=provider, service_name="my-agent"))
```

## Coexisting with OpenInference / Phoenix

OpenInference ships its own `TracerProvider` processors. The bridge
plays nicely with them — they just read the GenAI semconv attributes
the bridge emits. Either:

1. Install OpenInference's processor as an `additional_processor` in
   `setup_otel(...)`, or
2. Construct the provider yourself with OpenInference's recipe and
   hand it to `OTelTracer(provider=...)`.

## Examples

- `examples/tracing/otel_console.py` — minimal local run with
  `console=True`.
- `examples/tracing/otel_otlp.py` — ship to an OTLP collector (Jaeger
  default).
- `examples/tracing/multi_tracer.py` — fan-out to OTel plus an
  in-memory recorder.

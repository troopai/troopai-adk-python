# Tracing Module

Provider-agnostic observability layer. Opt-in: the default tracer is
`NoOpTracer`, so until an application calls `set_tracer(...)` every
factory returns a zero-cost no-op span.

## Directory Structure

```
tracing/
  tracer.py                 # Tracer protocol + NoOpTracer + registry
  spans.py                  # Span[TData] + NoOpSpan + *_span() factories + current_span()
  multi_tracer.py           # MultiTracer — fan-out composite over N tracers
  logging.py                # log_event() + EVENT_* structured-log constants
  otel/
    __init__.py             # OTelTracer, OTelSpan, setup_otel re-exports
    otel_tracer.py          # Tracer impl backed by opentelemetry.trace
    otel_span.py            # Span wrapping opentelemetry.trace.Span
    setup.py                # setup_otel() — TracerProvider + OTLP/console wiring
  openinference/
    __init__.py             # OpenInference attribute-mapper re-exports
    conventions.py          # SpanData → OpenInference attribute dicts
  metrics/
    __init__.py             # MetricsTracer, setup_metrics, Instruments re-exports
    tracer.py               # MetricsTracer + MetricSpan — records instruments at finish()
    instruments.py          # Instruments — owns OTel meter instruments + SpanData recording
    setup.py                # setup_metrics() — MeterProvider + OTLP metric export wiring
  exporters/
    __init__.py             # setup_phoenix, setup_logfire, setup_langsmith, setup_helicone
    phoenix.py              # setup_phoenix — setup_otel with OPENINFERENCE convention
    logfire.py              # setup_logfire — setup_otel with Logfire OTLP headers
    langsmith.py            # setup_langsmith — setup_otel with LangSmith OTLP headers
    helicone.py             # setup_helicone — gateway base_url + auth (not a span exporter)
```

Typed span-data payload classes live under
`src/troopai/adk/types/tracing/span_data.py`: `SpanData`,
`AgentSpanData`, `FunctionSpanData`, `GenerationSpanData`,
`ResponseSpanData`, `HandoffSpanData`, `GuardrailSpanData`,
`CustomSpanData`. `TracingConvention` lives under
`src/troopai/adk/types/tracing/convention.py`.

## Architectural Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | `Tracer` is a Protocol with seven typed `*_span()` factories | Keeps the surface minimal and typed — every call site receives `Span[ConcreteSpanData]`, not `Span[Any]`. |
| 2 | `NoOpTracer` is the default; two independent opt-in paths exist | Path A: set a real span tracer + `tracing_enabled=True` (spans). Path B: compose a `MetricsTracer` + `metrics_enabled=True` (instruments). Each path is independently cost-free until enabled. |
| 3 | Per-call-site `disabled=not (config.tracing_enabled or config.metrics_enabled)` kwarg instead of a registry swap | Registry swap is not thread-safe across concurrent `Runner.arun()` calls. Per-call-site bypass is contextvar-scoped and composable. |
| 4 | Parent tracking via `contextvars.ContextVar` in `spans.py` | Safe across `await` and `asyncio.gather`. `NoOpSpan` overrides `start/finish` as pass-through so the disabled path never touches the ContextVar. |
| 5 | `MultiTracer` + OTel bridge both ship | Users can fan out via OTel's own processor chain (one tracer, N exporters) **or** via `MultiTracer` (N tracers, each with its own lifecycle). MultiTracer's `CompositeSpan` does not touch the framework ContextVar — only child spans registered on inner tracers do, so parent chaining stays correct. |
| 6 | MCP spans reuse `FunctionSpanData.mcp_data` — no new span kind | Adds zero surface to the `Tracer` protocol; the OTel bridge name-switches `tool.<name>` → `mcp.<name>` based on the presence of `mcp_data`. |
| 7 | OpenTelemetry is an **optional extra** (`pip install 'troopai-adk-python[otel]'`) | The core framework has zero runtime dependency on `opentelemetry`. Soft imports in `otel/*` raise `TracingDependencyError` with the install command when the extra is missing. |
| 8 | OTel attribute mapping uses GenAI semconv where possible (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`) plus `troopai.*` for framework-specific fields | Makes spans ingestable by Phoenix/Langwatch/Honeycomb's GenAI dashboards without an additional adapter. |
| 9 | `TracingConvention` is a setup-time enum choice on `setup_otel(convention=...)` — `DEFAULT` emits GenAI semconv; `OPENINFERENCE` emits OpenInference keys read natively by Phoenix/Arize | Selects attribute vocabulary once at provider setup; span-factory code stays convention-agnostic. |
| 10 | `MetricsTracer` records OTel instruments from typed `SpanData` at `Span.finish()`; compose via `MultiTracer` | No new protocol method needed — the metrics path hooks the existing `finish()` lifecycle. `CompositeSpan` forwards `.data` rebinds to children before `finish()` fires. |
| 11 | `metrics_enabled` is independent of `tracing_enabled` | Users may want token-count metrics without shipping spans (cost), or spans without metrics (dashboards not yet configured). |
| 12 | Exporters (`phoenix`, `logfire`, `langsmith`) are thin `setup_otel` wrappers with pre-filled headers/convention; `helicone` is a gateway helper, not a `Tracer` | Keeps each exporter to a few lines; the heavy `TracerProvider` wiring lives in `otel/setup.py` once. Helicone intercepts LLM HTTP traffic — it never sees OTel spans. |
| 13 | `logging.py` defines canonical `EVENT_*` string constants and a `log_event(logger, event, **fields)` helper that uses `extra=` for structured handlers | Standardizes log-record field shapes without coupling callers to a specific logging framework or OTel log bridge. |

## Public API

| Symbol | Purpose |
|--------|---------|
| `Tracer` | Protocol — seven `*_span()` factory methods |
| `NoOpTracer` | Default tracer; every span is a `NoOpSpan` |
| `MultiTracer` | Fan-out composite over any number of inner tracers |
| `get_tracer()` / `set_tracer()` | Module-level registry |
| `Span[TData]` | Generic span wrapping a typed payload; context manager |
| `NoOpSpan` | Span that records nothing |
| `current_span()` | Read the current span from the contextvar stack |
| `agent_span`, `function_span`, `generation_span`, `response_span`, `handoff_span`, `guardrail_span`, `custom_span` | Typed span factories |
| `TracingConvention` | Span-attribute vocabulary selector (`DEFAULT` vs `OPENINFERENCE`); always available |
| `log_event()` | Emit a structured log record with event name + key/value fields; always available |
| `OTelTracer`, `setup_otel` | Conditionally re-exported when the `otel` extra is installed |
| `MetricsTracer`, `setup_metrics` | Conditionally re-exported when the `otel` extra is installed |

## Pointers

- `docs/tracing/tracing.md` — opt-in walkthrough, `RunConfig` integration, custom tracer recipe.
- `docs/tracing/otel.md` — OpenTelemetry bridge guide, Jaeger/Honeycomb/Datadog setup, attribute mapping table.
- `docs/tracing/custom_span.md` — authoring developer-facing spans via `custom_span(...)`.
- `docs/observability/observability.md` — full observability guide: `TracingConvention`, metrics, exporters, structured logging.
- `examples/observability/` — runnable example: OpenInference + metrics in a MultiTracer.
- `examples/tracing/` — single-file console/OTLP/multi-tracer examples.

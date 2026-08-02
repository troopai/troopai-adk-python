# Custom Spans

The tracing module exposes typed factory functions for every built-in span
kind (agent, function, generation, response, handoff, guardrail) plus
`custom_span()` for application-authored instrumentation.

## The tracer registry

Tracing is opt-in. Until an application calls `set_tracer(...)`, every span
factory returns a `NoOpSpan` — zero cost on the hot path.

```python
from troopai.adk.tracing import set_tracer, get_tracer

class MyBackendTracer:
    def agent_span(self, data): ...
    def function_span(self, data): ...
    # ... one method per span kind
    def custom_span(self, data): ...

set_tracer(MyBackendTracer())
```

The installed tracer implements the `Tracer` protocol (seven factory
methods, one per span kind) and is fetched lazily via `get_tracer()` inside
every factory, so swapping tracers at runtime is a single call.

## `custom_span(name, *, data=None, span_id=None, disabled=False)`

The only tracing factory the ADK exposes for application code.

```python
from troopai.adk.tracing import custom_span

with custom_span("rank_search_results", data={"n": len(results)}) as span:
    ranked = rank(results)
    span.data.data["elapsed_ms"] = elapsed
```

- `name` — short human-readable span name
- `data` — arbitrary JSON-safe payload (merged into the span's `CustomSpanData`)
- `span_id` — optional caller-assigned identifier
- `disabled` — return a `NoOpSpan` regardless of the installed tracer

The span is a context manager: `start()` runs on entry, `finish()` on exit,
and any exception is recorded via `set_error()` before re-raising.

## Typed span-data classes

Each built-in span kind has a frozen dataclass payload in
`troopai.adk.types.tracing.span_data`. This is the "G4" layer — typed
observability instead of untyped dict attributes:

| Factory | Data class | Fields |
|---------|-----------|--------|
| `agent_span` | `AgentSpanData` | `name`, `handoffs`, `tools`, `output_type`, `metadata` |
| `function_span` | `FunctionSpanData` | `name`, `input`, `output`, `mcp_data` |
| `generation_span` | `GenerationSpanData` | `input`, `output`, `model`, `model_config`, `usage` |
| `response_span` | `ResponseSpanData` | `response_id`, `input` |
| `handoff_span` | `HandoffSpanData` | `from_agent`, `to_agent` |
| `guardrail_span` | `GuardrailSpanData` | `name`, `triggered` |
| `custom_span` | `CustomSpanData` | `name`, `data` |

`AgentSpanData.metadata` is the home for arbitrary per-run tags passed
via `RunConfig.tracing_metadata` — see `docs/tracing/tracing.md`.

All seven classes subclass `SpanData` and are frozen dataclasses — safe to
pass across the tracer boundary without defensive copying.

## Recording errors without raising

```python
with custom_span("risky_step") as span:
    try:
        do_work()
    except RecoverableError as e:
        span.set_error(str(e), data={"type": type(e).__name__})
        return fallback()
```

`set_error()` populates the span's `error` dict; the context manager's
`__exit__` only records a fresh error if one hasn't already been set (via
the captured `exc_val`).

## See also

- `src/troopai/adk/tracing/tracer.py` — `Tracer` protocol + registry
- `src/troopai/adk/tracing/spans.py` — `Span`, `NoOpSpan`, factory functions
- `src/troopai/adk/types/tracing/span_data.py` — typed payload dataclasses
- `examples/tracing/custom_span_example.py` — runnable example
- `tests/unit/tracing/` — tracer, span, and span-data tests

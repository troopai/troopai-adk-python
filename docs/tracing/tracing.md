# Tracing

The tracing layer records structured, typed observability data for every
hot path in the runner: per-agent spans, per-turn LLM generation spans,
per-tool-call function spans, plus handoff and guardrail spans. The
design is **opt-in twice**: a user must flip
`RunConfig.tracing_enabled=True` *and* install a tracer via
`set_tracer(...)`. By default the framework runs with zero span
overhead.

## Why "opt-in twice"

| Opt-in | Default | What happens if missing |
|--------|---------|-------------------------|
| `RunConfig.tracing_enabled` | `False` | Every span factory passes `disabled=True` — no `SpanData` payload is built. |
| `set_tracer(...)` | `NoOpTracer` | Factories return `NoOpSpan`, which records nothing and never touches the contextvar stack. |

Both guards exist on the hot path: the first stops `SpanData` allocation
at the call site; the second guarantees that if a user set the flag but
forgot to install a tracer, nothing measurable is recorded.

## Quick start with the OpenTelemetry bridge

```python
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.otel import setup_otel

tracer = setup_otel(service_name="my-agent", console=True)
set_tracer(tracer)

config = RunConfig(tracing_enabled=True)
result = await Runner.arun(agent, "hello", run_config=config)
```

`setup_otel(console=True)` prints every finished span to stdout — handy
for local development. See `docs/tracing/otel.md` for the full guide,
including Jaeger, Honeycomb, and Datadog walkthroughs.

## Security — what ends up in a span

Span payloads are **recorded verbatim** by whichever tracer is installed
and, in a production setup, leave the application boundary (OTLP
collector, vendor backend, log files). Treat them as externally
visible.

| Span kind        | Verbatim-recorded field(s)                         |
|------------------|----------------------------------------------------|
| `function_span`  | `input` (raw tool arguments), `output` (tool result) |
| `generation_span`| `input` (messages), `output` (LLM response text)    |
| `agent_span`     | `metadata` (everything in `RunConfig.tracing_metadata`) |
| `response_span`  | `input` (raw response payload)                     |
| `custom_span`    | `data` (caller-provided dict)                      |

**Redaction is the application's responsibility.** Before enabling
tracing in production:

- Strip secrets, credentials, and API keys from tool inputs *before* the
  tool is called (e.g. a tool `agent_input_guardrail`).
- Strip PII from tool outputs *before* returning them (e.g. a tool
  `agent_output_guardrail`).
- Do not pass secrets through `RunConfig.tracing_metadata`. Use it for
  tenant IDs, feature flags, request IDs — never credentials.
- When fanning out via `MultiTracer`, remember that *every* wrapped
  tracer sees the full payload; a sanitizing wrapper on one branch does
  not protect the other.

The framework deliberately does not ship a built-in attribute filter —
sanitization is domain-specific (what counts as PII varies by
jurisdiction and application). A future release may add an opt-in
`RunConfig.span_attribute_filter` hook.

## Emitted spans

| Factory | When | Payload |
|---------|------|---------|
| `agent_span` | Once per `Runner.arun` call (wraps the whole dispatch) | `name`, `handoffs`, `tools`, `output_type`, `metadata` |
| `generation_span` | Every LLM turn inside the agent loop | `input`, `output`, `model`, `model_config`, `usage` |
| `function_span` | Every tool invocation (also used for MCP calls) | `name`, `input`, `output`, `mcp_data` |
| `handoff_span` | Every agent-to-agent handoff (both deterministic and LLM-driven) | `from_agent`, `to_agent` |
| `guardrail_span` | Every input/output guardrail evaluation | `name`, `triggered` |
| `response_span` | Provider-level raw-response tracking (opt-in; not wired into the runner loop) | `response_id`, `input` |
| `custom_span` | Application-authored via `custom_span(...)` | `name`, `data` |

Parent-child relationships are tracked through two channels: the OTel
bridge uses `opentelemetry.context` (native), and all framework-level
tracers use a `contextvars.ContextVar` maintained by `Span.start` /
`Span.finish`.

## Custom tracer

For test assertions or bespoke backends, implement the `Tracer`
protocol (seven typed factory methods) and install it:

```python
from troopai.adk.tracing import Span, set_tracer
from troopai.adk.types.tracing import AgentSpanData, SpanData

class Recorder:
    def __init__(self) -> None:
        self.spans: list[SpanData] = []

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        self.spans.append(data)
        return Span(data)
    # ... one method per span kind; see src/troopai/adk/tracing/tracer.py

recorder = Recorder()
set_tracer(recorder)
```

`examples/tracing/custom_span_example.py` shows a self-contained
in-memory recording tracer; `tests/unit/tracing/test_runner_tracing.py`
shows the same pattern used to assert span-tree shape.

## Attaching arbitrary metadata

`RunConfig.tracing_metadata: dict[str, Any]` is surfaced on the root
`AgentSpanData.metadata` field for every run. Use it to attach tenant
IDs, feature flags, or request IDs that should land on every emitted
span tree without plumbing through every factory:

```python
config = RunConfig(
    tracing_enabled=True,
    tracing_metadata={"tenant": "acme", "request_id": "req_42"},
)
```

The OTel bridge flattens metadata into `troopai.metadata.<key>`
attributes; the MultiTracer fans the same metadata to every wrapped
tracer.

## Fan-out to multiple backends

`MultiTracer` forwards every factory call to each wrapped tracer and
returns a `CompositeSpan` that propagates `start`, `finish`, and
`set_error` to all children. Useful when you want one backend shipping
to a collector and another capturing spans in-memory for test
assertions:

```python
from troopai.adk.tracing import MultiTracer, set_tracer
from troopai.adk.tracing.otel import setup_otel

otel = setup_otel(service_name="prod")
recorder = Recorder()
set_tracer(MultiTracer([otel, recorder]))
```

See `examples/tracing/multi_tracer.py` for a runnable walkthrough. When
both backends are OTel-compatible, prefer configuring a second
`SpanProcessor` on the shared `TracerProvider` instead — OTel's own
pipeline handles multi-exporter fan-out with proper batching and
sampling.

## See also

- `docs/tracing/otel.md` — OpenTelemetry bridge, vendor walkthroughs,
  attribute mapping.
- `docs/tracing/custom_span.md` — authoring application-level spans
  with `custom_span`.
- `examples/tracing/` — runnable examples.
- `tests/unit/tracing/` and `tests/unit/run/test_runner_tracing*.py` —
  test patterns.

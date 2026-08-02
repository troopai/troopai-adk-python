# Graph Observability

Two parallel observability surfaces let operators see what a graph run
is doing: `GraphHooks` for in-process control-flow reactions and
OpenTelemetry spans for distributed-trace correlation. Both fire at
the same lifecycle boundaries and are decoupled — use one, the other,
or both.

## Why Observe

A graph run can fan out into parallel agent nodes, suspend for
human-in-the-loop input, retry under flakiness, and run for minutes
before producing a final answer. Without observability:

- **Operators** cannot tell which node a long run is stuck on.
- **SREs** cannot correlate a graph span with a downstream incident.
- **Auditors** cannot reconstruct the order of approvals across a
  resume cycle.

The two surfaces together cover the spectrum:

- `GraphHooks` → in-process reactions (metrics, audit log writes,
  side-channel UI updates).
- OpenTelemetry spans → external trace correlation (Tempo, Honeycomb,
  Datadog, Jaeger), span-attribute querying, latency analysis.

## Surface 1: `GraphHooks`

`GraphHooks` is an async callback ABC. Subclass and override only the
methods you care about; unimplemented ones default to no-ops.

```python
from typing import override

from troopai.adk.graphs.hooks import GraphHooks


class AuditHooks(GraphHooks[Any]):
    @override
    async def on_node_interrupt(self, context, state, node_id, interrupt):
        audit_log.write({
            "graph_id": state.thread_id,
            "node_id": node_id,
            "interrupt_kind": interrupt.kind,
        })
```

Attach by passing the instance to `Runner.arun_graph`:

```python
result = await Runner.arun_graph(graph, "go", hooks=[AuditHooks()])
```

### Lifecycle Callbacks

| Method | Fires |
|---|---|
| `on_graph_start` | Once, before the first superstep. |
| `on_superstep_start` | At the top of every superstep, with the set of ready nodes. |
| `on_node_start` | Before each node's `Executable.invoke` runs. |
| `on_node_end` | After each clean node return. |
| `on_node_error` | When a node raises (not `InterruptException`). |
| `on_node_interrupt` | When a node raises `InterruptException` to suspend (HITL or nested-agent defer). |
| `on_superstep_end` | After all nodes in the superstep have applied results and fired their outgoing edges. |
| `on_graph_end` | Once, before `GraphRunResult` is returned. |

### Error Tolerance

Hook exceptions are logged but never abort the run — observability
should not break orchestration. A hook that wants to halt execution
must do so via a different channel (e.g. mutating context, raising
inside the orchestration code itself).

### Checkpointer Integration

The `Checkpointer` protocol extends `HookProvider`; checkpointers
subscribe to `on_node_end` / `on_graph_end` and persist state. The BSP
loop never calls `save()` directly — it just fires hooks. This keeps
persistence pluggable: any object implementing `HookProvider` can be
passed in the same `hooks=[...]` list as a plain `GraphHooks`
subclass, and the registry dispatches both.

## Surface 2: OpenTelemetry Spans

When an OTel tracer is installed, the BSP loop opens a three-level
span tree on every graph run:

```
graph.<id>                              [bracket: full run]
├── graph.superstep.0                   [bracket: one BSP superstep]
│   ├── graph.node.<a>                  [bracket: one node attempt]
│   └── graph.node.<b>
├── graph.superstep.1
│   └── graph.node.<c>                  [status=interrupted on suspend]
└── graph.superstep.2  (resume)
    └── graph.node.<c>  (resumed)
```

Parallel siblings inside a superstep open as siblings under their
parent superstep span — the BSP structure is visible in traces.

### Installing the Tracer

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.otel import OTelTracer

provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
set_tracer(OTelTracer(provider=provider, service_name="my-graphs"))
```

For production exporters (OTLP, Jaeger, Honeycomb), swap
`ConsoleSpanExporter` for the appropriate `SpanExporter` from
`opentelemetry-exporter-*`.

### Span Names

| Kind | OTel name |
|---|---|
| Graph-run root | `graph.<graph_id>` |
| BSP superstep boundary | `graph.superstep.<n>` |
| Per-node attempt | `graph.node.<node_name>` |

### Attribute Reference

All graph-tracing attributes live under the `troopai.graph.*` namespace.

**Graph-run span** (`graph.<id>`):

| Attribute | Type | Set | Meaning |
|---|---|---|---|
| `troopai.graph.id` | str | always | The graph identifier. |
| `troopai.graph.entry` | str | when set | Entry node id on the compiled graph. |

**Superstep span** (`graph.superstep.<n>`):

| Attribute | Type | Set | Meaning |
|---|---|---|---|
| `troopai.graph.id` | str | always | Parent graph identifier. |
| `troopai.graph.superstep.index` | int | always | Zero-or-one-based index (matches `state.superstep`). |
| `troopai.graph.superstep.ready_nodes` | list[str] | when non-empty | Nodes that were ready at superstep start. |
| `troopai.graph.superstep.fired_nodes` | list[str] | when non-empty | Nodes that fired in this superstep (stamped at close). |

**Per-node span** (`graph.node.<name>`):

| Attribute | Type | Set | Meaning |
|---|---|---|---|
| `troopai.graph.id` | str | always | Parent graph identifier. |
| `troopai.graph.node.name` | str | always | Node id. |
| `troopai.graph.node.status` | str | always (at close) | `success` / `failed` / `interrupted`. |
| `troopai.graph.node.attempts` | int | always (at close) | Final attempt count including retries (1 if no retries). |
| `troopai.graph.node.duration_ms` | int | optional | Wall-clock duration, set by the caller. |
| `troopai.graph.node.resume_attempt` | int | when resumed | Resume sequence number for resumed nodes. |

When a node raises a non-`InterruptException`, the span's OTel status
is also set to `ERROR` with the exception type and message recorded as
a span event (`exception.type`, `exception.message`).

### Cost-Conservative Defaults

No span is emitted unless a tracer is explicitly installed via
`set_tracer(...)`. The default `NoOpTracer` returns a `NoOpSpan` for
every factory call, and `NoOpSpan.start()` / `NoOpSpan.finish()` are
empty — the disabled path is zero-overhead.

## Custom Tracers

Building a backend that isn't OTel? Implement the `Tracer` protocol:

```python
from troopai.adk.tracing import Span, Tracer
from troopai.adk.types.tracing import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
)


class MyTracer:
    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        # Inspect data.data["type"] to recognise graph-tracing spans:
        #   "graph"           — the run-level root
        #   "graph_superstep" — a BSP superstep boundary
        #   "graph_node"      — one node attempt
        if data.data.get("type") == "graph":
            self.on_graph_start(data.data["graph_id"])
        return Span(data)

    # ... agent_span / function_span / etc. ...
```

Graph-tracing spans route through `custom_span` so custom tracers
don't need a graph-specific Tracer-protocol extension — the inner
`data["type"]` discriminator carries the kind.

## Worked Example

A runnable demo lives at `examples/graphs/observability.py`. It
combines `LoggingHooks` (prints every callback) with an `OTelTracer`
backed by `ConsoleSpanExporter` (prints every span). Topology is a
fan-out → join graph that exercises the parallel-superstep path.

```bash
python examples/graphs/observability.py
```

For HITL suspend + resume coverage of the same observability surfaces,
see `examples/graphs/hitl.py` — interrupt + resume cycles fire
`on_node_interrupt` and stamp `troopai.graph.node.status="interrupted"`
on the per-node span at close.

## Known Limitations

- The graph-run span (`graph.<id>`) currently records only
  `troopai.graph.id` and `troopai.graph.entry`. Status and
  `supersteps_total` are not stamped on the graph-run span itself
  today; they're available via the `on_graph_end` hook callback. A
  follow-up will extend the span surface to match.
- Per-node `attempts` defaults to 1 on exception paths and on
  nested-agent resume — the count is propagated from
  `run_node_with_reliability` only on the success path today.
- `resume_attempt` is allocated as an attribute slot but is not
  populated automatically by the BSP loop yet; callers that want
  resume-attempt visibility can stamp it manually via the inner
  payload dict before close.

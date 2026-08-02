(architecture/governance)=

# 🛡️ Governance

Cross-cutting concerns that apply at every stage of the pipeline:
multi-tenant isolation, audit, tool permissions, cost ledger, and tracing.

## Multi-tenant task-queue routing

Temporal workflows route per-tenant via a `TenantTaskQueueRouter`. The
maintainer dispatches a workflow with `start_tenant_workflow(...)`;
the router chooses the task queue for the tenant's worker pool.

```{mermaid}
flowchart LR
  client[client.start_tenant_workflow] --> router[TenantTaskQueueRouter]
  router -->|tenant A| qa[(queue-a)] --> wa[worker pool A]
  router -->|tenant B| qb[(queue-b)] --> wb[worker pool B]
  router -->|tenant C| qc[(queue-c)] --> wc[worker pool C]
```

Implementation lives at `src/troopai/adk/workflows/temporal/routing.py`.

## Tool permissions

Each tool execution path consults a `CanUseTool` callback that receives
a `ToolPermissionContext` (tool name, arguments, run context, tenant
identity) and returns a `PermissionResult` (`allow` or `deny` with
reason). The gate runs at every tool-execution call site, including
the Temporal-activity path. A denial does not silently drop — it
surfaces as a refused-tool item the model can react to.

The permission types live under `src/troopai/adk/types/permissions/`.

## Audit substrate

`AuditEvent`s are emitted at every governance boundary (handoff, tool
call, HITL resolution, tenant boundary cross). Sinks (all under
`src/troopai/adk/audit/`):

| Sink                   | Module                                      | Use for                          |
| ---------------------- | ------------------------------------------- | -------------------------------- |
| `InMemoryAuditSink`    | `src/troopai/adk/audit/sink.py`              | Tests, single-process dev.       |
| `JsonlFileAuditSink`   | `src/troopai/adk/audit/sink.py`              | Local dev, single-machine.       |
| `S3AuditSink`          | `src/troopai/adk/audit/sinks/s3.py`          | Cheap long-term retention.       |
| `PostgresAuditSink`    | `src/troopai/adk/audit/sinks/postgres.py`    | Queryable audit log.             |

## Cost ledger

Per-run cost accounting. Each LLM call appends a `CostEntry` to the
ledger (`CostLedger` is a Protocol — implementations live in
`src/troopai/adk/budgets/`). The `LLMRouter` ABC consults the ledger
when picking a model; shipped implementations:

- `CheapestFirstRouter` (`src/troopai/adk/llms/routing/cheapest_first.py`)
- `LatencyFirstRouter` (`src/troopai/adk/llms/routing/latency_first.py`)

Custom routers subclass `LLMRouter` (`src/troopai/adk/llms/routing/router.py`).

## Tracing (OpenTelemetry)

OpenInference semantic conventions on top of OpenTelemetry. Each
provider implementation emits OTel spans with proper redaction of
tool I/O (sensitive payload values masked). Exporters available:
Arize, Phoenix, Langfuse, generic OTLP.

## Why these are cross-cutting

Governance concerns intersect every stage of the pipeline. Audit fires
on Stage 2 guardrail rejections, Stage 3 tool calls, Stage 3 handoffs,
Stage 4 output rejections, and Stage 5 result emission. Putting them
in their own modules (rather than scattering them through the loop)
keeps the Runner small and the policies inspectable.

## Invariants

- The permission gate runs at EVERY tool-execution path (in-process and
  Temporal activity). No exceptions.
- Audit emission cannot be silenced. A run without an audit sink
  configured emits to the no-op sink, not nowhere.
- Tenant identity propagates through `RunContext`, never through tool
  arguments.

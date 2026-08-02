# Multi-tenant Governance

Run one fleet, serve many tenants safely. Three opt-in, off-by-default
capabilities, all keyed off `RunContext.tenant_id` (set via
`RunConfig.tenant_id`):

1. Per-tenant tool allowlists
2. Append-only tool-call audit logging
3. Per-tenant Temporal task-queue routing

## Per-tenant tool allowlists

Restrict which tools each tenant may call. The policy is a map from
`tenant_id` to the set of allowed tool names, set on `RunConfig`:

```python
from troopai.adk.run.config import RunConfig

config = RunConfig(
    tenant_id="free",
    tenant_tool_allowlist={"free": {"search"}, "admin": {"search", "delete_account"}},
)
```

A forbidden call **fails fast**: the executor raises
`ToolNotPermittedForTenant` before the tool runs (the tool body never
executes). The gate runs for every tool — including builtin/memory tools —
and on the human-in-the-loop resume path, so an approval cannot bypass it.

Two flags tune the behavior:

- `tenant_allowlist_default_deny=True` — a tenant absent from the map is
  denied all tools (fail-closed for tenants you did not configure).
- `tenant_allowlist_soft_deny=True` — instead of raising, return a denial
  message to the model so the run continues (it can pick another tool).

### Semantics

| Config / run state | Result |
|---|---|
| `tenant_tool_allowlist` is `None` | permitted (feature off) |
| untenanted run (`tenant_id` is `None`) | permitted (not tenant-governed) |
| tenant in map, tool in its set | permitted |
| tenant in map, tool not in set | denied |
| tenant in map → empty set | denied (all tools) |
| tenant absent, `default_deny=False` | permitted |
| tenant absent, `default_deny=True` | denied |

Untenanted runs are never governed by the allowlist, even under
`default_deny` — the map governs known tenants, not the absence of tenancy.

See `examples/governance/tenant_tool_allowlist.py`.

## Audit logging

Record every tool-call resolution to a pluggable, append-only sink:

```python
from troopai.adk.audit import JsonlFileAuditSink
from troopai.adk.run.config import RunConfig

config = RunConfig(tenant_id="acme", audit_sink=JsonlFileAuditSink("audit.jsonl"))
```

Each event is an `AuditEvent` with `tenant_id`, `agent_name`, `tool_name`,
`tool_call_id`, `args_hash`, `result_hash`, `outcome`, and `timestamp`.
**Only hashes are stored — never raw arguments or results** (sha256 of
canonical JSON), so the audit log is not a PII sink. `outcome` is one of
`ok` (executed), `denied` (allowlist rejection), or `error` (the tool
raised).

Built-in sinks:

| Sink | Use |
|---|---|
| `InMemoryAuditSink` | tests / single process |
| `JsonlFileAuditSink` | append-only JSON-Lines file |
| `S3AuditSink` | one object per event (`pip install 'troopai-adk-python[audit-s3]'`) |
| `PostgresAuditSink` | append-only `audit_events` table (`pip install 'troopai-adk-python[audit-postgres]'`) |

The `AuditSink` Protocol is `@runtime_checkable` — a custom sink (Kafka,
etc.) just needs `async def record(self, event: AuditEvent) -> None`.

By default audit is **best-effort**: a sink failure is logged and the run
continues. Set `audit_strict=True` to re-raise instead (fail-closed for
compliance deployments).

See `examples/governance/audit_logging.py`.

## Per-tenant Temporal task-queue routing

Isolate tenants at the durable-execution layer by dispatching each
tenant's workflow onto a tenant-specific task queue. Because Temporal
activities inherit their workflow's task queue, routing the workflow
isolates the whole run — premium tenants get a dedicated worker pool that
can't be starved by others.

```python
from troopai.adk.workflows.temporal.routing import (
    MappingTaskQueueRouter,
    start_tenant_workflow,
)

router = MappingTaskQueueRouter(
    mapping={"premium-tenant": "troopai-premium"},
    default="troopai-shared",
)

handle = await start_tenant_workflow(
    client, MyWorkflow, arg=prompt,
    tenant_id="premium-tenant", router=router, id="wf-1",
)
```

`start_tenant_workflow` is keyword-only after `workflow` and forwards
`**kwargs` to `client.start_workflow`, so pass workflow arguments the
Temporal way (`arg=<single>` or `args=[<multiple>]`). Operationally, run
one worker pool per queue/tier; isolation, prioritization, and rate
limiting fall out of pool sizing.

See `examples/temporal/multi_tenant_routing.py`.

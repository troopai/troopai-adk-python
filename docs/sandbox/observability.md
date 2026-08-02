# Sandbox Observability

Four surfaces feed downstream observability tooling: audit events, tracing
spans, run lifecycle hooks, and the usage accumulator.

## 1. Audit events

`AuditSink` is an ABC with one method `async emit(event:
SandboxAuditEvent)`. The sandbox lifecycle emits one event per
lifecycle transition:

| event_type | When |
|---|---|
| `"start"` | After `session.start()` succeeds |
| `"stop"` | Before `session.aclose()` runs (clean exit) |
| `"exec"` | After every `run_command` call completes (exit code, duration, command) |
| `"snapshot"` | After `SnapshotStore.save(...)` succeeds |
| `"violation"` | When a `SandboxCommandGuardrail` rejects a command |
| `"error"` | Any exception that exits the agent loop |

`exec` and `violation` events fire from the `run_command` tool via the
run-scoped `SandboxObservability` handle. Emission is best-effort: sink
errors are suppressed and logged at `DEBUG` level so an unavailable SIEM
does not abort the agent loop.

Built-in sinks:

- `NullAuditSink` (default) — discards events.
- `LoggingAuditSink(logger, level)` — routes events to a `logging.Logger`
  with the configured level per event type.

Composite sinks are first-class — wrap a list and fan out manually
if you need OTel + SIEM + logging concurrently.

## 2. Tracing spans

`sandbox_span(*, backend_id, command, ...)` mirrors `function_span()`
from the core tracing surface. Spans carry `SandboxSpanData` with
`backend_id`, `command`, `exit_code`, `duration_ms`,
`manifest_hash`, `resource_usage`, and `snapshot_id`.

The span data type is exported from `troopai.adk.types.tracing.span_data`
alongside the existing function / generation spans, so OTel exporters
ingest sandbox spans without configuration changes.

One span is emitted per `run_command` invocation. Spans are gated by
`RunConfig.tracing_enabled` and carry the resolved `backend_id` so spans
from different backends in one run are distinguishable.

## 3. Run lifecycle hooks

`RunHooks` (the run-lifecycle hook base class) carries four sandbox-specific
async callbacks. Override any of them in your `RunHooks` subclass:

### `on_sandbox_start(context, agent, session)`

Fires once after the sandbox session is acquired and `session.start()` has
returned successfully, before the first agent turn.

| Parameter | Type | Description |
|---|---|---|
| `context` | `RunContext[TContext]` | The active run context |
| `agent` | `Agent[TContext]` | The sandbox agent that owns the session |
| `session` | `BaseSandboxSession` | The live session handle |

### `on_sandbox_stop(context, agent, session, usage)`

Fires once during sandbox teardown, after live billing retrieval (when
`capture_live_cost=True`) and before `session.aclose()` runs.

| Parameter | Type | Description |
|---|---|---|
| `context` | `RunContext[TContext]` | The active run context |
| `agent` | `Agent[TContext]` | The sandbox agent that owned the session |
| `session` | `BaseSandboxSession` | The session being released |
| `usage` | `SandboxUsage` | Cumulative resource usage for the session |

### `on_sandbox_exec_start(context, agent, command)`

Fires before each non-PTY `run_command` call. PTY interactions stream
their own events and do not fire this hook.

| Parameter | Type | Description |
|---|---|---|
| `context` | `RunContext[TContext]` | The active run context |
| `agent` | `Agent[TContext]` | The sandbox agent owning the session |
| `command` | `str` | Command about to run (truncated to 1024 chars) |

### `on_sandbox_exec_end(context, agent, command, result)`

Fires after each non-PTY `run_command` call returns. Non-zero exit codes
are surfaced via `result.exit_code` — the hook is observation-only and
does not raise. If the backend cannot run the command at all (a transport
or connection failure, not a non-zero exit), `run_command` raises and this
end hook does not fire even though the matching `on_sandbox_exec_start`
did — pair start/end defensively if you key per-command state on the start.

| Parameter | Type | Description |
|---|---|---|
| `context` | `RunContext[TContext]` | The active run context |
| `agent` | `Agent[TContext]` | The sandbox agent owning the session |
| `command` | `str` | Command that ran (truncated to 1024 chars) |
| `result` | `ExecResult` | Captured stdout / stderr / exit code / duration |

### Wiring hooks

Pass a `RunHooks` subclass to `RunConfig`:

```python
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.config import RunConfig
from troopai.adk.types.sandbox.usage import SandboxUsage

class SandboxAuditHooks(RunHooks):
    async def on_sandbox_start(self, context, agent, session) -> None:
        print(f"sandbox started: agent={agent.name}")

    async def on_sandbox_stop(self, context, agent, session, usage: SandboxUsage) -> None:
        print(f"sandbox stopped: exec_count={usage.exec_count} "
              f"computed_cost_usd={usage.computed_cost_usd:.6f}")

    async def on_sandbox_exec_start(self, context, agent, command: str) -> None:
        print(f"exec start: {command!r}")

    async def on_sandbox_exec_end(self, context, agent, command: str, result) -> None:
        print(f"exec end: exit_code={result.exit_code} duration_ms={result.duration_ms}")

run_config = RunConfig(hooks=SandboxAuditHooks(), sandbox=sandbox_config)
```

## 4. Usage accumulator

`SandboxUsage` mirrors `LLMUsage`: `exec_count`, `total_duration_ms`,
`cpu_ms`, `memory_peak_mb`, `bytes_read`, `bytes_written`, plus a
per-exec breakdown list (`executions`). Supports `__add__` so cross-session
aggregation follows the same pattern as token usage.

The accumulator is populated by `SandboxObservability.after_exec`, which
fires from the `run_command` tool after every command. The per-command
`SandboxSingleExecUsage` record captures `command`, `exit_code`,
`duration_ms`, and `cost_usd` (computed from the backend rate card).

After `Runner.arun` returns, the aggregate is available on
`RunResult.sandbox_usage`:

```python
result = await Runner.arun(agent, prompt, run_config=run_config)
if result.sandbox_usage is not None:
    print(result.sandbox_usage.exec_count)
    print(result.sandbox_usage.computed_cost_usd)
```

See [cost.md](cost.md) for the cost fields (`computed_cost_usd`,
`billed_cost_usd`) and live billing configuration. See
[selection.md](selection.md) for cost-aware backend selection.

See `src/troopai/adk/sandbox/observability/` for the implementations.

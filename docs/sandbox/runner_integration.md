# Sandbox Runner Integration

The Runner detects `isinstance(agent, SandboxAgent)` or
`run_config.sandbox is not None`, then brackets the agent loop with
the sandbox lifecycle.

## Lifecycle (`sandbox_run_context`)

```
1. Acquire SandboxConcurrencyGuard (raises SandboxConcurrencyError on overlap)
2. Resolve session by priority:
     run_config.sandbox.session           # caller-provided live session
     RunState.sandbox_state               # HITL resumption
     run_config.sandbox.session_state     # serialized state -> client.resume()
     client.create(manifest, snapshot, options)
3. client._wrap_session(session, instrumentation)   # tracing + audit
4. session.start()
5. Fire on_sandbox_start hook
6. Run agent loop with capability tools merged into agent.tools
7. On exit: session.stop() (persist snapshot), session.aclose() (teardown)
8. Fire on_sandbox_stop hook
9. Release concurrency guard
```

The bracket lives in `runner_integration/lifecycle.py` and is invoked
from `Runner.arun` via an `AsyncExitStack`.

## Capability merge

Before the loop opens, the runner clones the agent (`dataclasses.replace`)
with capability-supplied tools appended to `agent.tools`. The
original `Agent` object is never mutated; the clone carries the
merged tool list for the duration of the run.

## Instructions composer

`runner_integration/instructions_composer.compose_sandbox_prompt(...)`
resolves the `SandboxAgent` placeholder marker, appends each capability's
`async instructions(manifest)` fragment, and renders the workspace
filesystem tree. The Runner calls it from `_resolve_system_prompt`
when a `_sandbox_handle` is attached to the run context.

## Concurrency guard

`SandboxConcurrencyGuard` is an `asyncio.Lock`-backed flag stored on
the `SandboxAgent` via `object.__setattr__(agent, "_concurrency_guard",
guard)`. External callers go through `agent.get_concurrency_guard()`
so the private attribute stays encapsulated. Re-entry by a concurrent
`Runner.arun(...)` on the same agent raises `SandboxConcurrencyError`.

## Hooks

`RunHooks` gains six no-op-by-default sandbox events:

- `on_sandbox_start`
- `on_sandbox_stop`
- `on_sandbox_exec_start` / `on_sandbox_exec_end`
- `on_sandbox_snapshot`
- `on_sandbox_error`

`AgentHooks` mirrors at per-agent scope. `CompositeRunHooks` fans out
preserving error-collection semantics.

See `src/troopai/adk/sandbox/runner_integration/` for the
implementation modules.

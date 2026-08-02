# Flows Module

Decorator-driven multi-step orchestration over typed shared state. A
`Flow` composes `Agent` / `Swarm` / `Graph` / `Task` calls as steps,
wires them via `@flow_start` / `@flow_listen` / `@flow_router` decorators, and carries
a developer-owned typed state object across the run.

Fills the gap between `Graph` (DAG with message-threading) and
`TaskPipeline` (sequential with no typed shared state). Coexists with
both; serves the different use case of *declarative business workflow
over mutable state*.

## Files

| File | Purpose |
|---|---|
| `flow.py` | `Flow[StateT]` base class + `FlowMeta` metaclass collecting decorator registrations at class creation |
| `flow_wrappers.py` | `FlowStep` descriptor class; supports `__or__` / `__and__` for fluent combinator construction |
| `decorators.py` | `@flow_start`, `@flow_listen`, `@flow_router` — each wraps the method in `FlowStep` |
| `combinators.py` | `Or` / `And` frozen dataclasses; produced by `|` / `&` operators on `FlowStep` |
| `registry.py` | `FlowStepRegistry` + `FlowTransitionTable` + `build_transition_table()` |
| `config.py` | `FlowConfig` (max_steps, max_total_tokens, error policy, fan-out cap) |
| `events.py` | Stream event dataclasses (`FlowStartEvent`, `FlowStepStartEvent`, etc.) |
| `result.py` | `FlowRunResult` + `FlowRunResultStreaming` |
| `executor.py` | `FlowExecutor` — the central driver |
| `exceptions.py` | `FlowDefinitionError`, `FlowMaxStepsExceeded`, `FlowStepError` |

## Key Architectural Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **Flow = config, Runner = execution** | No `Flow.run()` / `Flow.arun()`. Matches the codebase-wide rule established by `Agent`, `Swarm`, `Graph`, `Task`. Driver lives at `Runner.arun_flow`. |
| 2 | **Step methods take ONLY `self`** | Strictly enforced at class-definition time via `FlowMeta` signature validation. Prevents CrewAI's hidden behavior of injecting the previous step's return value via signature introspection (`crewai/flow/flow.py:3117-3139`). Return values are dropped on the floor (except `@flow_router` returns, which are typed `str`). |
| 3 | **State is explicit — never inferred from generic** | Two explicit paths: pass `initial_state=...` to the constructor (state class or pre-built instance — wins when both are present) or declare `state_factory = State` as a class attribute for zero-arg construction. A non-callable `state_factory` raises `FlowDefinitionError`. The framework NEVER reflects on the `Flow[StateT]` generic parameter to construct state. CrewAI does this; we reject it. |
| 4 | **Combinators are operator-only** | `@flow_listen(method_a | method_b)` / `@flow_listen(method_a & method_b)`. No `or_()` / `and_()` helper functions — operators on `FlowStep` instances are the fluent API. Matches the `TerminationCondition.__or__` precedent in `swarms/termination.py`. |
| 5 | **OR gates fire ONCE per flow run** | The first arrival of any trigger fires the listener; subsequent arrivals do NOT re-fire (gate is consumed). Matches the intuitive Python meaning of "or". CrewAI re-fires OR listeners on cyclic re-entry via `_clear_or_listeners()`; we deliberately keep simpler single-fire semantics. Cyclic re-firing is not implemented. |
| 6 | **AND gates fire ONCE per flow run** | After every required trigger has arrived, the listener fires once. Same single-fire semantic as OR. |
| 7 | **Mixed-type operator chains rejected** | `Or & x` and `And | x` raise `TypeError`. Mixed-type chains are ambiguous. For mixed shapes, construct `Or(...)` / `And(...)` directly with the dataclass constructor. |
| 8 | **No nested combinators** | CrewAI supports `or_(and_(a, b), c)` via recursive TypedDicts. We require flat gates. For complex shapes, restructure as multiple listeners or use a `@flow_router`. |
| 9 | **`@flow_start` is unconditional only** | CrewAI allows `@flow_start("trigger_method")` which fires the start after another method completes. We omit this — it overlaps with `@flow_listen` and adds confusion. If you need conditional entry, use `@flow_listen`. |
| 10 | **`@flow_router` returns a non-empty `str` label** | Empty / non-string returns raise `FlowDefinitionError` at runtime. Downstream `@flow_listen("label")` methods fire on the returned label. Plain `@flow_listen` methods' return values are IGNORED — only `@flow_router` drives dispatch. Prevents CrewAI's hidden "bare string return → next step" pattern. |
| 11 | **No source-code introspection of routers** | CrewAI parses router method source via `get_possible_return_constants` to know possible return values (used for visualization). We don't. Routers don't need to declare their possible labels. |
| 12 | **Persistence is explicit** | No `@persist` decorator. Developers call `FlowCheckpoint.to_json()` themselves and resume via `Runner.arun_flow_from_checkpoint(...)`. CrewAI's auto-write-after-every-step is forbidden. |
| 13 | **Parallel listener concurrency is the developer's responsibility** | When multiple listeners fire in parallel (via AND/OR gates or multiple `@flow_start` methods), they share `self.state` without framework-level locking. Developers write to disjoint fields or wrap mutations in their own `asyncio.Lock`. Mirrors the `TaskGroup` decision #15 contract. |
| 14 | **`FlowMeta` walks `cls.__dict__` only — NO inheritance** | Decorated methods on a parent class are NOT registered on the child's registry. Inheritance adds subtle ordering / override semantics conflicting with the no-hidden-behavior rule. If composition is needed, build a `FlowExecutable` adapter and nest the parent flow inside a `Graph` (or vice versa). |
| 15 | **Default `max_steps = 100`** | Cost-conservative cap matching CrewAI's `max_method_calls`. Raise via `FlowConfig.max_steps` for longer workflows. `max_total_tokens` defaults to `None` (no cap) — opt-in. |
| 16 | **`max_listeners_per_step = 20` default, enforced at build time** | Fan-out cap prevents accidental explosions. Validated by `build_transition_table` (not runtime), so misconfigured flows fail at `Runner.arun_flow` flow_start, not deep in execution. |
| 17 | **Error policy: `"halt"` default, `"route_to_error_handler"` opt-in** | On step exception, default halts the flow with `status="failed"`. With `"route_to_error_handler"`, the executor fires `@flow_listen("__error__")` listeners if any; otherwise falls back to halt. |
| 18 | **`run_context` shared via opt-in** | `Runner.arun_flow` attaches a `RunContext` as `flow.run_context`. Step bodies that want cumulative usage tracking pass `context=self.run_context` to their inner `Runner.arun(...)` calls. NEVER auto-injected. |
| 19 | **Streaming uses the same executor with event callback** | `Runner.arun_flow_streamed` constructs a `FlowExecutor` whose `on_event` pushes to an `asyncio.Queue`. Same code path as non-streaming; just observes events. |
| 20 | **`FlowRole = Literal["start", "listen", "router"]`** | Typed discriminator. Lets `match` statements be exhaustive and prevents `__flow_role__` from accumulating arbitrary strings. |
| 21 | **`FLOW_ERROR_TRIGGER` constant for the error route** | The `"__error__"` literal behind `error_policy="route_to_error_handler"` lives in `triggers.py` as `FLOW_ERROR_TRIGGER` and is re-exported from `troopai.adk.flows` and `troopai.adk`. Magic strings in user code are a readability and typo hazard; the executor uses the constant internally. |
| 22 | **`approval_policy=` decorator kwarg carries `FlowApprovalPolicy` onto the deferral** | All three decorators accept `approval_policy: FlowApprovalPolicy \| None = None`; the executor stamps it as `FlowDeferredStep.policy` (deadline stays `None` so the policy's relative `deadline_seconds` governs). The executor NEVER evaluates quorum/roles/deadline — the policy is informational for the out-of-band approval driver. |
| 23 | **Public API style mirrors swarms/graphs readability conventions** | `Flow` / `FlowRunResult` have one-line `__repr__`s (parts-list, capped previews, never dumping prompts/state); `FlowStep` exposes `role` / `triggers` / `approval_policy` as public read-only properties (the `__flow_*__` dunders stay for internals); `FlowCheckpoint.approve` / `.reject` accept a pending step NAME or its `FlowDeferredStep` (tolerant boundary), raising `ValueError` that lists valid pending names. |

## Differences from CrewAI Flow (source: `lib/crewai/src/crewai/flow/flow.py`)

CrewAI's Flow has at least seven hidden behaviors this ADK forbids:

1. Auto-instantiation of `Flow[StateT]` state from the generic parameter.
2. `_execute_single_listener` introspecting `inspect.signature(method).parameters` to inject the previous step's return value (`flow.py:3117-3139`).
3. `@persist` decorator auto-writing state after every step.
4. Auto-routing on bare `str` returns from non-`@flow_router` methods.
5. `kickoff_for_each(inputs)` auto-building state instances from input dicts.
6. `Flow.train()` / `test()` / `replay()` modes changing execution behavior implicitly.
7. `get_possible_return_constants` source-code parsing routers to determine route paths (used for visualization).

Plus three CrewAI features intentionally omitted:

- **Nested combinators** (`or_(and_(a, b), c)`) — flat gates only.
- **Conditional `@flow_start`** — use `@flow_listen` instead.
- **Cyclic re-firing of OR listeners** via `_clear_or_listeners` — fire-once-per-flow simpler.

See `docs/flows/flows.md` for usage. See `examples/flows/` for runnable
examples. See `tests/unit/flows/` for tests. Topology diagrams:
`Flow.to_mermaid()` / `Flow.to_dot()` — see `docs/visualization/visualization.md`.

## Composition with Existing Primitives

| Direction | Mechanism |
|---|---|
| Flow inside Graph | `FlowExecutable` adapter — wraps a `Flow` so it can be a `GraphNode` |
| Graph inside Flow | Inline: `await Runner.arun_graph(graph)` from a step body |
| Swarm inside Flow | Inline: `await Runner.arun_swarm(swarm)` from a step body |
| Agent inside Flow | Inline: `await Runner.arun(agent, prompt)` from a step body |
| Task inside Flow | Inline: `await Runner.arun_task(task)` from a step body |

## Future: Temporal `Workflow`

The name `Workflow` is RESERVED for the future Temporal-style durable
execution runtime. Temporal wraps any orchestration (Flow, Graph,
Swarm, TaskPipeline) and provides crash recovery, deterministic
replay, signals, queries, and long-running execution. The two layers
compose: a Temporal Workflow contains a Flow as its orchestration
topology.

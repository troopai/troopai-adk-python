# Tasks Module

Declarative units of work executed by `Runner.arun_task` /
`Runner.arun_task_streamed` / `Runner.arun_task_pipeline` /
`Runner.arun_task_group`. Purely additive layer on top of the existing
`Runner.arun(...)` surface.

## Files

| File | Purpose |
|---|---|
| `task.py` | `Task[TContext]` — frozen dataclass: description + agent + per-call overrides (output schema, guardrails, budgets, `skip_if`, `depends_on`) |
| `topology.py` | `topological_levels()` Kahn-grouped resolver + `TaskPipelineDefinitionError` |
| `task_output.py` | `TaskOutput` — frozen result: `final_output`, `new_items`, `usage`, `skipped`, `error`, `metadata` |
| `task_pipeline.py` | `TaskPipeline[TContext]` + `TaskPipelineResult[TContext]` — sequential composition with conditional skip + usage aggregation; the pipeline does NOT rewrite prompts at runtime |
| `task_group.py` | `TaskGroup[TContext]` + `TaskGroupResult[TContext]` — parallel fan-out under `asyncio.gather` with optional semaphore-bounded concurrency and `collect_all` / `halt_on_first` error policies |
| `task_pipeline_state.py` | `TaskPipelineState` — serializable mid-pipeline checkpoint (`pipeline_id`, recorded `slots`, `resume_index`, `completed_task_ids`). JSON round-trip via `to_json` / `from_json` (no version field; `from_json` raises `ValueError` on a missing required field). Resume via `Runner.arun_task_pipeline_from_state(pipeline, state)`. |

## Architectural Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | `Task` is `@dataclass(frozen=True, kw_only=True)` with NO `run()`/`arun()` method | Agent = config, Runner = execution. Task follows the same cardinal rule. |
| 2 | No `expected_output` field | CrewAI silently appends `expected_output` to the LLM prompt — hidden behavior. Developers put output expectations in their own prompt or `Task.output_schema`. |
| 3 | `Task.description` IS the user prompt; the framework NEVER rewrites prompts at runtime | CrewAI auto-joins prior task outputs with `"\n\n----------\n\n"` when `context=NOT_SPECIFIED`. We reject any runtime prompt transformation — including the originally-shipped `chain_inputs` formatter — to keep the developer's mental model unambiguous: what you write in `description` is exactly what the agent sees. Cross-task data flow is the developer's responsibility (run the upstream task, embed its result in the downstream `description`). |
| 4 | Per-task `output_schema` override uses `dataclasses.replace(agent, output_schema=task.output_schema)` to build a transient agent | Zero changes to `resolve_output_schema` / `call_llm` / tracing-span `_output_type_name_for_span`. The transient agent picks up the override naturally. |
| 5 | `arun_task_pipeline` accumulates per-task `TaskOutput.usage` into a fresh pipeline-level `RunContext` via `LLMUsage.__add__` | The simpler alternative — sharing one `RunContext` through the internal `arun` path — would couple `arun_task` to runner internals. Aggregating from `TaskOutput.usage` keeps `arun_task` a thin shim over the public `cls.arun` boundary; cost is one extra `LLMUsage` add per non-skipped task. The hook-side `pre_ctx` carries the user context but its `usage` field is the pre-task snapshot — read `TaskOutput.usage` for the per-task total or `TaskPipelineResult.context.usage` for the cumulative pipeline total. |
| 6 | Run-config guardrails run BEFORE task guardrails when merged | Matches `RunConfig.guardrails` documented contract (run-scope first, agent-scope second). Reversing silently changes policy. Duplicates are NOT de-duplicated. |
| 7 | `Task.max_turns` is passed as a kwarg to the internal arun path; NOT folded into the transient `RunConfig` | `max_turns` is a `Runner.arun` kwarg, not a `RunConfig` field — putting it in `RunConfig` would be a layering violation. |
| 8 | `skip_if` is a single `Callable` on `Task`, not a separate `ConditionalTask` class | One field replaces an entire class hierarchy + the hidden `get_skipped_task_output()` slot-filler. Skipped tasks remain in `TaskPipelineResult.task_outputs` with `skipped=True` so positional indexing stays stable. |
| 9 | `arun_task_pipeline` halts on error — no retries | `Session.add` is not idempotent across replays; retrying a task could write duplicate events. Errors surface in `TaskOutput.error`; the developer drives retry policy. |
| 10 | Pipeline streaming uses a **stream-of-streams** shape | `arun_task_pipeline_streamed` yields `(task_index, RunResultStreaming \| None)` pairs in input order. Skip slots yield `None` so positional indexing matches input tasks. Consumers drive each inner stream's `stream_events()`. No aggregated `TaskPipelineResult` — per-task `final_output` / `usage` lives on each inner stream. Rejected: merged event-stream (loses task-boundary observability) and silent skip (loses skip-firing visibility). |
| 11 | `Task.agent: Agent \| Swarm \| Graph` — union type | `arun_task` dispatches via `isinstance` to `arun` / `arun_swarm` / `arun_graph` and projects the inner result type into `TaskOutput`. `output_schema` rejected at construction for Swarm / Graph (those targets manage their own output shape). For Graph targets: user `RunHooks` are NOT propagated (the graph layer uses `GraphHooks`); attach those to the Graph directly. Streamed Task entry point (`arun_task_streamed`) is Agent-only. |
| 12 | New `RunHooks.on_task_start` / `on_task_end` lifecycle methods | Lifts task identity from verbose-only to user-land observability. Verbose `emit_task_*` continues to fire alongside. |
| 13 | `TaskGroup.max_concurrent` defaults to `None` (unbounded — caps at `len(tasks)`) | Matches the no-hidden-behavior principle: developers opt INTO a bounded concurrency, never out. Cost trade-off is documented in the dataclass; defaults remain cost-conservative by NOT introducing implicit retries / rate limiting. |
| 14 | `TaskGroup.error_policy` defaults to `"collect_all"` | Default keeps every task running on sibling failure — minimal-surprise semantics. `"halt_on_first"` cancels still-running siblings via `asyncio.Task.cancel()`; cancellation is best-effort because provider HTTP calls may finish anyway. Cancelled slots carry an explanatory `error` field so positional indexing stays stable. |
| 15 | `TaskGroup` hooks fire concurrently across tasks | The framework does NOT serialise `on_task_start` / `on_task_end` callbacks under group execution. Hooks holding shared mutable state MUST lock; the contract is documented on `TaskGroup`. |
| 16 | Pipeline persistence checkpoints at task boundaries only — no mid-turn HITL resume | `TaskPipelineState` records completed slots + `resume_index` + `completed_task_ids`. Mid-LLM-call pauses route through `RunState` instead. `Task.agent` / `Task.skip_if` / `Task.metadata` are NOT serialized — the resuming side reconstructs the pipeline definition. `new_items` is dropped from `TaskOutput.to_dict` (audit trail lives on `Session`); `final_output` is JSON-encoded when native, else `str(...)`. `from_json` raises `ValueError` when a required field is missing — partial / truncated payloads cannot silently rehydrate. |
| 17 | `Task.depends_on` opt-in DAG ordering | When any task in the pipeline declares `depends_on=(...)`, `arun_task_pipeline` runs in topological order: tasks at the same depth gather concurrently; downstream tasks wait for upstream completion. Pipelines without `depends_on` keep the sequential path. Validation (duplicate / unknown / missing `task_id` / cycles) raises `TaskPipelineDefinitionError` at `TaskPipeline` construction. DAG halt semantics: an error in one task of a level lets its siblings finish (predictable usage accounting); no later level fires. The returned `task_outputs` tuple is sorted back to declaration order so positional indexing matches the sequential path. Resume via `TaskPipelineState.completed_task_ids`. |

## Wiring

- `Runner.arun_task`, `Runner.arun_task_streamed`,
  `Runner.arun_task_pipeline`, and `Runner.arun_task_group` live in
  `run/runner.py`. They share `RunContext` plumbing with the existing
  `arun` / `arun_swarm` paths, including the verbose `emit_task_start`
  / `emit_task_end` panels.
- `RunHooks.on_task_start` and `on_task_end` are no-op base methods.
  `CompositeRunHooks` fans them out to all members. Under
  `arun_task_group` they fire concurrently — see decision #15.
- Existing `task_id` / `task_name` sites in `runner.py` (`arun`,
  `arun_swarm`, streamed) are unchanged — classic API consumers see
  identical behavior.

## Cost-Conservative Defaults

Every field defaults to None or an empty collection. Developer opts
INTO every cost: `output_schema=None`, `max_turns=None`,
`usage_limits=None`, empty guardrail lists, no streaming, no retries.

## Public API Style (mirrors graphs/handoffs/swarms)

- Every public dataclass (`Task`, `TaskDependency`, `TaskPipeline`,
  `TaskPipelineResult`, `TaskGroup`, `TaskGroupResult`, `TaskOutput`,
  `TaskPipelineState`) has a one-line `__repr__` built from a parts
  list — descriptions and outputs render as 60-char capped,
  newline-stripped `…`-ellipsized previews; full prompts are never
  dumped.
- Wiring stays `depends_on=[...]` only. `>>` / `|` operators were
  considered and rejected: frozen dataclasses make operator mutation
  or return-new semantics surprising, and a second wiring idiom would
  split the mental model the module works to keep unambiguous.
- `ErrorPolicy` and `TaskPipelineDefinitionError` are re-exported from
  the top-level `troopai.adk` package (parity with
  `FlowDefinitionError`).

See `docs/tasks/tasks.md` for usage and `examples/tasks/` for runnable
examples. See `tests/unit/tasks/` and `tests/integration/test_task_pipeline_e2e.py`
for tests.

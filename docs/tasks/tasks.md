# Tasks

Declarative units of work executed by `Runner.arun_task` and
`Runner.arun_task_pipeline`. Purely additive — every existing
`Runner.arun(...)` call continues to work unchanged.

## When to use Task

Pick `Task` when you want:

- A **named, documented work unit** with explicit metadata, surfaced
  in `RunHooks.on_task_start` / `on_task_end` and verbose Task panels.
- **Per-call overrides** for output schema, guardrails, max_turns, or
  usage budget — without mutating the underlying `Agent` definition.
- **Sequential composition** via `TaskPipeline` with explicit (never
  implicit) context-chaining between steps.
- **Conditional execution** via `Task.skip_if` without losing
  positional indexing in the resulting outputs.

Otherwise, classic `Runner.arun(agent, "prompt", ...)` remains the
shortest path.

## Quick start

```python
from troopai.adk import Agent, Task, Runner

summariser = Agent(name="Summariser", system_prompt="...")

task = Task(
    description="Summarise the meeting notes below.",
    agent=summariser,
)
output = await Runner.arun_task(task)
print(output.final_output)
```

## API surface

### `Task`

Frozen dataclass. Every field besides `description` and `agent` is
optional and defaults to a cost-conservative value.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `description` | `str` | required | User prompt fed to the agent |
| `agent` | `Agent[TContext]` | required | The agent to execute |
| `name` | `str \| None` | `None` | Display name (hooks + tracing); defaults to truncated description |
| `task_id` | `str \| None` | `None` | Stable identity. When `None`, the Runner generates a full `str(uuid.uuid4())` (36-char canonical UUID with hyphens). The verbose Task panel renders only the first 8 chars; the full UUID flows through hooks, tracing, and `TaskOutput.task_id`. |
| `output_schema` | `type \| AgentOutputSchemaBase \| None` | `None` | Per-call structured-output schema |
| `guardrails` | `AgentGuardrails` | `AgentGuardrails()` | `input` + `output` lists appended after RunConfig guardrails (mirrors `Agent.guardrails`) |
| `max_turns` | `int \| None` | `None` | Per-task ceiling; falls back to `DEFAULT_MAX_TURNS` |
| `usage_limits` | `LLMUsageLimits \| None` | `None` | Per-task LLM usage budget |
| `skip_if` | `Callable[[Sequence[TaskOutput]], bool] \| None` | `None` | Pipeline-skip predicate (pipeline-only) |
| `metadata` | `dict[str, Any]` | `{}` | Surfaced verbatim on `TaskOutput.metadata` |

`Task` raises `ValueError` at construction if `description` is empty
or `max_turns` is non-positive.

`Task` and its companions (`TaskDependency`, `TaskPipeline`,
`TaskGroup`, `TaskOutput`, `TaskPipelineResult`, `TaskGroupResult`,
`TaskPipelineState`) all have human-readable one-line reprs that never
dump full prompts — descriptions and outputs are capped previews:

```python
>>> facts
Task(name='facts', agent='researcher', depends_on=1)
>>> pipeline
TaskPipeline(tasks=4, dag=True)
```

### `TaskOutput`

Frozen result of one `Task` execution.

| Field | Type | Notes |
|---|---|---|
| `task_id` | `str` | Identity for this run — full `str(uuid.uuid4())` unless `Task.task_id` was explicitly set. Display truncates to 8 chars in the verbose panel; the value here is always the full UUID. |
| `task_name` | `str` | Display name |
| `final_output` | `Any` | Agent's final output; `None` on skip/error |
| `new_items` | `tuple[RunItem, ...]` | Layer-3 conversation trail |
| `usage` | `LLMUsage \| None` | Per-task token usage |
| `skipped` | `bool` | `True` when `skip_if` returned `True` |
| `error` | `str \| None` | Stringified exception, mutually exclusive with `skipped` |
| `metadata` | `dict[str, Any]` | Copy of `Task.metadata` |

### `TaskPipeline` + `TaskPipelineResult`

```python
from troopai.adk import Task, TaskPipeline, Runner

classify = Task(description="Detect language.", agent=classifier)
translate = Task(
    description="Translate to English.",
    agent=translator,
    skip_if=lambda prior: prior[-1].metadata.get("lang") == "en",
)
review = Task(description="Comment in one sentence.", agent=reviewer)

pipeline = TaskPipeline(tasks=(classify, translate, review))
result = await Runner.arun_task_pipeline(pipeline)
print(result.final_output)
print("Total tokens:", result.context.usage.total_tokens)
```

`TaskPipeline` validates that `tasks` is non-empty at construction.
Each task's `description` is fed verbatim as the user prompt; the
pipeline does NOT rewrite prompts at runtime.

### Declarative DAG ordering (`Task.depends_on`)

When any task in the pipeline declares `depends_on=[...]`,
`TaskPipeline` switches from sequential-by-declaration order to
topological DAG execution: tasks at the same depth run concurrently
via `asyncio.gather`, and downstream tasks wait until all their
upstream dependencies finish. Pipelines with no `depends_on` keep
the sequential path unchanged.

`depends_on` accepts a `Sequence` of `Task` instances or `task_id`
strings (mix freely). The default `None` means no dependency wiring.

```python
intake = Task(description="...", agent=intake_agent, task_id="intake")
facts = Task(description="...", agent=facts_reviewer,
             task_id="facts", depends_on=[intake])
style = Task(description="...", agent=style_reviewer,
             task_id="style", depends_on=[intake])
synthesise = Task(description="...", agent=synthesiser,
                  task_id="synthesise", depends_on=[facts, style])

pipeline = TaskPipeline(tasks=(intake, facts, style, synthesise))
result = await Runner.arun_task_pipeline(pipeline)
```

Validation happens at `TaskPipeline` construction:

- Every task that declares `depends_on` MUST have an explicit
  `task_id` so the resolver can name it.
- `task_id` values must be unique within the pipeline.
- Every entry in `depends_on` must resolve to a `task_id` present in
  the pipeline.
- No cycles. The validator reports the involved IDs.

Invalid pipelines raise `TaskPipelineDefinitionError`
(a `UserError` subclass, importable from `troopai.adk`) at
construction — before any task runs.

`TaskPipeline.topological_levels()` exposes the depth-grouped tasks
for introspection or diagram generation. See
`examples/tasks/dag_pipeline.py` for a runnable diamond.

### Explicit input forwarding (`TaskDependency` + `TaskInputFilter`)

By default `Task.description` is the user prompt verbatim — the
framework NEVER auto-injects upstream outputs (we reject CrewAI's
hidden auto-aggregation). To forward an upstream task's output into
a downstream input, wrap that upstream in a `TaskDependency` and
attach an `input_filter`:

```python
from troopai.adk import Task, TaskDependency
from troopai.adk.tasks.task_filters import forward_final_output

synthesise = Task(
    description="Combine the reviewer feedback above.",
    agent=synthesiser,
    task_id="synthesise",
    depends_on=[
        TaskDependency(task=facts, input_filter=forward_final_output),
        TaskDependency(task=style, input_filter=forward_final_output),
    ],
)
```

`TaskDependency` carries per-edge policy:

- `task` — `Task` instance or `task_id` string.
- `input_filter` — optional `TaskInputFilter` (a
  `Callable[[TaskInputData], TaskInputData]`) shaping that upstream's
  contribution. When `None`, the dependency is pure ordering — wait
  for the upstream to complete, do not read its output.

A bare `Task` or `task_id` string entry in `depends_on` is equivalent
to `TaskDependency(task=..., input_filter=None)` — pure ordering.
Mix bare and wrapped entries freely.

#### Filter contract (`TaskInputFilter`)

A filter receives one `TaskInputData` per upstream (per the wrapper
it sits on) — `task_id`, `output` (the upstream `TaskOutput`),
`items` (the upstream `RunItem` stream) — and returns a new
`TaskInputData` (via `.clone(forwarded=...)`) with `forwarded` set
to the `RunItem` subset that flows into the downstream input. The
runner concatenates `forwarded` items across all wrapped
dependencies, converts each via `RunItem.to_param()`, and prepends
them BEFORE the message(s) derived from `Task.description`. The
downstream agent's user prompt becomes a single
`list[LLMInputContentItem]` with forwarded messages first, then the
description.

`Task.description: UserPrompt` accepts either a plain string
(wrapped into a single user message at runtime) or a
`list[LLMInputContentItem]` directly when you need full control.

#### Built-in filters

`troopai.adk.tasks.task_filters` provides common patterns:

- `forward_final_output` — forward only the upstream's
  `final_output` as one user message.
- `forward_new_items` — forward the upstream's entire `RunItem`
  stream (system / user / assistant / tool).
- `forward_messages_only` — forward only the assistant
  `MessageOutputItem`s; strip system / user / tool internals.
- `keep_last_n(n)` — forward only the last n upstream items.
- `compose(*filters)` — chain multiple filters in a pipeline.

Write a custom filter when you need richer shaping:

```python
def summarise(data: TaskInputData) -> TaskInputData:
    summary = build_summary(data.output)
    item = UserItem(raw={"role": "user", "content": summary})
    return data.clone(forwarded=(item,))
```

`TaskPipelineResult` carries:

- `task_outputs: tuple[TaskOutput, ...]` — one slot per pipeline task in order.
  Skipped tasks appear with `skipped=True`; failed tasks appear with `error`
  set.
- `final_output: Any` — the last non-skipped task's `final_output`, or `None`.
- `context: RunContext[TContext] | None` — shared `RunContext`; its
  `usage.total_tokens` is the cumulative pipeline total.

## Explicit chaining (no runtime prompt rewriting)

`Task.description` is the user prompt verbatim — the framework
never transforms it. If a downstream task needs an upstream task's
output, the developer wires the chain explicitly:

```python
# Step 1 — run the upstream task on its own.
research_out = await Runner.arun_task(
    Task(description="Research Apollo 11.", agent=researcher),
)

# Step 2 — embed the upstream result in the downstream description.
summary_out = await Runner.arun_task(
    Task(
        description=f"Compress this into one sentence:\n\n{research_out.final_output}",
        agent=summariser,
    ),
)
```

This pattern matches CrewAI's `task.description = user prompt`
mental model and keeps things unambiguous: what you read in the
description is exactly what the agent sees. `TaskPipeline` remains
the right abstraction when N tasks need sequential execution with
conditional skip and usage aggregation — when there is no
upstream-to-downstream prompt data flow at all.

## Conditional execution: `skip_if`

```python
translate = Task(
    description="Translate to English.",
    agent=translator,
    skip_if=lambda prior: prior[-1].metadata.get("lang") == "en",
)
```

When `skip_if` returns `True`, the runner inserts a
`TaskOutput(skipped=True, ...)` slot at that position. **Slots are
never dropped** — positional indexing matches the input pipeline so
downstream consumers (audit logs, later `skip_if` predicates that
inspect prior outputs) can rely on tuple length.

The predicate MUST be a pure function of its inputs. Capturing
mutable closure state is the caller's responsibility and is not
validated by the framework. A predicate that raises halts the
pipeline; the error is captured in the corresponding `TaskOutput`.

## Guardrail merge order

The transient `RunConfig` passed to inner `arun` is:

```text
input_guardrails  = run_config.guardrails.input + task.guardrails.input
output_guardrails = run_config.guardrails.output + task.guardrails.output
```

Run-scope guardrails run FIRST, task-scope SECOND. The order is not
reversible. Duplicates are NOT de-duplicated — the same guardrail
appearing in both lists runs twice.

## Per-call output schema override

`Task.output_schema` overrides `Agent.output_schema` for that one
call. The runner builds a transient agent via
`dataclasses.replace(agent, output_schema=task.output_schema)`; the
original `Agent` definition is untouched.

## Error handling

`Runner.arun_task` raises `Exception` on inner failure (after firing
`on_task_end` with the error-set `TaskOutput`). `BaseException`
subclasses like `KeyboardInterrupt` and `asyncio.CancelledError`
propagate untouched — cooperative cancellation is preserved.

`Runner.arun_task_pipeline` does NOT raise — it captures the
exception into a `TaskOutput(error=...)`, halts the pipeline, and
returns the partial `TaskPipelineResult`. The stringified exception
is truncated (cap: 500 chars) to limit credential / response-body
leakage from provider exceptions. The full exception is also logged
via `logger.error` / `logger.warning` for debugging fidelity.

**No retries** — `Session.add` is not idempotent across replays. If
a task fails, the developer chooses the retry strategy at their
layer.

## Security: prior task outputs are untrusted

When the developer embeds an upstream `TaskOutput.final_output` in a
downstream task's `description` (the explicit-chaining pattern above),
they are forwarding LLM-produced text into another LLM's user prompt.
That content MUST be treated as untrusted — a hostile or hallucinated
prior output can redirect the downstream agent (prompt-injection).

Mitigations are the developer's responsibility:

- Apply `guardrails.output` on upstream tasks to detect / strip
  injection attempts before forwarding.
- Use `Task.output_schema` to constrain the upstream output to a
  structured shape and extract only specific fields when building the
  downstream description (avoid forwarding raw freeform text).
- Sanitise the prior output yourself before constructing the
  downstream `description`.

The framework does not auto-sanitise. `Task.skip_if` predicates also
receive untrusted prior outputs — apply the same caution there.

## Lifecycle hooks

```python
class MyHooks(RunHooks):
    async def on_task_start(self, context, agent, task):
        print(f"Starting {task.name}")

    async def on_task_end(self, context, agent, task, *, output):
        if output.error is not None:
            print(f"{task.name} failed: {output.error}")
```

These fire in addition to the existing run-level hooks
(`on_agent_start`, `on_llm_start`, etc.) and the verbose Task panel.

## Rejected CrewAI patterns

This implementation intentionally REJECTS several CrewAI Task
behaviors that inject framework actions the developer did not opt
into:

- No auto-manager-agent on hierarchical process (no hierarchical
  process at all).
- No runtime prompt rewriting at all — `Task.description` is the
  user prompt verbatim; cross-task data flow is the developer's
  responsibility via explicit chaining.
- No string-guardrails that silently spawn an LLM guardrail agent —
  guardrails must be explicit `AgentInputGuardrail` /
  `AgentOutputGuardrail` instances.
- No `expected_output` field that mutates the LLM prompt — put
  output expectations in `description` or `output_schema`.
- No `output_file` — file IO is the developer's responsibility.

See `examples/tasks/` for runnable examples.

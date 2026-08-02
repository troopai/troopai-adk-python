# Graph Checkpointing and Resume

Crash recovery and pause/resume for long-running graph executions — so a
multi-superstep pipeline that fails halfway through can continue from where
it stopped rather than restart from scratch.

## Why Checkpoint

A graph run may span many supersteps, invoke expensive LLM nodes, or
integrate with slow external systems. Without persistence, any crash —
process kill, OOM, network partition — loses all progress and forces a
full re-run from the entry node.

Checkpointing solves this at two granularities:

- **Per-node**: a snapshot is taken after every node completes, so a crash
  loses at most the nodes that had not yet finished in the current
  superstep.
- **Graph-end flush**: a final snapshot is written when the loop exits
  cleanly, making the terminal state available for inspection or for
  repeating the resume path in a test.

A checkpointer is a `HookProvider` that subscribes to `on_node_end` and
`on_graph_end`. The graph loop contains zero persistence code — swapping
the checkpointer is the only change needed to move from in-memory to
SQLite (or any future store).

## Attaching a Checkpointer

Pass the checkpointer in the `hooks=` list. It is a `HookProvider`, so it
registers its own callbacks on the `HookRegistry`; it is additive to any
`GraphHooks` observers already in the list.

### `InMemoryCheckpointer`

Dict-backed, process-local, zero setup. State is lost when the process
exits. Appropriate for:

- Unit and integration tests.
- Notebooks.
- Single-process demos where crash recovery is not needed.

```python
from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
from troopai.adk.run.runner import Runner

checkpointer = InMemoryCheckpointer()

result = await Runner.arun_graph(
    pipeline,
    "Summarize the quarterly report.",
    hooks=[checkpointer],
    thread_id="run-001",
)
```

### `SQLiteCheckpointer`

Durable, single-file, backed by `aiosqlite`. Survives process restarts
and is accessible from multiple processes that open the same file. Use
for production and crash-recoverable runs.

```python
from troopai.adk.graphs.checkpointers.sqlite import SQLiteCheckpointer
from troopai.adk.run.runner import Runner

checkpointer = SQLiteCheckpointer("runs.db")

result = await Runner.arun_graph(
    pipeline,
    "Summarize the quarterly report.",
    hooks=[checkpointer],
    thread_id="run-001",
)

await checkpointer.close()
```

### `thread_id` — opt-in per-run identity

`thread_id` is the key under which checkpoints are stored and loaded. It
is opt-in: when a checkpointer is attached but no `thread_id` is given,
the loop auto-generates a `thread-XXXX` id (12 hex characters) for the
run. Passing an explicit `thread_id` lets you later retrieve and resume
that exact run by name.

## Resuming a Run

Three entry points support resume. All load the persisted `GraphState` for
the given `thread_id`, re-seed the join barriers from the
`produced_at`/`versions_seen` maps, and continue from where the run
stopped.

### `Runner.arun_graph_from_checkpoint` (async)

```python
async def arun_graph_from_checkpoint(
    graph: Graph[Any],
    *,
    checkpointer: Checkpointer,
    thread_id: str,
    user_prompt: UserPrompt | None = None,
    context: TContext | None = None,
    hooks: list[GraphHooks[Any] | HookProvider] | None = None,
    run_config: RunConfig | None = None,
) -> GraphRunResult[Any]: ...
```

The checkpointer is appended to `hooks` automatically if it is not already
present, so the resumed run continues to checkpoint as nodes complete.

```python
result = await Runner.arun_graph_from_checkpoint(
    pipeline,
    checkpointer=checkpointer,
    thread_id="run-001",
)
```

### `Runner.run_graph_from_checkpoint` (sync)

Synchronous wrapper with identical parameters. Uses the same
running-loop / `ThreadPoolExecutor` strategy as `Runner.run_graph`.

```python
result = Runner.run_graph_from_checkpoint(
    pipeline,
    checkpointer=checkpointer,
    thread_id="run-001",
)
```

### `Runner.configure().graph(graph).resume_from(checkpointer, thread_id).arun()`

`GraphRunner` exposes `.resume_from(checkpointer, thread_id)`. When set, the
terminal `.run()` / `.arun()` call delegates to `run_graph_from_checkpoint`
instead of starting fresh. `user_prompt` is optional on resume.

```python
result = await (
    Runner.configure()
    .graph(pipeline)
    .resume_from(checkpointer, "run-001")
    .arun()
)
```

### Why `user_prompt` is optional on resume

The entry node normally does not re-fire on resume — its output is already
in `GraphState.node_results` and its upstream output was already consumed
before the checkpoint was taken. The driver only re-fires nodes whose
upstream output was produced but not yet consumed. If the entry node is
among those, pass `user_prompt` to give it input; in the common case it
is not re-fired and the value is ignored.

### Complete SQLite resume example

```python
import asyncio
import logging
import os

from troopai.adk.graphs import Graph, GraphConfig
from troopai.adk.graphs.checkpointers.sqlite import SQLiteCheckpointer
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "/tmp/demo_resume.db"
THREAD_ID = "demo-run-001"


# -- Graph definition -------------------------------------------------------

async def step_a(text: str) -> str:
    return f"a-done:{text}"

async def step_b(text: str) -> str:
    return f"b-done:{text}"

async def step_c(text: str) -> str:
    return f"c-done:{text}"

pipeline = (
    Graph.new("resume-demo", description="a → b → c with checkpointing")
    .node("a", step_a)
    .node("b", step_b)
    .node("c", step_c)
    .pipe("a", "b", "c")
    .entry("a")
    .terminal("c")
    # Cap to 2 supersteps so the first run stops before "c" fires.
    .with_config(GraphConfig(max_supersteps=2))
    .compile()
)


# -- First run (capped at 2 supersteps) -------------------------------------

async def first_run() -> None:
    checkpointer = SQLiteCheckpointer(DB_PATH)
    logger.info("Starting first run (capped at 2 supersteps).")
    result = await Runner.arun_graph(
        pipeline,
        "hello",
        hooks=[checkpointer],
        thread_id=THREAD_ID,
    )
    logger.info("First run status: %s", result.status.value)
    logger.info("Completed nodes: %s", list(result.node_results.keys()))
    await checkpointer.close()


# -- Resume run (fresh instance, no superstep cap) --------------------------

async def resume_run() -> None:
    # Rebuild the graph without the superstep cap for the resumed run.
    full_pipeline = (
        Graph.new("resume-demo", description="a → b → c with checkpointing")
        .node("a", step_a)
        .node("b", step_b)
        .node("c", step_c)
        .pipe("a", "b", "c")
        .entry("a")
        .terminal("c")
        .compile()
    )
    checkpointer = SQLiteCheckpointer(DB_PATH)
    logger.info("Resuming from checkpoint thread_id=%s.", THREAD_ID)
    result = await Runner.arun_graph_from_checkpoint(
        full_pipeline,
        checkpointer=checkpointer,
        thread_id=THREAD_ID,
    )
    logger.info("Resumed run status: %s", result.status.value)
    logger.info("Final output: %s", result.final_output)
    await checkpointer.close()
    os.unlink(DB_PATH)


async def main() -> None:
    await first_run()
    await resume_run()


if __name__ == "__main__":
    asyncio.run(main())
```

Expected log output (abbreviated):

```
INFO  Starting first run (capped at 2 supersteps).
INFO  First run status: max_supersteps
INFO  Completed nodes: ['a', 'b']
INFO  Resuming from checkpoint thread_id=demo-run-001.
INFO  Resumed run status: completed
INFO  Final output: c-done:[b]
b-done:[a]
a-done:hello
```

The default `Merge.concat_text` strategy labels upstream outputs with their
source node id in brackets, producing the layered string above. Node `c`
fired only on resume — `a` and `b` were not re-executed.

## Selective Re-fire (Idempotency)

On resume the driver does not blindly re-run every node. It compares two
maps stored in the checkpoint:

- **`produced_at[node_id]`** — the superstep at which `node_id`'s
  current `node_results` entry was written.
- **`versions_seen[node_id][upstream_id]`** — the superstep at which
  `node_id` last consumed input from `upstream_id`.

For each edge `(upstream → downstream)`, the barrier for `downstream` is
re-armed only when `produced_at[upstream] > versions_seen[downstream][upstream]`.
In other words: the upstream produced output that `downstream` has not yet
consumed. Nodes that had already consumed all their upstreams' output do not
re-execute.

This is the same channel-version comparison used by LangGraph's Pregel engine
(`_algo.py::prepare_next_tasks`), adapted here to the ADK's per-node barrier
model.

### Linear example: crash after node b

```
Graph: a → b → c
```

1. Superstep 1 fires `a`. Checkpoint: `produced_at = {a: 1}`, `versions_seen = {}`.
2. Superstep 2 fires `b`. `b` consumes `a`'s output.
   Checkpoint: `produced_at = {a: 1, b: 2}`, `versions_seen = {b: {a: 2}}`.
3. Process crashes before superstep 3.

On resume:

- Edge `a → b`: `produced_at[a]=1`, `versions_seen[b][a]=2`. Since `1 ≤ 2`,
  `b`'s barrier is NOT re-armed. `b` does not re-fire.
- Edge `b → c`: `produced_at[b]=2`, `versions_seen[c]` is absent (`-1`).
  Since `2 > -1`, `c`'s barrier IS re-armed with `b`'s stored result.
  `c` fires and the graph completes.

`a` and `b` are not re-executed. Only `c` fires, consuming the already-stored
output of `b`. No inner agent or swarm double-executes.

### Cyclic example: crash mid-cycle

```
Graph: a → b → c → b  (b loops back via a conditional edge)
```

Suppose the cycle ran once (`b` fired at superstep 2, `c` at superstep 3,
`b` queued again) and the process crashed before superstep 4.

`produced_at = {a: 1, b: 2, c: 3}`, `versions_seen = {b: {a: 2, c: 3}}`.
Wait — `c` fired at superstep 3 and its output is directed back to `b`.
`versions_seen[b][c]` was recorded when `b` consumed `c`'s output in a
prior iteration, or it is absent if `b` hadn't consumed it yet.

If `b` had NOT yet consumed `c`'s output (`versions_seen[b][c]` is absent
or less than `produced_at[c]`), the edge `c → b` re-arms `b`'s barrier,
and `b` re-fires. The loop continues normally from that point.

If `b` had already consumed `c`'s output before the crash (both maps agree),
`b` does not re-fire. The cycle proceeds to whichever node is next according
to `b`'s outgoing edges and the stored result.

## Cumulative Budgets

`GraphConfig.max_supersteps` and `GraphConfig.max_total_tokens` are not reset
on resume. The resumed run loads the `cumulative_usage` and `superstep` from
the checkpoint and continues counting from those values.

```python
from troopai.adk.graphs import Graph, GraphConfig

pipeline = (
    Graph.new("budget-demo")
    ...
    .with_config(GraphConfig(
        max_supersteps=100,
        max_total_tokens=500_000,
    ))
    .compile()
)
```

If the original run consumed 80 supersteps and 300 000 tokens before
checkpointing, the resumed run has 20 supersteps and 200 000 tokens
remaining. This ensures a resumed run cannot silently exceed the cost cap
configured at compile time — the budget is a property of the graph, not
of a single execution segment.

## Crash Semantics

Checkpoints are written:

1. **After each node completes** (`on_node_end`).
2. **When the graph loop exits** (`on_graph_end`), whether by terminal
   completion, budget exhaustion, or unhandled error.

A mid-superstep crash — one where some nodes in a parallel superstep had
completed and others had not — loses only the nodes that had not yet written
their checkpoint entry. On resume those nodes re-fire from their barrier
arrivals. Nodes that had completed before the crash are not re-executed.

A clean exit (all terminals fired) also writes a final checkpoint. Loading
that checkpoint and calling `arun_graph_from_checkpoint` produces an
immediate result — `_seed_barriers_from_checkpoint` finds no unconsumed
edges and the loop exits at once with `COMPLETED`.

## `SQLiteCheckpointer` Specifics

- **One row per `thread_id`, latest wins.** Each `save` is an upsert keyed
  on `thread_id`. There is no time-travel or replay-from-any-superstep;
  only the most recent checkpoint for a thread is retained.
- **Durable across processes.** Any process that can open the same file path
  can load and resume a run.
- **Connection lifecycle.** The connection opens lazily on first use and is
  held for the lifetime of the instance. The caller owns the instance and its
  connection; `Runner` does not close a caller-supplied checkpointer. Pass the
  same instance for both the initial run and any later resume call within a
  process. Call `await checkpointer.close()` at application shutdown or when
  the instance goes out of scope.

```python
# Process A: initial run
checkpointer = SQLiteCheckpointer("shared.db")
await Runner.arun_graph(pipeline, "input", hooks=[checkpointer], thread_id="t-1")
await checkpointer.close()

# Process B (separate process, same file): resume
checkpointer = SQLiteCheckpointer("shared.db")
result = await Runner.arun_graph_from_checkpoint(
    pipeline, checkpointer=checkpointer, thread_id="t-1"
)
await checkpointer.close()
```

- **Graph-id mismatch raises `ValueError`.** `load` validates that the
  stored `graph_id` matches `graph.id`. Supplying the wrong graph raises
  immediately rather than silently deserialising into a mismatched state.

## Caveats

- **Same graph required at resume.** The `Graph` supplied at resume must
  have the same `id` and the same node ids as the graph that produced the
  checkpoint. Node executables are code, not data, and do not round-trip
  through the checkpoint — the caller is responsible for supplying an
  equivalent graph.
- **Tolerant loader.** `GraphState.from_dict` reads every field with
  `dict.get(key, default)`. Persisted payloads from an older format load
  to safe defaults for new fields; no version field is stored or checked.
  A structural break (removed or renamed field) requires renaming the
  loader, not adding a version discriminator.
- **`pending_sends` is reserved.** The `GraphCheckpoint.pending_sends`
  field is always empty. It is reserved for dynamic fan-out packets once
  that feature is implemented.
- **Non-streaming path only.** Checkpointing applies to the standard
  `arun_graph` / `run_graph` / `arun_graph_from_checkpoint` /
  `run_graph_from_checkpoint` execution path. The `arun_graph_streamed`
  path does not support checkpointing.

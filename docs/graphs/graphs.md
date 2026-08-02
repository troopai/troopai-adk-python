# Graphs

Composable multi-agent orchestration — an Agent, a Swarm, another Graph, or a
plain Python callable, all as interchangeable nodes in a directed graph executed
under BSP (Bulk Synchronous Parallel) supersteps.

## Why Graph Exists

TroopAI already ships two multi-agent primitives:

- `Handoff` — one-shot linear delegation (Agent A → Agent B, run ends).
- `Swarm` — iterative collaboration with cycles (A ↔ B ↔ C until an explicit
  stop signal).

Neither composes. A developer cannot today embed a Swarm inside a larger DAG, or
route between an Agent-with-handoffs and a nested review graph. The `Graph`
primitive closes that gap.

It is the **composition layer** that unifies the three existing primitives:

| Node type | Auto-wrapped to |
|-----------|----------------|
| `Agent` | `AgentExecutable` |
| `Swarm` | `SwarmExecutable` |
| `Graph` | `Executable` directly (nested) |
| plain callable | `CallableExecutable` |

All four are interchangeable at the node level — the graph loop calls
`node.executable.invoke(input, context, config)` uniformly, regardless of what
sits inside.

### Comparison with LangGraph and Strands Agents

**LangGraph** is technically deep (BSP supersteps, channels, full checkpointing)
but its API is verbose: `Annotated[list, add_messages]` reducers declared on
state fields, `START`/`END` sentinels, and a dual routing mechanism
(`add_conditional_edges` + `Command(goto=)`). Per-node cost attribution is absent
from the result type.

**Strands Agents** has a readable builder but ships OR-default joins (silently
runs targets with partial inputs), a broken-parallel bug in older versions, and
no `graph_path` tag on nested events so inner-graph activity is opaque.

**TroopAI Graph** adds what neither ships:

1. Single routing mechanism — `edge(from, to, when=predicate)` with an optional
   pure predicate. No dual routing.
2. AND-join by default — the safer choice; OR is an explicit opt-in.
3. `Merge` declared on the receiving node — visible, local, no magic.
4. `graph_path: tuple[str, ...]` on every stream event — O(1) nesting depth.
5. `per_node_usage` on the result type — per-node LLM cost attribution.

## Quick Start

```python
import asyncio
from troopai.adk.graphs import Graph
from troopai.adk.run.runner import Runner

# Two callable nodes, one edge, one run.
async def summarize(text: str) -> str:
    return f"Summary: {text[:80]}"

async def format_output(text: str) -> str:
    return text.strip().upper()

pipeline = (
    Graph.new("quick-start", description="Summarize then format")
    .node("summarizer", summarize)
    .node("formatter", format_output)
    .pipe("summarizer", "formatter")
    .entry("summarizer")
    .terminal("formatter")
    .compile()
)

result = asyncio.run(Runner.arun_graph(pipeline, "Some long document text here."))
print(result.final_output)
print(result.status)          # GraphRunStatus.COMPLETED
print(result.total_supersteps)  # 2
```

## The Builder API

`Graph.new(id, *, description=None, metadata=None)` returns a `GraphBuilder`.
Every method except `.compile()` returns `self` for chaining.

### `.node(node_id, executable, *, merge=None, join=None, description=None, metadata=None)`

Register a node. The `executable` argument is auto-wrapped:

- An `Agent` → `AgentExecutable`.
- A `Swarm` → `SwarmExecutable`.
- A `Graph` (or any `Executable`) → used as-is; no wrapper needed.
- Any callable → `CallableExecutable` (arity detected at wrap time).

`merge` overrides the default fan-in strategy (see [Join Semantics and Merge
Strategies](#join-semantics-and-merge-strategies)). `join` overrides AND-join
for this node only.

```python
# Agent node
graph.node("triage", triage_agent)

# Swarm node
graph.node("research", research_swarm)

# Nested graph node — no adapter boilerplate
graph.node("legal", legal_subgraph)

# Callable node — zero LLM cost
graph.node("reformat", lambda text: text.strip())

# With explicit fan-in strategy
from troopai.adk.graphs import Merge, JoinSemantics

graph.node("synthesizer", synthesizer_agent, merge=Merge.concat_text)
graph.node("first-wins", voting_agent, join=JoinSemantics.OR)
```

### `.edge(source, target, *, when=None, label=None, priority=0)`

Register a directed edge. Both endpoints must already be registered as nodes.

`when` is an optional pure predicate over the upstream `NodeResult`. The
predicate can be sync or async; the graph loop awaits if needed. Returning a
truthy value fires the edge; falsy skips it.

```python
# Unconditional
graph.edge("writer", "reviewer")

# Conditional — route based on the upstream node's output
graph.edge(
    "checker",
    "approver",
    when=lambda result: result.output == "approved",
)

# With a label propagated to the downstream ExecutableInput.edge_label
graph.edge("router", "fast-path", when=lambda r: r.metadata.get("priority") == "high", label="high-priority")
```

### `.pipe(*node_ids)`

Sugar for chaining two or more nodes linearly. At least two ids are required.

```python
# Equivalent to .edge("a", "b").edge("b", "c").edge("c", "d")
graph.pipe("a", "b", "c", "d")
```

### `.entry(node_id)`

Declare the single entry node. Exactly one is allowed. The entry node must have
no incoming edges (validated at compile time).

### `.terminal(*node_ids)`

Declare one or more terminal nodes. When the graph has one terminal, `final_output`
is that node's output. When it has multiple, `final_output` is a dict keyed by
terminal id. Terminal nodes must have no outgoing edges.

### `.with_config(config)`

Attach a `GraphConfig` (budgets and knobs). Overrides the default `GraphConfig()`.

```python
from troopai.adk.graphs import GraphConfig

graph.with_config(GraphConfig(
    max_supersteps=20,
    max_total_tokens=500_000,
    fail_fast=True,
))
```

`max_supersteps` and `max_total_tokens` are **cumulative across a checkpoint
resume** — a resumed run continues counting from the checkpoint rather than
resetting to zero, so a configured cap is never silently exceeded.
See `docs/graphs/checkpointing.md` for the full resume contract.

### `.with_hooks(*hooks)`

Attach one or more `GraphHooks` at compile time. These merge with per-run hooks
passed to `Runner.arun_graph`.

### `.compile()`

Validate and return a frozen `Graph`. The compiled artifact is an `Executable`
itself — it can be used as a node in another `Graph` without any adapter.

Validation performed at compile time:

1. At least one node is registered.
2. `.entry()` has been called.
3. `.terminal()` has been called at least once.
4. Every node is reachable from the entry.
5. Every non-entry node can reach at least one terminal.
6. Entry has no incoming edges; terminals have no outgoing edges.
7. No duplicate node ids.

## Routing: `when=` Predicates

Every routing decision is exactly one `edge()` call with an optional `when=`
predicate. The predicate receives the upstream `NodeResult` and returns a boolean.

```python
from troopai.adk.orchestration.executable import NodeResult

def route_approved(result: NodeResult) -> bool:
    # Output is whatever the upstream node returned
    return isinstance(result.output, str) and result.output.startswith("APPROVED")

pipeline = (
    Graph.new("approval-flow")
    .node("reviewer", review_agent)
    .node("publisher", publish_agent)
    .node("drafter", draft_agent)
    .edge("reviewer", "publisher", when=route_approved)
    .edge("reviewer", "drafter", when=lambda r: not route_approved(r))
    .entry("reviewer")
    .terminal("publisher", "drafter")
    .compile()
)
```

When multiple conditional edges from the same source all return `False`, the
downstream targets receive no arrivals. If no terminal fires and no nodes become
ready, the loop exits with `GraphRunStatus.NO_READY_NODES` — a sign that the
routing conditions need adjustment.

## Join Semantics and Merge Strategies

When a node has multiple incoming edges (fan-in), two orthogonal concerns apply:

- **Join semantics**: which upstream arrivals are required before the node fires.
- **Merge strategy**: how the arrived upstream outputs are folded into one input.

### Join Semantics (`join=` on `.node()`)

`JoinSemantics.AND` (default): the node fires only after **all** expected upstreams
have arrived. The safer choice — prevents running with partial inputs.

`JoinSemantics.OR`: the node fires as soon as **any** expected upstream arrives.
Use for "first-to-respond wins" patterns.

```python
from troopai.adk.graphs import JoinSemantics

# OR-join: fires when either fast_path or slow_path arrives first
graph.node("responder", respond_agent, join=JoinSemantics.OR)
```

### Merge Strategies (`merge=` on `.node()`)

The merge function folds upstream `NodeResult` values into the downstream node's
`ExecutableInput.content`. Strategies are declared on the **receiving** node —
visible, local, no `Annotated` magic.

| Strategy | Behaviour |
|----------|-----------|
| `Merge.concat_text` | Join upstream `final_text` values with double newlines, labelled by source id. **Default.** |
| `Merge.last_wins` | Take the lexicographically-last source id's text; discard the rest. |
| `Merge.first_wins` | Inverse of `last_wins` — take the first source-sorted result. |
| `Merge.extend_items` | Concatenate Layer 1 replay params from every upstream (full turn structure). Heavier on tokens; preserves tool calls and reasoning. |
| `Merge.custom(fn)` | Wrap a user-supplied `(results, sources) -> str | list[LLMInputContentItem]` function. |

The merge result order is deterministic: the graph loop sorts upstream arrivals
by source node id before calling the merge function, regardless of task completion
order.

```python
from troopai.adk.graphs import Merge

# Fan-in from two parallel researchers: label + join their outputs
graph.node("synthesizer", synthesizer_agent, merge=Merge.concat_text)

# Fan-in where only the most authoritative source matters
graph.node("final", publish_agent, merge=Merge.last_wins)

# Full history fan-in (expensive)
graph.node("auditor", audit_agent, merge=Merge.extend_items)

# Custom reducer
def my_merge(results, sources):
    return "\n---\n".join(
        f"=== {src} ===\n{r.final_text or ''}"
        for r, src in zip(results, sources)
        if r.final_text is not None
    )

graph.node("combiner", combiner_agent, merge=Merge.custom(my_merge))
```

## Running Graphs

Three entry points with identical semantics:

### `Runner.arun_graph(graph, user_prompt, *, context=None, hooks=None, run_config=None, thread_id=None)`

Async. Returns `GraphRunResult`.

```python
import asyncio
from troopai.adk.run.runner import Runner

result = asyncio.run(Runner.arun_graph(pipeline, "Draft a legal brief on X."))
```

### `Runner.run_graph(graph, user_prompt, *, context=None, hooks=None, run_config=None, thread_id=None)`

Synchronous wrapper. Offloads to a `ThreadPoolExecutor` when called inside a
running event loop; otherwise calls `asyncio.run` directly.

```python
result = Runner.run_graph(pipeline, "Draft a legal brief on X.")
```

### `Runner.configure().graph(graph)` — Profile Runner

Returns a `GraphRunner` for method-chained configuration:

```python
result = await (
    Runner.configure(context=my_context)
    .model("gpt-4o")
    .graph(pipeline)
    .hooks([my_hooks, InMemoryCheckpointer()])
    .thread("run-2025-04-18")
    .arun("Draft a legal brief on X.")
)
```

`GraphRunner` methods include `hooks(hooks_list)`, `thread(thread_id)`,
`resume_from(checkpointer, thread_id)`, and shared profile methods such as
`model(model)`, `context(ctx)`, and `with_config(run_config)`.

Terminal methods: `.arun(user_prompt)` (async), `.run(user_prompt)` (sync).

## `GraphRunResult` Fields

| Field | Type | Description |
|-------|------|-------------|
| `final_output` | `Any` | Terminal output. `str` when one terminal; `dict[str, Any]` when multiple. |
| `status` | `GraphRunStatus` | `COMPLETED`, `FAILED`, `MAX_SUPERSTEPS`, `MAX_TOKENS`, `NO_READY_NODES`. |
| `user_prompt` | `UserPrompt` | Original input passed to `arun_graph`. |
| `new_items` | `list[RunItem]` | Layer 3 items from all nodes, in completion order. |
| `state` | `Optional[GraphState]` | Final graph state. Serialisable via `state.to_json()`. |
| `node_results` | `dict[str, NodeResult]` | Latest `NodeResult` per node id at loop exit. |
| `context` | `Optional[RunContext]` | The shared `RunContext` (carries usage totals). |
| `per_node_usage` | `dict[str, LLMUsage]` | Per-node LLM cost attribution keyed by node id. |
| `cumulative_usage` | `LLMUsage` | Graph-wide total — equal to the sum of `per_node_usage`. |
| `total_supersteps` | `int` | Number of BSP supersteps executed. |
| `error` | `Optional[str]` | Serialised error when `status == FAILED`. |

```python
result = await Runner.arun_graph(pipeline, "input")

# Single terminal
print(result.final_output)

# Multiple terminals
if isinstance(result.final_output, dict):
    approved = result.final_output.get("approver")
    draft = result.final_output.get("drafter")

# Per-node cost attribution
for node_id, usage in result.per_node_usage.items():
    print(f"{node_id}: {usage.total_tokens} tokens")

# All produced items
for item in result.new_items:
    print(item)
```

## Hooks and Observability

### `GraphHooks`

Subclass `GraphHooks` and override the lifecycle methods you care about:

```python
import logging
from troopai.adk.graphs import GraphHooks
from troopai.adk.graphs.result import GraphRunStatus

logger = logging.getLogger(__name__)

class MyGraphObserver(GraphHooks):

    async def on_graph_start(self, context, state):
        logger.info("Graph started. entry=%s", state.graph.entry)

    async def on_superstep_start(self, context, state, ready_nodes):
        logger.info("Superstep %d: %s", state.superstep, ready_nodes)

    async def on_node_start(self, context, state, node_id, input):
        logger.debug("Node %s starting", node_id)

    async def on_node_end(self, context, state, node_id, result):
        logger.info("Node %s: output=%s tokens=%d",
                    node_id, result.final_text, result.usage.total_tokens)

    async def on_node_error(self, context, state, node_id, error):
        logger.error("Node %s raised: %s", node_id, error)

    async def on_superstep_end(self, context, state, fired_nodes, items):
        logger.info("Superstep %d done. Fired: %s", state.superstep, fired_nodes)

    async def on_graph_end(self, context, state, status, final_output):
        logger.info("Graph done. status=%s supersteps=%d", status, state.superstep)

result = await Runner.arun_graph(
    pipeline,
    "input",
    hooks=[MyGraphObserver()],
)
```

### `HookProvider` Protocol

A `HookProvider` can register multiple callbacks on a `HookRegistry` at once.
Checkpointers implement this protocol. User-defined providers can use it to
attach a combination of observers without subclassing `GraphHooks` directly:

```python
from troopai.adk.graphs import HookProvider, HookRegistry

class MetricsSink(HookProvider):
    def register(self, registry: HookRegistry) -> None:
        # Attach multiple hooks from one provider
        registry.add(self._make_hooks())

    def _make_hooks(self):
        ...
```

### Stream Events

Every `GraphStreamEvent` subclasses `dict` — zero serialisation overhead.
Discriminate by the `type` key or by `isinstance`:

| Event class | `type` constant | Emitted when |
|-------------|-----------------|--------------|
| `GraphStartEvent` | `GRAPH_START` | Before the first superstep |
| `SuperstepStartEvent` | `SUPERSTEP_START` | Top of each superstep |
| `NodeStartEvent` | `NODE_START` | Before `Executable.invoke` |
| `NodeEndEvent` | `NODE_END` | After `Executable.invoke` returns cleanly |
| `NodeErrorEvent` | `NODE_ERROR` | When a node raises |
| `NodeStreamEvent` | `NODE_STREAM` | Per-token events from inner executables |
| `SuperstepEndEvent` | `SUPERSTEP_END` | After a superstep completes |
| `GraphEndEvent` | `GRAPH_END` | After the last superstep |

Every event carries `graph_path: tuple[str, ...]` — a stack of graph ids from
the outermost to the innermost graph. Nesting depth is `len(event["graph_path"])`.

```python
from troopai.adk.graphs import NodeEndEvent, GRAPH_END

# Dict access (wire-safe, works after json.dumps/loads)
if ev["type"] == GRAPH_END:
    print(ev["status"], ev["total_supersteps"])

# isinstance dispatch (Pythonic)
if isinstance(ev, NodeEndEvent):
    print(ev["node_id"], ev["result"].final_text)
```

Graph streaming is available via `Runner.arun_graph_streamed(graph, prompt)` and
`Runner.configure().graph(graph).arun(prompt, stream=True)`. Both return a
`GraphRunResultStreaming` immediately; events are consumed with
`async for ev in result.stream_events()`. The stream delivers structural events
(`graph.start`, `graph.node_start`, `graph.node_end`, `graph.node_error`,
`graph.end`, etc.) and, for agent nodes, interior `graph.node_stream` envelopes
carrying token-level events. Cancellation is available via
`result.cancel("immediate")` or `result.cancel("after_superstep")`. See
`docs/graphs/streaming.md` for the full streaming contract.

## Checkpointing

`InMemoryCheckpointer` is the default checkpointer. It persists state after
each node fires and after the graph ends, using the hook-provider pattern —
the graph loop itself contains no persistence code.

```python
from troopai.adk.graphs import InMemoryCheckpointer

checkpointer = InMemoryCheckpointer()

# Run with checkpointing opted in via thread_id
result = await Runner.arun_graph(
    pipeline,
    "input",
    hooks=[checkpointer],
    thread_id="my-run-001",
)

# Inspect the persisted state
thread_ids = await checkpointer.list_checkpoints()
print(thread_ids)  # ["my-run-001"]

state = await checkpointer.load("my-run-001", pipeline)
if state is not None:
    print(state.superstep)
    print(state.node_results.keys())
```

### Resume from Checkpoint

Pass a rehydrated `GraphState` as `initial_state` to `run_graph_loop` to resume
from where a previous run left off. The loop skips initialisation and resumes
from `state.superstep + 1`.

```python
from troopai.adk.run.graph_loop import run_graph_loop

state = await checkpointer.load("my-run-001", pipeline)
if state is not None:
    # Resume the run (same context + config as the original)
    result = await run_graph_loop(
        graph=pipeline,
        user_prompt="",        # ignored when resuming — state carries prior input context
        context=run_context,
        config=run_config,
        initial_state=state,
    )
```

### `GraphCheckpoint` Fields

| Field | Type | Description |
|-------|------|-------------|
| `thread_id` | `str` | Logical run identifier. |
| `graph_id` | `str` | Must match the `Graph.id` at load time. |
| `state` | `dict[str, Any]` | Serialised `GraphState.to_dict()` payload. |
| `created_at` | `float` | Unix timestamp of when the checkpoint was produced. |
| `superstep` | `int` | Superstep at which the checkpoint was taken. |

### Checkpointer Protocol

Implement the `Checkpointer` protocol against your own store:

```python
from troopai.adk.graphs import Checkpointer, GraphCheckpoint, HookRegistry

class RedisCheckpointer(Checkpointer):

    def register(self, registry: HookRegistry) -> None:
        # Subscribe to the hooks you care about
        ...

    async def save(self, checkpoint: GraphCheckpoint) -> None:
        ...

    async def load(self, thread_id: str, graph) -> Optional["GraphState"]:
        ...

    async def list_checkpoints(self) -> list[str]:
        ...

    async def delete(self, thread_id: str) -> None:
        ...
```

Human-in-the-loop (HITL) interrupt/resume is fully supported. A node calls
`request_human_input(inp, question, *, kind, **metadata)` to pause the run;
the BSP loop captures the `Interrupt` onto `GraphState.pending_interrupts`,
writes a checkpoint, and returns `status=INTERRUPTED` with no exception raised.
Resume is via `Runner.arun_graph_from_checkpoint(..., resume=GraphResume(replies={...}))`
or the profile runner's `resume_from(cp, tid).arun(..., resume=...)`.
Streaming runs emit a `graph.node_interrupt` event before
`graph.end(status=interrupted)`. See `docs/graphs/hitl.md` for the full
surface.

## Error Handling

### Fail-Fast (default)

When `GraphConfig.fail_fast=True` (the default), the first node exception in a
superstep cancels all sibling tasks immediately via `asyncio.wait(FIRST_COMPLETED)`.
The graph loop still fires `on_node_error` on the failed node and then records
`status=FAILED` on the result.

```python
result = await Runner.arun_graph(pipeline, "input")
if result.status == GraphRunStatus.FAILED:
    print(result.error)  # "RuntimeError: boom"
```

### Non-Fail-Fast

Set `fail_fast=False` on `GraphConfig` to let siblings finish. Nodes that
depended on the failed node's output do not fire (their `JoinBarrier` never
becomes ready), but unaffected parallel branches complete normally.

```python
from troopai.adk.graphs import GraphConfig

pipeline = (
    Graph.new("tolerant")
    .with_config(GraphConfig(fail_fast=False))
    ...
    .compile()
)
```

### `GraphRunStatus` Values

| Status | Meaning |
|--------|---------|
| `COMPLETED` | At least one terminal fired; loop exited cleanly. |
| `FAILED` | A node raised and `fail_fast` surfaced the error. |
| `INTERRUPTED` | A node raised `InterruptException`; the run is paused for human input. See `docs/graphs/hitl.md`. |
| `MAX_SUPERSTEPS` | `GraphConfig.max_supersteps` was hit. |
| `MAX_TOKENS` | `GraphConfig.max_total_tokens` was hit. |
| `NO_READY_NODES` | No terminal fired and no more nodes were schedulable (all conditional edges returned `False`). |

### Node Reliability: Retry and Timeout

Per-node retry (`GraphNode.retry: NodeRetryPolicy | None`) and per-attempt
timeout (`GraphNode.timeout: float | None`) are opt-in and default-off —
`NodeRetryPolicy()` defaults to `max_attempts=1` (no retries) and
`GraphConfig.per_node_timeout` defaults to `None` (no timeout). They never add
cost the developer did not choose. Graph-level defaults are set on
`GraphConfig.default_retry` and `GraphConfig.per_node_timeout`; per-node fields
override the defaults when set to a non-`None` value. See
`docs/graphs/reliability.md` for the full retry/timeout contract, exception
types, and the failure-boundary order.

## Decision Tree

```
Need to run multiple agents?
│
├── One-shot transfer (Agent A → Agent B, no return)
│   └── Use Handoff
│
├── Delegate sub-task (A asks B a question, resumes with B's answer)
│   └── Use Agent.as_tool()
│
├── Iterative collaboration with cycles (A ↔ B ↔ C until stop)
│   └── Use Swarm
│
└── DAG orchestration (parallel fan-out, conditional routing, fan-in,
    mixing Agent + Swarm + subgraph nodes)
    └── Use Graph
         │
         ├── Nodes run in parallel in the same superstep? Yes → automatic via BSP
         ├── Fan-in: wait for ALL upstreams? → AND-join (default)
         ├── Fan-in: fire on first arrival? → join=JoinSemantics.OR
         └── Merge multiple upstreams? → declare merge= on receiving node
```

See `docs/graphs/composition.md` for the in-depth composition guide.
See `examples/graphs/` for runnable examples (linear, parallel fan-out,
conditional routing, composition with Swarm, nested subgraph).

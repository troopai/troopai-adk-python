# Graph Streaming

Real-time incremental visibility into multi-superstep graph runs —
structural progress events, agent-node interior token events, and
cooperative or immediate cancellation.

## Why Stream

A long graph run may span many supersteps, fan out into parallel agent
nodes, and run for minutes before producing a final answer. Without
streaming, the caller waits in silence until `arun_graph` returns.
Streaming lets you:

- **Display incremental progress** — show which nodes started, which
  completed, and which superstep the run is on, without polling.
- **React to individual node outputs** — read a node's result as soon as
  its `graph.node_end` event arrives, before the rest of the graph
  finishes.
- **Observe agent-interior tokens** — agent nodes forward their per-token
  stream events inside `graph.node_stream` envelopes, so you can surface
  LLM generation in real time.
- **Cancel at a superstep boundary** — cooperative cancel lets the current
  superstep finish cleanly; immediate cancel aborts in-flight node tasks.

## Event Taxonomy

Every event is a `GraphStreamEvent` — a `dict` subclass. Access fields by
key (`ev["type"]`) or by attribute after narrowing with `isinstance`.

### Structural events

| Event class | `type` constant | `type` string | Emitted when |
|---|---|---|---|
| `GraphStartEvent` | `GRAPH_START` | `"graph.start"` | Before the first superstep |
| `SuperstepStartEvent` | `SUPERSTEP_START` | `"graph.superstep_start"` | Top of each superstep |
| `NodeStartEvent` | `NODE_START` | `"graph.node_start"` | Before `Executable.invoke` |
| `NodeEndEvent` | `NODE_END` | `"graph.node_end"` | After `Executable.invoke` returns cleanly |
| `NodeErrorEvent` | `NODE_ERROR` | `"graph.node_error"` | When a node raises |
| `NodeStreamEvent` | `NODE_STREAM` | `"graph.node_stream"` | Interior event forwarded from an agent node |
| `SuperstepEndEvent` | `SUPERSTEP_END` | `"graph.superstep_end"` | After a superstep completes |
| `GraphEndEvent` | `GRAPH_END` | `"graph.end"` | After the last superstep |

### Key fields shared by all events

- **`type`** — discriminator string (the `type` constant above).
- **`graph_path`** — `tuple[str, ...]` identifying the graph that emitted
  this event. For a top-level run `len(ev["graph_path"]) == 1`. A nested
  `Graph` used as a node runs non-streaming via `Graph.invoke()` and does
  not surface inner events into the outer stream; the outer consumer sees
  only that node's `graph.node_start` / `graph.node_end` boundaries, and
  every observable event carries the outer graph's id.

### Per-event payload keys

| Event | Additional keys |
|---|---|
| `GraphStartEvent` | `graph_id`, `description`, `entry_node`, `terminal_nodes` |
| `GraphEndEvent` | `graph_id`, `status` (`GraphRunStatus`), `final_output`, `total_supersteps` |
| `SuperstepStartEvent` | `superstep` (1-indexed), `ready_nodes` |
| `SuperstepEndEvent` | `superstep`, `fired_nodes`, `errored_nodes` |
| `NodeStartEvent` | `node_id`, `superstep`, `from_nodes`, `edge_label`, `input` |
| `NodeEndEvent` | `node_id`, `superstep`, `result` (`NodeResult`) |
| `NodeErrorEvent` | `node_id`, `superstep`, `error_type` (exception class name), `error_message` |
| `NodeStreamEvent` | `node_id`, `inner` (original event from the inner executable) |

### `NodeStreamEvent` — interior envelope

When an agent node's `stream_async` yields an event, the graph loop
re-emits it wrapped in a `NodeStreamEvent`. `graph_path` and `node_id`
identify the node; `inner` is the original inner event.

```python
if ev["type"] == "graph.node_stream":
    print(ev["graph_path"], ev["node_id"], ev["inner"])
```

Constants and event classes are importable from `troopai.adk.graphs.events`.

## Consuming a Stream

### `Runner.arun_graph_streamed`

```python
import asyncio
import logging

from troopai.adk.graphs import Graph
from troopai.adk.graphs.events import GRAPH_END, GRAPH_START, NODE_END, NODE_ERROR, NODE_START, NODE_STREAM
from troopai.adk.run.runner import Runner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def step_a(text: str) -> str:
    return f"a:{text}"


async def step_b(text: str) -> str:
    return f"b:{text}"


pipeline = (
    Graph.new("streaming-demo")
    .node("a", step_a)
    .node("b", step_b)
    .edge("a", "b")
    .entry("a")
    .terminal("b")
    .compile()
)


async def main() -> None:
    result = await Runner.arun_graph_streamed(pipeline, "hello")

    async for ev in result.stream_events():
        match ev["type"]:
            case "graph.start":
                logger.info("Graph started: entry=%s", ev["entry_node"])
            case "graph.superstep_start":
                logger.info("Superstep %d: nodes=%s", ev["superstep"], ev["ready_nodes"])
            case "graph.node_start":
                logger.info("  Node started: %s", ev["node_id"])
            case "graph.node_end":
                logger.info("  Node done: %s", ev["node_id"])
            case "graph.node_error":
                logger.warning("  Node error: %s — %s (%s)", ev["node_id"], ev["error_message"], ev["error_type"])
            case "graph.node_stream":
                logger.debug("  Interior event from %s: %s", ev["node_id"], ev["inner"])
            case "graph.end":
                logger.info("Graph ended: status=%s supersteps=%d", ev["status"].value, ev["total_supersteps"])

    # Terminal fields are populated once stream_events() returns.
    logger.info("final_output=%s", result.final_output)
    logger.info("status=%s", result.status.value)  # e.g. "completed"


if __name__ == "__main__":
    asyncio.run(main())
```

### Profile runner form — `configure().graph(...).arun(stream=True)`

The graph runner exposes `stream=True` on `.arun()`:

```python
result = await Runner.configure().graph(pipeline).arun("hello", stream=True)
async for ev in result.stream_events():
    logger.info("%s %s", ev["type"], ev.get("node_id", ""))
logger.info("status=%s output=%s", result.status.value, result.final_output)
```

### Terminal fields

After `stream_events()` returns (the async-for loop exits), the following
fields on `GraphRunResultStreaming` are populated:

| Field | Description |
|---|---|
| `final_output` | Aggregate graph output (terminal node's result, or `{id: output}` for multiple terminals) |
| `status` | `GraphRunStatus` — `.value` is `"completed"`, `"failed"`, `"max_supersteps"`, `"no_ready_nodes"`, or `"max_tokens"` |
| `state` | Final `GraphState` |
| `per_node_usage` | Per-node cost attribution keyed by node id |
| `cumulative_usage` | Graph-wide cumulative usage |
| `total_supersteps` | Number of supersteps executed |

`GraphRunResultStreaming` has no `.error` field. Node failures are
observed via the `graph.node_error` event (which carries `error_type` and
`error_message`) and the terminal `status.value == "failed"`.

## Cancellation

`GraphRunResultStreaming.cancel(mode)` accepts two modes:

```python
result.cancel("immediate")       # abort now; cancels driver + in-flight tasks
result.cancel("after_superstep") # cooperative; current superstep finishes
```

### `"immediate"` mode

Drops pending events from the queue, cancels the background driver task,
and cancels every in-flight node task tracked at the time of the call.
Concurrent fan-out nodes in the same superstep are all cancelled together.
The `stream_events()` consumer wakes and the async-for loop exits promptly.

```python
result = await Runner.arun_graph_streamed(pipeline, "go")
it = result.stream_events()
first = await it.__anext__()   # graph.start

result.cancel("immediate")

# Drain completes promptly — the driver was cancelled, not awaited.
remaining = [ev async for ev in it]
```

### `"after_superstep"` mode

Cooperative: the current superstep finishes normally, all its node tasks
run to completion, and the driver stops before starting the next superstep.
Use this when you need the current batch of work to complete (for example,
to checkpoint its output) before stopping.

```python
result = await Runner.arun_graph_streamed(pipeline, "go")
async for ev in result.stream_events():
    if ev["type"] == "graph.superstep_end" and ev["superstep"] >= 3:
        result.cancel("after_superstep")
        # Stream drains through the end of superstep 3; no superstep 4.
```

## What Each Node Type Streams

The level of interior detail in the event stream depends on the node type.

### Agent nodes — full interior forwarding

An `Agent` wrapped via `AgentExecutable` overrides `stream_async` and
forwards each agent-level event (token deltas, tool calls, tool outputs)
as a `graph.node_stream` envelope:

```python
async for ev in result.stream_events():
    if ev["type"] == "graph.node_stream" and ev["node_id"] == "my-agent":
        inner = ev["inner"]   # agent StreamEvent (raw_response_event, etc.)
```

The agent node also emits the structural `graph.node_start` and
`graph.node_end` boundaries like every other node.

### Callable nodes — structural boundaries only

Plain-callable nodes (Python functions or lambdas) are synchronous or
async functions; they have no interior event stream. The graph loop emits
`graph.node_start` when the call begins and `graph.node_end` (or
`graph.node_error`) when it returns or raises. No `graph.node_stream`
events are emitted for callable nodes.

### Swarm nodes — structural boundaries and terminal result

A `Swarm` wrapped in a `SwarmExecutable` does not forward interior swarm
events through the graph stream. The graph loop emits `graph.node_start`
and `graph.node_end` for the swarm node as a unit, contributing its
terminal result at `graph.node_end`. Interior swarm observability is
available via `GraphHooks.on_node_end` or through a dedicated swarm run.

### Nested-graph nodes — structural boundaries and terminal result

A nested `Graph` sitting inside a graph node emits `graph.node_start` and
`graph.node_end` for the nested-graph node as a unit from the outer
graph's perspective. It does not forward its own internal structural events
(its own `graph.start`, `graph.superstep_start`, etc.) into the outer
stream. The terminal result of the inner graph becomes the `result` field
on `graph.node_end`.

## Composition

### Streamed resume

Combine streaming with checkpoint resume via a profile runner:

```python
from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer

cp = InMemoryCheckpointer()

# Initial run (may be non-streaming or streaming with a superstep cap).
await Runner.arun_graph(pipeline, "go", hooks=[cp], thread_id="run-001")

# Resume via the streaming profile-runner path.
result = await (
    Runner.configure()
    .graph(pipeline)
    .resume_from(cp, "run-001")
    .arun(stream=True)
)
async for ev in result.stream_events():
    logger.info("%s", ev["type"])
logger.info("status=%s output=%s", result.status.value, result.final_output)
```

The resume path re-fires only the nodes whose upstream output was produced
before the checkpoint but whose downstream node had not consumed it yet —
the same selective re-fire semantics as the non-streaming resume path. See
`docs/graphs/checkpointing.md` for the full selective re-fire contract.

### Node errors in the stream

A node timeout or retry exhaustion raises a structured exception that
surfaces as a `graph.node_error` event before the graph ends with
`status.value == "failed"`. Read the error detail off the event, not off
the result (which has no `.error` field):

```python
async for ev in result.stream_events():
    if ev["type"] == "graph.node_error":
        logger.error(
            "Node %s failed: %s — %s",
            ev["node_id"],
            ev["error_type"],     # e.g. "GraphNodeTimeoutError"
            ev["error_message"],
        )

# After the stream ends:
if result.status.value == "failed":
    logger.error("Run failed after %d superstep(s).", result.total_supersteps)
```

`error_type` is the exception class name (e.g. `"GraphNodeTimeoutError"`,
`"NodeRetriesExhaustedError"`). See `docs/graphs/reliability.md` for the
full timeout and retry exception contract.

## See Also

- `docs/graphs/graphs.md` — profile runner API, event taxonomy overview, `fail_fast`,
  `GraphHooks`.
- `docs/graphs/checkpointing.md` — selective re-fire, cumulative budgets,
  `SQLiteCheckpointer`.
- `docs/graphs/reliability.md` — per-node timeout / retry, `GraphNodeTimeoutError`,
  `NodeRetriesExhaustedError`.
- `docs/graphs/composition.md` — nested graphs, fan-out patterns, nested-graph
  streaming contract.
- `src/troopai/adk/graphs/events.py` — discriminator constants + event classes.
- `src/troopai/adk/graphs/result.py` — `GraphRunResultStreaming`, `GraphRunStatus`.
- `examples/graphs/` — runnable graph examples.

# Graph Human-in-the-Loop Interrupts

Pause a graph run mid-superstep to collect a human decision, then resume
with the human's reply. The run suspends cleanly, persists its state via
a checkpointer, and re-fires only the interrupted node on resume — no
arbitrary re-execution of prior work.

## Why HITL

A graph run may encounter a decision that a model should not make alone:
approve a tool action, choose a route, or fetch data from an external
approval workflow. The framework's interrupt/resume design captures the
run's exact state at the moment the interrupt is raised, stores it in a
checkpoint, and re-fires only the interrupted node when the human supplies
a reply. Other nodes whose outputs were already produced are not re-executed.
This deterministic re-fire contract eliminates the correctness hazards that
arise from naive idempotency assumptions: the node that requested input is
called exactly once with the human's answer, and no other prior computation
is repeated.

## The Shape

All primitives live in `troopai.adk.graphs.interrupt` (except `GraphRunStatus`
which is in `troopai.adk.graphs.result`):

| Type | Description |
|---|---|
| `Interrupt` | Frozen dataclass: `node_id`, `question`, `kind="generic"`, `metadata={}`. Describes what the human must decide. |
| `InterruptException(TroopAIError)` | Raised by a node to signal a pause. Carries `.interrupt: Interrupt`. |
| `GraphResume` | Frozen dataclass: `replies: dict[str, Any]`, `rejected: dict[str, str]`. Carries the human's answers on resume. |
| `GraphRunStatus.INTERRUPTED` | Status value `"interrupted"` — the run is suspended and waiting for input. |
| `GraphState.pending_interrupts` | `dict[str, Interrupt]` keyed by `node_id`; populated while the run is suspended. |
| `GraphRunResult.interrupts` | `tuple[Interrupt, ...]` — the interrupts collected in the most recent superstep. |
| `GraphRunResultStreaming.interrupts` | Same field on the streaming result type; populated after `stream_events()` drains. |

## Requesting Input from a Node

Call `request_human_input(input, question, *, kind="generic", **metadata)`
from inside a callable node body. On the first invocation the helper raises
`InterruptException` (signalling the BSP loop to suspend). On a resumed
invocation the loop injects the human's reply into
`ExecutableInput.metadata["__resume_reply__"]` and the helper returns it.

Presence of the reserved key — not its truthiness — is the signal: a reply
of `None` (an "abstain" answer) is valid and is returned as-is.

```python
from troopai.adk.graphs.interrupt import request_human_input
from troopai.adk.orchestration.executable import ExecutableInput
from typing import Any


def ask_node(inp: ExecutableInput, ctx: Any) -> str:
    """Node that pauses for human approval before proceeding.

    The two-arg (ExecutableInput, context) signature tells the
    CallableExecutable dispatcher to pass the full envelope, giving
    access to inp.metadata where the loop injects the reply on resume.
    """
    reply = request_human_input(inp, "Approve the action?", kind="tool_approval", tool="deploy")
    return f"action approved with reply: {reply}"
```

The `**metadata` kwargs are forwarded verbatim into `Interrupt.metadata`
so consumers of the streaming event or `result.interrupts` can read
kind-specific detail (e.g. `tool_call_id`, `options: [...]`).

## Suspending a Run

Run the graph normally. When a node raises `InterruptException` the BSP
loop records the interrupt on `GraphState.pending_interrupts`, writes a
checkpoint (if a checkpointer is attached), and returns a result with
`status=INTERRUPTED`. No exception propagates to the caller.

```python
import asyncio
from troopai.adk.graphs import Graph
from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
from troopai.adk.graphs.interrupt import GraphResume, request_human_input
from troopai.adk.graphs.result import GraphRunStatus
from troopai.adk.orchestration.executable import ExecutableInput
from troopai.adk.run.runner import Runner
from typing import Any

cp = InMemoryCheckpointer()


def ask_node(inp: ExecutableInput, ctx: Any) -> str:
    reply = request_human_input(inp, "Approve?", kind="tool_approval")
    return f"approved:{reply}"


g = (
    Graph.new("hitl-suspend")
    .node("ask", ask_node)
    .entry("ask")
    .terminal("ask")
    .compile()
)


async def main() -> None:
    result = await Runner.arun_graph(g, "go", hooks=[cp], thread_id="run-1")

    # No exception raised. Status signals the suspension.
    assert result.status == GraphRunStatus.INTERRUPTED

    for iv in result.interrupts:
        # iv.node_id, iv.question, iv.kind, iv.metadata are all available.
        pass

    # State mirrors the same set of pending interrupts.
    assert result.state is not None
    assert "ask" in result.state.pending_interrupts
```

## Resuming a Run

Supply a `GraphResume` to the checkpoint-resume entry-points. Two
equivalent paths are available:

**Functional form:**

```python
second = await Runner.arun_graph_from_checkpoint(
    g,
    checkpointer=cp,
    thread_id="run-1",
    resume=GraphResume(replies={"ask": "the-answer"}),
)
assert second.status == GraphRunStatus.COMPLETED
```

**Profile runner form:**

```python
second = await (
    Runner.configure()
    .graph(g)
    .resume_from(cp, "run-1")
    .arun("go", resume=GraphResume(replies={"ask": "the-answer"}))
)
```

**`replies` vs `rejected`:**

- `replies={"ask": value}` — the human approved; `value` is returned by
  `request_human_input` inside the node.
- `rejected={"ask": "denied by reviewer"}` — the human declined; the
  rejection message string is delivered as the reply value (model-visible,
  analogous to the agent `state.reject(message=...)` idiom).

If a pending interrupt is supplied neither a reply nor a rejection, the
loop re-suspends with `status=INTERRUPTED` and the same
`pending_interrupts` entry persists for the next resume attempt.

## Streaming

Both `Runner.arun_graph_streamed` and `GraphRunner.arun(stream=True)`
emit a `graph.node_interrupt` event when a node interrupts. The event is
emitted **before** the final `graph.end(status=interrupted)` event. The
consumer's `async for` loop exits normally — no exception is raised by the
stream driver.

```python
from troopai.adk.graphs.events import GRAPH_END, NODE_INTERRUPT
from troopai.adk.run.runner import Runner

result = await Runner.arun_graph_streamed(g, "go")

async for ev in result.stream_events():
    if ev["type"] == NODE_INTERRUPT:
        node_id: str = ev["node_id"]
        iv = ev["interrupt"]   # Interrupt instance
        graph_path = ev["graph_path"]  # e.g. ("hitl-suspend",)
    elif ev["type"] == GRAPH_END:
        pass  # ev["status"].value == "interrupted"

# After the stream drains, result fields are populated.
assert result.status.value == "interrupted"
assert len(result.interrupts) == 1
```

### Event payload keys for `graph.node_interrupt`

| Key | Type | Description |
|---|---|---|
| `type` | `str` | Always `"graph.node_interrupt"` |
| `graph_path` | `tuple[str, ...]` | Single-element tuple `(graph_id,)` for top-level runs |
| `node_id` | `str` | Id of the node that raised `InterruptException` |
| `interrupt` | `Interrupt` | The structured interrupt payload |

### Streamed resume

Combine streaming with checkpoint-resume via a profile runner:

```python
from troopai.adk.graphs.interrupt import GraphResume

streamed = await (
    Runner.configure()
    .graph(g)
    .resume_from(cp, "run-1")
    .arun("go", stream=True, resume=GraphResume(replies={"ask": "yes"}))
)

async for ev in streamed.stream_events():
    pass  # drain; the resumed run completes normally

assert streamed.status.value == "completed"
```

## Concurrent Fan-Out

When multiple nodes interrupt in the same superstep, all their `Interrupt`
instances are collected. Non-interrupting sibling nodes that completed in the
same superstep have their outputs recorded in `GraphState.node_results` —
their work is not discarded. The downstream join node does not fire because
the superstep did not complete cleanly; the run suspends.

```python
# Topology: root → (a interrupts  ∥  b completes) → join
#
# After the first run:
#   result.status == INTERRUPTED
#   result.state.pending_interrupts == {"a": <Interrupt>}
#   result.state.node_results contains "b" (its output was recorded)
#   "join" did NOT fire
#
# Resume: supply replies keyed by each interrupting node_id.

resume = GraphResume(replies={"a": "approved"})
second = await Runner.arun_graph_from_checkpoint(
    g,
    checkpointer=cp,
    thread_id="run-fanout",
    resume=resume,
)
```

## Composition

HITL interrupts compose with all three complementary graph subsystems:

- **Checkpointing** — the same `arun_graph_from_checkpoint` / `resume_from`
  surface used for crash-recovery resume also carries `GraphResume`. A
  `SQLiteCheckpointer` makes the suspended state durable across process
  restarts. See `docs/graphs/checkpointing.md`.

- **Reliability** — the per-node retry wrapper never retries an
  `InterruptException`. An interrupt always propagates to the BSP loop
  immediately, regardless of `NodeRetryPolicy.max_attempts` or
  `retry_on`. See `docs/graphs/reliability.md`.

- **Streaming** — the `graph.node_interrupt` event is emitted before
  `graph.end(status=interrupted)` in the streaming driver. The consumer
  drains without raising. Streamed resume works via the profile runner's
  `resume_from(...).arun(stream=True, resume=...)` chain. See
  `docs/graphs/streaming.md`.

## Scope

**Supported:** Callable and agent nodes that explicitly raise
`InterruptException` (typically via `request_human_input`); single and
concurrent (fan-out) interrupts within a superstep; non-streaming and
streaming runs; checkpointed suspension with keyed resume via `replies`
and `rejected`.

**Nested-agent tool approvals:** a tool inside an Agent node that
defers is lifted to a graph-level interrupt via
`NestedAgentInterrupt` — see `docs/graphs/nested-agent-bridge.md`.

## See Also

- `docs/graphs/checkpointing.md` — durable checkpointers, selective re-fire
  contract, cumulative budgets.
- `docs/graphs/reliability.md` — per-node retry/timeout, `InterruptException`
  passthrough guarantee.
- `docs/graphs/streaming.md` — full event taxonomy, `graph.node_interrupt`
  discriminator, streamed resume.
- `docs/graphs/composition.md` — nested graphs, fan-out patterns.
- `docs/graphs/nested-agent-bridge.md` — lifting an `Agent` node's tool
  deferral to a `NestedAgentInterrupt`, `NestedAgentReply` payload, and
  partial-resume semantics.
- `src/troopai/adk/graphs/interrupt.py` — `Interrupt`, `InterruptException`,
  `GraphResume`, `request_human_input`.
- `src/troopai/adk/graphs/events.py` — `NodeInterruptEvent`, `NODE_INTERRUPT`.
- `src/troopai/adk/graphs/result.py` — `GraphRunStatus.INTERRUPTED`,
  `GraphRunResult.interrupts`, `GraphRunResultStreaming.interrupts`.
- `examples/graphs/hitl.py` — runnable end-to-end demonstration.

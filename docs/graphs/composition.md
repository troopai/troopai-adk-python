# Graph Composition

How `Agent`, `Swarm`, `Graph`, and plain callables compose into unified
multi-agent pipelines via the `Executable[TContext]` seam.

## Why Composition Matters

Production agentic systems rarely consist of a single pattern. A real legal-brief
pipeline might need:

1. A **triage agent** that classifies the incoming request (single Agent).
2. A **research swarm** where a researcher and a critic iterate until both agree
   (Swarm with cycles).
3. A **writer agent** that drafts based on research output (single Agent).
4. A **legal review subgraph** that runs a compliance checker and a legal
   approver in sequence (nested Graph).

Pre-composition, each step has to be wired manually — the developer manages state
threading, usage attribution, and error propagation across the four patterns.
With `Graph`, all four are nodes. The graph loop manages the rest.

## The `Executable[TContext]` Seam

Every node in a `Graph` holds an `Executable[TContext]`. This is the single
abstract base class in `src/troopai/adk/orchestration/executable.py` that every
composable primitive plugs into.

```
Executable[TContext]
│
├── abstract invoke(input, context, config) -> NodeResult[TContext]
└── stream_async(input, context, config) -> AsyncIterator[dict]   # default impl
```

The only abstract method is `invoke`. The default `stream_async` calls `invoke`
and yields a single terminal event — callables and other cheap primitives get
streaming for free without each adapter reimplementing an async iterator.

`Graph` itself inherits from `Executable` directly, which is what makes nested
graphs compose without any extra adapter: the outer loop calls
`graph.invoke(input, context, config)` on the inner graph the same way it calls
any other node.

## The Three Adapters

`Agent`, `Swarm`, and plain callables are NOT subclasses of `Executable`. The
"Agent = config" rule forbids adding an `invoke()` method to `Agent` or `Swarm`
— they are immutable configuration, not execution engines. Three thin adapters in
`src/troopai/adk/graphs/adapters.py` bridge the gap.

### `AgentExecutable`

Wraps an `Agent`. Calls `Runner.arun(agent, prompt, context=..., run_config=...)`
and converts the resulting `RunResult` into a `NodeResult`.

```python
from troopai.adk.graphs import AgentExecutable
from troopai.adk.agents.agent import Agent

triage = Agent(name="triage", system_prompt="Classify the request.")

# Explicit construction — use when you want to override max_turns
node_exec = AgentExecutable(agent=triage, max_turns=3)

# The builder does this automatically:
graph.node("triage", triage)  # equivalent to: graph.node("triage", AgentExecutable(agent=triage))
```

`NodeResult.usage` carries the inner `RunContext.usage` delta so the graph can
attribute cost to this node specifically. `NodeResult.output` holds the agent's
`final_output`; `final_text` mirrors it when the output is a string.

### `SwarmExecutable`

Wraps a `Swarm`. Calls `Runner.arun_swarm(swarm, prompt, context=..., run_config=...)`
and converts the resulting `SwarmRunResult` into a `NodeResult`.

```python
from troopai.adk.graphs import SwarmExecutable
from troopai.adk.swarms import Swarm

research_swarm = Swarm(
    members=(researcher, critic),
    entry=researcher,
    termination=ExplicitDoneTermination() | MaxTurnsTermination(10),
)

# Explicit construction
node_exec = SwarmExecutable(swarm=research_swarm)

# The builder does this automatically:
graph.node("research", research_swarm)
```

The full `SwarmRunResult` is preserved on `NodeResult.output`. Downstream edge
predicates can inspect it:

```python
from troopai.adk.swarms.result import SwarmRunResult

def research_succeeded(result):
    swarm_result = result.output
    if isinstance(swarm_result, SwarmRunResult):
        return swarm_result.final_output is not None
    return False

graph.edge("research", "writer", when=research_succeeded)
```

### `CallableExecutable`

Wraps any callable. Zero LLM cost — `NodeResult.usage` is always an empty
`LLMUsage`. Arity is detected at wrap time via `inspect.signature`:

| Callable signature | What the adapter passes |
|--------------------|------------------------|
| `() -> Any` | Nothing (pure producer). |
| `(text: str) -> Any` | Best-effort text extracted from upstream content. |
| `(text: str, context: RunContext) -> Any` | Text + the shared `RunContext`. |
| `(input: ExecutableInput, context: RunContext) -> Any` | Full input envelope + context. |

The heuristic for detecting the full-input variant: arity == 2 AND the first
parameter is annotated as `ExecutableInput` (or the string `"ExecutableInput"`).

```python
from troopai.adk.orchestration.executable import ExecutableInput
from troopai.adk.run.context import RunContext

# 0-arg: pure producer
graph.node("timestamp", lambda: "2025-04-18T00:00:00Z")

# 1-arg: text transformer
graph.node("upper", lambda text: text.upper())

# 2-arg with text: text + context
def log_and_pass(text: str, ctx: RunContext) -> str:
    logger.info("Processing for context: %s tokens used", ctx.usage.total_tokens)
    return text

graph.node("audited-step", log_and_pass)

# 2-arg with ExecutableInput: full control
def route_by_label(inp: ExecutableInput, ctx: RunContext) -> str:
    if inp.edge_label == "high-priority":
        return "FAST"
    return "SLOW"

graph.node("router", route_by_label)
```

Sync and async callables are both supported. The adapter awaits if the return
value is an awaitable.

### `to_executable(obj)` — Auto-dispatch

`to_executable` is the function `GraphBuilder.node()` calls internally. It
dispatches on type:

```
Executable  → returned as-is (including Graph, nested)
Agent       → AgentExecutable wrapper
Swarm       → SwarmExecutable wrapper
callable    → CallableExecutable wrapper
else        → TypeError
```

The dispatch order matters: `Agent` and `Swarm` are checked before `callable`
because both are technically callable in Python.

## Case Study: Agent + Swarm + Subgraph Pipeline

The following example is derived directly from the architecture plan. It
composes four node types in one pipeline:

```python
import asyncio
import logging

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs import Graph, Merge
from troopai.adk.run.runner import Runner
from troopai.adk.swarms import (
    Swarm,
    LLMHandoffPolicy,
    ExplicitDoneTermination,
    MaxTurnsTermination,
)

logger = logging.getLogger(__name__)

# ── 1. Individual agents ────────────────────────────────────────────
triage_agent = Agent(
    name="triage",
    system_prompt=(
        "You classify incoming legal requests. "
        "Output: 'brief', 'memo', or 'unknown'."
    ),
)

researcher = Agent(
    name="researcher",
    system_prompt="You research case law and precedents thoroughly.",
)

critic = Agent(
    name="critic",
    system_prompt=(
        "You critically evaluate research quality. "
        "Call swarm_done when satisfied."
    ),
)

writer_agent = Agent(
    name="writer",
    system_prompt="You draft legal documents from research notes.",
)

compliance_agent = Agent(
    name="compliance",
    system_prompt="You review documents for regulatory compliance.",
)

legal_agent = Agent(
    name="legal-approver",
    system_prompt="You give final legal sign-off. Output 'APPROVED' or 'REJECTED'.",
)

# ── 2. A Swarm for iterative research ───────────────────────────────
research_swarm = Swarm(
    members=(researcher, critic),
    entry=researcher,
    policy=LLMHandoffPolicy(),
    termination=ExplicitDoneTermination() | MaxTurnsTermination(10),
)

# ── 3. A nested subgraph for legal review ───────────────────────────
legal_subgraph = (
    Graph.new("legal-review", description="Compliance check + legal approval")
    .node("checker", compliance_agent)
    .node("approver", legal_agent)
    .pipe("checker", "approver")
    .entry("checker")
    .terminal("approver")
    .compile()
)

# ── 4. Top-level pipeline ────────────────────────────────────────────
#       triage → research (Swarm) → writer → legal (nested Graph)
pipeline = (
    Graph.new(
        "legal-brief-pipeline",
        description="Research → draft → legal sign-off",
    )
    .node("triage", triage_agent)           # Agent → AgentExecutable
    .node("research", research_swarm)       # Swarm → SwarmExecutable
    .node("writer", writer_agent)
    .node("legal", legal_subgraph)          # Graph → Executable (no adapter)
    .pipe("triage", "research", "writer", "legal")
    .entry("triage")
    .terminal("legal")
    .compile()
)

# ── 5. Run ───────────────────────────────────────────────────────────
async def main() -> None:
    result = await Runner.arun_graph(pipeline, "Draft a legal brief on X.")

    logger.info("Status: %s", result.status)
    logger.info("Final output: %s", result.final_output)
    logger.info("Total supersteps: %d", result.total_supersteps)

    # Per-node cost attribution — neither LangGraph nor Strands surfaces this
    for node_id, usage in result.per_node_usage.items():
        logger.info("  %s: %d tokens", node_id, usage.total_tokens)

    logger.info("Graph-wide total: %d tokens", result.cumulative_usage.total_tokens)

asyncio.run(main())
```

Key observations from this example:

- `triage_agent`, `research_swarm`, `writer_agent`, and `legal_subgraph` are
  created independently and don't know about each other. The graph is the only
  place that wires them.
- `.pipe("triage", "research", "writer", "legal")` is four nodes chained with
  three unconditional edges — equivalent to three `.edge()` calls.
- The `legal_subgraph` nested graph runs its own BSP superstep loop when the
  outer loop invokes it. The outer loop treats it as a black box.
- `per_node_usage` will contain four entries (`triage`, `research`, `writer`,
  `legal`), each with the tokens consumed by that node's run. Callable nodes
  would show zero usage.

## Nested Streaming and `graph_path`

Every `GraphStreamEvent` carries `graph_path: tuple[str, ...]` identifying
the graph that emitted the event. For a top-level run every event carries
a single-element tuple: `graph_path=(graph.id,)`.

A nested `Graph` used as a node runs non-streaming via `Graph.invoke()`.
The outer stream does not receive the inner graph's own structural events
(`graph.start`, `graph.superstep_start`, and so on). What the outer
consumer observes for that node is the same pair of structural boundaries
produced for any other node type: a `graph.node_start` event when the node
begins and a `graph.node_end` event when it completes, with the inner
graph's terminal result on `graph.node_end["result"]`. Every event in the
outer stream carries `graph_path=(outer_graph.id,)`.

Interior token-level streaming is an agent-node behavior — see the
`AgentExecutable` section above and the per-node-type contract in
`docs/graphs/streaming.md`.

## Per-Node Usage Attribution Across Nested Boundaries

Usage attribution flows upward through nested boundaries:

1. An inner `AgentExecutable.invoke` call returns `NodeResult(usage=inner_delta)`.
2. The outer graph loop calls `state.record(node_id, result)`, which adds
   `result.usage` to `state.per_node_usage[node_id]`.
3. When a nested `Graph` is the node, `Graph.invoke` returns a `NodeResult`
   whose `usage` is the inner graph's `cumulative_usage`.
4. The outer `per_node_usage["legal"]` therefore holds the sum of all token
   consumption inside the `legal-review` subgraph.

For top-level per-node breakdown of a nested graph's internals, inspect
`result.node_results["legal"].metadata["per_node_usage"]` — the inner
`GraphRunResult.per_node_usage` dict is preserved there.

```python
inner_usage = result.node_results["legal"].metadata.get("per_node_usage", {})
for inner_node, inner_tokens in inner_usage.items():
    logger.info("  legal.%s: %s", inner_node, inner_tokens)
```

## Limitations

### Asymmetric Composition

Composition is asymmetric: a `Graph` can contain a `Swarm` node, but a
`Swarm` cannot contain a `Graph` as one of its members. Swarm routing works by
injecting `transfer_to_<name>` LLM tools at dispatch time — this requires a
list of `Agent` members, not an `Executable` list.

Symmetric composition (Swarm-of-Graphs, where `SwarmPolicy` accepts
`Executable` members) is not currently supported. It would require refactoring
`SwarmPolicy` and the tool-injection dispatch site in `run/swarm_loop.py`.

The primary use case — a `Graph` that contains a `Swarm` node — is fully
supported.

### Swarm Interior Streaming

Graph streaming surfaces structural events and agent-node interior token
events. A `Swarm` node sitting inside a graph emits structural
`graph.node_start` and `graph.node_end` boundaries at the graph level;
interior swarm-step events are out of scope for the graph streaming surface
and are available via dedicated swarm observability hooks. See
`docs/graphs/streaming.md` for the per-node-type streaming contract.

### HITL Interrupt/Resume

Graph-level HITL interrupts are fully supported. A node calls
`request_human_input(inp, question, *, kind, **metadata)` to pause the run;
the BSP loop records the `Interrupt` on `GraphState.pending_interrupts` and
returns `status=INTERRUPTED`. Resume supplies human replies via
`GraphResume(replies={node_id: value})` through `arun_graph_from_checkpoint`
or the profile runner's `resume_from` path. Concurrent fan-out interrupts (multiple
nodes pausing in one superstep) are also collected. See `docs/graphs/hitl.md`
for the full surface.

A nested `Agent` node whose tool defers for approval is lifted to a
graph-level interrupt via `NestedAgentInterrupt`. The bridge embeds the
sub-agent's mid-run `RunState` in the graph checkpoint and re-injects it
into the agent loop on resume — see `docs/graphs/nested-agent-bridge.md`
for the public surface (`NestedAgentApproval`, `NestedAgentRejection`,
`NestedAgentReply`), partial-resume semantics, and current limitations
(streaming forwarding of resumed agent events, depth-2 inner-graph
deferrals).

A nested `Graph` node whose inner graph suspends on a **plain** `Interrupt`
(raised by `request_human_input`, not a sub-agent tool approval) is lifted as
a `NestedGraphInterrupt` — a distinct kind carrying no `agent_name`. The
distinction matters across a checkpoint: `GraphState.from_dict` rehydrates a
`NestedGraphInterrupt` without the non-empty-`agent_name` guard a tool
approval requires, and resume forwards a plain reply value
(`GraphResume.replies[node_id]`) verbatim into the inner graph.

### Self-Loops Not Allowed

`GraphEdge` rejects edges where `source == target`. Genuine cyclical behaviour
should be modelled as a `Swarm` (which is designed for cycles) or by routing
through an intermediate node.

### Per-node reliability

`NodeRetryPolicy` (graph-level `GraphConfig.default_retry` or per-node
`GraphNode.retry`) and `GraphNode.timeout` / `GraphConfig.per_node_timeout`
are enforced by the graph loop: each node firing is retried with exponential
backoff and `retry_on` filtering, and bounded by a per-attempt timeout. Both
are opt-in and default-off. See `docs/graphs/reliability.md` for the full
contract (failure-boundary order, exceptions, parity guarantee).

See `docs/graphs/graphs.md` for the full API reference.
See `examples/graphs/` for runnable examples.

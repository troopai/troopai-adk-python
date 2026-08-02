(architecture/graphs)=

# 🕸️ Graphs

The third multi-agent axis: state-machine orchestration with explicit
nodes, edges, checkpointers, and HITL.

## Shape

```{mermaid}
stateDiagram-v2
  [*] --> Triage
  Triage --> Search: needs_search
  Triage --> Answer: has_answer
  Search --> Synthesize
  Synthesize --> Answer
  Answer --> HITL: requires_review
  HITL --> Answer: approved
  HITL --> Revise: rejected
  Revise --> Synthesize
  Answer --> [*]
```

A graph is a directed multigraph of `GraphNode`s. What the node *does*
is bound to it — an agent run, a deterministic Python function, a
human-in-the-loop pause, a branching predicate. The Runner walks the
nodes; transitions resolve against the graph's edges.

## Runner integration

`Runner.arun_graph(graph, initial_state)` executes the state machine
to completion (or to the next HITL pause). `Runner.arun_graph_streamed(...)`
emits a `GraphStreamEvent` per transition. Stream-event variants
include `GraphStartEvent`, `SuperstepStartEvent`, `NodeStartEvent`,
`NodeEndEvent`, `NodeErrorEvent`, `NodeStreamEvent`,
`NodeInterruptEvent`, `SuperstepEndEvent`, `GraphEndEvent`.

## Checkpointers

Graphs persist their state via the `Checkpointer` Protocol. Backends
shipped today, all under `src/troopai/adk/graphs/checkpointers/`:

| Backend       | Module          | Use for                              |
| ------------- | --------------- | ------------------------------------ |
| In-memory     | `in_memory.py`  | Tests, single-process dev.           |
| SQLite        | `sqlite.py`     | Single-machine durability.           |
| Postgres      | `postgres.py`   | Horizontal scaling.                  |
| Redis         | `redis.py`      | Hot tier of a tiered composite.      |
| S3            | `s3.py`         | Cold tier of a tiered composite.     |
| Tiered        | `tiered.py`     | Composite hot/warm/cold (Redis → Postgres → S3). |

All Postgres / Redis / S3 backends use optimistic locking with a
`CheckpointConflictError` (in `troopai.adk.exceptions`) on contention.

## HITL (Human-In-The-Loop)

When a HITL node calls `request_human_input(...)`, the loop raises
`InterruptException` carrying an `Interrupt` payload. The Runner:

1. Persists the graph state via the checkpointer.
2. Emits a `NodeInterruptEvent` on the streaming channel.
3. Returns control to the caller with the interrupt payload.

The caller resolves the interrupt by calling
`Runner.arun_graph_from_checkpoint(checkpointer, thread_id, input)`.
The graph resumes from the persisted state.

## Resume semantics

- **Deep resume**: resume reconstructs every nested agent's loop state,
  not just the top-level node position.
- **Streaming resume**: a streamed graph resumes its event stream from
  the next transition; events already emitted are not re-emitted.
- **Idempotency**: the same interrupt may be resolved at most once;
  subsequent resolves error rather than re-running.

## OpenTelemetry

Every graph transition emits a span. The graph emits a root span
(`graph.run`) and per-node child spans. See
[Governance](governance.md) for cross-cutting observability.

## When to use a graph (vs. handoff / swarm)

| Need                                       | Pick     |
| ------------------------------------------ | -------- |
| Long workflow with branching + HITL        | Graph    |
| Pure routing, no human gate                | Handoff  |
| Iterate-and-refine cycle                   | Swarm    |
| One agent calling tools and returning      | Single agent |

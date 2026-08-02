# Graphs Module

Composable multi-agent orchestration primitive. A `Graph` is a directed graph
of nodes executed under BSP (Bulk Synchronous Parallel) supersteps. Each node
hosts an `Agent`, a `Swarm`, another `Graph`, or a plain Python callable — all
uniformly, via the `Executable[TContext]` seam in `orchestration/executable.py`.

## Files

- `graph.py` — `Graph` frozen dataclass (implements `Executable`); `Graph.new(id, *, description, metadata)` entry point
- `builder.py` — `GraphBuilder` fluent API: `.node() .edge() .pipe() .entry() .terminal() .with_config() .with_hooks() .compile()`
- `config.py` — `GraphConfig` (budgets + knobs), `NodeInputStrategy` enum, `NodeRetryPolicy` (per-node retry config; enforced by `run/node_reliability.py`)
- `node.py` — `GraphNode` (carries `retry: NodeRetryPolicy | None` and `timeout: float | None` per-node overrides), `GraphEdge`, `EdgeCondition` type alias
- `state.py` — `GraphState` (per-run mutable; `to_json`/`from_json` plain JSON, no version field; `produced_at` + `versions_seen` together drive selective re-fire on checkpoint resume)
- `result.py` — `GraphRunResult`, `GraphRunResultStreaming`, `GraphRunStatus` enum
- `merge.py` — `Merge` namespace: `last_wins`, `concat_text` (default), `extend_items`, `first_wins`, `custom(fn)`
- `join.py` — `JoinSemantics` (`AND` default, `OR`), `JoinBarrier` (NamedBarrierValue-inspired)
- `hooks.py` — `GraphHooks[TContext]` lifecycle callbacks, `HookProvider` Protocol, `HookRegistry`
- `events.py` — `GraphStreamEvent(dict)` and eight typed subclasses; discriminator constants
- `checkpointer.py` — `Checkpointer` Protocol (extends `HookProvider`), `GraphCheckpoint` (no version field)
- `checkpointers/in_memory.py` — `InMemoryCheckpointer` (default; dict-backed, adequate for tests and single-process runs)
- `checkpointers/sqlite.py` — `SQLiteCheckpointer` (durable `aiosqlite`-backed store for multi-process or crash-recoverable runs)
- `checkpointers/hooks.py` — `CheckpointerHooks` bridge (shared `GraphHooks` subclass used by both checkpointer implementations)
- `adapters.py` — `AgentExecutable`, `SwarmExecutable`, `CallableExecutable`, `to_executable()`
- `node_input.py` — `prepare_node_input` (mirrors `swarms/shared_context.py`)
- `interrupt.py` — `Interrupt`, `InterruptException`, `GraphResume`, `request_human_input` (HITL primitives; BSP loop wired; re-fire-with-reply contract: on resume the loop injects the human's value via `ExecutableInput.metadata["__resume_reply__"]` and re-fires only the interrupted node)

## Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| **`Executable[TContext]` as composition seam** | Single ABC; thin adapters preserve "Agent = config" rule without adding `invoke()` to `Agent`/`Swarm` |
| **`Graph` implements `Executable` directly** | Nested graphs compose without an intermediate adapter — `GraphNode` holds a `Graph` and the outer loop calls `graph.invoke()` identically to any other node |
| **BSP supersteps via `asyncio.wait(FIRST_COMPLETED)`** | Fail-fast errors cancel sibling tasks without waiting for slow siblings; happy-path degrades to gather semantics |
| **AND-join default, per-node OR override** | AND-join is the safer default — Strands' OR-default silently ran nodes with partial inputs; here, per-node `join=JoinSemantics.OR` opts in explicitly |
| **`JoinBarrier` owns the ready decision** | Barrier holds `expected` + `arrivals`; graph loop only asks `barrier.is_ready()`. No engine-level edge counting |
| **Single routing mechanism** | `edge(from, to, when=predicate)` only — no `Command(goto=)` duality (LangGraph). Every routing decision is one `edge()` call |
| **`Merge` on receiving node** | Fan-in strategy (`concat_text` default, `last_wins`, `extend_items`, `custom(fn)`) is declared at the receiving node — no `Annotated[list, add_messages]` magic |
| **Path-sorted write application** | After a superstep, results are applied sorted by node id — deterministic reducer order regardless of `asyncio.Task` completion order (mirrors LangGraph `_algo.py`) |
| **`TypedEvent(dict)` pattern** | Stream events subclass `dict`; zero `.model_dump()` overhead on token hot paths (Strands insight) |
| **`Checkpointer` as `HookProvider`** | `Checkpointer.register()` subscribes to `on_node_end`/`on_graph_end`; the graph loop contains ZERO persistence code — swappable without touching the driver |
| **`graph_path: tuple[str, ...]` on events** | Every `GraphStreamEvent` carries the emitting graph's id as a single-element tuple `(graph.id,)`; it is never extended. Nested `Graph` nodes run non-streaming via `Graph.invoke()` and surface only structural boundaries, so events do not carry combined outer→inner paths |
| **State: message-threading** | Each node receives merged `list[LLMInputContentItem]` from upstream — no shared state dict; a `GraphChannels` opt-in is not yet implemented |
| **Per-node usage attribution** | `NodeResult.usage` → `GraphState.per_node_usage` → `GraphRunResult.per_node_usage`; neither LangGraph nor Strands surfaces per-node cost attribution on the result type |
| **Checkpoint/resume with selective re-fire** | `_seed_barriers_from_checkpoint` in `run/graph_loop.py` reconstructs `JoinBarrier`s from the restored state; a node re-fires only when its `produced_at` superstep is older than the consuming node's `versions_seen` entry — the same LangGraph Pregel channel-version trick. `Runner.arun_graph_from_checkpoint`/`run_graph_from_checkpoint` and `Runner.configure().graph(graph).resume_from(...)` are the public resume surface. `SQLiteCheckpointer` provides durable crash-recoverable persistence. Cumulative budgets (`max_supersteps`, `max_total_tokens`) continue counting from the checkpoint; they are not reset on resume. |
| **Snapshot-restore HITL interrupt (re-fire-with-reply contract)** | `request_human_input(inp, question, *, kind, **metadata)` raises `InterruptException`; the BSP loop captures it onto `GraphState.pending_interrupts`, checkpoints, and returns `status=INTERRUPTED`. On resume via `GraphResume(replies={node_id: value})`, the loop injects the reply into `ExecutableInput.metadata["__resume_reply__"]` and re-fires only the interrupted node — no prior computation repeats. Key-presence (not truthiness) determines reply availability, so `None` is a valid reply. `InterruptException` is never retried by the reliability wrapper. Streaming: `graph.node_interrupt` event emitted before `graph.end(status=interrupted)`. |
| **Graph streaming** | Twin BSP driver `run_graph_loop_streamed` shares body with `run_graph_loop` via an emit seam. `GraphRunResultStreaming.stream_events()` yields `GraphStreamEvent` instances; `.cancel("immediate"/"after_superstep")` aborts the run. `AgentExecutable.stream_async` forwards interior agent events as `NodeStreamEvent` (`graph_path`/`node_id`/`inner`) envelopes. Callable, swarm, and nested-graph nodes emit structural boundaries (`graph.node_start`/`graph.node_end`) and contribute their terminal result only. |
| **Advisory / not-yet-enforced fields** | `GraphRunStatus` values `MAX_SUPERSTEPS`/`MAX_TOKENS` (enum members exist; statuses emitted by loop). |
| **Per-node reliability** | `run/node_reliability.py` (`resolve_node_reliability`, `run_node_with_reliability`) applies per-attempt `asyncio.timeout` and bounded retry with exponential backoff to every node invocation in `_invoke_node`. `GraphNode.retry`/`.timeout` override `GraphConfig.default_retry`/`per_node_timeout` when set. Final-attempt timeout → `GraphNodeTimeoutError`; exhausted retryable multi-attempt → `NodeRetriesExhaustedError`; no-policy path re-raises original unchanged (parity guarantee). |

## Integration Seams (read-only pointers)

- `run/graph_loop.py` — `run_graph_loop` BSP driver; `run_graph_loop_streamed` streaming twin; `_invoke_node` wires reliability; `_seed_barriers_from_checkpoint` selective re-fire on resume
- `run/node_reliability.py` — `resolve_node_reliability`, `run_node_with_reliability` (per-attempt timeout + retry wrapper)
- `run/runner.py` — `Runner.arun_graph`, `Runner.run_graph`, `Runner.configure`, `Runner.arun_graph_streamed`, `Runner.arun_graph_from_checkpoint`, `Runner.run_graph_from_checkpoint`
- `run/profile.py` — `RunnerProfile` and `GraphRunner` (including `resume_from`, `arun(stream=True)`)
- `orchestration/executable.py` — `Executable` ABC, `ExecutableInput`, `NodeResult`

## OR-exit Terminal Semantics

The loop exits `COMPLETED` as soon as **any** terminal fires. Conditional graphs
with mutually exclusive terminal branches exit after the winning branch fires.
An AND-exit `GraphConfig.require_all_terminals` is not yet implemented.

## Cost Levers

1. `GraphConfig.max_supersteps` — superstep cap (default 50)
2. `GraphConfig.max_total_tokens` — graph-wide cumulative token cap
3. `GraphConfig.node_input` (`NodeInputStrategy`) — default context window per node (`LAST_OUTPUT` cheapest)
4. `GraphConfig.default_retry` / `GraphNode.retry` — opt-in retry (default `max_attempts=1`, no retries); `GraphConfig.per_node_timeout` / `GraphNode.timeout` — opt-in per-attempt timeout (default `None`); both default-off
5. `FunctionTool.max_result_tokens` / `HandoffConfig.budget` — propagate from inner Agent/Swarm nodes unchanged
6. `GraphRunResult.per_node_usage` — per-node attribution for post-run cost analysis

See `docs/graphs/graphs.md` for usage. See `docs/graphs/checkpointing.md` for the
checkpoint/resume contract (SQLite durability, selective re-fire, cumulative budgets).
See `docs/graphs/reliability.md` for the per-node timeout/retry contract and exception types.
See `docs/graphs/streaming.md` for the streaming event contract, cancellation, and per-node-type
streaming behaviour. See `docs/graphs/composition.md` for nested-graph patterns.
See `docs/graphs/hitl.md` for the HITL interrupt/resume surface.
See `examples/graphs/` for runnable code.
Topology diagrams: `Graph.to_mermaid()` / `Graph.to_dot()` — see
`docs/visualization/visualization.md`.

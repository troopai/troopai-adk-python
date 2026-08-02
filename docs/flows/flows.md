# Flows

Decorator-driven multi-step orchestration over typed shared state.
A `Flow` composes Agent / Swarm / Graph / Task calls as ordered
steps, wires them with `@flow_start` / `@flow_listen` / `@flow_router` decorators,
and carries a developer-owned typed state object across the run.

## When to Use Flow

| Pattern | Best primitive |
|---|---|
| One-shot transfer of control (agent A → agent B, no return) | `Handoff` |
| Delegate-and-resume (A asks B a question, continues with answer) | `Agent.as_tool()` |
| Iterative cyclic collaboration (A ↔ B ↔ C until done) | `Swarm` |
| Explicit DAG with explicit edges + message-threading | `Graph` |
| Linear sequence of tasks with conditional skip | `TaskPipeline` |
| Parallel fan-out with no ordering | `TaskGroup` |
| **Declarative event-driven workflow over typed state with `@flow_listen` / `@flow_router`** | **`Flow`** |

## Anti-Hidden-Behavior Contract

TroopAI Flow deliberately rejects every hidden behavior CrewAI Flow
introduces. The contract is enforced at class-definition time where
structurally possible:

| CrewAI does this | TroopAI Flow does NOT |
|---|---|
| `inspect.signature(method)` injects previous step's return value | Step methods take ONLY `self`. Return values are dropped (except `@flow_router`). |
| Auto-instantiates state from `Flow[StateT]` generic | Requires explicit `initial_state=` or `state_factory` class attribute. |
| `@persist` decorator auto-writes state after every step | Persistence is explicit: developer calls `FlowCheckpoint.to_json()`. |
| Bare `str` return from non-router → next step's name | Routing only via `@flow_router`-decorated methods. |
| Source-code introspection of router returns | Routers dispatch on the literal returned string. |
| `train()` / `test()` / `replay()` modes | One execution path. |
| `kickoff_for_each(inputs)` | Developer loops `await Runner.arun_flow(...)` themselves. |

## Quick Start

```python
import asyncio
from pydantic import BaseModel
from troopai.adk import Flow, Runner, flow_start, flow_listen, flow_router

class ResearchState(BaseModel):
    topic: str = ""
    research: str = ""

class ResearchFlow(Flow[ResearchState]):
    state_factory = ResearchState  # explicit, never inferred from generic

    @flow_start
    async def kickoff(self) -> None:
        self.state.topic = "decentralized AI"

    @flow_listen(kickoff)
    async def research(self) -> None:
        # Build prompt from self.state and call an Agent inline:
        # result = await Runner.arun(agent, f"Research {self.state.topic}")
        # self.state.research = str(result.final_output)
        self.state.research = "synthetic finding"

    @flow_router(research)
    async def classify(self) -> str:
        return "high_risk" if "danger" in self.state.research else "low_risk"

    @flow_listen("high_risk")
    async def escalate(self) -> None: ...

    @flow_listen("low_risk")
    async def approve(self) -> None: ...

flow = ResearchFlow()
result = asyncio.run(Runner.arun_flow(flow))
print(result.final_state.research, result.status)
```

## Operator-Based Combinators

Combinators are constructed **operator-only** — there are no
`or_()` / `and_()` helper functions. Use the `|` and `&` operators
on `FlowStep` instances (the wrapped form of decorated methods):

```python
class MyFlow(Flow[State]):
    state_factory = State

    @flow_start
    async def a(self) -> None: ...

    @flow_start
    async def b(self) -> None: ...

    # OR gate — fires ONCE on first arrival; gate is consumed.
    @flow_listen(a | b)
    async def either(self) -> None: ...

    # AND gate — fires ONCE after BOTH complete.
    @flow_listen(a & b)
    async def both(self) -> None: ...

    # Chaining flattens left-associatively:
    @flow_listen(a | b | "external_label")
    async def any_of_three(self) -> None: ...
```

Mixed-type chains (`Or & something` or `And | something`) raise
`TypeError` to keep operator-chain semantics unambiguous. For complex
shapes, construct `Or(...)` / `And(...)` directly with the dataclass
constructor.

## State

State is a developer-controlled typed object. The framework NEVER
auto-instantiates it from the generic parameter. Two construction
paths, both explicit:

```python
class State(BaseModel):
    counter: int = 0

# Option 1: state_factory class attribute (works without args)
class A(Flow[State]):
    state_factory = State
    @flow_start
    async def kickoff(self): ...

flow_a = A()  # state_factory() builds a fresh State()

# Option 2: explicit initial_state
class B(Flow[State]):
    @flow_start
    async def kickoff(self): ...

flow_b = B(initial_state=State(counter=42))
```

Both Pydantic `BaseModel` and `@dataclass` states work. The framework
only needs the state to be JSON-serializable if you want to use
`FlowCheckpoint`.

When both paths are present, an explicit `initial_state=` argument
always wins over the class-level `state_factory`. A `state_factory`
that is not callable raises `FlowDefinitionError` at construction.

## Inspecting a Flow

`Flow` and `FlowRunResult` have human-readable one-line reprs that
never dump prompts or full state:

```python
>>> flow
ResearchFlow(flow_id='flow-a1b2c3d4', steps=3, routers=1, state=ResearchState)
>>> result
FlowRunResult(flow_id='flow-a1b2c3d4', status='completed', steps=3, final_state=...)
```

Decorated methods expose their wiring through public read-only
properties — no dunder access needed:

```python
>>> ResearchFlow.research.role       # "start" | "listen" | "router"
'listen'
>>> ResearchFlow.research.triggers   # trigger names declared by the decorator
('kickoff',)
```

## Execution

```python
# Async
result = await Runner.arun_flow(flow, config=FlowConfig(max_steps=200))

# Sync wrapper
result = Runner.run_flow(flow, config=FlowConfig(max_steps=200))

# Streaming events
streaming = Runner.arun_flow_streamed(flow)
async for event in streaming.stream_events():
    print(event.type, getattr(event, "step_name", ""))
print(streaming.status, streaming.final_state)

# Resume from checkpoint
result = await Runner.arun_flow_from_checkpoint(flow, checkpoint)
```

### Cost / Safety Bounds

| `FlowConfig` field | Default | Purpose |
|---|---|---|
| `max_steps` | `100` | Hard cap on total step invocations |
| `max_total_tokens` | `None` (no cap) | Optional cumulative token cap |
| `max_listeners_per_step` | `20` | Fan-out cap per trigger — enforced at table-build time |
| `error_policy` | `"halt"` | `"halt"` or `"route_to_error_handler"` |

All defaults are cost-conservative. Raise via explicit `FlowConfig(...)`.

With `error_policy="route_to_error_handler"`, the executor fires
listeners declared on the error route — write
`@flow_listen(FLOW_ERROR_TRIGGER)` (the exported constant for the
`"__error__"` route literal, importable from `troopai.adk` or
`troopai.adk.flows`) instead of spelling the magic string yourself.
Without such a listener the policy falls back to `"halt"` semantics.

## Streaming Events

Flow events are step-granularity (NOT token-granularity). Inner
agent-run token events from `Runner.arun_streamed(...)` calls inside
step bodies are NOT forwarded through the flow stream by default —
that would require the framework to subscribe step bodies to a hidden
event sink, outside the developer's declared opt-in. Developers who
want token-level streaming consume the inner runner's events directly.

| Event | When |
|---|---|
| `flow.start` | Once at run start |
| `flow.step_start` | Just before each step body invokes |
| `flow.step_end` | After each non-router step completes |
| `flow.route_evaluated` | After a router returns a label |
| `flow.step_error` | Step raised (both halt + route policies emit this) |
| `flow.end` | Once at run completion |

## Persistence (Checkpoint + Resume)

```python
from troopai.adk import FlowCheckpoint

# Capture (typically from a hook or after partial run)
checkpoint = FlowCheckpoint(
    flow_id=flow.flow_id,
    completed_steps=("step1", "step2"),
    pending_steps=("step3",),
    and_gate_arrivals={},
    consumed_gates=(),
    state_data=flow.state.model_dump_json(),
)
blob = checkpoint.to_json()  # store anywhere — DB, file, queue

# Resume in another process:
loaded = FlowCheckpoint.from_json(blob)  # raises on schema mismatch
state_back = MyState.model_validate_json(loaded.state_data)
flow_resumed = MyFlow(initial_state=state_back)
result = await Runner.arun_flow_from_checkpoint(flow_resumed, loaded)
```

The Flow class definition itself is NOT serialized — the resuming
side reconstructs the same class. Same contract as `TaskPipelineState`.

## Step-level HITL (`requires_approval`)

Per-step `requires_approval` gates pause a Flow before a sensitive
step runs and emit a `FlowCheckpoint` carrying the deferred steps.
The contract mirrors the function-tool HITL surface exactly:
decisions are recorded on the checkpoint itself via
`checkpoint.approve(...)` / `checkpoint.reject(...)` — the same shape
as `RunState.approve(...)` / `RunState.reject(...)` for tool HITL —
and resume goes through `Runner.arun_flow_from_checkpoint(flow,
checkpoint)`. There is **no live-inject channel** on the streaming
result; the stream ends on the deferred event, the decision is
recorded off-stream, and a new run resumes from the checkpoint.

```python
from troopai.adk.flows import Flow, FlowStepContext, flow_listen, flow_start

def needs_review(ctx: FlowStepContext[CartState]) -> bool:
    return ctx.flow_state.amount >= 1000

class RefundFlow(Flow[CartState]):
    @flow_start
    async def intake(self) -> None: ...

    @flow_listen("intake", requires_approval=needs_review)
    async def refund(self) -> None: ...

result = await Runner.arun_flow(flow)
if result.requires_action:
    checkpoint = result.checkpoint
    # Approve — by step name (or pass the FlowDeferredStep object itself):
    checkpoint.approve(
        "refund",
        approver_id="alice@example.com",
        approver_role="ops_lead",
        reason="amount within manual-review band",
    )
    # Or reject (message is routed to the @flow_listen(FLOW_ERROR_TRIGGER) handler):
    checkpoint.reject("refund", "fraud signal exceeded threshold")
    resumed = await Runner.arun_flow_from_checkpoint(flow_again, checkpoint)
```

Passing an unknown step name raises `ValueError` listing the pending
step names — decisions can only target pending deferrals.

### Typed primitives

The HITL surface uses typed dataclasses, not weak strings / dicts,
so callers can branch on provenance without parsing:

| Type | Purpose |
|---|---|
| `FlowDeferredStep` | One deferred step (with `kind`, `triggers`, `policy`, `deadline`, `metadata`). |
| `FlowApprovalDecision` | One decision: `approved`, routed `message`, audit (`approver_id`, `approver_role`, `reason`), `decision_time`. Derived `.status` ∈ `{"approved","rejected","expired"}`. |
| `FlowApprovalPolicy` | Declarative quorum / role-restricted / SLA semantics. |
| `FlowDeferralKind` | `"approval"` (HITL) or `"external_execution"` (run elsewhere). |

Attach a policy declaratively on the decorator via `approval_policy=`;
it rides on `FlowDeferredStep.policy` through the checkpoint so the
out-of-band approval driver (UI, Slack bot, approval service) can
enforce it:

```python
from troopai.adk.flows import FlowApprovalPolicy

class RefundFlow(Flow[CartState]):
    @flow_listen(
        "intake",
        requires_approval=needs_review,
        approval_policy=FlowApprovalPolicy(quorum=2, allowed_roles=("ops_lead",)),
    )
    async def refund(self) -> None: ...
```

The executor does NOT evaluate quorum / roles / deadline itself — the
policy is an informational primitive; the driver enforces it when
constructing the resume decision. `approval_policy` defaults to `None`
(bare single-approver case).

### Other step-level controls

Three companion attributes on every decorator (`@flow_start` /
`@flow_listen` / `@flow_router`) cover the rest of the
function-tool-style configuration surface:

| Attribute | Behaviour |
|---|---|
| `enabled` (bool / callable → bool) | When `False`, step body is silently skipped and successors are NOT dispatched. Emits `FlowStepSkippedEvent`. |
| `max_retries` (int / None) | On a body exception, retry up to N extra times before `FlowConfig.error_policy` engages. Cancellation-class exceptions and internal control-flow signals (HITL deferral, enablement skip, guardrail/rejection) never retry — a deferral is not a failure. |
| `timeout` (float / None) | Wraps the body in `asyncio.wait_for(...)`. Timeouts route through `error_policy` like any other exception. |

All four attributes default to "no change vs. today" — opting in adds
cost; the framework never adds a deferral, retry, or timeout the
developer didn't request.

## Step-Level Governance (`rate_limit` / `guardrails` / `cache`)

Three Tier-2 polymorphic-config attributes mirror their
`FunctionTool` analogues at the Flow step layer. Each is opt-in
and defaults to `None`:

| Attribute | Type | Purpose |
|---|---|---|
| `rate_limit` | `FlowStepRateLimit \| None` | Per-step 60-second sliding-window cap. `rpm` is the throughput ceiling; `behavior` is `"wait"` (default) or `"error"`; `max_wait_seconds` caps a `"wait"` acquire. |
| `guardrails` | `FlowStepGuardrails \| None` | Typed verdict surface — `pre` callables run after `enabled` + before the body; `post` callables run after a successful body. Each returns a `FlowStepGuardrailVerdict.allow()` / `reject_content(msg)` / `raise_exception(exc)`. |
| `cache` | `FlowStepCachePolicy \| None` | Per-step LRU + TTL result cache. Hits restore a deep copy of the cached state snapshot and skip the body; misses run the body and write the new snapshot. The `cache_key_fn` derives the key from `FlowStepContext`. |

```python
from troopai.adk.flows import (
    Flow, FlowStepCachePolicy, FlowStepGuardrails,
    FlowStepGuardrailVerdict, FlowStepRateLimit,
    flow_listen, flow_start,
)

class Pipeline(Flow[PipelineState]):
    @flow_start(rate_limit=FlowStepRateLimit(rpm=120))
    async def intake(self) -> None: ...

    @flow_listen(
        "intake",
        guardrails=FlowStepGuardrails(pre=(amount_under_threshold,)),
    )
    async def risk_check(self) -> None: ...

    @flow_listen(
        "risk_check",
        cache=FlowStepCachePolicy(cache_key_fn=lambda ctx: tier_of(ctx.flow_state)),
    )
    async def classify(self) -> None: ...
```

Evaluation order inside the executor:
1. `enabled` gate.
2. Pre-queued approval decision (resume path).
3. `requires_approval` gate.
4. Cache lookup — a hit short-circuits the body and emits a balanced
   `FlowStepStartEvent` / `FlowStepEndEvent` pair.
5. `rate_limit` acquire (waits or raises on saturation).
6. `guardrails.pre` chain.
7. Body, wrapped in `timeout` + `max_retries`.
8. `guardrails.post` chain.
9. Cache write.

## Agent-Internal HITL Bridge (`arun_flow_agent`)

When a step body calls an agent and that agent's run defers via a
tool's `requires_approval` gate, the bridge propagates the deferral
up to the flow layer so the whole flow halts gracefully:

```python
from troopai.adk.flows import arun_flow_agent

class CartFlow(Flow[CartState]):
    @flow_start
    async def review(self) -> None:
        result = await arun_flow_agent(self, agent, "approve $5000", defer_key="purchase")
        # If the agent's tool deferred, arun_flow_agent raises
        # FlowAgentDeferred and the executor halts before this line.
        self.state.outcome = result.final_output
```

The executor catches `FlowAgentDeferred`, captures the serialised
`RunState` into `FlowDeferredStep.agent_run_state`, and halts with
`status="deferred"`. Resume:

```python
ds = result.deferred_steps[0]
resumed_state = RunState.from_dict(json.loads(ds.agent_run_state))
# resumed_state.approve(call) / .reject(call, message) — same shape as plain tool HITL
resumed = await Runner.arun_flow_from_checkpoint(
    flow_again,
    result.checkpoint,
    agent_resolutions={ds.defer_key: json.dumps(resumed_state.to_dict())},
)
```

- The bridge is a free function rather than a method on `Flow` —
  `Flow` stays pure config (no `run()` / `arun()` methods).
- `defer_key` defaults to the calling step's method name when
  unset. Supply an explicit key when one step runs multiple agents.
- Multiple agent deferrals in the same step share `defer_key`s on
  one `FlowDeferredStep`; resume the whole set in one
  `agent_resolutions` mapping.

## Batch Fan-Out (`Runner.arun_flow_for_each`)

Fan out a Flow over a sequence of initial states with one call:

```python
results = await Runner.arun_flow_for_each(
    lambda state: SentimentFlow(state),
    [HeadlineState(headline=h) for h in headlines],
    concurrency=3,   # cap parallel runs; default 1 = sequential
)
for r in results:
    if r.status == "completed":
        print(r.final_state.label)
    else:
        print("failed:", r.error)
```

- The factory produces one fresh `Flow` per input state, so no two
  runs share a mutable `flow.state` reference.
- Default `concurrency=1` runs strictly sequentially — no
  `asyncio.gather` overhead, no implicit fan-out of LLM calls. Pass
  `concurrency=N` (N≥2) to bound a parallel batch via
  `asyncio.Semaphore`. `concurrency < 1` raises `ValueError`.
- Errors are isolated per item: a step exception or a factory raise
  produces `FlowRunResult(status="failed", error=...)` at the
  corresponding index without aborting the batch.

## Distributed Execution (`FlowWorkerBackend`)

Multiple workers / processes can share one Flow run through a
pluggable `FlowWorkerBackend`. The Protocol distributes at the
**batch boundary**: one worker claims an entire BSP superstep,
runs every step in that batch in parallel within its process,
and writes the resulting `FlowCheckpoint` back atomically.

```python
from troopai.adk.flows import SqliteFlowWorkerBackend
from troopai.adk.run.runner import Runner

backend = SqliteFlowWorkerBackend(path="/var/state/flows.db")
result = await Runner.arun_flow_distributed(
    flow, backend, worker_id="hostA-pid1234"
)
```

Built-in implementations:

| Backend | Scope |
|---|---|
| `InMemoryFlowWorkerBackend` | Single-process default; tests + single-host one-worker runs. |
| `SqliteFlowWorkerBackend` | File-backed; serialises claims via `BEGIN IMMEDIATE`. Suitable for multiple worker processes on one host. |

The Protocol shape is small (`claim_batch`, `heartbeat`,
`release_batch`, `load_checkpoint`, `save_checkpoint`,
`list_claims`) so a Redis/Postgres backend is a straightforward
extension.

**Rate-limit caveat**: `FlowStepRateLimit` enforcement is
per-executor. When the same step fires across batches claimed
by different workers, the rate-limit bucket resets between
claims — the documented `rpm` cap therefore applies *per batch
window*, not globally across a distributed deployment. Set
`rate_limit=None` and enforce the limit at the worker pool's
boundary when a globally coordinated limit is required.

## Composition with Other Primitives

A `Flow` is composable with the rest of the orchestration stack:

```python
# Flow inside Graph (via FlowExecutable adapter)
from troopai.adk.graphs import GraphBuilder
graph = (
    GraphBuilder.new("g")
    .node("flow_node", my_flow)  # auto-wraps via to_executable()
    .node("agent_node", my_agent)
    .edge("flow_node", "agent_node")
    .entry("flow_node")
    .terminal("agent_node")
    .compile()
)

# Graph / Swarm / Agent inside a Flow step body
class MyFlow(Flow[State]):
    state_factory = State

    @flow_start
    async def kickoff(self) -> None:
        # Run a Graph inside the step
        graph_result = await Runner.arun_graph(some_graph)
        self.state.intermediate = str(graph_result.final_output)
```

## Diverging from CrewAI (intentional choices)

| CrewAI feature | TroopAI Flow choice |
|---|---|
| Nested combinators `or_(and_(a, b), c)` | Flat gates only — use `Or(...)` / `And(...)` constructors for complex shapes |
| Conditional `@flow_start("trigger")` | Use `@flow_listen` for delayed entry |
| OR-listener re-firing on cyclic re-entry (`_clear_or_listeners`) | Single-fire per flow run |
| `@flow_listen` signature introspection injecting prev result | Forbidden — step methods take only `self` |
| `@persist` auto-write | Explicit `FlowCheckpoint.to_json()` |
| `Flow.plot()` visualization | Build externally from `FlowStepRegistry` |

## Future: Temporal `Workflow`

The name `Workflow` is reserved for a future Temporal-based durable
execution runtime. Temporal wraps any orchestration (Flow, Graph,
Swarm, TaskPipeline) and provides crash recovery, deterministic
replay, signals, queries, and long-running execution. The two layers
compose: a Temporal Workflow contains a Flow as its orchestration
topology.

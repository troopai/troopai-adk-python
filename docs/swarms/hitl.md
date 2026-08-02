# Human-in-the-Loop (HITL) in Swarms

Swarms support the same cooperative-pause/resume pattern as graphs: a
member agent's tool can defer for human approval, the swarm parks
state cleanly, and the caller resumes via a typed reply.

## When a member tool defers

A swarm member is just an `Agent`. When its tool defers via the
agent-level deferral mechanism, the swarm loop catches the
`AgentToolDeferral` and lifts it to a `NestedAgentInterrupt`:

```python
from troopai.adk.run.runner import Runner

result = await Runner.arun_swarm(swarm, "review-and-approve")

if result.stop_reason.kind == "interrupted":
    for interrupt in result.interrupts:
        print(f"Member {interrupt.node_id} wants approval:")
        print(f"  Question: {interrupt.question}")
        print(f"  Tool call ids: {interrupt.tool_call_ids}")
```

The swarm returns a `SwarmRunResult` with:

- `stop_reason.kind == "interrupted"` — clean signal, never raised
- `interrupts: tuple[Interrupt, ...]` — the parked interrupts
- `state.pending_interrupts` — same data keyed by member name
- `state.nested_agent_snapshots` — the deferring agent's `RunState`,
  ready for resume

The interrupt's `node_id` is the member's name (consistent with how
graphs use `node_id`).

## When a member calls `request_human_input` directly

If a member's tool raises `InterruptException` (e.g., via
`request_human_input` reused from the graphs API), the swarm catches
it and parks the interrupt the same way — except no agent state is
snapshotted, because no tool deferral occurred.

## Persisting and resuming

```python
from troopai.adk.swarms import SwarmResume
from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer

# Auto-save on every turn boundary — pass the checkpointer to the
# runner and the driver wires the swarm hook registry itself:
checkpointer = InMemorySwarmCheckpointer(thread_id="run-42")

first = await Runner.arun_swarm(swarm, "go", checkpointer=checkpointer)
if first.stop_reason.kind == "interrupted":
    # Caller composes reply...
    resume_payload = SwarmResume(replies={"approver": "yes"})

    # Resume:
    second = await Runner.arun_swarm_from_checkpoint(
        swarm,
        checkpointer=checkpointer,
        thread_id="run-42",
        resume=resume_payload,
    )
```

The `SwarmResume.replies` / `SwarmResume.rejected` dicts are keyed by
member name. Each member's reply has the same shape as the graph
substrate uses.

## Lifecycle hook

Custom `SwarmHooks` can observe interrupt events via the
`on_swarm_turn_interrupt` callback:

```python
from typing import override
from troopai.adk.swarms.hooks import SwarmHooks

class ApprovalAuditor(SwarmHooks):
    @override
    async def on_swarm_turn_interrupt(self, context, state, member_name, interrupt):
        audit_log.append({
            "member": member_name,
            "kind": interrupt.kind,
            "question": interrupt.question,
        })
```

Attach by passing `hooks=ApprovalAuditor()` at construction (or
`.with_hooks(ApprovalAuditor())` on the builder) — a `Swarm` is frozen,
so hooks are set at build time, not assigned later. Checkpointer
persistence hooks are wired automatically when you pass
`checkpointer=` to the runner; the driver fans them out with your
swarm hooks through its internal `HookRegistry`.

## Persistence backends

The shipped reference is `InMemorySwarmCheckpointer` — process-local,
suitable for tests and single-process workflows. An SQLite-backed
implementation for cross-process resume can implement the
`SwarmCheckpointer` protocol:

```python
class SwarmCheckpointer(Protocol):
    async def save(self, checkpoint: SwarmCheckpoint) -> None: ...
    async def load(self, thread_id: str, swarm: Swarm) -> SwarmCheckpoint | None: ...
    def register(self, registry: SwarmHookRegistry) -> None: ...
```

The `SwarmCheckpoint` dataclass is JSON-safe — `state` is
`SwarmState.to_dict()` output; `from_dict` rehydrates against the
caller-supplied `Swarm`.

## Deep resume

`arun_swarm_from_checkpoint(swarm, ..., resume=SwarmResume(replies={...}))`
applies the caller-supplied reply to the parked member so the
deferred work continues from where it paused. The mechanism dispatches
on the parked-interrupt kind:

- **Nested-agent-defer** (the deferring tool produced a
  `NestedAgentInterrupt` and a `RunState` snapshot): the driver
  wraps the member in an `AgentExecutable` and calls
  `resume_from_snapshot(snapshot, reply, ...)` with the
  `NestedAgentReply` payload. The deferred tool's decisions are
  applied to the paused `RunState`, the agent continues the same
  turn, and the swarm flows into the next turn normally.
- **Pure HITL** (the member's tool raised `InterruptException`
  directly, no snapshot): the driver seeds the reply onto the run
  context and re-fires the member. The member's tool reads the
  reply via `request_human_input_in_swarm(ctx_wrapper, member_name,
  question, ...)` — the swarm-substrate companion to graphs'
  `request_human_input`. Key-presence (not truthiness) determines
  reply availability, so `None` is a valid abstain answer.

Both paths re-park naturally on re-deferral: a fresh
`InterruptException` from the resumed run surfaces back through the
swarm loop's existing handler, which records the new interrupt
under the same member name and returns
`stop_reason.kind == "interrupted"`.

```python
# Nested-agent-defer resume
result = await Runner.arun_swarm_from_checkpoint(
    swarm,
    checkpointer=cp,
    thread_id="run-42",
    resume=SwarmResume(
        replies={
            "approver": NestedAgentReply(
                decisions=(NestedAgentApproval(tool_call_id="c1"),),
            ),
        },
    ),
)

# Pure-HITL resume — reply is whatever the originating tool expects
result = await Runner.arun_swarm_from_checkpoint(
    swarm,
    checkpointer=cp,
    thread_id="run-42",
    resume=SwarmResume(replies={"approver": "yes"}),
)
```

The deferring agent's tokens are folded into `per_member_usage` and
`state.cumulative_usage` so cost-attribution accounting is consistent
across the suspend/resume boundary.

When the caller resumes without a `SwarmResume` payload (or omits the
parked member from `replies`), the driver falls back to the clear-and-
restart path: the parked entries are dropped and the parked turn re-runs
from scratch. That path is useful when the caller wants to abandon the
parked decision but continue the swarm.

## Tracing

Swarms emit a typed OTel span tree when a tracer is installed:

- One `swarm.<swarm_id>` root span per run, with attributes
  `troopai.swarm.id`, `troopai.swarm.entry`, `troopai.swarm.status`,
  `troopai.swarm.turns_total`.
- One `swarm.turn.<index>` span per iteration that runs a member
  turn, with attributes `troopai.swarm.id`, `troopai.swarm.turn.index`,
  `troopai.swarm.turn.member`, `troopai.swarm.turn.status`,
  `troopai.swarm.turn.duration_ms`, and (on resumed turns)
  `troopai.swarm.turn.resume_attempt`.
- `troopai.swarm.id` is a UUID generated at runner entry and persisted
  on `SwarmState.swarm_id` so suspend + resume share one root-span
  identity. Tracing dashboards can correlate the full lifecycle by
  joining on this attribute.

Per-member `agent_span`s opened by the inner agent loop nest under
the active `swarm_turn_span` via OTel context propagation — no
extra factory needed for that layer.

Disable tracing entirely via `RunConfig(tracing_enabled=False)`;
factories return `NoOpSpan` and no spans are emitted.

## Streaming

`Runner.arun_swarm_streamed(swarm, prompt, ...)` returns a
`SwarmRunResultStreaming` immediately; the swarm runs in a
background `asyncio.Task` that pushes events into the result's
internal queue. Consumers iterate the queue via
`async for ev in result.stream_events()`.

The event vocabulary is the same as `Runner.arun_streamed`'s
per-agent stream plus these swarm-scoped variants:

- `SwarmStartEvent` — emitted once at run start; carries
  `entry_agent` + `member_names`.
- `SwarmTurnStartEvent` — emitted before each member turn; carries
  `agent` + `turn` (1-indexed).
- Per-agent `raw_response_event` / `run_item_stream_event` /
  `agent_updated_stream_event` — flow through unchanged between
  `SwarmTurnStartEvent` and the turn's closing event.
- `SwarmHandoffEvent` — emitted when a `SwarmHandoff` yield
  resolves; carries `from_agent` / `to_agent` / `message`.
- `SwarmTurnEndEvent` — emitted at successful turn end; carries
  `agent` + `items`.
- `SwarmTurnInterruptEvent` — **replaces** `SwarmTurnEndEvent`
  for any turn that suspends on `InterruptException` (pure HITL)
  or `AgentToolDeferral` (nested-agent-defer, lifted to a
  `NestedAgentInterrupt`). Carries `agent` + `turn` + `interrupt`.
- `SwarmDoneEvent` — emitted exactly once at run end; carries the
  `StopReason` and `final_output`.

Suspending mid-stream:

```python
from troopai.adk.swarms.events import SwarmTurnInterruptEvent

result = await Runner.arun_swarm_streamed(swarm, "review-and-approve")
async for ev in result.stream_events():
    if isinstance(ev, SwarmTurnInterruptEvent):
        print(f"Member {ev.agent} suspended on turn {ev.turn}: {ev.interrupt.question}")
        # Build a SwarmResume and resume via a second arun_swarm_streamed
        # call with initial_state=result.state and resume=SwarmResume(...).
```

Resume-through-stream: pass the persisted `SwarmState` as
`initial_state` and the caller's `SwarmResume` as `resume` to a
second `arun_swarm_streamed` call. The same deep-resume splice
from `swarm_resume.py` fires inside the streamed loop, and the
same `troopai.swarm.id` flows through both root spans for
end-to-end trace correlation.

Cancellation: `result.cancel()` cancels the background driver and
drains the queue; the consumer's next `await` exits cleanly.
Slow consumers grow the queue unbounded — same trade-off as
`Runner.arun_graph_streamed`.

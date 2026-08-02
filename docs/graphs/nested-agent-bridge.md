# Nested-Agent Tool Approval Bridge

Lift a sub-agent's tool-approval deferral to a graph-level interrupt so
the same `GraphResume` / checkpoint surface that powers human-in-the-loop
callables also covers `Agent` nodes whose tools require approval.

## Overview

When an `Agent` node's tool defers for human approval, the bridge
translates the sub-agent's `AgentToolDeferral` into a
`NestedAgentInterrupt` keyed by the graph node id. The BSP loop parks
the interrupt on `GraphState.pending_interrupts`, deposits the sub-agent's
mid-run `RunState` on `GraphState.nested_agent_snapshots`, and writes a
checkpoint. The caller resumes by supplying a typed `NestedAgentReply`
on `GraphResume.replies[node_id]`. On resume the loop reads the snapshot
back, applies each `NestedAgentApproval` / `NestedAgentRejection` to the
sub-agent's `RunState`, and re-enters `Runner.arun` with the staged
state — the sub-agent picks up exactly where it paused.

No new authoring step is required for the agent itself: tools already
declared with `requires_approval=True` defer normally, and the bridge
takes over at the graph boundary.

## Public surface

All types are importable from `troopai.adk.graphs`:

| Type | Description |
|---|---|
| `NestedAgentInterrupt(Interrupt)` | Specialisation of `Interrupt` carrying `node_id`, `question`, `kind="nested_agent_tool_approval"`, `agent_name`, and `tool_call_ids: tuple[str, ...]`. |
| `NestedAgentApproval(tool_call_id, approver_id=None, reason=None)` | Approve one deferred tool call. Forwarded to `RunState.approve`. |
| `NestedAgentRejection(tool_call_id, message=None, approver_id=None, reason=None)` | Reject one deferred tool call. Forwarded to `RunState.reject`. |
| `NestedAgentDecision` | `NestedAgentApproval \| NestedAgentRejection` — discriminated union of the two decisions above. |
| `NestedAgentReply(decisions: tuple[NestedAgentDecision, ...])` | Typed payload supplied via `GraphResume.replies[node_id]` for a `NestedAgentInterrupt`. An empty tuple is permitted — every pending call re-defers. |
| `NestedAgentResumeError` | Raised when a `NestedAgentReply` targets unknown `tool_call_id`s, or contains duplicates. |
| `NestedAgentSerializationError` | Raised at deferral time when the sub-agent's `RunState` carries a non-JSON-serialisable field (e.g. closures, threadlocals). |
| `GraphResumeError` | Raised by the BSP loop on resume-payload / state mismatch (wrong reply shape for the parked interrupt kind, missing snapshot, both `replies` and `rejected` keyed on the same `node_id`, ...). |

`NESTED_AGENT_TOOL_APPROVAL_KIND` is the `Interrupt.kind` discriminator
value — useful when a downstream consumer needs to branch on the
interrupt kind without an `isinstance` check.

## Minimal example

A single-node graph wraps an `Agent` whose `delete_user` tool requires
approval. The first run defers; the caller approves the call and resumes.

```python
import asyncio

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs import Graph
from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
from troopai.adk.graphs.interrupt import (
    GraphResume,
    NestedAgentApproval,
    NestedAgentInterrupt,
    NestedAgentReply,
)
from troopai.adk.graphs.result import GraphRunStatus
from troopai.adk.run.runner import Runner
from troopai.adk.tools.function_tool import function_tool


@function_tool(
    name="delete_user",
    description="Permanently delete a user account. Irreversible.",
    requires_approval=True,
)
def delete_user(user_id: str) -> str:
    return f"User {user_id} has been permanently deleted."


account_manager = Agent(
    name="account-manager",
    llm="gpt-4o-mini",
    system_prompt="Use delete_user to remove a user when asked.",
    tools=[delete_user],
)

graph = (
    Graph.new("nested-hitl-demo")
    .node("manager", account_manager)
    .entry("manager")
    .terminal("manager")
    .compile()
)


async def main() -> None:
    cp = InMemoryCheckpointer()

    first = await Runner.arun_graph(
        graph,
        "Please delete the user with id 2.",
        hooks=[cp],
        thread_id="nested-1",
    )
    assert first.status == GraphRunStatus.INTERRUPTED

    # One interrupt, carrying the deferred tool_call_id(s) and the
    # sub-agent name. The matching RunState lives on
    # first.state.nested_agent_snapshots["manager"].
    interrupt = first.interrupts[0]
    assert isinstance(interrupt, NestedAgentInterrupt)
    decisions = tuple(
        NestedAgentApproval(tool_call_id=tcid, approver_id="cli-user")
        for tcid in interrupt.tool_call_ids
    )

    resumed = await Runner.arun_graph_from_checkpoint(
        graph,
        checkpointer=cp,
        thread_id="nested-1",
        resume=GraphResume(
            replies={"manager": NestedAgentReply(decisions=decisions)},
        ),
    )
    assert resumed.status == GraphRunStatus.COMPLETED


asyncio.run(main())
```

The same shape works with the profile runner form
(`Runner.configure().graph(graph).resume_from(cp, "nested-1").arun(...)`).
See `docs/graphs/hitl.md` for the two equivalent resume entry points.

## Partial resume (concurrent fan-out)

When two `Agent` nodes both defer in the same superstep, the loop collects
both interrupts under their respective node ids and parks both snapshots.
The caller can resume one at a time:

- `GraphResume(replies={"a": NestedAgentReply(...)})` applies the
  decision to node `a`. Node `a`'s sub-agent re-enters and runs to
  completion (or re-defers); node `b`'s interrupt re-surfaces on the
  resumed result with the same `pending_interrupts["b"]` and the same
  parked snapshot.
- The next resume can target node `b` (or both at once). Until every
  node receives a reply or a rejection, the run stays in
  `GraphRunStatus.INTERRUPTED`.

A node that received a reply has its `pending_interrupts[node_id]` and
`nested_agent_snapshots[node_id]` cleared; its terminal output (if any)
moves to `state.terminal_outputs[node_id]`. The canonical executable
contract for partial resume lives in
`tests/integration/graphs/test_nested_resume.py::test_concurrent_fanout_partial_resume_then_completion`.

## Re-deferral semantics

A sub-agent that resumes from one approval and then defers again on a
subsequent tool call (e.g. a chain of gated operations) surfaces a fresh
`NestedAgentInterrupt` keyed by the same graph node id. The bridge is
idempotent: the loop deposits a new snapshot, parks a new interrupt, and
checkpoints; the caller can iterate approve → resume → approve → resume
without any special handling. The interrupt's `tool_call_ids` reflect the
newly deferred set, NOT the historical chain. Reference test:
`tests/integration/graphs/test_nested_resume.py::test_re_deferral_after_resume_re_checkpoints`.

## Mutual exclusion: `replies` vs `rejected`

`GraphResume.replies[node_id]` and `GraphResume.rejected[node_id]` are
mutually exclusive for the same `node_id`. Supplying both raises
`GraphResumeError` before the resumed executable runs, so the caller can
fix the payload and retry against the same checkpoint.

- `replies[node_id] = NestedAgentReply(decisions=(...))` — typed
  decisions, one per deferred `tool_call_id`. Use this for any non-blanket
  outcome (mixed approvals/rejections, per-call rationale, partial
  decision sets).
- `rejected[node_id] = "denied by reviewer"` — single message applied as
  a blanket rejection to every pending call on that node. Equivalent to
  one `NestedAgentRejection(message=...)` per `tool_call_id`, useful when
  the reviewer denies the whole batch with one rationale.

## Limitations and follow-up work

### Streaming variant

A `Runner.arun_graph_streamed` run that hits a `NestedAgentInterrupt`
emits the `graph.node_interrupt` event normally and exits the stream
cleanly. The resumed turn, however, currently flows through
non-streaming `Runner.arun` internally — interior `agent_event` items
produced by the resumed sub-agent are not re-emitted to the streaming
consumer. A streaming variant of `resume_from_snapshot` that forwards
those events into the outer stream is a follow-up.

### Depth-2 nested-graph resume

A `Graph` used as an inner node whose own agent defers does NOT yet
propagate the interrupt to the outer graph. `Graph.invoke` translates an
inner `GraphRunResult.status == INTERRUPTED` into an ordinary
`NodeResult` rather than re-raising `InterruptException(NestedAgentInterrupt)`
with the `node_id` rewritten to the outer scope, and
`GraphState.nested_agent_snapshots` is typed `dict[str, RunState]` with
no slot for an inner `GraphState`. The strict-xfail at
`tests/integration/graphs/test_nested_resume_deep.py:203` is the on-record
diagnostic — when depth-2 support lands, the marker flips and the test
starts passing.

### Snapshot serialisation

The sub-agent's `RunState` is serialised via `RunState.to_dict()` /
`RunState.from_dict()`. Fields that carry non-JSON-serialisable values
(closures, threadlocals, framework-internal handles) surface as
`NestedAgentSerializationError` at deferral time, raised by the
`AgentExecutable` before the BSP loop deposits a half-formed snapshot.
The bridge MUST NOT silently fall back — losing the snapshot would
defeat the HITL contract by erasing the caller's pending decision.

### Two-phase validate-then-stage on resume

The BSP loop currently validates resume payloads while mutating
in-memory state. A `GraphResumeError` raised mid-validation leaves the
in-process `GraphState` partially mutated; recovery requires reloading
the run from the checkpointer.
`Runner.arun_graph_from_checkpoint` does this automatically — each
invocation re-loads `GraphState` via `checkpointer.load`, so a fixed
payload on the next call runs against an untouched copy. Callers that
hand a `GraphState` directly to the loop should reload from the
checkpointer rather than retry against the same in-memory state.

## See also

- `docs/graphs/hitl.md` — graph-level human-in-the-loop interrupts, the
  `request_human_input` helper, and the `GraphResume` shape this bridge
  reuses for nested-agent decisions.
- `docs/graphs/composition.md` — how `Agent`, `Swarm`, `Graph`, and
  plain callables compose at the `Executable[TContext]` seam; nested
  graph patterns.
- `docs/graphs/checkpointing.md` — durable checkpointers and the
  selective re-fire contract that makes resume deterministic.
- `src/troopai/adk/graphs/interrupt.py` — `NestedAgentInterrupt`,
  `NestedAgentApproval`, `NestedAgentRejection`, `NestedAgentReply`, and
  the error hierarchy.
- `src/troopai/adk/graphs/adapters.py` — the `AgentExecutable` invoke /
  resume path that lifts deferrals to interrupts and applies decisions
  on resume.
- `tests/integration/graphs/test_nested_resume.py` — single-level
  defer/resume, concurrent fan-out, partial resume, re-deferral.

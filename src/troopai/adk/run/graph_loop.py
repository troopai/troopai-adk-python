"""``run_graph_loop`` — BSP superstep driver for :class:`Graph` execution.

This is the graph-world counterpart of :func:`run_swarm_loop`. The
driver owns:

- **Scheduling** — at the top of each superstep, it computes the set of
  ready nodes (entry on the first pass; all nodes whose
  :class:`JoinBarrier` is satisfied thereafter).
- **Concurrent execution** — ready nodes run concurrently via
  :func:`asyncio.wait` with ``FIRST_COMPLETED`` so a fail-fast error
  cancels siblings as early as possible. On happy paths the loop
  degrades to :func:`asyncio.gather` semantics (all results before
  barrier).
- **Deterministic state-apply** — after all ready tasks complete, the
  loop applies their results into :class:`GraphState` sorted by node
  id. This matches LangGraph's path-sorted write application and
  guarantees reducers run in a stable order regardless of completion
  order.
- **Edge evaluation** — for each completed node, outgoing edges fire
  (subject to their ``when`` predicate). Each firing records the
  upstream's :class:`NodeResult` on the target's :class:`JoinBarrier`.
- **Terminal tracking** — terminal outputs are copied off as they fire
  and the loop exits once every terminal has fired at least once
  (OR-semantics — exit on ``any``; an AND-all-terminals flag is not
  yet implemented).
- **Budget guards** — :attr:`GraphConfig.max_supersteps` /
  :attr:`GraphConfig.max_total_tokens` trip the loop.
- **Hook fan-out** — :class:`HookRegistry` is wired once at top,
  fired at every documented boundary. :class:`Checkpointer`\\ s subscribe
  via :meth:`HookProvider.register` and persist state on
  ``on_node_end`` — the loop itself never calls ``save()``.

Deliberate non-goals:

- No dynamic fan-out (``Send``-equivalent).

The loop never re-enters ``run_agent_loop`` with a graph-specific
next-step. :meth:`Executable.invoke` is the only delegation seam.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import uuid
from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, Any, Protocol, cast

from troopai.adk.graphs.adapters import AgentExecutable
from troopai.adk.graphs.events import (
    GraphEndEvent,
    GraphStartEvent,
    NodeEndEvent,
    NodeErrorEvent,
    NodeInterruptEvent,
    NodeStartEvent,
    NodeStreamEvent,
    SuperstepEndEvent,
    SuperstepStartEvent,
)
from troopai.adk.graphs.hooks import HookProvider, HookRegistry
from troopai.adk.graphs.interrupt import (
    GraphResume,
    GraphResumeError,
    Interrupt,
    InterruptException,
    NestedAgentInterrupt,
    NestedAgentRejection,
    NestedAgentReply,
    NestedGraphInterrupt,
)
from troopai.adk.graphs.join import JoinBarrier
from troopai.adk.graphs.node_input import prepare_node_input
from troopai.adk.graphs.result import GraphRunResult, GraphRunResultStreaming, GraphRunStatus, StructuredInterrupts
from troopai.adk.graphs.state import GraphState
from troopai.adk.orchestration.executable import ExecutableInput, NodeResult
from troopai.adk.run.node_reliability import resolve_node_reliability, run_node_with_reliability
from troopai.adk.run.state import RunState
from troopai.adk.run.stream import CancelMode
from troopai.adk.tracing.spans import Span, graph_node_span, graph_span, graph_superstep_span
from troopai.adk.types.tracing.span_data import CustomSpanData

if TYPE_CHECKING:
    from troopai.adk.graphs.graph import Graph
    from troopai.adk.graphs.hooks import GraphHooks
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.types import UserPrompt


logger = logging.getLogger(__name__)

EventEmitter = Callable[[object], Awaitable[None]]


async def _noop_emitter(event: object) -> None:
    """Default emit seam — drops the event (non-streaming path)."""
    del event
    return None


class NodeRunner(Protocol):
    """Callable signature for per-node execution strategies."""

    async def __call__(
        self,
        *,
        graph: Graph[Any],
        node_id: str,
        input: ExecutableInput,
        context: RunContext[Any],
        config: RunConfig,
    ) -> NodeResult[Any]: ...


class TaskTracker(Protocol):
    """Hook points for registering/discarding in-flight node tasks."""

    def register_node_task(self, task: asyncio.Task[NodeResult[Any]]) -> None: ...

    def discard_node_task(self, task: asyncio.Task[NodeResult[Any]]) -> None: ...


def _generate_thread_id() -> str:
    """Auto-generate a ``thread-XXXX`` id when checkpointing is opted in.

    Matches :func:`troopai.adk.graphs.graph._generate_graph_id` in format
    so ``graph-<id>`` and ``thread-<id>`` read alike in logs. 12 hex
    chars (~48 bits of entropy) is more than enough to avoid collisions
    within a single process lifetime.
    """
    return f"thread-{uuid.uuid4().hex[:12]}"


def _build_join_barriers(
    graph: Graph[Any],
) -> dict[str, JoinBarrier]:
    """Compute a :class:`JoinBarrier` for every node with ≥1 incoming edge.

    Entry nodes have zero incoming edges and therefore no barrier
    (they fire on the first superstep by construction). Single-parent
    nodes still get a barrier — it makes the loop's scheduling logic
    uniform ("ask the barrier; if it says ready, fire"). The memory
    cost is negligible.

    The barrier's ``semantics`` is read from the target
    :class:`GraphNode`. Most graphs keep the AND default; fan-in-
    heavy workflows (race-to-respond patterns) set OR on the join node.
    """
    barriers: dict[str, JoinBarrier] = {}
    for node in graph.nodes:
        sources = frozenset(e.source for e in graph.incoming_edges(node.id))
        if len(sources) == 0:
            continue
        barriers[node.id] = JoinBarrier(
            target=node.id,
            expected=sources,
            semantics=node.join,
        )
    return barriers


async def _eval_edge_predicate(
    predicate: Any,
    result: NodeResult[Any],
) -> bool:
    """Evaluate a possibly-async :class:`EdgeCondition` predicate.

    :class:`EdgeCondition` MAY be sync or async. We call it, detect an
    awaitable via :func:`inspect.isawaitable`, await if needed, and
    coerce the return to ``bool`` so downstream can rely on a strict
    type.
    """
    import inspect

    value = predicate(result)
    if inspect.isawaitable(value):
        value = await value
    return bool(value)


async def _invoke_node(
    *,
    graph: Graph[Any],
    node_id: str,
    input: ExecutableInput,
    context: RunContext[Any],
    config: RunConfig,
) -> NodeResult[Any]:
    """Invoke a single node's :meth:`Executable.invoke`.

    Thin wrapper: looks up the node, calls ``.executable.invoke(...)``.
    Kept out of the main loop body so that :func:`asyncio.create_task`
    sees a single awaitable per node and the task name is easy to
    attribute in traces.
    """
    node = graph.get_node(node_id)
    logger.debug(
        "graph_loop: invoking node_id=%s executable=%s",
        node_id,
        type(node.executable).__name__,
    )
    policy, timeout = resolve_node_reliability(graph, node)
    return await run_node_with_reliability(
        node_id=node_id,
        policy=policy,
        timeout=timeout,
        invoke=lambda: node.executable.invoke(input, context, config),
    )


async def _stream_node(
    *,
    graph: Graph[Any],
    node_id: str,
    input: ExecutableInput,
    context: RunContext[Any],
    config: RunConfig,
    graph_path: tuple[str, ...],
    result: GraphRunResultStreaming[Any],
) -> NodeResult[Any]:
    """Run one node via :meth:`Executable.stream_async`, forwarding interior events.

    The default :meth:`Executable.stream_async` yields exactly one terminal
    ``{"type": "result", "result": NodeResult}``; any non-result item is an
    interior event forwarded wrapped in :class:`NodeStreamEvent`. Wrapped by
    the SP2 reliability policy so retry/timeout semantics are preserved.
    """
    node = graph.get_node(node_id)
    logger.debug(
        "graph_loop: streaming node_id=%s executable=%s",
        node_id,
        type(node.executable).__name__,
    )
    policy, timeout = resolve_node_reliability(graph, node)

    async def _attempt() -> NodeResult[Any]:
        terminal: NodeResult[Any] | None = None
        async for item in node.executable.stream_async(input, context, config):
            if isinstance(item, dict) and item.get("type") == "result":
                terminal = item["result"]
                continue
            await result.put_event(NodeStreamEvent(graph_path=graph_path, node_id=node_id, inner=item))
        if terminal is None:
            raise RuntimeError(f"stream_async for node {node_id!r} produced no terminal result")
        return terminal

    return await run_node_with_reliability(
        node_id=node_id,
        policy=policy,
        timeout=timeout,
        invoke=_attempt,
    )


async def _seed_barriers_from_checkpoint(
    *,
    graph: Graph[Any],
    state: GraphState[Any],
    barriers: dict[str, JoinBarrier],
) -> None:
    """Re-deliver unconsumed edges after a checkpoint restore.

    JoinBarrier arrivals are never serialised, so on resume barriers
    are rebuilt empty. For each edge ``(u -> d)`` where
    ``state.produced_at[u] > state.versions_seen[d].get(u, -1)``, this
    records ``u``'s persisted :class:`NodeResult` on ``d``'s barrier
    (or a skip when ``edge.when`` is False / raises). Already-consumed
    edges are not re-delivered, so completed nodes do not re-execute.
    """
    for node in graph.nodes:
        for edge in graph.outgoing_edges(node.id):
            upstream = edge.source
            downstream = edge.target
            node_result = state.node_results.get(upstream)
            if node_result is None:
                continue
            produced = state.produced_at.get(upstream, -1)
            last_consumed = state.versions_seen.get(downstream, {}).get(upstream, -1)
            if produced <= last_consumed:
                continue
            target_barrier = barriers.get(downstream)
            if target_barrier is None:
                logger.warning(
                    "graph_loop: resume edge (%s -> %s) has no target barrier; ignoring.",
                    upstream,
                    downstream,
                )
                continue
            should_fire = True
            predicate_failed = False
            if edge.when is not None:
                try:
                    should_fire = await _eval_edge_predicate(edge.when, node_result)
                except Exception as exc:
                    logger.error(
                        "graph_loop: resume edge predicate (%s -> %s) raised %s: %s; "
                        "recording failure (downstream will not fire).",
                        upstream,
                        downstream,
                        type(exc).__name__,
                        exc,
                    )
                    should_fire = False
                    predicate_failed = True
            if should_fire:
                target_barrier.record(upstream, node_result)
            elif predicate_failed:
                target_barrier.record_fail(upstream)
            else:
                target_barrier.record_skip(upstream)
    logger.info(
        "graph_loop: reconstructed %d barrier(s) from checkpoint at superstep=%d",
        len(barriers),
        state.superstep,
    )


def _setup_graph_run(
    *,
    graph: Graph[Any],
    hooks: list[GraphHooks[Any] | HookProvider] | None,
    thread_id: str | None,
    initial_state: GraphState[Any] | None,
) -> tuple[HookRegistry, GraphState[Any]]:
    """Assemble the :class:`HookRegistry` and initial :class:`GraphState`.

    Returns the registry and state ready for the BSP loop. Callers must
    await :func:`_seed_barriers_from_checkpoint` when ``initial_state``
    is not ``None``.
    """
    registry = HookRegistry()
    for h in graph.hooks:
        registry.add(h)
    if hooks is not None:
        for item in hooks:
            if isinstance(item, HookProvider):
                registry.add_provider(item)
            else:
                registry.add(item)

    if initial_state is not None:
        state = initial_state
        logger.info(
            "run_graph_loop: resuming from checkpoint thread_id=%s superstep=%s",
            state.thread_id,
            state.superstep,
        )
    else:
        state = GraphState(
            graph=graph,
            thread_id=thread_id
            if thread_id is not None
            else (_generate_thread_id() if _has_checkpointer(hooks) else None),
        )
        logger.info(
            "run_graph_loop: starting graph_id=%s thread_id=%s",
            graph.id,
            state.thread_id,
        )
    return registry, state


def _assemble_final_output(
    *,
    state: GraphState[Any],
    terminals_set: set[str],
    status: GraphRunStatus,
) -> Any:
    """Derive ``final_output`` from terminal outputs and update state."""
    if len(terminals_set) == 1:
        only_terminal = next(iter(terminals_set))
        final_output = state.terminal_outputs.get(only_terminal)
    else:
        final_output = dict(state.terminal_outputs)
    state.final_output = final_output

    if status == GraphRunStatus.COMPLETED and state.status != "failed":
        state.status = "completed"
    return final_output


def _require_nested_snapshot(
    *,
    node_id: str,
    state: GraphState[Any],
) -> RunState:
    """Return the parked sub-agent snapshot for ``node_id``; raise if absent.

    Caller is responsible for removal via :func:`_stage_nested_reply`.
    The cross-reference check in :meth:`GraphState.from_dict` already
    guards loaded payloads — this is the in-process counterpart that
    refuses to dispatch a nested-resume without a paired snapshot.
    """
    snap = state.nested_agent_snapshots.get(node_id)
    if snap is None:
        raise GraphResumeError(
            f"node {node_id!r} paused on NestedAgentInterrupt but "
            f"nested_agent_snapshots has no matching entry — "
            f"checkpoint inconsistency"
        )
    return snap


def _stage_nested_reply(
    *,
    node_id: str,
    reply: NestedAgentReply,
    snap: RunState,
    prepared_input: ExecutableInput,
    state: GraphState[Any],
) -> None:
    """Write the staged reply + snapshot onto the prepared input + state.

    Centralised so the approval and rejection branches both clear
    :attr:`GraphState.pending_interrupts` and
    :attr:`GraphState.nested_agent_snapshots` identically — splitting the
    metadata-write from the state-pop avoided dropping one branch's
    cleanup as the function grew.
    """
    prepared_input.metadata["__nested_agent_reply__"] = reply
    prepared_input.metadata["__nested_agent_snapshot__"] = snap
    state.pending_interrupts.pop(node_id, None)
    state.nested_agent_snapshots.pop(node_id, None)


def _require_agent_executable_for_nested_resume(
    *,
    graph: Graph[Any],
    node_id: str,
) -> AgentExecutable[Any]:
    """Return the node's executable narrowed to :class:`AgentExecutable`.

    Raises :class:`GraphResumeError` synchronously when the node parked
    on a :class:`NestedAgentInterrupt` is now backed by a non-Agent
    executable — the graph shape must have changed between checkpoint
    and resume.
    """
    executable = graph.get_node(node_id).executable
    if not isinstance(executable, AgentExecutable):
        raise GraphResumeError(
            f"node {node_id!r} paused on NestedAgentInterrupt but executable "
            f"is {type(executable).__name__}, not AgentExecutable — graph "
            f"shape changed between checkpoint and resume."
        )
    return executable


def _synthesise_rejection_reply(
    *,
    node_id: str,
    snap: RunState,
    message: str,
) -> NestedAgentReply:
    """Build a :class:`NestedAgentReply` rejecting every pending approval.

    Raises:
        GraphResumeError: When ``snap.deferred_tool_requests.approvals``
            is empty — the caller likely retried against an
            already-resolved snapshot.
    """
    if len(snap.deferred_tool_requests.approvals) == 0:
        raise GraphResumeError(
            f"node {node_id!r}: GraphResume.rejected supplied but snapshot has "
            f"no pending approvals to reject — caller likely retried against "
            f"an already-resolved snapshot."
        )
    return NestedAgentReply(
        decisions=tuple(
            NestedAgentRejection(tool_call_id=c.tool_call_id, message=message)
            for c in snap.deferred_tool_requests.approvals
        ),
    )


def _inject_nested_agent_resume(
    *,
    graph: Graph[Any],
    node_id: str,
    resume: GraphResume,
    prepared_input: ExecutableInput,
    state: GraphState[Any],
) -> None:
    """Stage a :class:`NestedAgentInterrupt` resume payload on the node's input.

    Translates the human-supplied :class:`NestedAgentReply` (or a bare
    decline message under :attr:`GraphResume.rejected`) into reserved
    metadata keys, runs the executable-type guard synchronously so any
    mismatch propagates to the outer
    ``except GraphResumeError: raise`` rather than being absorbed by
    the per-task error collector, and pops the matching
    ``pending_interrupts`` / ``nested_agent_snapshots`` entries.

    Raises:
        GraphResumeError: On type-incompatible reply, missing snapshot,
            or executable that is no longer an :class:`AgentExecutable`.
    """
    _require_agent_executable_for_nested_resume(graph=graph, node_id=node_id)
    if node_id in resume.replies:
        reply_value = resume.replies[node_id]
        if not isinstance(reply_value, NestedAgentReply):
            raise GraphResumeError(
                f"node {node_id!r} paused on NestedAgentInterrupt — "
                f"GraphResume.replies[{node_id!r}] must be a "
                f"NestedAgentReply, got {type(reply_value).__name__}"
            )
        snap = _require_nested_snapshot(node_id=node_id, state=state)
        _stage_nested_reply(
            node_id=node_id,
            reply=reply_value,
            snap=snap,
            prepared_input=prepared_input,
            state=state,
        )
        return
    if node_id in resume.rejected:
        snap = _require_nested_snapshot(node_id=node_id, state=state)
        # ``GraphResume.rejected`` delivers a single decline message — apply
        # to ALL pending tool calls on the snapshot so the resumed sub-agent
        # treats the whole deferral as rejected.
        synth_reply = _synthesise_rejection_reply(
            node_id=node_id,
            snap=snap,
            message=resume.rejected[node_id],
        )
        _stage_nested_reply(
            node_id=node_id,
            reply=synth_reply,
            snap=snap,
            prepared_input=prepared_input,
            state=state,
        )


def _inject_nested_graph_resume(
    *,
    node_id: str,
    resume: GraphResume,
    prepared_input: ExecutableInput,
    state: GraphState[Any],
) -> None:
    """Stage a graph-backed nested resume payload on the node's input.

    Sibling of :func:`_inject_nested_agent_resume` for nodes whose
    executable is a :class:`Graph` (not an :class:`Agent`) and whose
    inner graph suspended. The outer reply (a :class:`NestedAgentReply`)
    is forwarded to the inner-graph's deferring agent by the dispatch
    helper :func:`_dispatch_inner_graph_resume`.

    Pops ``state.pending_interrupts[node_id]`` and
    ``state.nested_graph_snapshots[node_id]`` once staged.

    Raises:
        GraphResumeError: When ``state.nested_graph_snapshots`` has no
            matching entry, or when the supplied reply is not a
            :class:`NestedAgentReply`.
    """
    if node_id not in state.nested_graph_snapshots:
        raise GraphResumeError(
            f"node {node_id!r}: routed as graph-backed resume but "
            f"nested_graph_snapshots has no entry — checkpoint inconsistency."
        )
    inner_state = state.nested_graph_snapshots[node_id]
    inner_sorted = sorted(inner_state.pending_interrupts.items())
    if len(inner_sorted) == 0:
        raise GraphResumeError(
            f"node {node_id!r}: graph-backed resume staged but inner GraphState has no pending interrupts."
        )
    # The inner interrupt type is authoritative: a NestedAgentInterrupt takes a
    # NestedAgentReply (tool-approval decisions); a plain Interrupt (lifted as a
    # NestedGraphInterrupt) takes a plain reply value forwarded verbatim.
    inner_iv = inner_sorted[0][1]

    if node_id in resume.replies:
        reply_value = resume.replies[node_id]
        if isinstance(inner_iv, NestedAgentInterrupt) and not isinstance(reply_value, NestedAgentReply):
            raise GraphResumeError(
                f"node {node_id!r} paused on inner-graph NestedAgentInterrupt — "
                f"GraphResume.replies[{node_id!r}] must be a NestedAgentReply, "
                f"got {type(reply_value).__name__}"
            )
        prepared_input.metadata["__nested_graph_reply__"] = reply_value
        prepared_input.metadata["__nested_graph_snapshot__"] = inner_state
        state.pending_interrupts.pop(node_id, None)
        state.nested_graph_snapshots.pop(node_id, None)
        return
    if node_id in resume.rejected:
        forwarded_reply: Any
        if isinstance(inner_iv, NestedAgentInterrupt):
            # Build a NestedAgentReply rejecting every pending inner approval —
            # mirror of the agent-backed _synthesise_rejection_reply path. The
            # deferring agent's tool_call_ids are on the inner NestedAgentInterrupt.
            forwarded_reply = NestedAgentReply(
                decisions=tuple(
                    NestedAgentRejection(tool_call_id=tcid, message=resume.rejected[node_id])
                    for tcid in inner_iv.tool_call_ids
                ),
            )
        else:
            # Inner plain Interrupt: forward the rejection message as the plain
            # reply, mirroring the plain-interrupt path's __resume_reply__.
            forwarded_reply = resume.rejected[node_id]
        prepared_input.metadata["__nested_graph_reply__"] = forwarded_reply
        prepared_input.metadata["__nested_graph_snapshot__"] = inner_state
        state.pending_interrupts.pop(node_id, None)
        state.nested_graph_snapshots.pop(node_id, None)


def _inject_resume_for_node(
    *,
    graph: Graph[Any],
    node_id: str,
    resume: GraphResume,
    prepared_input: ExecutableInput,
    state: GraphState[Any],
) -> None:
    """Dispatch one node's resume payload kind-aware.

    Branches on the parked :class:`Interrupt` subtype: a
    :class:`NestedAgentInterrupt` routes through
    :func:`_inject_nested_agent_resume`; any other interrupt uses the
    plain ``__resume_reply__`` channel consumed by
    :func:`request_human_input`. Both branches pop the consumed
    ``pending_interrupts`` entry. A node missing from ``resume.replies``
    and ``resume.rejected`` re-suspends naturally on the next invocation.

    Graph-backed nodes (where the executable is itself a Graph and the
    inner graph suspended) take precedence: when
    ``state.nested_graph_snapshots`` has an entry for ``node_id`` the
    graph-backed seed path runs, populating the
    ``__nested_graph_*__`` metadata keys consumed by
    :func:`_dispatch_inner_graph_resume`.

    Raises:
        GraphResumeError: When the same ``node_id`` appears in BOTH
            ``resume.replies`` and ``resume.rejected`` — these are
            mutually-exclusive intents.
    """
    pending = state.pending_interrupts.get(node_id)
    if pending is None:
        return
    if node_id in resume.replies and node_id in resume.rejected:
        raise GraphResumeError(
            f"node {node_id!r}: GraphResume.replies AND GraphResume.rejected both "
            f"contain an entry — these are mutually exclusive intents; supply only one."
        )
    # Graph-backed case has precedence: if this node has an inner-graph
    # snapshot, route to the graph-backed seed.
    if node_id in state.nested_graph_snapshots:
        _inject_nested_graph_resume(
            node_id=node_id,
            resume=resume,
            prepared_input=prepared_input,
            state=state,
        )
        return
    if isinstance(pending, NestedAgentInterrupt):
        _inject_nested_agent_resume(
            graph=graph,
            node_id=node_id,
            resume=resume,
            prepared_input=prepared_input,
            state=state,
        )
        return
    if node_id in resume.replies:
        prepared_input.metadata["__resume_reply__"] = resume.replies[node_id]
        state.pending_interrupts.pop(node_id, None)
    elif node_id in resume.rejected:
        prepared_input.metadata["__resume_reply__"] = resume.rejected[node_id]
        state.pending_interrupts.pop(node_id, None)
    # else: no reply supplied — node re-suspends naturally when
    # request_human_input raises InterruptException.


def _seed_interrupt_side_channel(
    *,
    graph: Graph[Any],
    ready_nodes: list[str],
    prepared_inputs: dict[str, ExecutableInput],
    resume: GraphResume | None,
    state: GraphState[Any],
) -> None:
    """Stage every ready node's interrupt side-channel + any resume payload.

    Combines two passes the BSP loop would otherwise run back-to-back:

    1. Seed two reserved metadata keys on each prepared input —
       ``__interrupt_node_id__`` (read by :func:`request_human_input`
       to build the parked :class:`Interrupt`) and
       ``__nested_agent_snapshots__`` (a reference to the same
       :attr:`GraphState.nested_agent_snapshots` dict the
       :meth:`AgentExecutable.invoke` catch path writes into — no copy,
       no marshalling).
    2. When ``resume`` is supplied, dispatch each node's resume payload
       kind-aware via :func:`_inject_resume_for_node`. Nodes missing
       from both ``resume.replies`` and ``resume.rejected`` re-suspend
       naturally on the next invocation.

    Raises:
        GraphResumeError: Propagated from
            :func:`_inject_nested_agent_resume` when the resume payload
            is incompatible with the parked interrupt.
    """
    for node_id in ready_nodes:
        meta = prepared_inputs[node_id].metadata
        meta["__interrupt_node_id__"] = node_id
        meta["__nested_agent_snapshots__"] = state.nested_agent_snapshots
        if resume is not None:
            _inject_resume_for_node(
                graph=graph,
                node_id=node_id,
                resume=resume,
                prepared_input=prepared_inputs[node_id],
                state=state,
            )


async def _dispatch_inner_graph_resume(
    *,
    outer_graph: Graph[Any],
    outer_node_id: str,
    inner_graph_state: GraphState[Any],
    outer_reply: Any,
    context: RunContext[Any],
    config: RunConfig,
) -> NodeResult[Any]:
    """Re-enter ``run_graph_loop`` to resume a paused inner graph.

    Forwards the outer caller's reply to the lexicographically-first inner
    pending interrupt (the one that surfaced as the outer interrupt). The
    reply is a :class:`NestedAgentReply` when the inner interrupt is a
    :class:`NestedAgentInterrupt`, or a plain value when the inner interrupt
    is a plain :class:`Interrupt` (lifted as a :class:`NestedGraphInterrupt`)
    — the inner graph's own ``_inject_resume_for_node`` routes each kind. If
    the inner still has pending
    interrupts after the resume (multi-interrupt fan-out case), lifts
    the next inner interrupt to a fresh outer
    :class:`InterruptException` so the outer node fires again for the
    next caller reply.

    Args:
        outer_graph: Compiled outer graph (caller of ``Graph.invoke``).
        outer_node_id: Outer node whose executable is a :class:`Graph`.
        inner_graph_state: The parked inner :class:`GraphState` (read
            from the outer state's
            ``nested_graph_snapshots[outer_node_id]`` by the seed
            phase, then staged on prepared_input.metadata).
        outer_reply: The :class:`NestedAgentReply` the caller composed
            against ``outer_node_id``.
        context: Outer run context, forwarded into the inner re-entry.
        config: Outer run config, forwarded.

    Returns:
        A :class:`NodeResult` when the inner graph completes.

    Raises:
        InterruptException: When the inner still has pending
            interrupts after the resume (multi-interrupt
            serialization). The fresh exception carries the next inner
            interrupt's metadata and stashes the post-resume inner
            ``GraphState`` on ``_nested_graph_state`` for re-parking by
            the outer catch.
        Exception: When the inner fails (re-raised as a wrapped
            :class:`RuntimeError`).
    """
    inner_graph = cast("Graph[Any]", outer_graph.get_node(outer_node_id).executable)

    # Forward the outer reply to the lexicographically-first inner
    # pending interrupt — that's the one that surfaced as the outer
    # interrupt by Graph.invoke's lift convention.
    inner_sorted = sorted(inner_graph_state.pending_interrupts.items())
    if len(inner_sorted) == 0:
        raise GraphResumeError(
            f"node {outer_node_id!r}: graph-backed resume staged but inner "
            f"GraphState has no pending interrupts — checkpoint inconsistency."
        )
    inner_node_id = inner_sorted[0][0]
    inner_resume = GraphResume(replies={inner_node_id: outer_reply})

    inner_result = await run_graph_loop(
        graph=inner_graph,
        user_prompt="",
        context=context,
        config=config,
        initial_state=inner_graph_state,
        resume=inner_resume,
    )

    if inner_result.status == GraphRunStatus.INTERRUPTED:
        # Multi-interrupt: another inner interrupt is still pending.
        # Lift the lexicographically-first one and raise so the outer
        # node fires again with the next outer reply.
        next_inner_state = inner_result.state
        if next_inner_state is None:
            raise RuntimeError(
                f"inner graph {inner_graph.id!r} returned INTERRUPTED with no state "
                "after resume — cannot lift multi-interrupt"
            )
        next_inner_id, next_inner_iv = sorted(next_inner_state.pending_interrupts.items())[0]
        next_outer_metadata = {
            **next_inner_iv.metadata,
            "inner_graph_id": inner_graph.id,
            "inner_node_id": next_inner_id,
        }
        next_outer_iv: Interrupt
        if isinstance(next_inner_iv, NestedAgentInterrupt):
            next_outer_iv = NestedAgentInterrupt(
                node_id="",
                question=next_inner_iv.question,
                metadata=next_outer_metadata,
                agent_name=next_inner_iv.agent_name,
                tool_call_ids=next_inner_iv.tool_call_ids,
            )
        else:
            # Inner plain Interrupt — re-lift as NestedGraphInterrupt (mirror of
            # Graph.invoke's cold-lift) so the re-parked checkpoint stays
            # resumable.
            next_outer_iv = NestedGraphInterrupt(
                node_id="",
                question=next_inner_iv.question,
                metadata=next_outer_metadata,
            )
        exc = InterruptException(next_outer_iv)
        exc._nested_graph_state = next_inner_state  # type: ignore[attr-defined]
        raise exc

    if inner_result.status == GraphRunStatus.FAILED:
        raise RuntimeError(f"inner graph {inner_graph.id!r} failed during resume: {inner_result.error}")

    if inner_result.status != GraphRunStatus.COMPLETED:
        # MAX_SUPERSTEPS / MAX_TOKENS / NO_READY_NODES after resume — the
        # inner graph hit a budget cap or deadlocked. Surface explicitly so
        # the outer node does not record a partial/empty success (mirror of
        # the cold-invoke guard in Graph.invoke).
        raise RuntimeError(
            f"inner graph {inner_graph.id!r} did not complete during resume: status={inner_result.status.value}."
        )

    # COMPLETED — return a NodeResult mirroring Graph.invoke's success path.
    final_output = inner_result.final_output
    final_text = final_output if isinstance(final_output, str) else None
    return NodeResult(
        output=final_output,
        new_items=list(inner_result.new_items),
        usage=inner_result.cumulative_usage,
        final_text=final_text,
        metadata={
            "adapter": "graph",
            "graph_id": inner_graph.id,
            "status": inner_result.status.value,
            "total_supersteps": inner_result.total_supersteps,
            "per_node_usage": dict(inner_result.per_node_usage),
        },
    )


async def _dispatch_nested_resume(
    *,
    graph: Graph[Any],
    node_id: str,
    nested_reply: NestedAgentReply,
    nested_snap: RunState,
    context: RunContext[Any],
    config: RunConfig,
    state: GraphState[Any],
    is_streaming: bool,
) -> NodeResult[Any]:
    """Route a staged nested-agent resume through :meth:`resume_from_snapshot`.

    Verifies the node's executable is still an :class:`AgentExecutable`
    (the seed-phase guard already rejected the mismatch case; this is a
    defensive re-check) and emits a warning when a streamed driver
    routes through the non-streaming resume bridge — interior
    ``agent_event`` items will not be forwarded as
    :class:`NodeStreamEvent` during the resumed turn.
    """
    executable = graph.get_node(node_id).executable
    if not isinstance(executable, AgentExecutable):
        raise GraphResumeError(
            f"node {node_id!r}: executable is "
            f"{type(executable).__name__}, not AgentExecutable — graph "
            f"shape changed between seed phase and node dispatch."
        )
    if is_streaming:
        logger.warning(
            "graph_loop: node=%s streaming run resuming via non-streaming "
            "resume_from_snapshot — interior agent events will not be forwarded "
            "as NodeStreamEvent during the resumed turn. Full streaming variant "
            "is a follow-up.",
            node_id,
        )
    return await executable.resume_from_snapshot(
        snapshot=nested_snap,
        reply=nested_reply,
        context=context,
        config=config,
        node_id=node_id,
        nested_agent_snapshots=state.nested_agent_snapshots,
    )


async def _call_error_handler(
    *,
    graph: Graph[Any],
    node_id: str,
    exc: BaseException,
) -> NodeResult[Any] | None:
    """Call the effective error handler for ``node_id`` and return its result.

    Resolution order:
    1. :attr:`~troopai.adk.graphs.node.GraphNode.on_error` on the node.
    2. :attr:`~troopai.adk.graphs.config.GraphConfig.default_error_handler`.
    3. ``None`` (caller re-raises original exception).

    The handler is awaited when its return value is awaitable so both sync
    and async handler callables work identically.  Any exception raised
    INSIDE the handler propagates directly — no suppression.

    Args:
        graph: The compiled :class:`~troopai.adk.graphs.graph.Graph`.
        node_id: Id of the node whose execution failed.
        exc: The exception raised after all retry attempts were exhausted.

    Returns:
        A fallback :class:`~troopai.adk.orchestration.executable.NodeResult`
        when the handler returns one, or ``None`` when no handler is
        configured or the handler itself returns ``None``.
    """
    node = graph.get_node(node_id)
    handler = node.on_error if node.on_error is not None else graph.config.default_error_handler
    if handler is None:
        return None
    result = handler(node_id, exc)
    if inspect.isawaitable(result):
        result = await result
    return result  # type: ignore[return-value]


async def _dispatch_node(
    *,
    graph: Graph[Any],
    node_id: str,
    prepared_input: ExecutableInput,
    context: RunContext[Any],
    config: RunConfig,
    state: GraphState[Any],
    node_runner: NodeRunner,
    is_streaming: bool,
) -> NodeResult[Any]:
    """Dispatch one node either via ``node_runner`` or its resume bridge.

    Pops the staged nested-agent reply + snapshot from the prepared
    input's metadata. When both are present, routes through
    :func:`_dispatch_nested_resume` so the resumed sub-agent
    re-applies decisions to the paused :class:`RunState` instead of
    starting over. When both are absent, delegates to ``node_runner``
    so the standard invoke / stream paths run unchanged. A half-formed
    pair (exactly one of the two reserved keys present) raises
    :class:`GraphResumeError` — the seed phase normally rejects such
    payloads earlier.

    Raises:
        GraphResumeError: When exactly one of the two reserved
            staging keys is present.
    """
    # Wrap the dispatch with a graph_node_span so each node attempt
    # gets its own OTel span, nested under the active graph_span. The
    # span is opened in this task's body (not the BSP loop's main
    # coroutine), so parallel siblings dispatched in the same
    # superstep are sibling spans under graph_span rather than nested
    # under each other.
    #
    # The node_status flag is stamped on the span's inner payload dict
    # right before close, so OTel attribute mapping surfaces
    # troopai.graph.node.status. InterruptException is a cooperative
    # pause (status="interrupted") rather than a failure — let it
    # propagate without set_error so the span doesn't get an error
    # attribute too.
    span = graph_node_span(graph_id=graph.id, node_name=node_id)
    span.start()
    # Detect resume from either the SP4 HITL channel
    # (``__resume_reply__``) or the SP5 nested-agent channel
    # (``__nested_agent_reply__``) and increment the per-node resume
    # count. The current count is stamped onto the span as
    # troopai.graph.node.resume_attempt; original firings leave the
    # counter at zero and don't surface the attribute.
    is_resume = (
        "__resume_reply__" in prepared_input.metadata
        or "__nested_agent_reply__" in prepared_input.metadata
        or "__nested_graph_reply__" in prepared_input.metadata
    )
    if is_resume:
        state.resume_counts[node_id] = state.resume_counts.get(node_id, 0) + 1
        cast(CustomSpanData, span.data).data["resume_attempt"] = state.resume_counts[node_id]
    node_status = "success"
    node_result: NodeResult[Any] | None = None
    exc_attempts: int | None = None
    try:
        # Inner-graph (PA4) resume takes precedence over inner-agent (SP5)
        # resume because the seed phase only stages one or the other. The
        # SNAPSHOT is the authoritative signal: the reply may be a
        # NestedAgentReply (inner agent) OR a plain value (inner plain Interrupt
        # lifted as a NestedGraphInterrupt), and a plain reply can legitimately
        # be None — so use key-presence and dispatch on the snapshot, not the
        # reply type.
        has_graph_snap = "__nested_graph_snapshot__" in prepared_input.metadata
        has_graph_reply = "__nested_graph_reply__" in prepared_input.metadata
        nested_graph_snap = prepared_input.metadata.pop("__nested_graph_snapshot__", None)
        nested_graph_reply = prepared_input.metadata.pop("__nested_graph_reply__", None)
        if has_graph_snap:
            if not has_graph_reply or not isinstance(nested_graph_snap, GraphState):
                raise GraphResumeError(
                    f"node {node_id!r}: __nested_graph_snapshot__ staged but the "
                    f"paired __nested_graph_reply__ is missing or the snapshot is not a "
                    f"GraphState — refusing to dispatch a half-formed inner-graph resume."
                )
            node_result = await _dispatch_inner_graph_resume(
                outer_graph=graph,
                outer_node_id=node_id,
                inner_graph_state=nested_graph_snap,
                outer_reply=nested_graph_reply,
                context=context,
                config=config,
            )
            return node_result
        if has_graph_reply:
            raise GraphResumeError(
                f"node {node_id!r}: __nested_graph_reply__ staged without "
                f"__nested_graph_snapshot__ — refusing to dispatch a half-formed inner-graph resume."
            )

        nested_reply = prepared_input.metadata.pop("__nested_agent_reply__", None)
        nested_snap = prepared_input.metadata.pop("__nested_agent_snapshot__", None)
        both_absent = nested_reply is None and nested_snap is None
        if both_absent:
            node_result = await node_runner(
                graph=graph,
                node_id=node_id,
                input=prepared_input,
                context=context,
                config=config,
            )
            return node_result
        if not isinstance(nested_reply, NestedAgentReply) or not isinstance(nested_snap, RunState):
            raise GraphResumeError(
                f"node {node_id!r}: __nested_agent_reply__ / "
                f"__nested_agent_snapshot__ were not both staged as the expected "
                f"types (reply={type(nested_reply).__name__}, snap={type(nested_snap).__name__}) — "
                f"refusing to dispatch a half-formed nested-agent resume."
            )
        node_result = await _dispatch_nested_resume(
            graph=graph,
            node_id=node_id,
            nested_reply=nested_reply,
            nested_snap=nested_snap,
            context=context,
            config=config,
            state=state,
            is_streaming=is_streaming,
        )
        return node_result
    except InterruptException:
        node_status = "interrupted"
        raise
    except asyncio.CancelledError:
        # Cooperative cancellation (fail-fast sibling cancel or an
        # immediate streamed cancel) is NOT a node failure — never route
        # it through the error handler. Stamp the span and re-raise so
        # the task is genuinely cancelled rather than completing with a
        # handler-supplied fallback. The reliability wrapper deliberately
        # lets CancelledError propagate (it catches Exception, not
        # BaseException); this clause preserves that contract here.
        node_status = "cancelled"
        raise
    except BaseException as exc:
        # Attempt per-node or graph-level error handler before marking
        # the node as failed.  The handler runs AFTER all retries are
        # exhausted (reliability wrapper already ran); exceptions raised
        # INSIDE the handler propagate immediately so no silent double-
        # failure can occur.  When the handler returns a fallback
        # NodeResult the loop treats the node as succeeded with that
        # result; when it returns None or no handler is configured the
        # original exception propagates.
        # Mark failed up front so the span is stamped correctly even if
        # the handler path exits via cancellation or another control-flow
        # exception; the fallback branch restores success explicitly.
        node_status = "failed"
        fallback: NodeResult[Any] | None = None
        try:
            fallback = await _call_error_handler(graph=graph, node_id=node_id, exc=exc)
        except BaseException:
            # Handler itself raised — propagate that exception (not the
            # original) so the double-failure is surfaced, not hidden.
            raise
        if fallback is not None:
            node_status = "success"
            node_result = fallback
            logger.info(
                "graph_loop: node_id=%s error handler returned fallback result; treating node as succeeded.",
                node_id,
            )
            return node_result
        span.set_error(str(exc), data={"type": type(exc).__name__})
        # NodeRetriesExhaustedError and GraphNodeTimeoutError both
        # carry an .attempts field with the final attempt count after
        # the retry loop bailed. Surface it via the same span
        # attribute as the success path uses, so trace queries for
        # attempts > 1 catch flaky-then-fail nodes as well as
        # flaky-then-succeed ones.
        candidate = getattr(exc, "attempts", None)
        if isinstance(candidate, int):
            exc_attempts = candidate
        raise
    finally:
        span_payload = cast(CustomSpanData, span.data).data
        span_payload["status"] = node_status
        # Resolution order: success-path metadata > exception .attempts
        # > default 1 (nested-agent resume, GraphResumeError, etc.).
        if node_result is not None:
            span_payload["attempts"] = node_result.metadata.pop("__attempts__", 1)
        elif exc_attempts is not None:
            span_payload["attempts"] = exc_attempts
        else:
            span_payload["attempts"] = 1
        span.finish()


async def _reconstruct_arrivals_from_state(
    *,
    graph: Graph[Any],
    node_id: str,
    state: GraphState[Any],
) -> tuple[list[NodeResult[Any]], list[str]]:
    """Rebuild a parked node's upstream arrivals from :attr:`GraphState.node_results`.

    Used when a node parked on an :class:`Interrupt` re-readies on resume
    but its :class:`JoinBarrier` is empty — the upstream(s) fired during
    the suspending superstep, the barrier was consumed there, and
    :func:`_seed_barriers_from_checkpoint` correctly skipped re-delivery
    because ``produced_at == versions_seen``. Without this reconstruction
    the resumed node would receive empty input, masking the parked
    interrupt's question for plain :func:`request_human_input` HITL.

    For each incoming edge of ``node_id``, looks up the upstream's
    persisted :class:`NodeResult` in ``state.node_results`` and includes
    it — but only when the edge's ``when`` predicate fires, exactly as
    :func:`_seed_barriers_from_checkpoint` decides live delivery. An edge
    whose predicate returns ``False`` (or raises) is excluded, so the
    resumed node never sees content from a branch that never fired.
    Upstreams with no persisted result are skipped (a defensive guard —
    by construction, a parked downstream must have seen its upstreams fire
    to reach the suspending superstep at all).

    Args:
        graph: The compiled :class:`Graph` (used for edge lookup).
        node_id: Id of the parked node whose input we are rebuilding.
        state: Current :class:`GraphState`.

    Returns:
        ``(results, sources)``: Parallel lists of upstream
        :class:`NodeResult` values and their source node ids, sorted by
        source id for determinism (matches :meth:`JoinBarrier.consume`).
    """
    upstream_results: dict[str, NodeResult[Any]] = {}
    for edge in graph.incoming_edges(node_id):
        upstream_id = edge.source
        upstream_result = state.node_results.get(upstream_id)
        if upstream_result is None:
            continue
        if edge.when is not None:
            try:
                fires = await _eval_edge_predicate(edge.when, upstream_result)
            except Exception as exc:
                logger.error(
                    "graph_loop: reconstruct edge predicate (%s -> %s) raised %s: %s; excluding upstream.",
                    upstream_id,
                    node_id,
                    type(exc).__name__,
                    exc,
                )
                fires = False
            if not fires:
                continue
        upstream_results[upstream_id] = upstream_result
    sorted_sources = sorted(upstream_results.keys())
    results = [upstream_results[s] for s in sorted_sources]
    return results, sorted_sources


async def _cancel_pending_node_tasks(
    tasks: Iterable[asyncio.Task[NodeResult[Any]]],
) -> None:
    """Cancel and await any node tasks still in flight.

    Called on every exit from a superstep's task-wait loop — normal
    completion (a no-op: all tasks already done), a fail-fast break, an
    operator-actionable :class:`GraphResumeError` re-raised from inside the
    loop, or an external :class:`asyncio.CancelledError` (the driver task
    cancelled from outside). :func:`asyncio.wait` never cancels the futures
    it waits on, so a cancelled driver would otherwise orphan its running
    node tasks — surfacing as "Task was destroyed but it is pending" /
    "Task exception was never retrieved". Cancelling first (synchronous,
    always happens) then gathering with ``return_exceptions=True``
    guarantees no leak even if the gather is itself interrupted.
    """
    leftover = [t for t in tasks if not t.done()]
    if len(leftover) == 0:
        return
    for t in leftover:
        t.cancel()
    await asyncio.gather(*leftover, return_exceptions=True)


def _resolve_edge_label(
    *,
    graph: Graph[Any],
    node_id: str,
    arrived_sources: list[str],
) -> str | None:
    """Return the label of the single triggering edge, or ``None``.

    Only meaningful when exactly one upstream delivered — this mirrors
    :attr:`ExecutableInput.from_node`, which is likewise ``None`` for a
    multi-input fan-in. When one source arrived, returns the
    :attr:`GraphEdge.label` of the ``source -> node_id`` edge that fired so
    the downstream adapter can route on which branch delivered (e.g.
    "approved" vs "rejected"). The label is left ``None`` on the entry
    node (no incoming edge) and on any fan-in.
    """
    if len(arrived_sources) != 1:
        return None
    src = arrived_sources[0]
    for edge in graph.incoming_edges(node_id):
        if edge.source == src:
            return edge.label
    return None


async def _prepare_superstep_inputs(
    *,
    graph: Graph[Any],
    ready_nodes: list[str],
    barriers: dict[str, JoinBarrier],
    state: GraphState[Any],
    user_prompt: UserPrompt,
) -> dict[str, ExecutableInput]:
    """Build each ready node's :class:`ExecutableInput` for this superstep.

    Entry (barrier-less) nodes bootstrap from ``user_prompt``. Barrier nodes
    consume their arrivals; when a parked-interrupt node re-readies with an
    empty barrier (its upstreams fired in the suspending superstep and the
    barrier was consumed there), the arrivals are rebuilt from
    ``state.node_results`` — honouring each edge's ``when`` predicate — so
    the resumed node receives the same merged content it would have on the
    first firing. Plain-Interrupt HITL nodes depend on this;
    NestedAgentInterrupt nodes go through the staged-snapshot dispatch path
    but the same code serves both. Consumed versions are marked and the
    firing edge's label is threaded onto the input.
    """
    prepared_inputs: dict[str, ExecutableInput] = {}
    for node_id in ready_nodes:
        node = graph.get_node(node_id)
        barrier = barriers.get(node_id)
        if barrier is None:
            prepared_inputs[node_id] = prepare_node_input(
                state=state,
                node=node,
                arrived_results=[],
                arrived_sources=[],
                strategy=graph.config.node_input,
                edge_label=None,
                initial_prompt=user_prompt,
            )
            continue
        arrived_results, arrived_sources = barrier.consume()
        if len(arrived_results) == 0 and node_id in state.pending_interrupts:
            arrived_results, arrived_sources = await _reconstruct_arrivals_from_state(
                graph=graph,
                node_id=node_id,
                state=state,
            )
        for src in arrived_sources:
            state.mark_version_consumed(node_id, src)
        prepared_inputs[node_id] = prepare_node_input(
            state=state,
            node=node,
            arrived_results=arrived_results,
            arrived_sources=arrived_sources,
            strategy=graph.config.node_input,
            edge_label=_resolve_edge_label(graph=graph, node_id=node_id, arrived_sources=arrived_sources),
            initial_prompt=None,
        )
    return prepared_inputs


async def _apply_completed_results(
    *,
    graph: Graph[Any],
    completed_results: dict[str, NodeResult[Any]],
    terminals_set: set[str],
    state: GraphState[Any],
    barriers: dict[str, JoinBarrier],
    registry: HookRegistry,
    context: RunContext[Any],
    graph_path: tuple[str, ...],
    emit: EventEmitter,
) -> None:
    """Record each completed node, fire its outgoing edges, track terminals.

    Applied in node-id-sorted order for deterministic reducer behaviour.
    Each result is recorded on ``state``, ``on_node_end`` fires, a
    :class:`NodeEndEvent` is emitted, terminal outputs are captured, and
    every outgoing edge evaluates its ``when`` predicate — recording a real
    arrival, a fail-closed failure (predicate raised), or a skip (predicate
    ``False``) on the target node's :class:`JoinBarrier`.
    """
    for nid in sorted(completed_results.keys()):
        node_result = completed_results[nid]
        state.record(nid, node_result)
        await registry.on_node_end(context, state, nid, node_result)
        end_ev = NodeEndEvent(
            graph_path=graph_path,
            node_id=nid,
            superstep=state.superstep,
            result=node_result,
        )
        logger.debug("graph_loop: emitted %s", end_ev["type"])
        await emit(end_ev)

        if nid in terminals_set:
            state.terminal_outputs[nid] = node_result.output

        for edge in graph.outgoing_edges(nid):
            should_fire = True
            predicate_failed = False
            if edge.when is not None:
                try:
                    should_fire = await _eval_edge_predicate(edge.when, node_result)
                except Exception as exc:
                    logger.error(
                        "graph_loop: edge predicate on (%s -> %s) raised %s: %s; "
                        "recording failure (downstream will not fire).",
                        edge.source,
                        edge.target,
                        type(exc).__name__,
                        exc,
                    )
                    should_fire = False
                    predicate_failed = True
            target_barrier = barriers.get(edge.target)
            if target_barrier is None:
                logger.warning(
                    "graph_loop: edge (%s -> %s) fired but target has no barrier. Ignoring.",
                    edge.source,
                    edge.target,
                )
                continue
            if should_fire:
                target_barrier.record(edge.source, node_result)
            elif predicate_failed:
                target_barrier.record_fail(edge.source)
            else:
                target_barrier.record_skip(edge.source)


async def _run_bsp_loop(
    *,
    graph: Graph[Any],
    user_prompt: UserPrompt,
    context: RunContext[Any],
    config: RunConfig,
    state: GraphState[Any],
    barriers: dict[str, JoinBarrier],
    registry: HookRegistry,
    graph_path: tuple[str, ...],
    emit: EventEmitter,
    node_runner: NodeRunner,
    task_tracker: TaskTracker | None = None,
    after_superstep_cancel: Callable[[], bool] | None = None,
    drain_cancel: Callable[[], bool] | None = None,
    resume: GraphResume | None = None,
    is_streaming: bool = False,
) -> tuple[GraphRunStatus, str | None]:
    """Execute the BSP superstep loop and return ``(status, error_msg)``.

    Parameterised by ``emit``, ``node_runner``, an optional ``task_tracker``
    (for streaming cancel support), an optional ``after_superstep_cancel``
    predicate (returns ``True`` when the caller requests cooperative stop at
    the next superstep boundary), an optional ``drain_cancel`` predicate
    (returns ``True`` when the caller requests drain-mode stop — in-flight
    tasks complete but no new superstep is started), an optional ``resume``
    payload that delivers human replies to pending interrupt nodes on the
    first superstep, and ``is_streaming`` (``True`` when driven from
    :func:`run_graph_loop_streamed`) so the dispatch layer can warn when a
    streaming run routes through the non-streaming
    :meth:`AgentExecutable.resume_from_snapshot` bridge.
    """
    first_pass = state.superstep == 0
    status: GraphRunStatus = GraphRunStatus.COMPLETED
    error_msg: str | None = None
    accumulated_errors: dict[str, str] = {}
    terminals_set = set(graph.terminals)
    # The currently-open superstep span, tracked at function scope so the
    # outer ``finally`` can close it on EVERY escape path — a user hook /
    # emit seam raising mid-superstep, a CancelledError, or a
    # GraphResumeError. Without this guard the span (and, on a real OTel
    # tracer, its attached context token) would leak. Cleared to ``None``
    # at each explicit close so the ``finally`` never double-finishes.
    open_superstep_span: Span[Any] | None = None
    # In-flight node tasks of the CURRENT superstep, tracked at function
    # scope so the outer ``finally`` can cancel them on EVERY escape path —
    # an external ``CancelledError`` (driver cancelled from outside), a
    # user hook / emit seam raising mid-superstep, or a GraphResumeError.
    # ``asyncio.wait`` never cancels the futures it waits on, so without
    # this a cancelled driver orphans its running node tasks. Reset to
    # empty once the superstep's wait loop has drained (all tasks done).
    open_node_tasks: dict[asyncio.Task[NodeResult[Any]], str] = {}

    try:
        while True:
            if state.superstep >= graph.config.max_supersteps:
                logger.warning(
                    "graph_loop: max_supersteps=%d reached; exiting.",
                    graph.config.max_supersteps,
                )
                status = GraphRunStatus.MAX_SUPERSTEPS
                break
            if (
                graph.config.max_total_tokens is not None
                and state.cumulative_usage.total_tokens >= graph.config.max_total_tokens
            ):
                logger.warning(
                    "graph_loop: max_total_tokens=%d reached (observed=%d); exiting.",
                    graph.config.max_total_tokens,
                    state.cumulative_usage.total_tokens,
                )
                status = GraphRunStatus.MAX_TOKENS
                break

            if after_superstep_cancel is not None and after_superstep_cancel():
                logger.info("graph_loop: cooperative after-superstep cancel requested; exiting.")
                status = GraphRunStatus.COMPLETED if len(state.terminal_outputs) > 0 else GraphRunStatus.NO_READY_NODES
                break

            if drain_cancel is not None and drain_cancel():
                # Drain mode: in-flight tasks from the CURRENT superstep have
                # already completed (the task-wait loop ran to completion).
                # Do NOT start a new superstep — checkpoint and exit cleanly.
                logger.info("graph_loop: drain cancel requested; exiting after draining in-flight nodes.")
                status = GraphRunStatus.COMPLETED if len(state.terminal_outputs) > 0 else GraphRunStatus.NO_READY_NODES
                break

            ready_nodes: list[str]
            if first_pass:
                ready_nodes = [graph.entry]
                first_pass = False
            else:
                ready_nodes = sorted(nid for nid, b in barriers.items() if b.is_ready())
                # Pending-interrupt nodes need explicit re-readying on the
                # first resume superstep — their barriers were consumed in
                # the suspending superstep and the seed phase correctly
                # skips re-delivery (``produced_at == versions_seen``).
                # Without this, a fan-out target parked on an interrupt
                # would silently drop off the schedule, leaving the loop
                # to exit NO_READY_NODES. Barrier-less parked nodes (entry
                # nodes that interrupted on their first firing) are added
                # for the same reason — they have no barrier to satisfy.
                for nid in sorted(state.pending_interrupts.keys()):
                    if nid in ready_nodes:
                        continue
                    has_reply = resume is not None and (nid in resume.replies or nid in resume.rejected)
                    has_barrier = nid in barriers
                    # A barrier-less parked node always re-readies so a
                    # plain Interrupt re-raises naturally via
                    # ``request_human_input``; a barrier-having parked
                    # node only re-readies when a reply/rejection is
                    # staged, so unanswered fan-out interrupts stay
                    # parked rather than re-executing their underlying
                    # body with stale or empty upstream content.
                    if not has_barrier or has_reply:
                        ready_nodes.append(nid)

            if len(ready_nodes) == 0:
                if len(state.pending_interrupts) > 0:
                    # Unanswered parked interrupts kept the loop from
                    # making progress this superstep — surface as
                    # INTERRUPTED so the caller can supply replies and
                    # retry. A previously-fired terminal still appears
                    # in ``terminal_outputs``; the resume status takes
                    # precedence so the run is not declared COMPLETED
                    # while a decision is still outstanding.
                    state.status = "interrupted"
                    status = GraphRunStatus.INTERRUPTED
                elif len(state.terminal_outputs) > 0:
                    status = GraphRunStatus.COMPLETED
                else:
                    logger.warning(
                        "graph_loop: no ready nodes and no terminal fired (expected=%s). Exiting NO_READY_NODES.",
                        sorted(terminals_set),
                    )
                    if len(accumulated_errors) > 0:
                        # At least one node failed; surface which ones so
                        # callers can distinguish a deadlocked graph from
                        # a predicate-routing exit.
                        status = GraphRunStatus.FAILED
                        error_msg = f"nodes failed: {sorted(accumulated_errors.keys())}"
                        state.status = "failed"
                        state.error = error_msg
                    else:
                        status = GraphRunStatus.NO_READY_NODES
                break

            state.superstep += 1
            ready_tuple = tuple(ready_nodes)

            # Open a per-superstep span so the BSP structure is visible
            # in traces (one graph span → N superstep spans → M node
            # spans). The span closes immediately after the matching
            # SuperstepEndEvent emit below, so a single iteration of
            # this loop is fully bracketed.
            superstep_tracing_span = graph_superstep_span(
                graph_id=graph.id,
                index=state.superstep,
                ready_nodes=ready_tuple,
            )
            superstep_tracing_span.start()
            open_superstep_span = superstep_tracing_span

            superstep_start_event = SuperstepStartEvent(
                graph_path=graph_path,
                superstep=state.superstep,
                ready_nodes=ready_tuple,
            )
            logger.debug("graph_loop: superstep=%d ready=%s", state.superstep, ready_tuple)
            await emit(superstep_start_event)
            await registry.on_superstep_start(context, state, ready_tuple)

            prepared_inputs = await _prepare_superstep_inputs(
                graph=graph,
                ready_nodes=ready_nodes,
                barriers=barriers,
                state=state,
                user_prompt=user_prompt,
            )

            _seed_interrupt_side_channel(
                graph=graph,
                ready_nodes=ready_nodes,
                prepared_inputs=prepared_inputs,
                resume=resume,
                state=state,
            )

            for node_id in ready_nodes:
                await registry.on_node_start(context, state, node_id, prepared_inputs[node_id])
                start_ev = NodeStartEvent(
                    graph_path=graph_path,
                    node_id=node_id,
                    superstep=state.superstep,
                    from_nodes=prepared_inputs[node_id].from_nodes,
                    edge_label=prepared_inputs[node_id].edge_label,
                    input=prepared_inputs[node_id],
                )
                logger.debug("graph_loop: emitted %s", start_ev["type"])
                await emit(start_ev)

            tasks: dict[asyncio.Task[NodeResult[Any]], str] = {}
            for node_id in ready_nodes:
                task = asyncio.create_task(
                    _dispatch_node(
                        graph=graph,
                        node_id=node_id,
                        prepared_input=prepared_inputs[node_id],
                        context=context,
                        config=config,
                        state=state,
                        node_runner=node_runner,
                        is_streaming=is_streaming,
                    ),
                    name=f"graph:{graph.id}:node:{node_id}:ss{state.superstep}",
                )
                tasks[task] = node_id
                if task_tracker is not None:
                    task_tracker.register_node_task(task)
            # Publish this superstep's tasks for the outer ``finally`` so a
            # cancel mid-wait cancels them instead of orphaning them.
            open_node_tasks = tasks

            completed_results: dict[str, NodeResult[Any]] = {}
            errored: dict[str, BaseException] = {}
            superstep_interrupted: dict[str, Interrupt] = {}

            pending = set(tasks.keys())
            fail_fast_triggered = False
            while len(pending) > 0:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    nid = tasks[task]
                    if task_tracker is not None:
                        task_tracker.discard_node_task(task)
                    exc = task.exception()
                    if exc is not None:
                        if isinstance(exc, InterruptException):
                            # Cooperative pause — not a failure. Collect for
                            # end-of-superstep handling; siblings still run.
                            # Rewrite the interrupt's node_id field to the
                            # outer scope so consumers see consistent ids
                            # regardless of whether the lift came from an
                            # AgentExecutable (empty inner id) or
                            # Graph.invoke (inner-graph metadata).
                            outer_iv = dataclasses.replace(exc.interrupt, node_id=nid)
                            superstep_interrupted[nid] = outer_iv
                            # Graph.invoke stashes the inner GraphState on
                            # exc._nested_graph_state when a graph-backed
                            # node suspends; park it for resume. Agent-
                            # backed snapshots populate
                            # nested_agent_snapshots via a separate
                            # adapters.py seed path.
                            inner_graph_state = getattr(exc, "_nested_graph_state", None)
                            if isinstance(inner_graph_state, GraphState):
                                state.nested_graph_snapshots[nid] = inner_graph_state
                            logger.info(
                                "graph_loop: node_id=%s raised InterruptException; suspending.",
                                nid,
                            )
                            await registry.on_node_interrupt(context, state, nid, outer_iv)
                            continue
                        if isinstance(exc, GraphResumeError):
                            # Operator-actionable mismatch — propagate so the
                            # outer ``except GraphResumeError: raise`` surfaces
                            # the error verbatim rather than absorbing it into
                            # ``state.error`` as a generic FAILED.
                            for p in pending:
                                p.cancel()
                            if len(pending) > 0:
                                await asyncio.gather(*pending, return_exceptions=True)
                            superstep_tracing_span.finish()
                            open_superstep_span = None
                            raise exc
                        errored[nid] = exc
                        logger.error(
                            "graph_loop: node_id=%s raised %s: %s",
                            nid,
                            type(exc).__name__,
                            exc,
                        )
                        await registry.on_node_error(context, state, nid, exc)
                        err_ev = NodeErrorEvent(
                            graph_path=graph_path,
                            node_id=nid,
                            superstep=state.superstep,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                        logger.debug("graph_loop: emitted %s", err_ev["type"])
                        await emit(err_ev)
                        if graph.config.fail_fast:
                            # Defer the sibling cancel until the WHOLE ``done``
                            # batch is drained. ``asyncio.wait`` can surface
                            # several tasks in ``done`` within one event-loop
                            # step; breaking here would silently drop a peer
                            # that already succeeded (lost result) or parked on
                            # an InterruptException (lost suspend state) when it
                            # happens to be visited after the errored task.
                            fail_fast_triggered = True
                    else:
                        completed_results[nid] = task.result()
                if fail_fast_triggered:
                    for p in pending:
                        p.cancel()
                    if len(pending) > 0:
                        await asyncio.gather(*pending, return_exceptions=True)
                    break
            # Wait loop drained — every task is done. Clear the tracker so a
            # cancel during the post-wait phase (edge eval / hooks / emit)
            # does not re-cancel already-finished tasks.
            open_node_tasks = {}

            await _apply_completed_results(
                graph=graph,
                completed_results=completed_results,
                terminals_set=terminals_set,
                state=state,
                barriers=barriers,
                registry=registry,
                context=context,
                graph_path=graph_path,
                emit=emit,
            )

            # Propagate errored-node failures to their downstream AND-join
            # barriers so the join knows it cannot fire (fail-closed).
            # Without this, barriers waiting on an errored upstream have
            # empty arrivals and empty failed/skipped sets, so is_ready()
            # never returns True and the loop exits NO_READY_NODES silently.
            for nid in sorted(errored.keys()):
                accumulated_errors[nid] = f"{type(errored[nid]).__name__}: {errored[nid]}"
                for edge in graph.outgoing_edges(nid):
                    target_barrier = barriers.get(edge.target)
                    if target_barrier is not None:
                        target_barrier.record_fail(edge.source)

            step_items = []
            for nid in sorted(completed_results.keys()):
                step_items.extend(completed_results[nid].new_items)

            await registry.on_superstep_end(
                context,
                state,
                tuple(sorted(completed_results.keys())),
                step_items,
            )
            end_step_ev = SuperstepEndEvent(
                graph_path=graph_path,
                superstep=state.superstep,
                fired_nodes=tuple(sorted(completed_results.keys())),
                errored_nodes=tuple(sorted(errored.keys())),
            )
            logger.debug(
                "graph_loop: emitted %s (fired=%s errored=%s)",
                end_step_ev["type"],
                end_step_ev["fired_nodes"],
                end_step_ev["errored_nodes"],
            )
            await emit(end_step_ev)

            # Stamp the fired-nodes set on the superstep span's inner
            # payload dict so the OTel attribute flattener picks it up
            # at finish-time. CustomSpanData.data is a mutable dict;
            # mutating it preserves the same object the flattener's
            # closure captured at OTelTracer.custom_span() time. The
            # cast surfaces the runtime type — graph_superstep_span()
            # advertises Span[GraphSuperstepSpanData] but routes through
            # custom_span, so the live instance carries CustomSpanData.
            cast(CustomSpanData, superstep_tracing_span.data).data["fired_nodes"] = sorted(completed_results.keys())
            superstep_tracing_span.finish()
            open_superstep_span = None

            if len(superstep_interrupted) > 0:
                # A cooperative interrupt pauses the run and is never dropped.
                # When a sibling ALSO hard-failed in the SAME superstep, the
                # run still pauses for the human decision, but the failure is
                # persisted onto the state rather than being silently
                # discarded — the fail-fast error is evaluated alongside the
                # interrupt instead of the interrupt short-circuiting it.
                state.pending_interrupts.update(superstep_interrupted)
                state.status = "interrupted"
                status = GraphRunStatus.INTERRUPTED
                if len(errored) > 0:
                    state.error = f"nodes failed alongside interrupt: {sorted(errored.keys())}"
                break

            if len(errored) > 0 and graph.config.fail_fast:
                first_err = next(iter(errored.values()))
                status = GraphRunStatus.FAILED
                error_msg = f"{type(first_err).__name__}: {first_err}"
                state.status = "failed"
                state.error = error_msg
                break

            # A partial resume can complete one parked node while leaving
            # siblings unanswered. Surface as INTERRUPTED so the caller
            # can supply the missing replies and retry — declaring
            # COMPLETED here would lose the outstanding decision.
            if len(state.pending_interrupts) > 0:
                state.status = "interrupted"
                status = GraphRunStatus.INTERRUPTED
                break

            if len(state.terminal_outputs) > 0:
                status = GraphRunStatus.COMPLETED
                break

    except asyncio.CancelledError:
        logger.info("graph_loop: cancelled externally.")
        raise
    except InterruptException as exc:
        # A lifecycle hook (e.g. a custom HITL checkpointer) raised a
        # cooperative pause rather than a node body. Surface it as
        # INTERRUPTED — the same terminal state a node-level
        # request_human_input produces — instead of letting the generic
        # handler below record it as a FAILED run. The interrupt names the
        # node it belongs to; park it so the caller can resume.
        iv = exc.interrupt
        state.pending_interrupts[iv.node_id] = iv
        state.status = "interrupted"
        status = GraphRunStatus.INTERRUPTED
        logger.info(
            "graph_loop: lifecycle hook raised InterruptException on node=%s; suspending run.",
            iv.node_id,
        )
    except GraphResumeError:
        # Operator-actionable: the resume payload is incompatible with the
        # parked interrupt. Propagate so the caller can fix the payload and
        # retry against the same checkpoint — folding it into FAILED status
        # would hide the mismatch behind a generic ``state.error`` string.
        raise
    except Exception as exc:
        logger.exception("graph_loop: unexpected exception: %s", exc)
        status = GraphRunStatus.FAILED
        error_msg = f"{type(exc).__name__}: {exc}"
        state.status = "failed"
        state.error = error_msg
    finally:
        # Cancel any node tasks still running from the current superstep. On
        # an external cancel the driver's ``asyncio.wait`` raises without
        # cancelling the tasks it awaited, so they would otherwise keep
        # executing detached; cancelling here (a no-op once the wait loop has
        # drained) guarantees no orphaned node task survives loop teardown.
        await _cancel_pending_node_tasks(open_node_tasks.keys())
        # Close any superstep span still open because the body exited via a
        # user hook / emit raising, a CancelledError, or an early break —
        # the per-superstep span has no with/finally of its own, unlike the
        # per-node span and the enclosing graph span.
        if open_superstep_span is not None:
            open_superstep_span.finish()

    return status, error_msg


async def run_graph_loop(
    *,
    graph: Graph[Any],
    user_prompt: UserPrompt,
    context: RunContext[Any],
    config: RunConfig,
    hooks: list[GraphHooks[Any] | HookProvider] | None = None,
    thread_id: str | None = None,
    initial_state: GraphState[Any] | None = None,
    emit: EventEmitter = _noop_emitter,
    resume: GraphResume | None = None,
) -> GraphRunResult[Any]:
    """Execute a :class:`Graph` end-to-end.

    The driver is BSP-structured:

    1. Build the :class:`HookRegistry`, fire :meth:`on_graph_start`.
    2. Initialise (or restore) :class:`GraphState` and
       :class:`JoinBarrier`\\ s.
    3. Superstep loop until (a) every terminal has fired at least once,
       (b) ``max_supersteps`` hit, (c) ``max_total_tokens`` hit, or
       (d) no more ready nodes and no terminals fired
       (``NO_READY_NODES``).
    4. Per superstep: compute ready nodes → prepare inputs → launch
       tasks → wait (fail-fast aware) → apply results sorted by id →
       fire outgoing edges → check terminals.
    5. On exit, populate ``final_output`` and fire
       :meth:`on_graph_end` / build :class:`GraphRunResult`.

    Args:
        graph: The compiled :class:`Graph` to run.
        user_prompt: Initial input for the entry node's first firing.
        context: Shared :class:`RunContext`. Usage from nested
            executables accumulates here.
        config: :class:`RunConfig` — threaded to every nested invoke.
        hooks: Optional list of :class:`GraphHooks` or
            :class:`HookProvider` (e.g. a :class:`Checkpointer`) to
            attach. Combined with :attr:`Graph.hooks`.
        thread_id: Opt-in identifier for checkpointing. When provided
            and a :class:`Checkpointer` is attached, per-node saves
            fire. When ``None`` and a checkpointer is attached, the
            loop auto-generates a ``thread-XXXX`` id; the checkpointer
            skips writes when state.thread_id is ``None``.
        initial_state: Optional pre-existing :class:`GraphState` from a
            restored checkpoint. When supplied, the loop skips
            initialisation and resumes from ``state.superstep + 1``.
        resume: Optional human replies for pending interrupt nodes.
            When supplied, the loop injects each reply into the matching
            node's :class:`ExecutableInput` metadata on the first
            superstep where that node is ready, then clears the consumed
            entry from :attr:`GraphState.pending_interrupts`. Nodes
            with no matching reply in ``resume`` re-suspend naturally.

    Returns:
        A :class:`GraphRunResult` with terminal output, status, and
        per-node attribution.

    Raises:
        ValueError: On malformed graph state (e.g. an unknown ready node).
    """
    registry, state = _setup_graph_run(
        graph=graph,
        hooks=hooks,
        thread_id=thread_id,
        initial_state=initial_state,
    )

    barriers: dict[str, JoinBarrier] = _build_join_barriers(graph)
    if initial_state is not None:
        await _seed_barriers_from_checkpoint(graph=graph, state=state, barriers=barriers)

    graph_path: tuple[str, ...] = (graph.id,)

    # Open the root graph_span around the whole run so per-node spans
    # opened inside _dispatch_node nest as children. The span closes
    # via the with-statement on every exit path including exceptions;
    # the __exit__ default records any exception via set_error.
    with graph_span(graph_id=graph.id, entry=graph.entry) as graph_tracing_span:
        await registry.on_graph_start(context, state)
        start_event = GraphStartEvent(
            graph_path=graph_path,
            graph_id=graph.id,
            description=graph.description,
            entry_node=graph.entry,
            terminal_nodes=tuple(sorted(graph.terminals)),
        )
        logger.debug("graph_loop: emitted %s", start_event["type"])
        await emit(start_event)

        status, error_msg = await _run_bsp_loop(
            graph=graph,
            user_prompt=user_prompt,
            context=context,
            config=config,
            state=state,
            barriers=barriers,
            registry=registry,
            graph_path=graph_path,
            emit=emit,
            node_runner=_invoke_node,
            resume=resume,
        )

        final_output = _assemble_final_output(state=state, terminals_set=set(graph.terminals), status=status)

        # Stamp terminal status + final superstep count on the graph
        # span's inner payload before the with-statement closes it, so
        # the OTel attribute mapping surfaces troopai.graph.status and
        # troopai.graph.supersteps_total. Same cast pattern as the
        # superstep span — runtime data is CustomSpanData despite the
        # factory's typed-handle return.
        graph_span_payload = cast(CustomSpanData, graph_tracing_span.data).data
        graph_span_payload["status"] = status.value
        graph_span_payload["supersteps_total"] = state.superstep

        await registry.on_graph_end(context, state, status, final_output)
        end_event = GraphEndEvent(
            graph_path=graph_path,
            graph_id=graph.id,
            status=status,
            final_output=final_output,
            total_supersteps=state.superstep,
        )
        logger.debug(
            "graph_loop: emitted %s status=%s total_supersteps=%d",
            end_event["type"],
            status.value,
            state.superstep,
        )
        await emit(end_event)

        pending_interrupts_tuple = tuple(state.pending_interrupts.values())
        return GraphRunResult(
            final_output=final_output,
            status=status,
            user_prompt=user_prompt,
            new_items=list(state.all_items),
            state=state,
            node_results=dict(state.node_results),
            context=context,
            per_node_usage=dict(state.per_node_usage),
            cumulative_usage=state.cumulative_usage,
            total_supersteps=state.superstep,
            error=error_msg,
            interrupts=pending_interrupts_tuple,
            structured_interrupts=StructuredInterrupts.from_interrupts(pending_interrupts_tuple),
        )


async def run_graph_loop_streamed(
    *,
    graph: Graph[Any],
    user_prompt: UserPrompt,
    context: RunContext[Any],
    config: RunConfig,
    result: GraphRunResultStreaming[Any],
    hooks: list[GraphHooks[Any] | HookProvider] | None = None,
    thread_id: str | None = None,
    initial_state: GraphState[Any] | None = None,
    resume: GraphResume | None = None,
) -> None:
    """Streaming twin of :func:`run_graph_loop`.

    Drives the same BSP superstep loop but forwards every
    :class:`GraphStreamEvent` to ``result`` via
    :meth:`GraphRunResultStreaming.put_event`. Interior events from
    :meth:`Executable.stream_async` are wrapped in
    :class:`NodeStreamEvent` before forwarding. On completion, terminal
    fields on ``result`` are populated and
    :meth:`GraphRunResultStreaming.complete` is called to unblock
    consumers of :meth:`GraphRunResultStreaming.stream_events`.

    ``result.cancel(mode="after_superstep")`` requests cooperative
    termination at the next superstep boundary; the driver checks
    :attr:`GraphRunResultStreaming.cancel_mode` before each superstep.
    ``result.cancel(mode="immediate")`` cancels the driver task and
    all in-flight node tasks directly (tracked via
    :meth:`~GraphRunResultStreaming.register_node_task` /
    :meth:`~GraphRunResultStreaming.discard_node_task`).

    Args:
        graph: The compiled :class:`Graph` to run.
        user_prompt: Initial input for the entry node.
        context: Shared :class:`RunContext`.
        config: :class:`RunConfig` threaded to every nested invoke.
        result: Pre-created :class:`GraphRunResultStreaming` that the
            caller has already handed to the consumer.
        hooks: Optional :class:`GraphHooks` / :class:`HookProvider` list.
        thread_id: Opt-in checkpointing thread identifier.
        initial_state: Optional restored :class:`GraphState` for resume.
        resume: Optional human replies for pending interrupt nodes.
            Threaded to :func:`_run_bsp_loop` which injects each reply
            into the matching node's :class:`ExecutableInput` metadata.
    """
    registry, state = _setup_graph_run(
        graph=graph,
        hooks=hooks,
        thread_id=thread_id,
        initial_state=initial_state,
    )

    barriers: dict[str, JoinBarrier] = _build_join_barriers(graph)
    if initial_state is not None:
        await _seed_barriers_from_checkpoint(graph=graph, state=state, barriers=barriers)

    graph_path: tuple[str, ...] = (graph.id,)

    async def streamed_node_runner(
        *,
        graph: Graph[Any],
        node_id: str,
        input: ExecutableInput,
        context: RunContext[Any],
        config: RunConfig,
    ) -> NodeResult[Any]:
        return await _stream_node(
            graph=graph,
            node_id=node_id,
            input=input,
            context=context,
            config=config,
            graph_path=graph_path,
            result=result,
        )

    def after_superstep_cancel() -> bool:
        return result.cancel_mode == CancelMode.AFTER_SUPERSTEP

    def drain_cancel() -> bool:
        return result.cancel_mode == CancelMode.DRAIN

    try:
        # Open the root graph_span around the whole run; per-node spans
        # opened inside _dispatch_node will nest as children. See the
        # non-streaming variant above for rationale.
        with graph_span(graph_id=graph.id, entry=graph.entry) as graph_tracing_span:
            await registry.on_graph_start(context, state)
            start_event = GraphStartEvent(
                graph_path=graph_path,
                graph_id=graph.id,
                description=graph.description,
                entry_node=graph.entry,
                terminal_nodes=tuple(sorted(graph.terminals)),
            )
            logger.debug("graph_loop: streamed emitted %s", start_event["type"])
            await result.put_event(start_event)

            status, error_msg = await _run_bsp_loop(
                graph=graph,
                user_prompt=user_prompt,
                context=context,
                config=config,
                state=state,
                barriers=barriers,
                registry=registry,
                graph_path=graph_path,
                emit=result.put_event,
                node_runner=streamed_node_runner,
                task_tracker=result,
                after_superstep_cancel=after_superstep_cancel,
                drain_cancel=drain_cancel,
                resume=resume,
                is_streaming=True,
            )

            final_output = _assemble_final_output(state=state, terminals_set=set(graph.terminals), status=status)

            if status == GraphRunStatus.INTERRUPTED:
                for nid, iv in sorted(state.pending_interrupts.items()):
                    await result.put_event(NodeInterruptEvent(graph_path=graph_path, node_id=nid, interrupt=iv))
                pending_tuple = tuple(state.pending_interrupts.values())
                result.interrupts = pending_tuple
                result.structured_interrupts = StructuredInterrupts.from_interrupts(pending_tuple)

            result.final_output = final_output
            result.status = status
            result.error = error_msg
            result.user_prompt = user_prompt
            result.new_items = list(state.all_items)
            result.state = state
            result.node_results = dict(state.node_results)
            result.context = context
            result.per_node_usage = dict(state.per_node_usage)
            result.cumulative_usage = state.cumulative_usage
            result.total_supersteps = state.superstep

            # Mirror the non-streaming variant: stamp terminal status +
            # supersteps_total on the graph span before close so OTel
            # attribute mapping surfaces them.
            streamed_graph_payload = cast(CustomSpanData, graph_tracing_span.data).data
            streamed_graph_payload["status"] = status.value
            streamed_graph_payload["supersteps_total"] = state.superstep

            await registry.on_graph_end(context, state, status, final_output)
            end_event = GraphEndEvent(
                graph_path=graph_path,
                graph_id=graph.id,
                status=status,
                final_output=final_output,
                total_supersteps=state.superstep,
            )
            logger.debug(
                "graph_loop: streamed emitted %s status=%s total_supersteps=%d",
                end_event["type"],
                status.value,
                state.superstep,
            )
            await result.put_event(end_event)

    except Exception as exc:
        result.set_exception(exc)
        raise
    finally:
        await result.complete()


def _has_checkpointer(
    hooks: list[GraphHooks[Any] | HookProvider] | None,
) -> bool:
    """Return ``True`` if any attached hook is a :class:`HookProvider`.

    Used to decide whether to auto-generate a ``thread_id`` when one
    was not supplied — the assumption being that if the caller
    attached a :class:`Checkpointer` they want per-node persistence
    even without explicitly naming the thread.
    """
    if hooks is None:
        return False
    return any(isinstance(h, HookProvider) for h in hooks)


__all__ = ["run_graph_loop", "run_graph_loop_streamed"]

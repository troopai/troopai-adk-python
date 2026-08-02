"""End-to-end BSP-loop tests for the nested-agent deferral bridge.

The BSP loop seeds ``__nested_agent_snapshots__`` into every prepared
input via :class:`ExecutableInput.metadata`, and on resume routes a
node parked on a :class:`NestedAgentInterrupt` to
:meth:`AgentExecutable.resume_from_snapshot` instead of re-invoking the
node from scratch. Mocks at the :class:`AgentExecutable` level so the
tests don't drive a real :meth:`Runner.arun` cycle.
"""

from __future__ import annotations

from typing import Any

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs.adapters import AgentExecutable
from troopai.adk.graphs.checkpointer import GraphCheckpoint
from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.interrupt import (
    GraphResume,
    GraphResumeError,
    NestedAgentApproval,
    NestedAgentInterrupt,
    NestedAgentReply,
)
from troopai.adk.graphs.result import GraphRunStatus
from troopai.adk.graphs.state import GraphState
from troopai.adk.orchestration.executable import ExecutableInput, NodeResult
from troopai.adk.run.runner import Runner
from troopai.adk.run.state import RunState
from troopai.adk.tools.deferred_tool import DeferredToolCall, DeferredToolRequests
from troopai.adk.types.tokens.llm_usage import LLMUsage


def _deferred_call(call_id: str) -> DeferredToolCall:
    """Build a placeholder :class:`DeferredToolCall` for snapshot fixtures."""
    return DeferredToolCall(
        tool_call_id=call_id,
        tool_name="t",
        tool_arguments={},
        raw_arguments="{}",
    )


def _trivial_graph() -> Graph[Any]:
    """Build a one-node graph whose node id is ``"n"`` and whose
    executable is an :class:`AgentExecutable`. The agent's body is
    never reached — every test in this module monkeypatches
    :meth:`AgentExecutable.invoke` (and sometimes
    :meth:`AgentExecutable.resume_from_snapshot`) at the class level.
    """
    return (
        Graph.new("test-bsp-nested")
        .node("n", Agent(name="planner", system_prompt="x"))
        .entry("n")
        .terminal("n")
        .compile()
    )


def _node_result(agent_name: str, output: str) -> NodeResult[Any]:
    """Build a minimal terminal :class:`NodeResult` for the fake
    ``invoke`` / ``resume_from_snapshot`` paths."""
    return NodeResult(
        output=output,
        new_items=[],
        usage=LLMUsage(),
        final_text=output,
        metadata={"adapter": "agent", "agent_name": agent_name, "last_agent_name": agent_name},
    )


async def test_bsp_loop_seeds_nested_agent_snapshots_side_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prepared input passed to every :class:`AgentExecutable` node
    must carry the ``__nested_agent_snapshots__`` side-channel — a
    reference to the loop's :attr:`GraphState.nested_agent_snapshots`
    dict. This is the channel :meth:`AgentExecutable.invoke` deposits
    into when the inner agent defers."""
    observed_channels: list[Any] = []

    async def probing_invoke(
        self: AgentExecutable[Any],
        input: ExecutableInput,
        context: Any,
        config: Any,
    ) -> NodeResult[Any]:
        observed_channels.append(input.metadata.get("__nested_agent_snapshots__"))
        return _node_result(self.agent.name, "done")

    monkeypatch.setattr(AgentExecutable, "invoke", probing_invoke)

    graph = _trivial_graph()
    result = await Runner.arun_graph(graph, "go")

    assert result.status == GraphRunStatus.COMPLETED
    assert len(observed_channels) == 1
    channel = observed_channels[0]
    assert isinstance(channel, dict)
    # Identity check — the BSP loop must NOT copy the dict; the
    # executable's catch path mutates the dict in place.
    assert result.state is not None
    assert channel is result.state.nested_agent_snapshots


async def test_bsp_loop_routes_resume_to_resume_from_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a checkpoint has a :class:`NestedAgentInterrupt` on node ``"n"``
    and ``GraphResume.replies["n"]`` is a :class:`NestedAgentReply`, the
    loop dispatches :meth:`AgentExecutable.resume_from_snapshot` — NOT
    :meth:`AgentExecutable.invoke`. The ``nested_agent_snapshots`` kwarg
    handed to the resumed executable must be a dict (the loop's
    in-memory side-channel) and the entry for ``"n"`` must have been
    popped by the staging helper before dispatch — the snapshot lives
    only in ``snapshot``, never both."""
    invoked: list[str] = []
    resumed: list[str] = []
    observed_kwarg_dicts: list[dict[str, RunState]] = []

    async def probing_invoke(
        self: AgentExecutable[Any],
        input: ExecutableInput,
        context: Any,
        config: Any,
    ) -> NodeResult[Any]:
        invoked.append(self.agent.name)
        return _node_result(self.agent.name, "invoke-output")

    async def probing_resume(
        self: AgentExecutable[Any],
        *,
        snapshot: RunState,
        reply: NestedAgentReply,
        context: Any,
        config: Any,
        node_id: str,
        nested_agent_snapshots: dict[str, RunState],
    ) -> NodeResult[Any]:
        resumed.append(node_id)
        observed_kwarg_dicts.append(nested_agent_snapshots)
        return _node_result(self.agent.name, "resumed-output")

    monkeypatch.setattr(AgentExecutable, "invoke", probing_invoke)
    monkeypatch.setattr(AgentExecutable, "resume_from_snapshot", probing_resume)

    graph = _trivial_graph()
    checkpointer = InMemoryCheckpointer()
    thread_id = "t"

    paused_state = GraphState[None](graph=graph, thread_id=thread_id)
    paused_state.nested_agent_snapshots["n"] = RunState(
        deferred_tool_requests=DeferredToolRequests(approvals=[_deferred_call("c1")]),
        current_agent_name="planner",
        turn_count=3,
    )
    paused_state.pending_interrupts["n"] = NestedAgentInterrupt(
        node_id="n",
        question="approve?",
        agent_name="planner",
        tool_call_ids=("c1",),
    )
    paused_state.status = "interrupted"
    await checkpointer.save(
        GraphCheckpoint(thread_id=thread_id, graph_id=graph.id, state=paused_state.to_dict()),
    )

    resumed_result = await Runner.arun_graph_from_checkpoint(
        graph=graph,
        checkpointer=checkpointer,
        thread_id=thread_id,
        resume=GraphResume(
            replies={"n": NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="c1"),))},
        ),
    )

    # invoke MUST NOT have been called for node 'n' — resume_from_snapshot was the dispatch.
    assert invoked == []
    assert resumed == ["n"]
    assert resumed_result.status == GraphRunStatus.COMPLETED
    # Contract: the kwarg the resumed executable received is the loop's
    # nested-agent-snapshots dict (a plain dict, not None), with the
    # entry for the resumed node already popped by the staging helper.
    assert len(observed_kwarg_dicts) == 1
    kwarg_dict = observed_kwarg_dicts[0]
    assert isinstance(kwarg_dict, dict)
    assert "n" not in kwarg_dict


async def test_bsp_loop_rejects_mismatched_reply_type() -> None:
    """If ``GraphResume.replies["n"]`` is NOT a :class:`NestedAgentReply`
    but the parked interrupt IS a :class:`NestedAgentInterrupt`, the
    loop raises :class:`GraphResumeError` BEFORE dispatching the
    executable. No monkeypatch needed — the failure path triggers
    before any invoke / resume call."""
    graph = _trivial_graph()
    checkpointer = InMemoryCheckpointer()
    thread_id = "t"

    paused_state = GraphState[None](graph=graph, thread_id=thread_id)
    paused_state.nested_agent_snapshots["n"] = RunState(
        deferred_tool_requests=DeferredToolRequests(approvals=[_deferred_call("c1")]),
        current_agent_name="planner",
    )
    paused_state.pending_interrupts["n"] = NestedAgentInterrupt(
        node_id="n",
        question="approve?",
        agent_name="planner",
        tool_call_ids=("c1",),
    )
    paused_state.status = "interrupted"
    await checkpointer.save(
        GraphCheckpoint(thread_id=thread_id, graph_id=graph.id, state=paused_state.to_dict()),
    )

    with pytest.raises(GraphResumeError, match="must be a NestedAgentReply"):
        await Runner.arun_graph_from_checkpoint(
            graph=graph,
            checkpointer=checkpointer,
            thread_id=thread_id,
            # Pass a plain string instead of a NestedAgentReply — should reject.
            resume=GraphResume(replies={"n": "approve"}),
        )


def _callable_graph() -> Graph[Any]:
    """One-node graph whose executable is a plain callable (not an
    :class:`AgentExecutable`). The builder auto-wraps it in a
    :class:`CallableExecutable`. Used to simulate a graph shape that
    has changed under a parked :class:`NestedAgentInterrupt` — for
    example, the operator edited the graph between checkpoint and
    resume."""
    return Graph.new("test-bsp-nested-callable").node("n", lambda text: text).entry("n").terminal("n").compile()


async def test_bsp_loop_rejects_executable_type_mismatch() -> None:
    """A :class:`NestedAgentInterrupt` parked on a node whose
    executable is no longer an :class:`AgentExecutable` (graph shape
    drifted between checkpoint and resume) MUST raise
    :class:`GraphResumeError` synchronously during the seed phase —
    not be absorbed into a generic FAILED status by the per-task error
    collector."""
    graph = _callable_graph()
    checkpointer = InMemoryCheckpointer()
    thread_id = "t"

    paused_state = GraphState[None](graph=graph, thread_id=thread_id)
    paused_state.nested_agent_snapshots["n"] = RunState(
        deferred_tool_requests=DeferredToolRequests(approvals=[_deferred_call("c1")]),
        current_agent_name="planner",
    )
    paused_state.pending_interrupts["n"] = NestedAgentInterrupt(
        node_id="n",
        question="approve?",
        agent_name="planner",
        tool_call_ids=("c1",),
    )
    paused_state.status = "interrupted"
    await checkpointer.save(
        GraphCheckpoint(thread_id=thread_id, graph_id=graph.id, state=paused_state.to_dict()),
    )

    with pytest.raises(GraphResumeError, match="not AgentExecutable"):
        await Runner.arun_graph_from_checkpoint(
            graph=graph,
            checkpointer=checkpointer,
            thread_id=thread_id,
            resume=GraphResume(
                replies={"n": NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="c1"),))},
            ),
        )


async def test_bsp_loop_rejects_replies_and_rejected_both_supplied() -> None:
    """``GraphResume.replies[node_id]`` and ``GraphResume.rejected[node_id]``
    are mutually-exclusive intents. Supplying both for the same node
    MUST raise :class:`GraphResumeError` — silently picking one would
    hide the operator's confusion."""
    graph = _trivial_graph()
    checkpointer = InMemoryCheckpointer()
    thread_id = "t"

    paused_state = GraphState[None](graph=graph, thread_id=thread_id)
    paused_state.nested_agent_snapshots["n"] = RunState(
        deferred_tool_requests=DeferredToolRequests(approvals=[_deferred_call("c1")]),
        current_agent_name="planner",
    )
    paused_state.pending_interrupts["n"] = NestedAgentInterrupt(
        node_id="n",
        question="approve?",
        agent_name="planner",
        tool_call_ids=("c1",),
    )
    paused_state.status = "interrupted"
    await checkpointer.save(
        GraphCheckpoint(thread_id=thread_id, graph_id=graph.id, state=paused_state.to_dict()),
    )

    with pytest.raises(GraphResumeError, match="mutually exclusive"):
        await Runner.arun_graph_from_checkpoint(
            graph=graph,
            checkpointer=checkpointer,
            thread_id=thread_id,
            resume=GraphResume(
                replies={"n": NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="c1"),))},
                rejected={"n": "no"},
            ),
        )


async def test_bsp_loop_rejects_empty_snapshot_under_rejection() -> None:
    """``GraphResume.rejected[node_id]`` requires the snapshot to carry
    at least one pending approval — otherwise the synthesised
    :class:`NestedAgentReply` would be empty and the caller likely
    retried against an already-resolved snapshot. The loop MUST raise
    :class:`GraphResumeError` rather than silently produce a no-op
    reply."""
    graph = _trivial_graph()
    checkpointer = InMemoryCheckpointer()
    thread_id = "t"

    paused_state = GraphState[None](graph=graph, thread_id=thread_id)
    # Snapshot with NO pending approvals — the only legitimate way this
    # can co-exist with a parked NestedAgentInterrupt is operator error
    # (e.g., a stale checkpoint after the snapshot was already drained).
    paused_state.nested_agent_snapshots["n"] = RunState(
        deferred_tool_requests=DeferredToolRequests(approvals=[]),
        current_agent_name="planner",
    )
    paused_state.pending_interrupts["n"] = NestedAgentInterrupt(
        node_id="n",
        question="approve?",
        agent_name="planner",
        tool_call_ids=("c1",),
    )
    paused_state.status = "interrupted"
    await checkpointer.save(
        GraphCheckpoint(thread_id=thread_id, graph_id=graph.id, state=paused_state.to_dict()),
    )

    with pytest.raises(GraphResumeError, match="no pending approvals"):
        await Runner.arun_graph_from_checkpoint(
            graph=graph,
            checkpointer=checkpointer,
            thread_id=thread_id,
            resume=GraphResume(rejected={"n": "no"}),
        )

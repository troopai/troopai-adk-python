"""Round-trip + factory tests for NestedAgentInterrupt and
the NestedAgentReply discriminated union, plus GraphState's
nested_agent_snapshots field."""

from __future__ import annotations

import pytest

from troopai.adk.exceptions import AgentToolDeferral
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.interrupt import (
    Interrupt,
    NestedAgentApproval,
    NestedAgentDecision,
    NestedAgentInterrupt,
    NestedAgentRejection,
    NestedAgentReply,
)
from troopai.adk.graphs.state import GraphState
from troopai.adk.run.state import RunState
from troopai.adk.tools.deferred_tool import DeferredToolCall, DeferredToolRequests


def _make_deferred_call(call_id: str) -> DeferredToolCall:
    return DeferredToolCall(
        tool_call_id=call_id,
        tool_name="t",
        tool_arguments={},
        raw_arguments="{}",
    )


def _make_graph_with_node_n() -> Graph[None]:
    """Build a minimal one-node graph whose only node id is ``"n"``.

    The single node is required for ``from_dict`` to accept payload
    fields keyed under ``"n"`` (``node_results``/``versions_seen``/
    ``produced_at``/``pending_interrupts``); the graph's edges and
    terminals are otherwise irrelevant to these tests.
    """
    return Graph.new("test-graph").node("n", lambda text: text).entry("n").terminal("n").compile()


def test_nested_agent_interrupt_is_interrupt_subtype() -> None:
    ni = NestedAgentInterrupt(
        node_id="n",
        question="approve?",
        agent_name="planner",
        tool_call_ids=("c1",),
    )
    assert isinstance(ni, Interrupt)
    assert ni.kind == "nested_agent_tool_approval"
    assert ni.node_id == "n"
    assert ni.agent_name == "planner"
    assert ni.tool_call_ids == ("c1",)


def test_nested_agent_interrupt_from_deferral_builds_payload() -> None:
    defer = AgentToolDeferral(
        agent_name="planner",
        deferred_requests=DeferredToolRequests(
            approvals=[_make_deferred_call("c1"), _make_deferred_call("c2")],
        ),
        state=RunState(),
    )
    ni = NestedAgentInterrupt.from_deferral(node_id="n", deferral=defer)
    assert ni.node_id == "n"
    assert ni.agent_name == "planner"
    assert ni.tool_call_ids == ("c1", "c2")
    assert ni.question.startswith("agent")


def test_nested_agent_interrupt_from_deferral_rejects_empty_approvals() -> None:
    """An empty approvals list would build an undecidable interrupt."""
    defer = AgentToolDeferral(
        agent_name="planner",
        deferred_requests=DeferredToolRequests(approvals=[]),
        state=RunState(),
    )
    with pytest.raises(ValueError, match="0 deferred approvals"):
        NestedAgentInterrupt.from_deferral(node_id="n", deferral=defer)


def test_nested_agent_reply_collects_approvals_and_rejections() -> None:
    """Proves the union is genuinely discriminated via a dispatch fn.

    Each ``NestedAgentDecision`` is routed by ``isinstance`` to either
    the approve or reject branch — narrowing the union literally rather
    than just splitting a 2-element tuple.
    """

    def dispatch(d: NestedAgentDecision) -> str:
        if isinstance(d, NestedAgentApproval):
            return "approve"
        if isinstance(d, NestedAgentRejection):
            return "reject"
        raise AssertionError(f"unhandled decision variant: {type(d).__name__}")

    approval = NestedAgentApproval(tool_call_id="c1", approver_id="alice")
    rejection = NestedAgentRejection(tool_call_id="c2", message="no")
    reply = NestedAgentReply(decisions=(approval, rejection))

    assert dispatch(approval) == "approve"
    assert dispatch(rejection) == "reject"
    branches = [dispatch(d) for d in reply.decisions]
    assert branches == ["approve", "reject"]

    approvals = [d for d in reply.decisions if isinstance(d, NestedAgentApproval)]
    rejections = [d for d in reply.decisions if isinstance(d, NestedAgentRejection)]
    assert len(approvals) == 1 and approvals[0].tool_call_id == "c1"
    assert len(rejections) == 1 and rejections[0].message == "no"


def test_graphstate_nested_agent_snapshots_roundtrip() -> None:
    graph = _make_graph_with_node_n()
    state = GraphState[None](graph=graph)
    snap = RunState(current_agent_name="planner", turn_count=2)
    state.nested_agent_snapshots["n"] = snap
    state.pending_interrupts["n"] = NestedAgentInterrupt(
        node_id="n",
        question="approve?",
        agent_name="planner",
        tool_call_ids=("c1",),
    )
    state.status = "interrupted"

    data = state.to_dict()
    assert "nested_agent_snapshots" in data
    assert data["nested_agent_snapshots"]["n"]["current_agent_name"] == "planner"
    assert data["pending_interrupts"]["n"]["kind"] == "nested_agent_tool_approval"
    assert tuple(data["pending_interrupts"]["n"]["tool_call_ids"]) == ("c1",)

    restored: GraphState[None] = GraphState.from_dict(data, graph=graph)
    assert isinstance(restored.nested_agent_snapshots["n"], RunState)
    assert restored.nested_agent_snapshots["n"].turn_count == 2
    restored_interrupt = restored.pending_interrupts["n"]
    assert isinstance(restored_interrupt, NestedAgentInterrupt)
    assert restored_interrupt.tool_call_ids == ("c1",)


def test_graphstate_from_dict_when_no_nested_interrupts_present() -> None:
    """Loading a payload without the field yields an empty snapshot dict."""
    graph = _make_graph_with_node_n()
    state = GraphState[None](graph=graph)
    data = state.to_dict()
    data.pop("nested_agent_snapshots", None)
    restored: GraphState[None] = GraphState.from_dict(data, graph=graph)
    assert restored.nested_agent_snapshots == {}


def test_graphstate_from_dict_rejects_orphan_nested_interrupt() -> None:
    """A ``NestedAgentInterrupt`` without its snapshot is a deadlock trap.

    The cross-reference check in ``from_dict`` MUST refuse to rehydrate
    a payload whose ``pending_interrupts`` carries a nested entry that
    ``nested_agent_snapshots`` cannot back. Without the snapshot the
    resume path has no ``RunState`` to apply the decisions to.
    """
    graph = _make_graph_with_node_n()
    state = GraphState[None](graph=graph)
    state.nested_agent_snapshots["n"] = RunState(current_agent_name="planner", turn_count=1)
    state.pending_interrupts["n"] = NestedAgentInterrupt(
        node_id="n",
        question="approve?",
        agent_name="planner",
        tool_call_ids=("c1",),
    )
    data = state.to_dict()
    # Drop the matching snapshot to create the orphan condition.
    data["nested_agent_snapshots"] = {}

    with pytest.raises(ValueError, match=r"pending_interrupts\['n'\] is a NestedAgentInterrupt"):
        GraphState.from_dict(data, graph=graph)

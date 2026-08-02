"""End-to-end integration of the nested-agent deferral bridge.

Each test drives a real graph through :meth:`Runner.arun_graph` and
:meth:`Runner.arun_graph_from_checkpoint`, exercising every layer of the
bridge — :meth:`AgentExecutable.invoke` -> ``AgentToolDeferral`` lift ->
BSP loop checkpoint -> resume staging -> :meth:`AgentExecutable.resume_from_snapshot`
-> completion (or re-deferral) — except the inner LLM step, which is
mocked at :meth:`Runner.arun` for honesty: simulating a real provider
inside a unit-style test would defeat the test's purpose.

Three shapes covered:

1. Single-node single-level defer -> resume -> completion.
2. Concurrent fan-out: two nodes each defer in the same superstep;
   partial resume leaves the other interrupt outstanding; second resume
   completes the graph.
3. Re-deferral: a resumed node defers AGAIN with a fresh tool-call id;
   the second resume completes the graph.
"""

from __future__ import annotations

from typing import Any

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.exceptions import AgentToolDeferral
from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.interrupt import (
    GraphResume,
    NestedAgentApproval,
    NestedAgentInterrupt,
    NestedAgentReply,
)
from troopai.adk.graphs.result import GraphRunStatus
from troopai.adk.run.runner import Runner
from troopai.adk.run.state import RunState
from troopai.adk.tools.deferred_tool import DeferredToolCall, DeferredToolRequests
from troopai.adk.types.tokens.llm_usage import LLMUsage

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _deferred_call(call_id: str, tool_name: str = "approve_me") -> DeferredToolCall:
    """Build a placeholder :class:`DeferredToolCall` used in fake deferrals."""
    return DeferredToolCall(
        tool_call_id=call_id,
        tool_name=tool_name,
        tool_arguments={},
        raw_arguments="{}",
    )


class _FakeAgentRef:
    """Minimal stand-in for the ``Agent`` instance referenced by ``RunResult.last_agent``.

    The bridge reads ``result.last_agent.name`` to populate
    :attr:`NodeResult.metadata['last_agent_name']`. A real :class:`Agent`
    works too but pulls more wiring into the fake than needed.
    """

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRunContextRef:
    """Stand-in for ``RunResult.context`` — only ``.usage`` is read."""

    def __init__(self) -> None:
        self.usage = LLMUsage()


class _CompletedRunResult:
    """Stand-in for a normal :class:`RunResult` returned by :meth:`Runner.arun`.

    Mirrors the shape :meth:`AgentExecutable.invoke` /
    :meth:`AgentExecutable.resume_from_snapshot` read after a non-deferred
    inner run: ``final_output``, ``new_items``, ``context.usage``,
    ``last_agent.name``, ``requires_action=False``, no
    ``deferred_requests``/``state``.
    """

    def __init__(self, final_output: Any, agent_name: str = "planner") -> None:
        self.final_output = final_output
        self.new_items: list[Any] = []
        self.context = _FakeRunContextRef()
        self.last_agent = _FakeAgentRef(agent_name)
        self.requires_action = False
        self.deferred_requests: DeferredToolRequests | None = None
        self.state: RunState | None = None


class _ReDeferringRunResult:
    """Stand-in for a :class:`RunResult` whose resumed run deferred AGAIN.

    The bridge's :meth:`AgentExecutable._handle_re_deferral` reads
    ``requires_action=True`` + ``deferred_requests`` + ``state`` to lift a
    fresh :class:`NestedAgentInterrupt`.
    """

    def __init__(self, new_state: RunState, agent_name: str = "planner") -> None:
        self.final_output = None
        self.new_items: list[Any] = []
        self.context = _FakeRunContextRef()
        self.last_agent = _FakeAgentRef(agent_name)
        self.requires_action = True
        self.deferred_requests = new_state.deferred_tool_requests
        self.state = new_state


def _single_node_graph(node_id: str = "n", agent_name: str = "planner") -> Graph[Any]:
    """Build a one-node graph wrapping an :class:`Agent` node.

    The agent body is never reached — the test monkeypatches
    :meth:`Runner.arun` at the class level so the inner LLM step never
    fires. The node id matches the bridge's reserved metadata channel.
    """
    return (
        Graph.new("integration-nested")
        .node(node_id, Agent(name=agent_name, system_prompt="x"))
        .entry(node_id)
        .terminal(node_id)
        .compile()
    )


def _two_node_fanout_graph() -> Graph[Any]:
    """Build a root -> (a, b) fan-out graph where both leaves are agents.

    The two agent nodes 'a' and 'b' are reachable in the same superstep
    via a callable root that emits a constant prompt. Both terminals;
    the run completes only when both agents return.
    """
    return (
        Graph.new("integration-nested-fanout")
        .node("root", lambda: "go")
        .node("a", Agent(name="agent-a", system_prompt="x"))
        .node("b", Agent(name="agent-b", system_prompt="x"))
        .edge("root", "a")
        .edge("root", "b")
        .entry("root")
        .terminal("a")
        .terminal("b")
        .compile()
    )


def _install_scripted_arun(
    monkeypatch: pytest.MonkeyPatch,
    script: list[Any],
) -> dict[str, list[Any]]:
    """Install a :meth:`Runner.arun` class-level mock driven by ``script``.

    Each invocation pops the next entry: if an :class:`Exception` instance,
    the call raises it (this is how :class:`AgentToolDeferral` is delivered
    to :meth:`AgentExecutable.invoke`); otherwise the call returns the
    entry as the :class:`RunResult` stand-in.

    Records every ``(agent_name, user_prompt_or_state)`` pair so a test can
    assert which leg of the bridge the call landed on (initial invoke
    receives a ``str``/``list``; resume_from_snapshot receives a
    :class:`RunState`).

    Returns the recording dict so the test body can inspect call shapes.
    """
    calls: dict[str, list[Any]] = {"agents": [], "prompts": [], "raised": []}

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        agent = args[0] if len(args) >= 1 else kwargs.get("agent")
        prompt = args[1] if len(args) >= 2 else kwargs.get("user_prompt")
        calls["agents"].append(getattr(agent, "name", None))
        calls["prompts"].append(prompt)
        nxt = script.pop(0)
        if isinstance(nxt, BaseException):
            calls["raised"].append(type(nxt).__name__)
            raise nxt
        calls["raised"].append(None)
        return nxt

    from troopai.adk.run import runner as runner_mod

    monkeypatch.setattr(runner_mod.Runner, "arun", classmethod(fake_arun))
    return calls


# ---------------------------------------------------------------------
# Test 1: single-level defer -> checkpoint -> resume -> completion
# ---------------------------------------------------------------------


async def test_single_level_defer_then_resume_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One agent node defers; the loop suspends with a single
    :class:`NestedAgentInterrupt` carrying the deferred tool-call id and
    a snapshot in ``state.nested_agent_snapshots``. Resuming with a
    matching :class:`NestedAgentApproval` lands the decision on the
    snapshot via ``state.approve`` and completes the graph with the
    resumed run's ``final_output``.
    """
    pre_defer_state = RunState(
        current_agent_name="planner",
        turn_count=2,
        deferred_tool_requests=DeferredToolRequests(approvals=[_deferred_call("c-init")]),
    )
    deferral = AgentToolDeferral(
        agent_name="planner",
        deferred_requests=pre_defer_state.deferred_tool_requests,
        state=pre_defer_state,
    )
    completion_result = _CompletedRunResult(final_output="approved-and-done")
    calls = _install_scripted_arun(monkeypatch, [deferral, completion_result])

    graph = _single_node_graph()
    checkpointer = InMemoryCheckpointer()
    thread_id = "single-1"

    first = await Runner.arun_graph(graph, "go", hooks=[checkpointer], thread_id=thread_id)

    assert first.status == GraphRunStatus.INTERRUPTED
    assert len(first.interrupts) == 1
    interrupt = first.interrupts[0]
    assert isinstance(interrupt, NestedAgentInterrupt)
    assert interrupt.node_id == "n"
    assert interrupt.agent_name == "planner"
    assert interrupt.tool_call_ids == ("c-init",)
    assert len(interrupt.tool_call_ids) > 0
    # The bridge MUST have deposited the sub-agent ``RunState`` into the
    # graph state's side-channel — the resume path reads it from there.
    assert first.state is not None
    assert "n" in first.state.nested_agent_snapshots
    parked_snap = first.state.nested_agent_snapshots["n"]
    assert isinstance(parked_snap, RunState)
    assert parked_snap.turn_count == 2
    # The initial invoke received a user prompt (str/list), NOT a RunState.
    assert calls["agents"] == ["planner"]
    assert not isinstance(calls["prompts"][0], RunState)

    resumed = await Runner.arun_graph_from_checkpoint(
        graph=graph,
        checkpointer=checkpointer,
        thread_id=thread_id,
        resume=GraphResume(
            replies={"n": NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="c-init"),))},
        ),
    )

    assert resumed.status == GraphRunStatus.COMPLETED
    # The approved tool call produced the completion; the final_output
    # reflects the resumed inner run's return value.
    assert resumed.final_output == "approved-and-done"
    # Resume routes through resume_from_snapshot, which calls Runner.arun
    # with a RunState (NOT a string prompt) carrying the applied decision.
    assert len(calls["agents"]) == 2
    assert calls["agents"][1] == "planner"
    resumed_state_arg = calls["prompts"][1]
    assert isinstance(resumed_state_arg, RunState)
    assert len(resumed_state_arg.approved_tools) == 1
    assert resumed_state_arg.approved_tools[0].tool_call_id == "c-init"
    # The pending interrupt + parked snapshot have been cleared.
    assert resumed.state is not None
    assert "n" not in resumed.state.pending_interrupts
    assert "n" not in resumed.state.nested_agent_snapshots


# ---------------------------------------------------------------------
# Test 2: concurrent fan-out, partial resume
# ---------------------------------------------------------------------


async def test_concurrent_fanout_partial_resume_then_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two parallel agent nodes each defer in the same superstep, yielding
    a single INTERRUPTED result with two distinct interrupts. Resuming
    with only one node's reply suspends the run again on the other; the
    second resume completes the graph.
    """
    defer_state_a = RunState(
        current_agent_name="agent-a",
        turn_count=1,
        deferred_tool_requests=DeferredToolRequests(approvals=[_deferred_call("ca-1")]),
    )
    defer_state_b = RunState(
        current_agent_name="agent-b",
        turn_count=1,
        deferred_tool_requests=DeferredToolRequests(approvals=[_deferred_call("cb-1")]),
    )
    deferral_a = AgentToolDeferral(
        agent_name="agent-a",
        deferred_requests=defer_state_a.deferred_tool_requests,
        state=defer_state_a,
    )
    deferral_b = AgentToolDeferral(
        agent_name="agent-b",
        deferred_requests=defer_state_b.deferred_tool_requests,
        state=defer_state_b,
    )
    # The BSP loop dispatches sibling tasks concurrently, so the script's
    # order against (a, b) must be agent-resolved at the fake. Use a
    # dispatch-by-agent-name strategy that does not depend on iteration order.
    script: dict[str, list[Any]] = {
        "agent-a": [deferral_a, _CompletedRunResult(final_output="a-final", agent_name="agent-a")],
        "agent-b": [deferral_b, _CompletedRunResult(final_output="b-final", agent_name="agent-b")],
    }
    seen: dict[str, list[Any]] = {"agents": [], "prompts": []}

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        agent = args[0] if len(args) >= 1 else kwargs.get("agent")
        prompt = args[1] if len(args) >= 2 else kwargs.get("user_prompt")
        name = getattr(agent, "name", "")
        seen["agents"].append(name)
        seen["prompts"].append(prompt)
        nxt = script[name].pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    from troopai.adk.run import runner as runner_mod

    monkeypatch.setattr(runner_mod.Runner, "arun", classmethod(fake_arun))

    graph = _two_node_fanout_graph()
    checkpointer = InMemoryCheckpointer()
    thread_id = "fanout-1"

    first = await Runner.arun_graph(graph, "go", hooks=[checkpointer], thread_id=thread_id)

    assert first.status == GraphRunStatus.INTERRUPTED
    interrupt_by_node = {iv.node_id: iv for iv in first.interrupts}
    assert set(interrupt_by_node.keys()) == {"a", "b"}
    iv_a = interrupt_by_node["a"]
    iv_b = interrupt_by_node["b"]
    assert isinstance(iv_a, NestedAgentInterrupt)
    assert isinstance(iv_b, NestedAgentInterrupt)
    assert iv_a.tool_call_ids == ("ca-1",)
    assert iv_b.tool_call_ids == ("cb-1",)
    assert first.state is not None
    assert "a" in first.state.nested_agent_snapshots
    assert "b" in first.state.nested_agent_snapshots

    # First resume answers ONLY node 'a'; node 'b' must re-surface as
    # interrupted with the same parked snapshot.
    intermediate = await Runner.arun_graph_from_checkpoint(
        graph=graph,
        checkpointer=checkpointer,
        thread_id=thread_id,
        resume=GraphResume(
            replies={"a": NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="ca-1"),))},
        ),
    )

    assert intermediate.status == GraphRunStatus.INTERRUPTED
    assert len(intermediate.interrupts) == 1
    remaining = intermediate.interrupts[0]
    assert isinstance(remaining, NestedAgentInterrupt)
    assert remaining.node_id == "b"
    assert remaining.tool_call_ids == ("cb-1",)
    # Node 'a' is resolved: its terminal output was recorded.
    assert intermediate.state is not None
    assert "a" not in intermediate.state.pending_interrupts
    assert "a" not in intermediate.state.nested_agent_snapshots
    assert "a" in intermediate.state.terminal_outputs
    assert intermediate.state.terminal_outputs["a"] == "a-final"
    # Node 'b' is still parked.
    assert "b" in intermediate.state.pending_interrupts
    assert "b" in intermediate.state.nested_agent_snapshots

    # Second resume answers node 'b' -> the graph completes.
    final = await Runner.arun_graph_from_checkpoint(
        graph=graph,
        checkpointer=checkpointer,
        thread_id=thread_id,
        resume=GraphResume(
            replies={"b": NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="cb-1"),))},
        ),
    )

    assert final.status == GraphRunStatus.COMPLETED
    assert final.state is not None
    assert len(final.state.pending_interrupts) == 0
    assert len(final.state.nested_agent_snapshots) == 0
    # Both terminals have their outputs.
    assert final.state.terminal_outputs.get("a") == "a-final"
    assert final.state.terminal_outputs.get("b") == "b-final"


# ---------------------------------------------------------------------
# Test 3: re-deferral after resume re-checkpoints
# ---------------------------------------------------------------------


async def test_re_deferral_after_resume_re_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first deferral surfaces; the user approves; the resumed agent
    defers AGAIN with a NEW pending tool-call id; the second resume
    completes the graph. Exercises
    :meth:`AgentExecutable._handle_re_deferral`'s integration with the
    BSP loop's checkpoint-on-suspend and side-channel deposit.
    """
    first_defer_state = RunState(
        current_agent_name="planner",
        turn_count=1,
        deferred_tool_requests=DeferredToolRequests(approvals=[_deferred_call("c-first")]),
    )
    deferral_one = AgentToolDeferral(
        agent_name="planner",
        deferred_requests=first_defer_state.deferred_tool_requests,
        state=first_defer_state,
    )

    second_defer_state = RunState(
        current_agent_name="planner",
        turn_count=4,
        deferred_tool_requests=DeferredToolRequests(approvals=[_deferred_call("c-second")]),
    )
    re_deferring_result = _ReDeferringRunResult(new_state=second_defer_state)
    completion_result = _CompletedRunResult(final_output="finally-done")

    calls = _install_scripted_arun(monkeypatch, [deferral_one, re_deferring_result, completion_result])

    graph = _single_node_graph()
    checkpointer = InMemoryCheckpointer()
    thread_id = "redefer-1"

    # 1) First run: first deferral with tool_call_id 'c-first'.
    first = await Runner.arun_graph(graph, "go", hooks=[checkpointer], thread_id=thread_id)

    assert first.status == GraphRunStatus.INTERRUPTED
    assert len(first.interrupts) == 1
    first_iv = first.interrupts[0]
    assert isinstance(first_iv, NestedAgentInterrupt)
    assert first_iv.tool_call_ids == ("c-first",)
    assert first.state is not None
    assert first.state.nested_agent_snapshots["n"].turn_count == 1

    # 2) First resume approves 'c-first'; the resumed agent defers AGAIN
    # with a new tool_call_id 'c-second'. The BSP loop must re-surface
    # INTERRUPTED carrying the SECOND interrupt's id, and must have
    # re-deposited the fresh snapshot.
    intermediate = await Runner.arun_graph_from_checkpoint(
        graph=graph,
        checkpointer=checkpointer,
        thread_id=thread_id,
        resume=GraphResume(
            replies={"n": NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="c-first"),))},
        ),
    )

    assert intermediate.status == GraphRunStatus.INTERRUPTED
    assert len(intermediate.interrupts) == 1
    second_iv = intermediate.interrupts[0]
    assert isinstance(second_iv, NestedAgentInterrupt)
    assert second_iv.node_id == "n"
    assert second_iv.tool_call_ids == ("c-second",)
    assert intermediate.state is not None
    assert "n" in intermediate.state.nested_agent_snapshots
    fresh_snap = intermediate.state.nested_agent_snapshots["n"]
    assert isinstance(fresh_snap, RunState)
    assert fresh_snap.turn_count == 4
    # The resume_from_snapshot leg was driven with a RunState arg.
    assert isinstance(calls["prompts"][1], RunState)

    # 3) Second resume approves 'c-second' -> the graph completes.
    final = await Runner.arun_graph_from_checkpoint(
        graph=graph,
        checkpointer=checkpointer,
        thread_id=thread_id,
        resume=GraphResume(
            replies={"n": NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="c-second"),))},
        ),
    )

    assert final.status == GraphRunStatus.COMPLETED
    assert final.final_output == "finally-done"
    assert final.state is not None
    assert "n" not in final.state.pending_interrupts
    assert "n" not in final.state.nested_agent_snapshots
    # Three total Runner.arun legs: initial defer, first resume (re-defers),
    # second resume (completes).
    assert len(calls["agents"]) == 3

"""Depth-2 stress test for the nested-agent deferral bridge.

Shape under test:

- An OUTER :class:`Graph` whose single node holds an INNER :class:`Graph`
  as its executable. The inner graph holds a single
  :class:`AgentExecutable` whose underlying agent's tool defers.

When the inner agent defers:

1. The inner agent's :class:`AgentToolDeferral` is lifted to a
   :class:`NestedAgentInterrupt` keyed by the INNER node id by
   :meth:`AgentExecutable.invoke`'s lift path.
2. The inner graph's BSP loop parks the interrupt on
   :attr:`GraphState.pending_interrupts` and the snapshot on
   :attr:`GraphState.nested_agent_snapshots`, exiting with
   status=INTERRUPTED.
3. The outer graph's call to :meth:`Graph.invoke` (the inner graph
   wrapped as a node) MUST bubble this interrupt up to the OUTER BSP
   loop, with the ``node_id`` rewritten to the OUTER node id (so the
   caller doesn't need to know the inner graph's structure to compose
   a :class:`GraphResume`).

Resume:

- The caller supplies ``GraphResume(replies={outer_node_id:
  NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id=...),))})``.
- The outer BSP loop must route that reply through the inner graph
  back to the deferring agent, applying the decision on its parked
  :class:`RunState`. The inner agent then re-enters and returns, the
  inner graph completes, the outer graph completes.

Pre-existing gap surfaced by this test. :meth:`Graph.invoke` (in
``src/troopai/adk/graphs/graph.py``) calls :func:`run_graph_loop` for the
inner graph and translates the resulting :class:`GraphRunResult` into a
:class:`NodeResult` UNCONDITIONALLY — including when
``inner_result.status == GraphRunStatus.INTERRUPTED``. The outer loop has
no signal that the inner graph paused, so a depth-2 deferral silently
"completes" the outer node with a non-final ``NodeResult`` carrying the
inner ``GraphRunResult`` as ``output`` and ``"status": "interrupted"`` in
metadata. Resume cannot work because:

1. No outer :class:`InterruptException` is raised, so no outer
   :class:`NestedAgentInterrupt` is parked, so the outer caller has no
   ``GraphResume.replies`` key to target.
2. Even if the lift were added at :meth:`Graph.invoke`, the inner
   :class:`GraphState` (carrying its own ``nested_agent_snapshots``)
   would need to be preserved across the outer/inner boundary so the
   resume path can re-enter the inner graph with both the staged
   reply and the saved inner state. The current side-channel under
   :attr:`GraphState.nested_agent_snapshots` is typed
   ``dict[str, RunState]`` — it has no slot for a nested
   :class:`GraphState`. Outer-loop dispatch (`_dispatch_node` /
   `_dispatch_nested_resume` in ``src/troopai/adk/run/graph_loop.py``)
   only knows how to call :meth:`AgentExecutable.resume_from_snapshot`.

This test is marked :func:`pytest.mark.xfail(strict=True)` — when the
bridge gains depth-2 support, the test will start passing and the strict
xfail surfaces the regression. Until then, the marker is the on-record
gap.
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
# Stand-ins for the ``Runner.arun`` return shape
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
    """Stand-in for the ``Agent`` instance referenced by ``RunResult.last_agent``.

    Only ``.name`` is read by :meth:`AgentExecutable.invoke` /
    :meth:`AgentExecutable.resume_from_snapshot`.
    """

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRunContextRef:
    """Stand-in for ``RunResult.context`` — only ``.usage`` is read."""

    def __init__(self) -> None:
        self.usage = LLMUsage()


class _CompletedRunResult:
    """Stand-in for a non-deferring :class:`RunResult` from :meth:`Runner.arun`."""

    def __init__(self, final_output: Any, agent_name: str = "planner") -> None:
        self.final_output = final_output
        self.new_items: list[Any] = []
        self.context = _FakeRunContextRef()
        self.last_agent = _FakeAgentRef(agent_name)
        self.requires_action = False
        self.deferred_requests: DeferredToolRequests | None = None
        self.state: RunState | None = None


# ---------------------------------------------------------------------
# Graph builders (depth-2)
# ---------------------------------------------------------------------


def _inner_single_agent_graph(
    inner_node_id: str = "inner_agent",
    agent_name: str = "planner",
) -> Graph[Any]:
    """Build a one-node inner graph wrapping a single agent node.

    Used as the executable of the outer graph's single node so the
    overall shape is ``outer_graph[Graph[AgentExecutable[Agent]]]``.
    """
    return (
        Graph.new("depth2-inner")
        .node(inner_node_id, Agent(name=agent_name, system_prompt="x"))
        .entry(inner_node_id)
        .terminal(inner_node_id)
        .compile()
    )


def _outer_wraps_inner_graph(
    inner: Graph[Any],
    outer_node_id: str = "outer_graph_node",
) -> Graph[Any]:
    """Build a one-node outer graph whose node holds ``inner`` as its executable.

    ``Graph`` implements :class:`Executable` directly, so the inner graph
    is placed as the node's executable without an adapter. This is the
    "graph of graphs" composition the framework already supports for
    non-deferring inner graphs.
    """
    return Graph.new("depth2-outer").node(outer_node_id, inner).entry(outer_node_id).terminal(outer_node_id).compile()


def _install_scripted_arun(
    monkeypatch: pytest.MonkeyPatch,
    script: list[Any],
) -> dict[str, list[Any]]:
    """Install a :meth:`Runner.arun` class-level mock driven by ``script``.

    Each invocation pops the next entry: an :class:`Exception` instance
    is raised, anything else is returned as the :class:`RunResult` stand-in.
    Records call shapes so the test body can assert which leg fired.
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
# Depth-2 stress test
# ---------------------------------------------------------------------


async def test_depth_2_nested_defer_then_resume_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outer graph wraps an inner graph; the inner agent's tool defers.

    The inner agent's deferral becomes a
    :class:`~troopai.adk.graphs.interrupt.NestedAgentInterrupt` at the
    inner graph layer, then bubbles to the OUTER graph layer with the
    OUTER node id (not the inner). The caller resumes via
    :class:`~troopai.adk.graphs.interrupt.GraphResume.replies` keyed by
    the outer node id; the resume routes through both layers and the
    run completes.

    The assertion on outer node id is the load-bearing contract — the
    inner graph's bubbling MUST rewrite the node id so the caller does
    not need to know the inner graph's internal node ids to compose a
    resume.
    """
    outer_node_id = "outer_graph_node"
    inner_node_id = "inner_agent"

    # The inner agent defers on its first call, then completes on the
    # second call when its parked RunState has been re-entered with the
    # approval applied.
    pre_defer_state = RunState(
        current_agent_name="planner",
        turn_count=1,
        deferred_tool_requests=DeferredToolRequests(approvals=[_deferred_call("deep-c1")]),
    )
    deferral = AgentToolDeferral(
        agent_name="planner",
        deferred_requests=pre_defer_state.deferred_tool_requests,
        state=pre_defer_state,
    )
    completion_result = _CompletedRunResult(final_output="deep-approved-and-done")
    _install_scripted_arun(monkeypatch, [deferral, completion_result])

    inner = _inner_single_agent_graph(inner_node_id=inner_node_id)
    outer = _outer_wraps_inner_graph(inner, outer_node_id=outer_node_id)

    checkpointer = InMemoryCheckpointer()
    thread_id = "depth2-1"

    # --- Pause leg --------------------------------------------------
    first = await Runner.arun_graph(outer, "go", hooks=[checkpointer], thread_id=thread_id)

    # The outer caller MUST see INTERRUPTED with a single interrupt
    # keyed by the OUTER node id. The inner graph's nesting is an
    # implementation detail the caller does not need to know.
    assert first.status == GraphRunStatus.INTERRUPTED
    assert len(first.interrupts) == 1
    outer_interrupt = first.interrupts[0]
    assert isinstance(outer_interrupt, NestedAgentInterrupt)
    assert outer_interrupt.node_id == outer_node_id, (
        "depth-2 bridge MUST rewrite the inner interrupt's node_id "
        "to the outer scope; caller must not need to know the inner "
        f"graph's node ids. Got node_id={outer_interrupt.node_id!r}, "
        f"expected {outer_node_id!r}."
    )
    assert outer_interrupt.agent_name == "planner"
    assert outer_interrupt.tool_call_ids == ("deep-c1",)
    assert first.state is not None
    # The outer GraphState MUST carry the parked inner-graph snapshot
    # under the OUTER node id so the resume dispatch can find it. The
    # PA4 design uses a parallel slot (nested_graph_snapshots) for
    # graph-backed inner state, distinct from nested_agent_snapshots
    # which holds agent-paused state at depth 1.
    assert outer_node_id in first.state.nested_graph_snapshots, (
        f"outer GraphState.nested_graph_snapshots must contain "
        f"{outer_node_id!r} so the resume path can re-enter the inner "
        f"graph. Keys present: {sorted(first.state.nested_graph_snapshots.keys())}"
    )

    # --- Resume leg -------------------------------------------------
    resumed = await Runner.arun_graph_from_checkpoint(
        graph=outer,
        checkpointer=checkpointer,
        thread_id=thread_id,
        resume=GraphResume(
            replies={
                outer_node_id: NestedAgentReply(
                    decisions=(NestedAgentApproval(tool_call_id="deep-c1"),),
                ),
            },
        ),
    )

    assert resumed.status == GraphRunStatus.COMPLETED
    assert resumed.final_output == "deep-approved-and-done"
    assert resumed.state is not None
    assert outer_node_id not in resumed.state.pending_interrupts
    assert outer_node_id not in resumed.state.nested_graph_snapshots

"""``GraphHooks.on_node_interrupt`` fires when a node suspends.

Covers the operator-visible callback path: when a node raises
``InterruptException`` (HITL `request_human_input`, nested-agent tool
deferral, etc.), the graph driver invokes ``on_node_interrupt`` on every
attached :class:`GraphHooks` before the superstep ends.

The hook complements the existing ``NodeInterruptEvent`` stream emission
— streaming subscribers receive the event; hook subscribers receive a
callback. Both fire at the same lifecycle boundary.
"""

from __future__ import annotations

from typing import Any, override

from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.hooks import GraphHooks
from troopai.adk.graphs.interrupt import Interrupt, request_human_input
from troopai.adk.graphs.result import GraphRunStatus
from troopai.adk.graphs.state import GraphState
from troopai.adk.orchestration.executable import ExecutableInput
from troopai.adk.run.context import RunContext
from troopai.adk.run.runner import Runner


class _Recorder(GraphHooks[Any]):
    def __init__(self) -> None:
        self.interrupt_calls: list[tuple[str, Interrupt]] = []
        self.error_calls: list[tuple[str, BaseException]] = []
        self.end_calls: list[str] = []

    @override
    async def on_node_interrupt(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        node_id: str,
        interrupt: Interrupt,
    ) -> None:
        del context, state
        self.interrupt_calls.append((node_id, interrupt))

    @override
    async def on_node_error(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        node_id: str,
        error: BaseException,
    ) -> None:
        del context, state
        self.error_calls.append((node_id, error))

    @override
    async def on_node_end(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        node_id: str,
        result: Any,
    ) -> None:
        del context, state, result
        self.end_calls.append(node_id)


def _interrupting_node(inp: ExecutableInput, ctx: Any) -> str:
    del ctx
    reply = request_human_input(inp, "approve?", kind="tool_approval", tool="x")
    return f"approved:{reply}"


async def test_on_node_interrupt_fires_with_node_id_and_interrupt_payload() -> None:
    g = Graph.new("hooks-interrupt-single").node("ask", _interrupting_node).entry("ask").terminal("ask").compile()
    recorder = _Recorder()

    result = await Runner.arun_graph(g, "go", hooks=[recorder])

    assert result.status == GraphRunStatus.INTERRUPTED
    assert len(recorder.interrupt_calls) == 1
    node_id, interrupt = recorder.interrupt_calls[0]
    assert node_id == "ask"
    assert interrupt.node_id == "ask"
    assert interrupt.question == "approve?"
    assert interrupt.kind == "tool_approval"
    # Interrupt is NOT an error and NOT a clean end — neither sibling
    # hook should fire for the suspended node.
    assert len(recorder.error_calls) == 0
    assert "ask" not in recorder.end_calls


async def test_on_node_interrupt_and_on_node_end_can_coexist_in_one_superstep() -> None:
    """Fan-out: one node suspends, sibling completes. Both hooks fire — once each."""

    def _completing(text: str) -> str:
        return f"done:{text}"

    g = (
        Graph.new("hooks-interrupt-fanout")
        .node("root", lambda: "go")
        .node("ask", _interrupting_node)
        .node("ok", _completing)
        .edge("root", "ask")
        .edge("root", "ok")
        .entry("root")
        .terminal("ask")
        .terminal("ok")
        .compile()
    )
    recorder = _Recorder()

    result = await Runner.arun_graph(g, "go", hooks=[recorder])

    assert result.status == GraphRunStatus.INTERRUPTED
    # Suspended node: on_node_interrupt fires, on_node_end does NOT.
    assert {nid for nid, _ in recorder.interrupt_calls} == {"ask"}
    assert "ask" not in recorder.end_calls
    # Completing sibling: on_node_end fires, on_node_interrupt does NOT.
    assert "ok" in recorder.end_calls
    assert "ok" not in {nid for nid, _ in recorder.interrupt_calls}

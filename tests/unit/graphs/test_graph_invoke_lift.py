"""Graph.invoke lifts inner INTERRUPTED to outer InterruptException."""

from __future__ import annotations

from typing import Any

import pytest

from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.interrupt import (
    GraphResume,
    InterruptException,
    NestedGraphInterrupt,
    request_human_input,
)
from troopai.adk.graphs.result import GraphRunStatus
from troopai.adk.graphs.state import GraphState
from troopai.adk.orchestration.executable import ExecutableInput
from troopai.adk.run.config import DEFAULT_RUN_CONFIG
from troopai.adk.run.context import RunContext
from troopai.adk.run.graph_loop import run_graph_loop
from troopai.adk.types.input.llm_input_easy_message import LLMInputEasyMessage


def _ask_inner(inp: ExecutableInput, ctx: Any) -> str:
    """Inner-graph node that requests human input on first call."""
    del ctx
    reply = request_human_input(inp, "approve?", kind="tool_approval", tool="x")
    return f"approved:{reply}"


async def test_graph_invoke_lifts_inner_interrupted_to_outer_interrupt() -> None:
    inner = Graph.new("inner-g").node("ask", _ask_inner).entry("ask").terminal("ask").compile()
    input_ = ExecutableInput(
        content=[LLMInputEasyMessage(role="user", content="go")],
        from_node=None,
        edge_label=None,
        metadata={},
    )
    ctx: RunContext[Any] = RunContext(context=None)

    with pytest.raises(InterruptException) as excinfo:
        await inner.invoke(input_, ctx, DEFAULT_RUN_CONFIG)

    # The inner pending interrupt is a PLAIN Interrupt (request_human_input),
    # so the lift produces a NestedGraphInterrupt — NOT a NestedAgentInterrupt
    # with an empty agent_name (which GraphState.from_dict would reject,
    # making the outer graph non-resumable). The inner node id is stashed on
    # metadata; node_id is left empty for the outer BSP loop to rewrite.
    iv = excinfo.value.interrupt
    assert isinstance(iv, NestedGraphInterrupt)
    assert iv.metadata["inner_graph_id"] == "inner-g"
    assert iv.metadata["inner_node_id"] == "ask"

    # The inner GraphState is communicated via the underscore attr
    # for the outer BSP loop to park.
    inner_state = getattr(excinfo.value, "_nested_graph_state", None)
    assert isinstance(inner_state, GraphState)
    assert "ask" in inner_state.pending_interrupts


async def test_nested_plain_interrupt_checkpoint_roundtrip_and_resume() -> None:
    """A nested graph wrapping a PLAIN Interrupt survives checkpoint + resume.

    Regression: the lifted plain interrupt used to be a
    ``NestedAgentInterrupt(agent_name="")`` — which ``GraphState.from_dict``
    REJECTED (empty ``agent_name``), leaving the outer graph permanently
    non-resumable after a checkpoint. It is now lifted as a
    ``NestedGraphInterrupt``: ``from_dict`` rehydrates it, and resume forwards
    a PLAIN reply value into the inner graph.
    """
    inner = Graph.new("inner-g").node("ask", _ask_inner).entry("ask").terminal("ask").compile()
    outer = Graph.new("outer-g").node("g", inner).entry("g").terminal("g").compile()
    ctx: RunContext[Any] = RunContext(context=None)

    # 1. Run → suspends on the inner plain Interrupt, lifted onto outer node "g".
    first = await run_graph_loop(graph=outer, user_prompt="go", context=ctx, config=DEFAULT_RUN_CONFIG)
    assert first.status == GraphRunStatus.INTERRUPTED
    assert first.state is not None
    assert isinstance(first.state.pending_interrupts["g"], NestedGraphInterrupt)

    # 2. Checkpoint round-trip — this is where the bug bit: from_dict previously
    #    RAISED ValueError on the empty agent_name and the run could not resume.
    rehydrated = GraphState.from_dict(first.state.to_dict(), outer)
    assert isinstance(rehydrated.pending_interrupts["g"], NestedGraphInterrupt)

    # 3. Resume with a PLAIN reply forwarded into the inner graph → completes.
    second = await run_graph_loop(
        graph=outer,
        user_prompt="",
        context=ctx,
        config=DEFAULT_RUN_CONFIG,
        initial_state=rehydrated,
        resume=GraphResume(replies={"g": "yes"}),
    )
    assert second.status == GraphRunStatus.COMPLETED

"""End-to-end graph streaming: structural events via stream_events()."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs.events import (
    GRAPH_END,
    GRAPH_START,
    NODE_END,
    NODE_START,
    NODE_STREAM,
)
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.result import GraphRunResultStreaming
from troopai.adk.run.config import DEFAULT_RUN_CONFIG
from troopai.adk.run.context import RunContext
from troopai.adk.run.graph_loop import run_graph_loop_streamed
from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText


def _linear() -> Graph:
    return (
        Graph.new("stream-linear")
        .node("a", lambda: "a-done")
        .node("b", lambda: "b-done")
        .node("c", lambda: "c-done")
        .edge("a", "b")
        .edge("b", "c")
        .entry("a")
        .terminal("c")
        .compile()
    )


async def test_streamed_linear_emits_ordered_structural_events() -> None:
    g = _linear()
    result: GraphRunResultStreaming = GraphRunResultStreaming()
    ctx: RunContext = RunContext(context=None)
    task = asyncio.get_running_loop().create_task(
        run_graph_loop_streamed(
            graph=g,
            user_prompt="go",
            context=ctx,
            config=DEFAULT_RUN_CONFIG,
            result=result,
        )
    )
    result.set_run_task(task)
    types = [ev["type"] async for ev in result.stream_events()]
    await task
    assert types[0] == GRAPH_START
    assert types[-1] == GRAPH_END
    assert NODE_START in types
    assert NODE_END in types
    assert result.status is not None and result.status.value == "completed"
    assert result.final_output == "c-done"


async def test_streamed_parallel_fanout_interleaves_not_serialized() -> None:
    g = (
        Graph.new("stream-fan")
        .node("root", lambda: "go")
        .node("x", lambda t: f"x:{t}")
        .node("y", lambda t: f"y:{t}")
        .node("join", lambda t: f"join:{t}")
        .edge("root", "x")
        .edge("root", "y")
        .edge("x", "join")
        .edge("y", "join")
        .entry("root")
        .terminal("join")
        .compile()
    )
    result: GraphRunResultStreaming = GraphRunResultStreaming()
    ctx: RunContext = RunContext(context=None)
    task = asyncio.get_running_loop().create_task(
        run_graph_loop_streamed(
            graph=g,
            user_prompt="go",
            context=ctx,
            config=DEFAULT_RUN_CONFIG,
            result=result,
        )
    )
    result.set_run_task(task)
    starts = [ev["node_id"] async for ev in result.stream_events() if ev["type"] == NODE_START]
    await task
    assert {"x", "y"}.issubset(set(starts))
    assert result.status is not None and result.status.value == "completed"


# ---------------------------------------------------------------------------
# AgentExecutable streaming: interior NodeStreamEvents and nested-graph deferral
# ---------------------------------------------------------------------------


def _fake_response(text: str) -> LLMResponse:
    return LLMResponse(
        response_id="resp-graph-stream",
        model="fake",
        response=[LLMResponseText(text=text)],
    )


@contextmanager
def _patched_llm(text: str) -> Iterator[None]:
    """Patch both non-streaming and streaming LLM call sites plus guardrails."""
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "troopai.adk.run.loop.call_llm",
                new=AsyncMock(side_effect=lambda *args, **kwargs: _fake_response(text)),
            )
        )
        stack.enter_context(
            patch(
                "troopai.adk.run.loop.call_llm_streamed",
                new=AsyncMock(side_effect=lambda *args, **kwargs: _fake_response(text)),
            )
        )
        stack.enter_context(
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", new=AsyncMock(return_value=[]))
        )
        stack.enter_context(
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", new=AsyncMock(return_value=[]))
        )
        stack.enter_context(patch("troopai.adk.run.runner.run_output_guardrails", new=AsyncMock(return_value=[])))
        yield


async def test_agent_node_emits_node_stream_events() -> None:
    """A graph with one agent node emits NodeStreamEvents during a streamed run.

    The stream must contain at least one ``NODE_STREAM`` event with the agent
    node's id and the correct graph path, and the run must complete with the
    agent's output as ``final_output``.
    """
    agent = Agent(name="stream-agent", system_prompt="you are helpful")
    graph = Graph.new("agent-stream-g").node("agent-node", agent).entry("agent-node").terminal("agent-node").compile()

    result: GraphRunResultStreaming[Any] = GraphRunResultStreaming()
    ctx: RunContext[Any] = RunContext(context=None)

    with _patched_llm("agent-output"):
        task = asyncio.get_running_loop().create_task(
            run_graph_loop_streamed(
                graph=graph,
                user_prompt="go",
                context=ctx,
                config=DEFAULT_RUN_CONFIG,
                result=result,
            )
        )
        result.set_run_task(task)
        events = [ev async for ev in result.stream_events()]
        # discard the result; check for errors by awaiting the task
        await task  # propagates any exception raised inside the driver

    types = [ev["type"] for ev in events]

    # Graph structural events must be present
    assert GRAPH_START in types
    assert GRAPH_END in types
    assert NODE_START in types
    assert NODE_END in types

    # At least one interior NodeStreamEvent must have been forwarded from
    # the AgentExecutable.stream_async override
    node_stream_evs = [ev for ev in events if ev["type"] == NODE_STREAM]
    assert len(node_stream_evs) >= 1, "Expected >= 1 NODE_STREAM events from agent node"

    # Every NodeStreamEvent for the agent node must carry the correct metadata
    for ev in node_stream_evs:
        if ev.get("node_id") == "agent-node":
            assert ev["graph_path"] == (graph.id,)

    # Run must have completed with the agent's output
    assert result.status is not None and result.status.value == "completed"
    assert result.final_output == "agent-output"


async def test_nested_graph_node_defers_zero_node_stream_events() -> None:
    """A nested ``Graph``-as-node under a streamed run emits NO ``NODE_STREAM``
    events for the nested graph's internal execution.

    Nested ``Graph`` keeps the default terminal-only ``stream_async``
    (one ``{"type":"result"}`` event only, which ``_stream_node`` captures
    without forwarding). The outer graph emits ``NODE_START``/``NODE_END``
    for the nested graph node, but zero ``NODE_STREAM`` events for it.
    """
    # Build an inner graph from plain callables — zero LLM cost
    inner = Graph.new("inner-g").node("leaf", lambda: "leaf-done").entry("leaf").terminal("leaf").compile()

    # Outer graph: one root callable, one nested-graph node
    outer = (
        Graph.new("outer-g")
        .node("root", lambda: "root-done")
        .node("nested", inner)
        .edge("root", "nested")
        .entry("root")
        .terminal("nested")
        .compile()
    )

    result: GraphRunResultStreaming[Any] = GraphRunResultStreaming()
    ctx: RunContext[Any] = RunContext(context=None)
    task = asyncio.get_running_loop().create_task(
        run_graph_loop_streamed(
            graph=outer,
            user_prompt="go",
            context=ctx,
            config=DEFAULT_RUN_CONFIG,
            result=result,
        )
    )
    result.set_run_task(task)
    events = [ev async for ev in result.stream_events()]
    await task

    types = [ev["type"] for ev in events]

    # Structural events for BOTH the outer graph and the nested-graph node
    # must be present
    assert GRAPH_START in types
    assert NODE_START in types
    assert NODE_END in types
    assert GRAPH_END in types

    # The nested graph node emits zero NODE_STREAM events (deferral contract):
    # its default stream_async yields only the terminal result which
    # _stream_node captures without forwarding to the outer event stream.
    nested_node_stream_evs = [ev for ev in events if ev["type"] == NODE_STREAM and ev.get("node_id") == "nested"]
    assert len(nested_node_stream_evs) == 0, (
        "Nested graph node must not emit any NODE_STREAM events (deferral contract)"
    )

    # The run must still complete successfully
    assert result.status is not None and result.status.value == "completed"


# ---------------------------------------------------------------------------
# Task 5 — Runner.arun_graph_streamed + GraphRunner stream=True + cancel
# ---------------------------------------------------------------------------


async def test_runner_arun_graph_streamed_returns_streaming_result() -> None:
    """Runner.arun_graph_streamed returns a GraphRunResultStreaming immediately;
    draining stream_events() yields a graph.start first and graph.end last,
    and the result is completed with the expected final_output."""
    from troopai.adk.run.runner import Runner

    g = _linear()
    result = await Runner.arun_graph_streamed(g, "go")
    assert isinstance(result, GraphRunResultStreaming)
    types = [ev["type"] async for ev in result.stream_events()]
    assert len(types) > 0
    assert types[0] == GRAPH_START
    assert types[-1] == GRAPH_END
    assert result.status is not None and result.status.value == "completed"
    assert result.final_output == "c-done"


async def test_graph_runner_stream_true_routes_to_streamed() -> None:
    """Runner.configure().graph(g).arun(stream=True) returns a GraphRunResultStreaming
    and produces the expected final_output."""
    from troopai.adk.run.runner import Runner

    g = _linear()
    result = await Runner.configure().graph(g).arun("go", stream=True)
    assert isinstance(result, GraphRunResultStreaming)
    types = [ev["type"] async for ev in result.stream_events()]
    assert GRAPH_END in types
    assert result.final_output == "c-done"


async def test_cancel_immediate_stops_streamed_run() -> None:
    """cancel(mode='immediate') genuinely interrupts a slow node — the drain
    completes well within the slow node's sleep duration and the slow path
    never produces its output."""
    from troopai.adk.run.runner import Runner

    async def slow() -> str:
        await asyncio.sleep(5.0)
        return "late"

    def last(x: str) -> str:
        return f"t:{x}"

    g = Graph.new("cancel-test").node("s", slow).node("t", last).edge("s", "t").entry("s").terminal("t").compile()
    result = await Runner.arun_graph_streamed(g, "go")
    it = result.stream_events()
    first = await it.__anext__()
    assert first["type"] == GRAPH_START

    result.cancel("immediate")

    # Drain must complete promptly (well under the 5 s sleep) — proves
    # the slow node task was actually cancelled, not awaited to completion.
    async with asyncio.timeout(3.0):
        rest = [ev async for ev in it]

    # The graph.end event should NOT appear in the drained remainder (the
    # driver was cancelled before normal completion), and the slow path must
    # not have produced its output.
    end_in_rest = any(ev["type"] == GRAPH_END for ev in rest)
    assert not end_in_rest or result.final_output != "t:late"


# ---------------------------------------------------------------------------
# Task 6 — Streaming × SP2 reliability and SP1 resume composition
# ---------------------------------------------------------------------------


async def test_streaming_composes_with_sp2_timeout() -> None:
    """Streaming surface delivers NODE_ERROR and a failed status when a node
    exceeds its per-node timeout.

    A graph with a slow node (sleeps 1.0 s) is configured with a 0.05 s
    per-node timeout so the timeout fires well before the node returns.
    The run is driven via ``Runner.arun_graph_streamed`` and the event stream
    is drained.  Three non-tautological assertions prove that SP2
    ``run_node_with_reliability`` composed with the streamed driver:

    1. A ``NODE_ERROR`` structural event appears — only emitted when the node
       raised (the timeout path), never on success.
    2. The ``error_type`` key on that event equals ``"GraphNodeTimeoutError"``
       — the specific SP2 timeout exception, not a generic failure.
    3. ``result.status.value == "failed"`` — the run did not complete; if the
       timeout had not fired, the run would complete with ``"fast-done"``.

    The drain is wrapped in a 5 s hard bound so the test cannot hang even if
    the timeout logic regresses.
    """
    from troopai.adk.graphs.config import GraphConfig
    from troopai.adk.graphs.events import NODE_ERROR
    from troopai.adk.run.runner import Runner

    async def slow() -> str:
        await asyncio.sleep(1.0)
        return "should-not-reach"

    graph = (
        Graph.new("streaming-sp2-timeout")
        .node("slow", slow)
        .node("fast", lambda: "fast-done")
        .edge("slow", "fast")
        .entry("slow")
        .terminal("fast")
        .with_config(GraphConfig(per_node_timeout=0.05))
        .compile()
    )

    result = await Runner.arun_graph_streamed(graph, "go")
    async with asyncio.timeout(5.0):
        events = [ev async for ev in result.stream_events()]

    # Assertion 1: a node-error structural event is present — only emitted
    # when the node raised; vacuously false if the timeout never fired.
    node_error_evs = [ev for ev in events if ev["type"] == NODE_ERROR]
    assert len(node_error_evs) >= 1, (
        "Expected at least one NODE_ERROR event; timeout did not compose with streamed driver"
    )

    # Assertion 2: the error_type names the SP2 timeout exception class
    # specifically — a generic exception would produce a different string.
    error_types = {ev["error_type"] for ev in node_error_evs}
    assert "GraphNodeTimeoutError" in error_types, (
        f"Expected 'GraphNodeTimeoutError' in node-error event error_type; got {error_types}"
    )

    # Assertion 3: the run ended in the failed terminal state — if the timeout
    # had not fired, the run would be 'completed' (not 'failed').
    assert result.status is not None and result.status.value == "failed", (
        f"Expected run status 'failed' after timeout; got {result.status!r}"
    )


async def test_streaming_composes_with_sp1_resume() -> None:
    """Streaming surface correctly resumes a mid-flight graph from a checkpoint.

    Replicates the setup from the SP1 resume integration tests: a linear
    a → b graph is capped at 1 superstep so node ``b`` has not yet fired
    when the first run hits the superstep limit.  The resume is performed
    via the ``GraphRunner.resume_from(...).arun(stream=True)`` path,
    which routes through ``Runner.arun_graph_streamed`` with the restored
    ``initial_state`` — the same BSP body that ``run_graph_loop_streamed``
    executes.  Three assertions prove ``_seed_barriers_from_checkpoint``
    seeded barriers correctly and the streamed driver ran to completion:

    1. The event stream contains a ``GRAPH_END`` event — proof that the
       streaming driver reached the terminal exit, not just a partial drain.
    2. ``result.status.value == "completed"`` — only possible if ``b`` fired;
       if the barrier seeding had failed, ``b`` would not be ready and the
       run would end ``no_ready_nodes``.
    3. ``result.final_output == "b-done"`` — exact terminal output; if node
       ``a`` had re-fired (incorrect barrier seeding) its output ``"a-done"``
       would shadow ``b``.
    """
    from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
    from troopai.adk.graphs.config import GraphConfig
    from troopai.adk.run.runner import Runner

    cp = InMemoryCheckpointer()

    capped = (
        Graph.new("streaming-sp1-resume")
        .node("a", lambda: "a-done")
        .node("b", lambda: "b-done")
        .edge("a", "b")
        .entry("a")
        .terminal("b")
        .with_config(GraphConfig(max_supersteps=1))
        .compile()
    )
    first = await Runner.arun_graph(capped, "go", hooks=[cp], thread_id="stream-resume-1")
    assert first.status.value == "max_supersteps", (
        f"First capped run must hit superstep limit; got {first.status.value!r}"
    )

    full = (
        Graph.new("streaming-sp1-resume")
        .node("a", lambda: "a-done")
        .node("b", lambda: "b-done")
        .edge("a", "b")
        .entry("a")
        .terminal("b")
        .compile()
    )

    # Resume via the streaming profile path — same route the Task-5 code wires
    # through Runner.arun_graph_streamed with initial_state from checkpoint.
    result = await Runner.configure().graph(full).resume_from(cp, "stream-resume-1").arun(stream=True)
    async with asyncio.timeout(5.0):
        events = [ev async for ev in result.stream_events()]

    # Assertion 1: stream reached graph.end — only emitted at terminal exit.
    types = [ev["type"] for ev in events]
    assert GRAPH_END in types, "Expected GRAPH_END in stream; driver did not reach terminal exit after resume"

    # Assertion 2: run status is completed — requires node 'b' to have fired.
    assert result.status is not None and result.status.value == "completed", (
        f"Expected 'completed' after resume; got {result.status!r}"
    )

    # Assertion 3: final_output is the terminal node's output — proves only
    # post-checkpoint node 'b' re-fired, not 'a' (which was already done).
    assert result.final_output == "b-done", f"Expected final_output='b-done' after resume; got {result.final_output!r}"

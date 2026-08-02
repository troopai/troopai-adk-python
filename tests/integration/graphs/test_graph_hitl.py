"""Graph HITL interrupt suspension (non-streaming and streaming)."""

from __future__ import annotations

from typing import Any

from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
from troopai.adk.graphs.events import GRAPH_END, NODE_INTERRUPT
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.interrupt import GraphResume, Interrupt, request_human_input
from troopai.adk.graphs.result import GraphRunStatus
from troopai.adk.orchestration.executable import ExecutableInput
from troopai.adk.run.runner import Runner

# ---- Helpers ---------------------------------------------------------


def _interrupting_node(inp: ExecutableInput, ctx: Any) -> str:
    """Single-attempt node that always requests a human approval.

    The two-arg (ExecutableInput, context) signature ensures the
    CallableExecutable dispatcher passes the full envelope — allowing
    access to ``inp.metadata`` where the loop injects the node id and,
    on resume, the human reply.
    """
    reply = request_human_input(inp, "approve?", kind="tool_approval", tool="x")
    return f"approved:{reply}"


# ---- Tests -----------------------------------------------------------


async def test_single_node_interrupt_surfaces_interrupted_and_pending() -> None:
    """A node that raises InterruptException suspends the run cleanly."""
    g = Graph.new("hitl-1").node("ask", _interrupting_node).entry("ask").terminal("ask").compile()
    result = await Runner.arun_graph(g, "go")
    assert result.status == GraphRunStatus.INTERRUPTED
    assert len(result.interrupts) == 1
    iv: Interrupt = result.interrupts[0]
    assert iv.node_id == "ask"
    assert iv.question == "approve?"
    assert iv.kind == "tool_approval"
    # pending_interrupts mirrored on the returned state
    assert result.state is not None
    assert "ask" in result.state.pending_interrupts
    assert result.state.status == "interrupted"
    # NOT a failure: no exception raised by the runner; no error populated
    assert result.error is None


async def test_concurrent_fanout_interrupts_collect_all_and_apply_sibling_outputs() -> None:
    """root → (a interrupts in parallel with b completing) → join.

    Node b completes normally; its output is recorded in state.node_results.
    Node a is collected into pending_interrupts.
    The join node never fires because the run suspends.
    """

    def _b(text: str) -> str:
        return f"b-done:{text}"

    def _join(text: str) -> str:
        return f"join:{text}"

    g = (
        Graph.new("hitl-fanout")
        .node("root", lambda: "go")
        .node("a", _interrupting_node)
        .node("b", _b)
        .node("join", _join)
        .edge("root", "a")
        .edge("root", "b")
        .edge("a", "join")
        .edge("b", "join")
        .entry("root")
        .terminal("join")
        .compile()
    )
    result = await Runner.arun_graph(g, "go")
    assert result.status == GraphRunStatus.INTERRUPTED
    assert {"a"} == {iv.node_id for iv in result.interrupts}
    assert result.state is not None
    assert "a" in result.state.pending_interrupts
    assert "b" not in result.state.pending_interrupts
    # Sibling b completed and was recorded — no fail-fast cancel on interrupt.
    assert "b" in result.state.node_results
    # join did NOT fire
    assert "join" not in result.state.node_results


async def test_no_interrupt_run_is_byte_unchanged() -> None:
    """An interrupt-free run yields a normal COMPLETED result.

    Confirms the interrupt branch is dead when no node raises
    InterruptException and parity with pre-existing behaviour is
    preserved.
    """
    g = (
        Graph.new("hitl-parity")
        .node("a", lambda: "a-done")
        .node("b", lambda t: f"b:{t}")
        .edge("a", "b")
        .entry("a")
        .terminal("b")
        .compile()
    )
    result = await Runner.arun_graph(g, "go")
    assert result.status == GraphRunStatus.COMPLETED
    # The default concat_text merge prepends the source label, so b receives
    # "[a]\na-done" and returns "b:[a]\na-done".
    assert result.final_output == "b:[a]\na-done"
    assert len(result.interrupts) == 0
    assert result.state is not None
    assert len(result.state.pending_interrupts) == 0


async def test_resume_with_replies_completes_run() -> None:
    """Interrupt the run, supply a reply via GraphResume, resume to completion.

    The interrupted node sees the human-supplied value and the run finishes.
    The concat_text merge labels the output with the source node id before
    passing it to the downstream node.
    """
    g = (
        Graph.new("hitl-resume-1")
        .node("ask", _interrupting_node)
        .node("after", lambda t: f"after:{t}")
        .edge("ask", "after")
        .entry("ask")
        .terminal("after")
        .compile()
    )
    cp = InMemoryCheckpointer()
    first = await Runner.arun_graph(g, "go", hooks=[cp], thread_id="run-1")
    assert first.status == GraphRunStatus.INTERRUPTED
    assert first.state is not None
    assert "ask" in first.state.pending_interrupts

    second = await Runner.arun_graph_from_checkpoint(
        g,
        checkpointer=cp,
        thread_id="run-1",
        resume=GraphResume(replies={"ask": "the-answer"}),
    )
    assert second.status == GraphRunStatus.COMPLETED
    # concat_text labels the source: "after" receives "[ask]\napproved:the-answer"
    assert second.final_output == "after:[ask]\napproved:the-answer"
    assert second.state is not None
    assert "ask" not in second.state.pending_interrupts


async def test_resume_via_graph_runner_resume_from_threads_resume_kw() -> None:
    """GraphRunner.resume_from(...).arun(resume=GraphResume(...)) reaches
    the same code path as arun_graph_from_checkpoint(..., resume=...).
    """
    g = Graph.new("hitl-resume-profile").node("ask", _interrupting_node).entry("ask").terminal("ask").compile()
    cp = InMemoryCheckpointer()
    first = await Runner.arun_graph(g, "go", hooks=[cp], thread_id="run-2")
    assert first.status == GraphRunStatus.INTERRUPTED

    second = (
        await Runner.configure()
        .graph(g)
        .resume_from(cp, "run-2")
        .arun("go", resume=GraphResume(replies={"ask": "yes"}))
    )
    assert second.status == GraphRunStatus.COMPLETED
    assert second.final_output == "approved:yes"


async def test_resume_rejected_delivers_message_as_reply() -> None:
    """GraphResume.rejected[node_id] delivers the rejection message as the
    reply value. Mirrors the agent state.reject(message=) idiom.
    """
    g = Graph.new("hitl-reject").node("ask", _interrupting_node).entry("ask").terminal("ask").compile()
    cp = InMemoryCheckpointer()
    first = await Runner.arun_graph(g, "go", hooks=[cp], thread_id="run-3")
    assert first.status == GraphRunStatus.INTERRUPTED

    second = await Runner.arun_graph_from_checkpoint(
        g,
        checkpointer=cp,
        thread_id="run-3",
        resume=GraphResume(rejected={"ask": "denied by reviewer"}),
    )
    assert second.status == GraphRunStatus.COMPLETED
    assert second.final_output == "approved:denied by reviewer"


async def test_resume_with_unanswered_pending_re_surfaces_interrupted() -> None:
    """A pending interrupt with no reply or rejection re-surfaces INTERRUPTED
    on resume and the same pending entry persists for a subsequent resume.
    """
    g = Graph.new("hitl-unanswered").node("ask", _interrupting_node).entry("ask").terminal("ask").compile()
    cp = InMemoryCheckpointer()
    first = await Runner.arun_graph(g, "go", hooks=[cp], thread_id="run-4")
    assert first.status == GraphRunStatus.INTERRUPTED

    second = await Runner.arun_graph_from_checkpoint(
        g,
        checkpointer=cp,
        thread_id="run-4",
        resume=GraphResume(),
    )
    assert second.status == GraphRunStatus.INTERRUPTED
    assert second.state is not None
    assert "ask" in second.state.pending_interrupts


# ---- Streaming HITL (T6) ------------------------------------------------


async def test_streamed_interrupt_emits_node_interrupt_event_and_completes_cleanly() -> None:
    """A streamed run with an interrupting node emits NodeInterruptEvent and
    GraphEndEvent(status=interrupted), populates result.status/.interrupts,
    and the consumer's async-for ends normally without raising.
    """
    g = Graph.new("hitl-stream-1").node("ask", _interrupting_node).entry("ask").terminal("ask").compile()
    result = await Runner.arun_graph_streamed(g, "go")
    events = [ev async for ev in result.stream_events()]
    types = [ev["type"] for ev in events]
    assert NODE_INTERRUPT in types
    ni = next(ev for ev in events if ev["type"] == NODE_INTERRUPT)
    assert ni["node_id"] == "ask"
    iv: Interrupt = ni["interrupt"]
    assert iv.node_id == "ask"
    assert iv.question == "approve?"
    end = next(ev for ev in events if ev["type"] == GRAPH_END)
    assert end["status"].value == "interrupted"
    assert result.status == GraphRunStatus.INTERRUPTED
    assert len(result.interrupts) == 1
    assert result.interrupts[0].node_id == "ask"


async def test_streamed_resume_via_builder_completes() -> None:
    """Streamed resume: builder.resume_from(cp, tid).arun(stream=True, resume=...) completes the run."""
    g = (
        Graph.new("hitl-stream-resume")
        .node("ask", _interrupting_node)
        .node("after", lambda t: f"after:{t}")
        .edge("ask", "after")
        .entry("ask")
        .terminal("after")
        .compile()
    )
    cp = InMemoryCheckpointer()
    first = await Runner.arun_graph(g, "go", hooks=[cp], thread_id="stream-1")
    assert first.status == GraphRunStatus.INTERRUPTED

    streamed = await (
        Runner.configure()
        .graph(g)
        .resume_from(cp, "stream-1")
        .arun("go", stream=True, resume=GraphResume(replies={"ask": "yes"}))
    )
    events = [ev async for ev in streamed.stream_events()]
    end = next(ev for ev in events if ev["type"] == GRAPH_END)
    assert end["status"].value == "completed"
    assert streamed.status == GraphRunStatus.COMPLETED
    assert streamed.final_output == "after:[ask]\napproved:yes"


async def test_streamed_interrupt_free_run_is_byte_unchanged() -> None:
    """An interrupt-free streamed run yields no NodeInterruptEvent and the
    existing structural-event surface is unchanged.
    """
    g = (
        Graph.new("hitl-stream-parity")
        .node("a", lambda: "a-done")
        .node("b", lambda t: f"b:{t}")
        .edge("a", "b")
        .entry("a")
        .terminal("b")
        .compile()
    )
    result = await Runner.arun_graph_streamed(g, "go")
    events = [ev async for ev in result.stream_events()]
    types = [ev["type"] for ev in events]
    assert NODE_INTERRUPT not in types
    assert result.status == GraphRunStatus.COMPLETED
    assert result.final_output == "b:[a]\na-done"
    assert len(result.interrupts) == 0

"""BSP loop opens graph + node spans around the run.

When a tracer is installed, executing a graph via ``Runner.arun_graph``
records one :class:`GraphSpanData` for the whole run plus one
:class:`GraphNodeSpanData` per node — with the node span CLOSED at
the matching ``on_node_*`` hook site (success, error, or interrupt).

The tests use the same custom-span-routing :class:`CustomSpanData`
shape as the factories (see ``tests/unit/tracing/test_graph_spans.py``)
so the recording tracer sees the discriminator + payload without
needing a graph-aware Tracer protocol.
"""

from __future__ import annotations

from typing import Any

from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.interrupt import request_human_input
from troopai.adk.graphs.result import GraphRunStatus
from troopai.adk.orchestration.executable import ExecutableInput
from troopai.adk.run.runner import Runner
from troopai.adk.tracing import Span, set_tracer
from troopai.adk.types.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SpanData,
)


class _Recorder:
    """In-memory tracer recording every span created — for assertion in tests."""

    def __init__(self) -> None:
        self.spans: list[SpanData] = []

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        self.spans.append(data)
        return Span(data)

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        self.spans.append(data)
        return Span(data)

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        self.spans.append(data)
        return Span(data)

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        self.spans.append(data)
        return Span(data)

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        self.spans.append(data)
        return Span(data)

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        self.spans.append(data)
        return Span(data)

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        self.spans.append(data)
        return Span(data)


def _graph_spans(recorder: _Recorder, kind: str) -> list[CustomSpanData]:
    """Filter recorded spans to graph-typed CustomSpanData matching ``kind``."""
    out: list[CustomSpanData] = []
    for s in recorder.spans:
        if isinstance(s, CustomSpanData) and s.data.get("type") == kind:
            out.append(s)
    return out


async def test_arun_graph_opens_one_graph_span_for_the_whole_run() -> None:
    recorder = _Recorder()
    set_tracer(recorder)
    try:
        g = Graph.new("t4-graph").node("a", lambda: "done").entry("a").terminal("a").compile()
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.COMPLETED

        graph_spans = _graph_spans(recorder, "graph")
        assert len(graph_spans) == 1
        assert graph_spans[0].name == "graph.t4-graph"
        assert graph_spans[0].data["graph_id"] == "t4-graph"
    finally:
        set_tracer(None)


async def test_graph_span_records_status_completed_on_success() -> None:
    recorder = _Recorder()
    set_tracer(recorder)
    try:
        g = Graph.new("pa1-ok").node("a", lambda: "done").entry("a").terminal("a").compile()
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.COMPLETED

        graph_spans = _graph_spans(recorder, "graph")
        assert len(graph_spans) == 1
        assert graph_spans[0].data["status"] == "completed"
    finally:
        set_tracer(None)


async def test_graph_span_records_status_interrupted_on_suspend() -> None:
    recorder = _Recorder()
    set_tracer(recorder)

    def _ask(inp: ExecutableInput, ctx: Any) -> str:
        del ctx
        reply = request_human_input(inp, "approve?", kind="tool_approval", tool="x")
        return f"approved:{reply}"

    try:
        g = Graph.new("pa1-int").node("ask", _ask).entry("ask").terminal("ask").compile()
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.INTERRUPTED

        graph_spans = _graph_spans(recorder, "graph")
        assert len(graph_spans) == 1
        assert graph_spans[0].data["status"] == "interrupted"
    finally:
        set_tracer(None)


async def test_graph_span_records_status_failed_on_error() -> None:
    recorder = _Recorder()
    set_tracer(recorder)

    def _boom() -> str:
        raise RuntimeError("boom")

    try:
        g = Graph.new("pa1-fail").node("a", _boom).entry("a").terminal("a").compile()
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.FAILED

        graph_spans = _graph_spans(recorder, "graph")
        assert len(graph_spans) == 1
        assert graph_spans[0].data["status"] == "failed"
    finally:
        set_tracer(None)


async def test_graph_span_records_supersteps_total() -> None:
    recorder = _Recorder()
    set_tracer(recorder)
    try:
        g = (
            Graph.new("pa1-multi")
            .node("a", lambda: "a-done")
            .node("b", lambda text: f"b-of-{text}")
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.COMPLETED

        graph_spans = _graph_spans(recorder, "graph")
        assert len(graph_spans) == 1
        # 2-node sequential graph → 2 supersteps.
        assert graph_spans[0].data["supersteps_total"] == 2
    finally:
        set_tracer(None)


async def test_arun_graph_opens_one_node_span_per_node_on_success() -> None:
    recorder = _Recorder()
    set_tracer(recorder)
    try:
        g = (
            Graph.new("t4-nodes")
            .node("a", lambda: "a-done")
            .node("b", lambda text: f"b-of-{text}")
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.COMPLETED

        node_spans = _graph_spans(recorder, "graph_node")
        assert {s.data["node_name"] for s in node_spans} == {"a", "b"}
        for span in node_spans:
            assert span.data["graph_id"] == "t4-nodes"
    finally:
        set_tracer(None)


async def test_node_span_closes_when_node_errors() -> None:
    recorder = _Recorder()
    set_tracer(recorder)

    def _boom() -> str:
        raise RuntimeError("boom")

    try:
        g = Graph.new("t4-error").node("a", _boom).entry("a").terminal("a").compile()
        result = await Runner.arun_graph(g, "go")
        # fail_fast defaults to True; the run surfaces FAILED.
        assert result.status == GraphRunStatus.FAILED

        node_spans = _graph_spans(recorder, "graph_node")
        assert len(node_spans) == 1
        assert node_spans[0].data["node_name"] == "a"
    finally:
        set_tracer(None)


async def test_node_span_closes_when_node_suspends() -> None:
    recorder = _Recorder()
    set_tracer(recorder)

    def _ask(inp: ExecutableInput, ctx: Any) -> str:
        del ctx
        reply = request_human_input(inp, "approve?", kind="tool_approval", tool="x")
        return f"approved:{reply}"

    try:
        g = Graph.new("t4-suspend").node("ask", _ask).entry("ask").terminal("ask").compile()
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.INTERRUPTED

        node_spans = _graph_spans(recorder, "graph_node")
        assert len(node_spans) == 1
        assert node_spans[0].data["node_name"] == "ask"
    finally:
        set_tracer(None)


async def test_arun_graph_opens_one_superstep_span_per_superstep() -> None:
    recorder = _Recorder()
    set_tracer(recorder)
    try:
        g = (
            Graph.new("t5-supersteps")
            .node("a", lambda: "a-done")
            .node("b", lambda text: f"b-of-{text}")
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.COMPLETED

        superstep_spans = _graph_spans(recorder, "graph_superstep")
        # 2-node sequential graph → 2 supersteps (a fires in #1, b in #2).
        assert len(superstep_spans) == 2
        indices = sorted(s.data["index"] for s in superstep_spans)
        assert indices == [1, 2]
        for span in superstep_spans:
            assert span.data["graph_id"] == "t5-supersteps"
    finally:
        set_tracer(None)


async def test_node_span_records_status_success_on_clean_completion() -> None:
    recorder = _Recorder()
    set_tracer(recorder)
    try:
        g = Graph.new("t6-ok").node("a", lambda: "a-done").entry("a").terminal("a").compile()
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.COMPLETED

        node_spans = _graph_spans(recorder, "graph_node")
        assert len(node_spans) == 1
        assert node_spans[0].data["status"] == "success"
    finally:
        set_tracer(None)


async def test_node_span_records_status_interrupted_on_suspend() -> None:
    recorder = _Recorder()
    set_tracer(recorder)

    def _ask(inp: ExecutableInput, ctx: Any) -> str:
        del ctx
        reply = request_human_input(inp, "approve?", kind="tool_approval", tool="x")
        return f"approved:{reply}"

    try:
        g = Graph.new("t6-suspend").node("ask", _ask).entry("ask").terminal("ask").compile()
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.INTERRUPTED

        node_spans = _graph_spans(recorder, "graph_node")
        assert len(node_spans) == 1
        assert node_spans[0].data["status"] == "interrupted"
    finally:
        set_tracer(None)


async def test_node_span_records_status_failed_on_error() -> None:
    recorder = _Recorder()
    set_tracer(recorder)

    def _boom() -> str:
        raise RuntimeError("boom")

    try:
        g = Graph.new("t6-fail").node("a", _boom).entry("a").terminal("a").compile()
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.FAILED

        node_spans = _graph_spans(recorder, "graph_node")
        assert len(node_spans) == 1
        assert node_spans[0].data["status"] == "failed"
    finally:
        set_tracer(None)


async def test_node_span_records_attempts_one_on_clean_first_try() -> None:
    recorder = _Recorder()
    set_tracer(recorder)
    try:
        g = Graph.new("t7-once").node("a", lambda: "a-done").entry("a").terminal("a").compile()
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.COMPLETED

        node_spans = _graph_spans(recorder, "graph_node")
        assert len(node_spans) == 1
        assert node_spans[0].data["attempts"] == 1
    finally:
        set_tracer(None)


async def test_node_span_records_attempts_after_retry_then_success() -> None:
    """A flaky node that succeeds on the second attempt records attempts=2."""
    from troopai.adk.graphs.config import GraphConfig, NodeRetryPolicy

    recorder = _Recorder()
    set_tracer(recorder)

    call_count = {"n": 0}

    def _flaky() -> str:
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise RuntimeError("flake")
        return "ok"

    try:
        g = (
            Graph.new("t7-retry")
            .node("a", _flaky)
            .entry("a")
            .terminal("a")
            .with_config(GraphConfig(default_retry=NodeRetryPolicy(max_attempts=3, initial_backoff=0.001)))
            .compile()
        )
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.COMPLETED

        node_spans = _graph_spans(recorder, "graph_node")
        assert len(node_spans) == 1
        assert node_spans[0].data["attempts"] == 2
    finally:
        set_tracer(None)


async def test_node_span_records_attempts_after_retries_exhausted() -> None:
    """An always-failing node with retry policy records attempts=max_attempts."""
    from troopai.adk.graphs.config import GraphConfig, NodeRetryPolicy

    recorder = _Recorder()
    set_tracer(recorder)

    def _always_fails() -> str:
        raise RuntimeError("nope")

    try:
        g = (
            Graph.new("pa2-exhausted")
            .node("a", _always_fails)
            .entry("a")
            .terminal("a")
            .with_config(GraphConfig(default_retry=NodeRetryPolicy(max_attempts=3, initial_backoff=0.001)))
            .compile()
        )
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.FAILED

        node_spans = _graph_spans(recorder, "graph_node")
        assert len(node_spans) == 1
        # All 3 attempts failed; the NodeRetriesExhaustedError exception
        # carries attempts=3 which the span surfaces.
        assert node_spans[0].data["attempts"] == 3
        assert node_spans[0].data["status"] == "failed"
    finally:
        set_tracer(None)


async def test_node_span_records_resume_attempt_after_hitl_resume() -> None:
    """A node that suspended and then resumed records resume_attempt=1 on the resumed span."""
    from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
    from troopai.adk.graphs.interrupt import GraphResume

    recorder = _Recorder()
    set_tracer(recorder)

    def _ask(inp: ExecutableInput, ctx: Any) -> str:
        del ctx
        reply = request_human_input(inp, "approve?", kind="tool_approval", tool="x")
        return f"approved:{reply}"

    try:
        cp = InMemoryCheckpointer()
        g = Graph.new("pa3-resume").node("ask", _ask).entry("ask").terminal("ask").compile()

        first = await Runner.arun_graph(g, "go", hooks=[cp], thread_id="pa3-thread")
        assert first.status == GraphRunStatus.INTERRUPTED

        second = await Runner.arun_graph_from_checkpoint(
            g,
            checkpointer=cp,
            thread_id="pa3-thread",
            resume=GraphResume(replies={"ask": "yes"}),
        )
        assert second.status == GraphRunStatus.COMPLETED

        # Two spans for "ask" across both runs: the first (interrupted) has
        # resume_attempt=None; the second (resumed) has resume_attempt=1.
        ask_spans = [s for s in _graph_spans(recorder, "graph_node") if s.data["node_name"] == "ask"]
        assert len(ask_spans) == 2
        resume_values = sorted(
            (s.data["resume_attempt"] for s in ask_spans),
            key=lambda v: -1 if v is None else v,
        )
        assert resume_values == [None, 1]
    finally:
        set_tracer(None)


async def test_superstep_span_records_fired_nodes() -> None:
    recorder = _Recorder()
    set_tracer(recorder)
    try:
        g = (
            Graph.new("t5-fired")
            .node("root", lambda: "go")
            .node("a", lambda text: f"a-of-{text}")
            .node("b", lambda text: f"b-of-{text}")
            .edge("root", "a")
            .edge("root", "b")
            .entry("root")
            .terminal("a")
            .terminal("b")
            .compile()
        )
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.COMPLETED

        superstep_spans = _graph_spans(recorder, "graph_superstep")
        # Superstep 1: root fires. Superstep 2: a and b fire in parallel.
        by_index = {s.data["index"]: s for s in superstep_spans}
        assert sorted(by_index.keys()) == [1, 2]
        assert by_index[1].data["fired_nodes"] == ["root"]
        assert sorted(by_index[2].data["fired_nodes"]) == ["a", "b"]
    finally:
        set_tracer(None)

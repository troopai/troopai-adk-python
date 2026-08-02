"""Regression tests for the graphs/ wave-2 bug-fix sweep.

Each test class targets a specific finding. Every test must FAIL against the
pre-fix code and PASS after the fix. Owned source: ``graphs/{state,graph,
hooks,node,result,nested_snapshot}.py`` and ``run/{graph_loop,node_reliability}.py``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any, cast, override

import pytest

from troopai.adk.exceptions import GraphNodeTimeoutError
from troopai.adk.graphs.config import GraphConfig, NodeRetryPolicy
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.hooks import GraphHooks, HookRegistry
from troopai.adk.graphs.interrupt import Interrupt, InterruptException
from troopai.adk.graphs.nested_snapshot import NestedSnapshot
from troopai.adk.graphs.result import GraphRunResultStreaming, GraphRunStatus
from troopai.adk.graphs.state import GraphState
from troopai.adk.orchestration.executable import ExecutableInput, NodeResult
from troopai.adk.run.config import DEFAULT_RUN_CONFIG
from troopai.adk.run.context import RunContext
from troopai.adk.run.graph_loop import (
    _build_join_barriers,
    _reconstruct_arrivals_from_state,
    _run_bsp_loop,
    run_graph_loop,
)
from troopai.adk.run.node_reliability import run_node_with_reliability
from troopai.adk.types.tokens.llm_usage import LLMUsage


def _noop() -> str:
    return "noop"


def _result(text: str) -> NodeResult:
    return NodeResult(output=text, usage=LLMUsage(), final_text=text)


async def _noop_emit(_: object) -> None:
    return None


# ---------------------------------------------------------------------------
# Finding: state.py — to_dict()/to_json() must be JSON-safe (HIGH)
# ---------------------------------------------------------------------------


class TestStateToJsonJsonSafe:
    """HIGH — state.py — non-JSON leaves (LLMUsage in nested-node metadata)
    used to crash json.dumps(state.to_dict()) and fail the durable
    checkpointer's save, failing the whole run."""

    def test_to_json_coerces_non_json_metadata(self) -> None:
        g = Graph.new("json-meta").node("n", _noop).entry("n").terminal("n").compile()
        state: GraphState = GraphState(graph=g)
        # A nested-graph node stamps ``per_node_usage`` = dict[str, LLMUsage]
        # onto its NodeResult.metadata — LLMUsage is a dataclass json cannot
        # serialise. Before the fix this raised TypeError inside to_json().
        state.record(
            "n",
            NodeResult(
                output="ok",
                final_text="ok",
                metadata={"per_node_usage": {"inner": LLMUsage(input_tokens=3, output_tokens=1)}},
            ),
        )
        raw = state.to_json()  # must NOT raise
        payload = json.loads(raw)
        coerced = payload["node_results"]["n"]["metadata"]["per_node_usage"]["inner"]
        assert isinstance(coerced, str)  # the LLMUsage leaf is str-coerced
        # And the whole thing rehydrates without error.
        restored = GraphState.from_json(raw, g)
        assert "n" in restored.node_results

    def test_to_json_coerces_non_json_final_output(self) -> None:
        g = Graph.new("json-final").node("n", _noop).entry("n").terminal("n").compile()
        state: GraphState = GraphState(graph=g)
        # Multi-terminal graphs put a dict on final_output; a non-JSON leaf
        # inside used to slip past the old "keep dict/list as-is" branch.
        state.final_output = {"terminal": LLMUsage(input_tokens=2)}
        payload = json.loads(state.to_json())  # must NOT raise
        assert isinstance(payload["final_output"]["terminal"], str)

    def test_json_safe_preserves_plain_structures(self) -> None:
        g = Graph.new("json-plain").node("n", _noop).entry("n").terminal("n").compile()
        state: GraphState = GraphState(graph=g)
        state.record("n", NodeResult(output={"k": [1, 2, "x"]}, final_text=None, metadata={"tag": "v"}))
        payload = json.loads(state.to_json())
        assert payload["node_results"]["n"]["output"] == {"k": [1, 2, "x"]}
        assert payload["node_results"]["n"]["metadata"] == {"tag": "v"}


# ---------------------------------------------------------------------------
# Finding: result.py:404 — stream_events must cancel the driver on early close
# ---------------------------------------------------------------------------


class TestStreamEventsCancelsDriverOnEarlyClose:
    """LOW — result.py — an early break/aclose used to await the FULL run in
    the finally, defeating early termination and keeping the graph running."""

    async def test_early_close_cancels_driver_and_node_tasks(self) -> None:
        r: GraphRunResultStreaming = GraphRunResultStreaming()
        completed = False
        node_cancelled = asyncio.Event()
        loop = asyncio.get_running_loop()

        async def driver() -> None:
            nonlocal completed
            await r.put_event({"type": "graph.start"})
            await asyncio.sleep(1.0)  # ongoing work
            completed = True
            await r.complete()

        async def fake_node() -> None:
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                node_cancelled.set()
                raise

        task = loop.create_task(driver())
        r.set_run_task(task)
        node = loop.create_task(fake_node())
        r.register_node_task(node)

        # stream_events is annotated AsyncIterator (no aclose) but is a real
        # async generator at runtime; cast so aclose() type-checks.
        gen = cast("AsyncGenerator[Any, None]", r.stream_events())
        first = await gen.__anext__()
        assert first == {"type": "graph.start"}
        await gen.aclose()  # consumer stops early
        await asyncio.gather(task, node, return_exceptions=True)

        # After the fix the driver is cancelled instead of run to completion,
        # and the in-flight node task is cancelled too.
        assert completed is False
        assert task.cancelled()
        assert node_cancelled.is_set()


# ---------------------------------------------------------------------------
# Finding: graph.py:381 — inner MAX_SUPERSTEPS/MAX_TOKENS/NO_READY_NODES must
# not be a silent empty success (MED)
# ---------------------------------------------------------------------------


class TestNestedGraphNonCompletedSurfaces:
    """MED — graph.py — a nested graph that hits a budget cap or deadlocks
    used to return a partial/empty NodeResult the outer graph read as
    success."""

    async def test_inner_max_supersteps_raises(self) -> None:
        # start -> end needs 2 supersteps; capped at 1 => MAX_SUPERSTEPS.
        inner = (
            Graph.new("inner-budget")
            .node("start", _noop)
            .node("end", _noop)
            .edge("start", "end")
            .entry("start")
            .terminal("end")
            .with_config(GraphConfig(max_supersteps=1))
            .compile()
        )
        input_ = ExecutableInput(content=[], from_node=None, edge_label=None, metadata={})
        ctx: RunContext[Any] = RunContext(context=None)
        with pytest.raises(RuntimeError, match="did not complete"):
            await inner.invoke(input_, ctx, DEFAULT_RUN_CONFIG)


# ---------------------------------------------------------------------------
# Finding: hooks.py:322 — a hook InterruptException must surface as INTERRUPTED
# ---------------------------------------------------------------------------


class _InterruptingHook(GraphHooks[Any]):
    @override
    async def on_node_end(self, context: Any, state: Any, node_id: str, result: Any) -> None:
        del context, state, result
        raise InterruptException(Interrupt(node_id=node_id, question="hook-pause"))


class TestHookInterruptSurfacesInterrupted:
    """MED — hooks.py — a lifecycle hook raising InterruptException used to be
    swallowed (propagate_errors=False) or downgraded to a generic FAILED."""

    async def test_hook_interrupt_yields_interrupted_not_failed(self) -> None:
        g = Graph.new("hook-int").node("n", _noop).entry("n").terminal("n").compile()
        ctx: RunContext[Any] = RunContext(context=None)
        result = await run_graph_loop(
            graph=g,
            user_prompt="go",
            context=ctx,
            config=DEFAULT_RUN_CONFIG,
            hooks=[_InterruptingHook()],
        )
        assert result.status == GraphRunStatus.INTERRUPTED
        assert result.error is None
        assert result.state is not None
        assert "n" in result.state.pending_interrupts


# ---------------------------------------------------------------------------
# Finding: node_reliability.py:150 — body TimeoutError must not be masked as
# GraphNodeTimeoutError(0.0) when no per-attempt timeout was configured (LOW)
# ---------------------------------------------------------------------------


class TestBodyTimeoutNotMaskedWithoutConfiguredTimeout:
    """LOW — node_reliability.py — a TimeoutError from the node body under no
    configured timeout used to be re-typed as GraphNodeTimeoutError(0.0),
    masking the original error."""

    async def test_body_timeout_reraised_unchanged(self) -> None:
        sentinel = TimeoutError("body timed out")

        async def invoke() -> NodeResult:
            raise sentinel

        with pytest.raises(TimeoutError) as ei:
            await run_node_with_reliability(
                node_id="a",
                policy=NodeRetryPolicy(max_attempts=1),
                timeout=None,
                invoke=invoke,
            )
        assert ei.value is sentinel
        assert not isinstance(ei.value, GraphNodeTimeoutError)


# ---------------------------------------------------------------------------
# Finding: nested_snapshot.py:122 — graph-kind from_dict must resolve the
# inner graph off the parent node executable (LOW)
# ---------------------------------------------------------------------------


class TestNestedSnapshotGraphKindRoundTrip:
    """LOW — nested_snapshot.py — graph-kind from_dict used to validate the
    inner GraphState against the PARENT graph, rejecting every inner node
    id."""

    def _graphs(self) -> tuple[Graph[Any], Graph[Any]]:
        inner = Graph.new("inner").node("x", _noop).entry("x").terminal("x").compile()
        parent = Graph.new("parent").node("g", inner).entry("g").terminal("g").compile()
        return inner, parent

    def test_graph_kind_round_trip_resolves_inner_graph(self) -> None:
        inner, parent = self._graphs()
        inner_state: GraphState = GraphState(graph=inner, superstep=2)
        inner_state.record("x", NodeResult(output="ok", final_text="ok"))
        snap = NestedSnapshot(kind="graph", graph_state=inner_state)
        restored = NestedSnapshot.from_dict(snap.to_dict(), parent, node_id="g")
        assert restored.kind == "graph"
        assert restored.graph_state is not None
        assert restored.graph_state.superstep == 2
        assert "x" in restored.graph_state.node_results

    def test_graph_kind_without_node_id_raises_clear_error(self) -> None:
        inner, parent = self._graphs()
        snap = NestedSnapshot(kind="graph", graph_state=GraphState(graph=inner))
        with pytest.raises(ValueError, match="requires node_id"):
            NestedSnapshot.from_dict(snap.to_dict(), parent)


# ---------------------------------------------------------------------------
# Finding: graph_loop.py:1597 — a same-superstep failure must be persisted,
# not discarded behind the interrupt (MED)
# ---------------------------------------------------------------------------


class TestFailureAlongsideInterruptPersisted:
    """MED — graph_loop.py — when one node interrupts and another errors in
    the same superstep, the run still pauses (INTERRUPTED — the interrupt is
    NOT dropped, per the prior deliberate decision) but the failure used to
    be silently discarded; it must now be persisted on the state.
    """

    async def test_failure_alongside_interrupt_is_persisted(self) -> None:
        g = (
            Graph.new("ff-persist")
            .node("start", _noop)
            .node("a", _noop)
            .node("b", _noop)
            .edge("start", "a")
            .edge("start", "b")
            .entry("start")
            .terminal("a", "b")
            .compile()
        )
        state: GraphState = GraphState(graph=g, thread_id=None)
        barriers = _build_join_barriers(g)
        registry = HookRegistry()
        a_interrupted = asyncio.Event()

        async def fake_node_runner(*, graph: Any, node_id: str, input: Any, context: Any, config: Any) -> NodeResult:
            del graph, input, context, config
            if node_id == "start":
                return NodeResult(output="start")
            if node_id == "a":
                a_interrupted.set()
                raise InterruptException(Interrupt(node_id="a", question="approve?"))
            if node_id == "b":
                await a_interrupted.wait()  # ensure a's interrupt is collected first
                raise ValueError("boom-b")
            return NodeResult(output=node_id)

        status, _err = await _run_bsp_loop(
            graph=g,
            user_prompt="hi",
            context=None,  # type: ignore[arg-type]
            config=None,  # type: ignore[arg-type]
            state=state,
            barriers=barriers,
            registry=registry,
            graph_path=(g.id,),
            emit=_noop_emit,
            node_runner=fake_node_runner,  # type: ignore[arg-type]
        )
        # The interrupt is not dropped: the run pauses.
        assert status == GraphRunStatus.INTERRUPTED
        assert "a" in state.pending_interrupts
        # But the sibling failure is persisted rather than silently discarded.
        assert state.error is not None
        assert "b" in state.error


# ---------------------------------------------------------------------------
# Finding: graph_loop.py:1156 — reconstruct must honour edge predicates (MED)
# ---------------------------------------------------------------------------


class TestReconstructHonoursEdgePredicates:
    """MED — graph_loop.py — reconstructing a parked node's arrivals used to
    include every upstream, ignoring the edge ``when`` predicate."""

    async def test_predicate_false_upstream_excluded(self) -> None:
        g = (
            Graph.new("recon-pred")
            .node("start", _noop)
            .node("a", _noop)
            .node("b", _noop)
            .node("c", _noop)
            .edge("start", "a")
            .edge("start", "b")
            .edge("a", "c")
            .edge("b", "c", when=lambda r: False)
            .entry("start")
            .terminal("c")
            .compile()
        )
        state: GraphState = GraphState(graph=g)
        state.record("a", _result("a-out"))
        state.record("b", _result("b-out"))
        results, sources = await _reconstruct_arrivals_from_state(graph=g, node_id="c", state=state)
        assert sources == ["a"]  # b excluded — its edge predicate returns False
        assert len(results) == 1
        assert results[0].final_text == "a-out"


# ---------------------------------------------------------------------------
# Finding: node.py:228 + graph_loop.py:1342/1371 — edge label must be threaded
# to the downstream ExecutableInput (MED)
# ---------------------------------------------------------------------------


class TestEdgeLabelThreaded:
    """MED — graph_loop.py — the firing edge's label was hard-coded None on
    prepare_node_input, so the downstream never saw which branch fired."""

    async def test_single_upstream_edge_label_threaded(self) -> None:
        g = (
            Graph.new("edge-label")
            .node("a", _noop)
            .node("b", _noop)
            .edge("a", "b", label="approved")
            .entry("a")
            .terminal("b")
            .compile()
        )
        state: GraphState = GraphState(graph=g, thread_id=None)
        barriers = _build_join_barriers(g)
        registry = HookRegistry()
        seen: dict[str, str | None] = {}

        async def fake_node_runner(*, graph: Any, node_id: str, input: Any, context: Any, config: Any) -> NodeResult:
            del graph, context, config
            seen[node_id] = input.edge_label
            return NodeResult(output=f"r-{node_id}")

        status, err = await _run_bsp_loop(
            graph=g,
            user_prompt="hi",
            context=None,  # type: ignore[arg-type]
            config=None,  # type: ignore[arg-type]
            state=state,
            barriers=barriers,
            registry=registry,
            graph_path=(g.id,),
            emit=_noop_emit,
            node_runner=fake_node_runner,  # type: ignore[arg-type]
        )
        assert status == GraphRunStatus.COMPLETED
        assert err is None
        assert seen["a"] is None  # entry node has no incoming edge
        assert seen["b"] == "approved"


# ---------------------------------------------------------------------------
# Finding: graph_loop.py:1624 — external cancel must cancel pending node tasks
# (MED)
# ---------------------------------------------------------------------------


class TestExternalCancelCancelsPendingNodeTasks:
    """MED — graph_loop.py — an external cancel of the driver used to orphan
    in-flight node tasks (asyncio.wait never cancels the futures it awaits)."""

    async def test_cancel_driver_cancels_in_flight_node(self) -> None:
        node_started = asyncio.Event()
        node_cancelled = asyncio.Event()

        async def blocking(inp: ExecutableInput, ctx: Any) -> str:
            del inp, ctx
            node_started.set()
            try:
                await asyncio.Event().wait()  # never fires
            except asyncio.CancelledError:
                node_cancelled.set()
                raise
            return "done"  # pragma: no cover

        g = Graph.new("cancel-node").node("n", blocking).entry("n").terminal("n").compile()
        ctx: RunContext[Any] = RunContext(context=None)
        driver = asyncio.create_task(run_graph_loop(graph=g, user_prompt="go", context=ctx, config=DEFAULT_RUN_CONFIG))
        await asyncio.wait_for(node_started.wait(), timeout=2.0)
        driver.cancel()
        with pytest.raises(asyncio.CancelledError):
            await driver
        # After the fix the loop teardown cancels the still-running node task;
        # before the fix it is orphaned and this times out.
        await asyncio.wait_for(node_cancelled.wait(), timeout=2.0)

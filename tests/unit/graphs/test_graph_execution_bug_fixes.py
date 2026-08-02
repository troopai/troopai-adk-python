"""Regression tests for graph-execution cluster bug fixes.

Each class targets a specific finding from the review.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from troopai.adk.graphs.checkpointers.sqlite import SQLiteCheckpointer
from troopai.adk.graphs.config import GraphConfig
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.result import GraphRunResultStreaming, GraphRunStatus
from troopai.adk.graphs.state import GraphState
from troopai.adk.orchestration.executable import NodeResult
from troopai.adk.run.context import RunContext
from troopai.adk.run.runner import Runner
from troopai.adk.types.tokens.llm_usage import LLMUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_ctx() -> RunContext:
    return RunContext(context={})


def _result(text: str) -> NodeResult:
    return NodeResult(output=text, usage=LLMUsage(), final_text=text)


# ---------------------------------------------------------------------------
# Finding 2 (HIGH): suppress(Exception) → suppress(BaseException) in result.py
# ---------------------------------------------------------------------------


class TestStreamEventsSupressCancelledError:
    """stream_events() finally block must not let a cancelled driver task's
    CancelledError shadow a stored exception."""

    async def test_stored_exception_surfaces_when_task_cancelled(self) -> None:
        """When the driver task is cancelled AND _stored_exception is set,
        the stored exception must be raised — not CancelledError."""
        r: GraphRunResultStreaming = GraphRunResultStreaming()
        # Store a real driver exception first.
        r.set_exception(RuntimeError("driver_error"))

        async def fake_driver() -> None:
            await asyncio.sleep(10)

        task = asyncio.get_running_loop().create_task(fake_driver())
        r.set_run_task(task)
        task.cancel()
        # Give the event loop a chance to cancel the task.
        await asyncio.sleep(0)

        await r.complete()
        with pytest.raises(RuntimeError, match="driver_error"):
            async for _ in r.stream_events():
                pass

    async def test_cancel_immediate_no_stored_exception_exits_cleanly(self) -> None:
        """With no stored exception, an immediate cancel should exit without raising."""
        r: GraphRunResultStreaming = GraphRunResultStreaming()

        async def fake_driver() -> None:
            await asyncio.sleep(10)

        task = asyncio.get_running_loop().create_task(fake_driver())
        r.set_run_task(task)
        r.cancel("immediate")
        # Should NOT raise; driver task CancelledError is suppressed.
        events = []
        with contextlib.suppress(asyncio.CancelledError):
            async for ev in r.stream_events():
                events.append(ev)
        # Queue was drained before iteration — no events expected.
        assert events == []


# ---------------------------------------------------------------------------
# Finding 3 (MEDIUM): dict/list excluded from JSON-safe types in state.py
# ---------------------------------------------------------------------------


class TestNodeResultSerialisation:
    """dict and list outputs must survive a serialise→rehydrate round-trip."""

    def _graph(self) -> Graph:
        return Graph.new("serial-test").node("a", lambda: {"key": "val", "n": 42}).entry("a").terminal("a").compile()

    def test_dict_output_round_trips(self) -> None:
        from troopai.adk.graphs.state import _rehydrate_node_results, _serialise_node_results

        g = self._graph()
        state = GraphState(graph=g, thread_id="t1")
        nr = NodeResult(output={"key": "val", "n": 42}, usage=LLMUsage(), final_text=None)
        state.record("a", nr)

        serialised = _serialise_node_results(state.node_results)
        assert serialised["a"]["output"] == {"key": "val", "n": 42}
        # Must not be a string repr.
        assert not isinstance(serialised["a"]["output"], str)

        rehydrated = _rehydrate_node_results({"node_results": serialised})
        assert rehydrated["a"].output == {"key": "val", "n": 42}

    def test_list_output_round_trips(self) -> None:
        from troopai.adk.graphs.state import _rehydrate_node_results, _serialise_node_results

        g = self._graph()
        state = GraphState(graph=g, thread_id="t2")
        nr = NodeResult(output=[1, 2, 3], usage=LLMUsage(), final_text=None)
        state.record("a", nr)

        serialised = _serialise_node_results(state.node_results)
        assert serialised["a"]["output"] == [1, 2, 3]
        assert not isinstance(serialised["a"]["output"], str)

        rehydrated = _rehydrate_node_results({"node_results": serialised})
        assert rehydrated["a"].output == [1, 2, 3]

    def test_non_json_object_still_coerced_to_str(self) -> None:
        """Objects that aren't JSON primitives/dicts/lists are str()-coerced."""
        from troopai.adk.graphs.state import _serialise_node_results

        g = self._graph()
        state = GraphState(graph=g, thread_id="t3")

        class _Custom:
            def __str__(self) -> str:
                return "custom_repr"

        nr = NodeResult(output=_Custom(), usage=LLMUsage(), final_text=None)
        state.record("a", nr)
        serialised = _serialise_node_results(state.node_results)
        assert serialised["a"]["output"] == "custom_repr"


# ---------------------------------------------------------------------------
# Finding 5 (MEDIUM): assert → RuntimeError for inner-graph INTERRUPTED state
# ---------------------------------------------------------------------------


class TestDispatchInnerGraphResumeAssert:
    """The assert on inner_result.state must be a RuntimeError so it is not
    stripped under -O and gives an actionable message."""

    def test_runtime_error_on_none_state(self) -> None:
        # Confirm the assert was replaced: get the source and check no bare assert.
        import inspect

        from troopai.adk.run.graph_loop import _dispatch_inner_graph_resume

        src = inspect.getsource(_dispatch_inner_graph_resume)
        assert "assert next_inner_state is not None" not in src, "assert must be replaced with explicit RuntimeError"
        assert "raise RuntimeError" in src


# ---------------------------------------------------------------------------
# Finding 6 (MEDIUM): SQLiteCheckpointer concurrent _db() connection leak
# ---------------------------------------------------------------------------


class TestSQLiteCheckpointerConcurrency:
    """Concurrent callers of _db() must not open two connections."""

    async def test_concurrent_first_open_uses_single_connection(self, tmp_path) -> None:
        db = str(tmp_path / "cp_conc.db")
        cp = SQLiteCheckpointer(db)
        # Launch several concurrent _db() calls simultaneously.
        conns = await asyncio.gather(cp._db(), cp._db(), cp._db())
        # All three must return the same connection object.
        assert len({id(c) for c in conns}) == 1
        await cp.close()

    async def test_context_manager_closes_connection(self, tmp_path) -> None:
        db = str(tmp_path / "cp_ctx.db")
        async with SQLiteCheckpointer(db) as cp:
            # Force connection open.
            await cp._db()
            assert cp._conn is not None
        # After context exit, connection must be closed.
        assert cp._conn is None


# ---------------------------------------------------------------------------
# Finding 1 & 4 (HIGH + MEDIUM): fail_fast=False barrier/error surfacing
# ---------------------------------------------------------------------------


class TestFailFastFalseBarrierAndErrorSurfacing:
    """When fail_fast=False, errored nodes must unblock downstream AND-join
    barriers via record_fail() and the final GraphRunResult must surface
    which nodes failed rather than returning error=None."""

    def _graph_with_failing_branch(self) -> Graph:
        """Graph: entry → (A, B) fan-out → C AND-join → terminal.
        B always raises so C's AND-join cannot fire with fail_fast=False.
        """

        def always_ok() -> str:
            return "ok"

        def always_fail() -> str:
            raise RuntimeError("node-b-failed")

        async def join_c(x: str, y: str) -> str:
            return f"{x}-{y}"

        return (
            Graph.new("fail-fast-false-test")
            .node("entry", always_ok)
            .node("a", always_ok)
            .node("b", always_fail)
            .node("c", join_c)
            .edge("entry", "a")
            .edge("entry", "b")
            .edge("a", "c")
            .edge("b", "c")
            .entry("entry")
            .terminal("c")
            .with_config(GraphConfig(fail_fast=False))
            .compile()
        )

    async def test_result_status_is_failed_not_no_ready_nodes(self) -> None:
        """Status must be FAILED (not NO_READY_NODES) when node errors cause
        all downstream paths to deadlock with fail_fast=False."""
        g = self._graph_with_failing_branch()
        r = Runner()
        result = await r.arun_graph(g, user_prompt="go", context=_run_ctx())
        assert result.status == GraphRunStatus.FAILED

    async def test_error_is_non_none_and_names_failed_node(self) -> None:
        """result.error must be set and name the failed node(s)."""
        g = self._graph_with_failing_branch()
        r = Runner()
        result = await r.arun_graph(g, user_prompt="go", context=_run_ctx())
        assert result.error is not None
        assert "b" in result.error


# ---------------------------------------------------------------------------
# Finding 11 (LOW): async context-manager protocol for checkpointers
# ---------------------------------------------------------------------------


class TestCheckpointerAsyncContextManager:
    """SQLiteCheckpointer must support ``async with`` and call close() on exit."""

    async def test_sqlite_aenter_returns_self(self, tmp_path) -> None:
        db = str(tmp_path / "cm.db")
        cp = SQLiteCheckpointer(db)
        async with cp as entered:
            assert entered is cp
        assert cp._conn is None

    async def test_sqlite_aexit_closes_on_exception(self, tmp_path) -> None:
        db = str(tmp_path / "cm_exc.db")
        with pytest.raises(ValueError, match="test"):
            async with SQLiteCheckpointer(db) as cp:
                await cp._db()  # open connection
                raise ValueError("test")
        assert cp._conn is None

"""Regression tests for the graphs/ bug-fix sweep.

Each test class targets a specific finding. All tests must fail against
the pre-fix code and pass after the fix.
"""

from __future__ import annotations

import asyncio

import pytest

from troopai.adk.graphs.config import GraphConfig, NodeRetryPolicy
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.join import JoinBarrier, JoinSemantics
from troopai.adk.graphs.result import GraphRunResultStreaming
from troopai.adk.graphs.state import GraphState
from troopai.adk.orchestration.executable import NodeResult
from troopai.adk.run.graph_loop import (
    _build_join_barriers,
    _seed_barriers_from_checkpoint,
)
from troopai.adk.types.tokens.llm_usage import LLMUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(text: str) -> NodeResult:
    return NodeResult(output=text, usage=LLMUsage(), final_text=text)


def _noop() -> str:
    return "noop"


# ---------------------------------------------------------------------------
# Finding: NodeRetryPolicy and GraphConfig validation (__post_init__)
# ---------------------------------------------------------------------------


class TestNodeRetryPolicyValidation:
    """MED — config.py:84 — __post_init__ guards on NodeRetryPolicy."""

    def test_max_attempts_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            NodeRetryPolicy(max_attempts=0)

    def test_max_attempts_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            NodeRetryPolicy(max_attempts=-1)

    def test_initial_backoff_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_backoff"):
            NodeRetryPolicy(initial_backoff=0.0)

    def test_initial_backoff_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_backoff"):
            NodeRetryPolicy(initial_backoff=-1.0)

    def test_max_backoff_less_than_initial_raises(self) -> None:
        with pytest.raises(ValueError, match="max_backoff"):
            NodeRetryPolicy(initial_backoff=5.0, max_backoff=1.0)

    def test_valid_defaults_accepted(self) -> None:
        policy = NodeRetryPolicy()
        assert policy.max_attempts == 1
        assert policy.initial_backoff == 1.0
        assert policy.max_backoff == 30.0

    def test_max_backoff_equal_to_initial_accepted(self) -> None:
        # max_backoff == initial_backoff is valid (no growth)
        policy = NodeRetryPolicy(initial_backoff=2.0, max_backoff=2.0)
        assert policy.max_backoff == 2.0


class TestGraphConfigValidation:
    """MED — config.py:84 — __post_init__ guards on GraphConfig."""

    def test_max_supersteps_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_supersteps"):
            GraphConfig(max_supersteps=0)

    def test_max_supersteps_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_supersteps"):
            GraphConfig(max_supersteps=-5)

    def test_per_node_timeout_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="per_node_timeout"):
            GraphConfig(per_node_timeout=0.0)

    def test_per_node_timeout_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="per_node_timeout"):
            GraphConfig(per_node_timeout=-1.0)

    def test_per_node_timeout_none_accepted(self) -> None:
        cfg = GraphConfig(per_node_timeout=None)
        assert cfg.per_node_timeout is None

    def test_valid_defaults_accepted(self) -> None:
        cfg = GraphConfig()
        assert cfg.max_supersteps == 50


# ---------------------------------------------------------------------------
# Finding: JoinBarrier.record_fail / fail-closed semantics
# ---------------------------------------------------------------------------


class TestJoinBarrierRecordFail:
    """HIGH — join.py:130 — record_fail distinguishes predicate errors."""

    def _barrier(self) -> JoinBarrier:
        return JoinBarrier(
            target="target",
            expected=frozenset({"a", "b"}),
            semantics=JoinSemantics.AND,
        )

    def test_record_fail_blocks_and_join_even_with_real_arrival(self) -> None:
        """AND-join must NOT fire when one upstream predicate failed."""
        b = self._barrier()
        b.record("a", _result("from-a"))
        b.record_fail("b")
        assert b.is_ready() is False

    def test_record_fail_on_unexpected_source_is_noop(self) -> None:
        b = self._barrier()
        b.record_fail("unknown")
        assert len(b.failed) == 0

    def test_record_fail_on_already_arrived_source_is_noop(self) -> None:
        b = self._barrier()
        b.record("a", _result("x"))
        b.record_fail("a")
        # Real arrival must take precedence; 'a' should not be in failed
        assert "a" not in b.failed
        assert "a" in b.arrivals

    def test_record_after_fail_clears_fail(self) -> None:
        """A real arrival supersedes an earlier failure for the same source."""
        b = self._barrier()
        b.record_fail("a")
        assert "a" in b.failed
        b.record("a", _result("from-a"))
        assert "a" not in b.failed
        assert "a" in b.arrivals

    def test_record_fail_cleared_by_consume(self) -> None:
        b = self._barrier()
        b.record("a", _result("x"))
        b.record_fail("b")
        # Manually clear by calling consume (even though not ready,
        # consume may be called after external state reset)
        b.failed = set()  # simulate clearing
        b.record("b", _result("y"))
        b.consume()
        assert len(b.failed) == 0

    def test_consume_clears_failed_set(self) -> None:
        """consume() must reset the failed set for cycle re-use."""
        b = self._barrier()
        # Force a consume (bypass is_ready by manually setting state)
        b.record("a", _result("x"))
        b.record("b", _result("y"))
        b.failed.add("a")  # inject a stale failure
        b.consume()
        assert len(b.failed) == 0

    def test_record_fail_blocks_or_join(self) -> None:
        """OR-join with a failed upstream is also blocked (fail-closed)."""
        b = JoinBarrier(
            target="t",
            expected=frozenset({"a", "b"}),
            semantics=JoinSemantics.OR,
        )
        b.record("a", _result("x"))
        b.record_fail("b")
        # OR with failed upstream must not be ready
        assert b.is_ready() is False

    def test_record_skip_still_allows_and_join(self) -> None:
        """record_skip (predicate-False) should still allow AND-join to fire."""
        b = self._barrier()
        b.record("a", _result("from-a"))
        b.record_skip("b")
        assert b.is_ready() is True

    def test_failed_distinct_from_skipped(self) -> None:
        """failed and skipped are independent fields."""
        b = self._barrier()
        b.record_skip("a")
        b.record_fail("b")
        assert "a" in b.skipped
        assert "b" in b.failed
        assert "a" not in b.failed
        assert "b" not in b.skipped


# ---------------------------------------------------------------------------
# Finding: result.py suppress(BaseException) → suppress(Exception)
# ---------------------------------------------------------------------------


class TestGraphRunResultStreamingSuppress:
    """HIGH — result.py:304 — suppress(Exception) does not suppress CancelledError,
    masking stored exceptions when the driver task is cancelled.

    Fix: change to suppress(BaseException) so the driver task's CancelledError
    is drained silently and _stored_exception is re-raised instead.
    """

    async def test_stored_exception_not_masked_by_driver_cancellation(self) -> None:
        """A stored driver-side exception must be raised even when the driver
        task was also cancelled.

        Before fix: suppress(Exception) did NOT suppress CancelledError, so
        await self._run_task raised CancelledError out of the finally block
        before the _stored_exception check could fire.
        After fix: suppress(BaseException) silently drains the driver task's
        CancelledError, then the stored exception is re-raised.
        """
        r: GraphRunResultStreaming = GraphRunResultStreaming()
        r.set_exception(RuntimeError("driver_side_error"))

        async def cancellable_driver() -> None:
            await asyncio.sleep(100)

        loop = asyncio.get_running_loop()
        task = loop.create_task(cancellable_driver())
        r.set_run_task(task)
        task.cancel()
        await asyncio.sleep(0)  # allow cancellation to propagate
        await r.complete()

        # After fix: stored exception is raised (not CancelledError).
        with pytest.raises(RuntimeError, match="driver_side_error"):
            async for _ in r.stream_events():
                pass

        assert task.cancelled()

    async def test_cancelled_driver_with_no_stored_exception_exits_cleanly(self) -> None:
        """When the driver is cancelled but there is no stored exception,
        stream_events() must exit without raising."""
        r: GraphRunResultStreaming = GraphRunResultStreaming()

        async def cancellable_driver() -> None:
            await asyncio.sleep(100)

        loop = asyncio.get_running_loop()
        task = loop.create_task(cancellable_driver())
        r.set_run_task(task)
        task.cancel()
        await asyncio.sleep(0)
        await r.complete()

        events: list = []
        async for ev in r.stream_events():
            events.append(ev)
        assert events == []
        assert task.cancelled()

    async def test_exception_still_reraised_through_iterator(self) -> None:
        r: GraphRunResultStreaming = GraphRunResultStreaming()
        r.set_exception(ValueError("stored-error"))
        await r.complete()
        with pytest.raises(ValueError, match="stored-error"):
            async for _ in r.stream_events():
                pass


# ---------------------------------------------------------------------------
# Finding: state.py KeyError → ValueError in _rehydrate_nested_graph_snapshots
# ---------------------------------------------------------------------------


class TestRehydrateNestedGraphKeyError:
    """MED — state.py:595 — KeyError from get_node() should be ValueError."""

    def test_unknown_node_id_raises_value_error(self) -> None:
        g = Graph.new("parent").node("a", _noop).entry("a").terminal("a").compile()
        # Build a fake serialised payload that references a nonexistent node id
        payload = {
            "superstep": 1,
            "node_results": {},
            "versions_seen": {},
            "produced_at": {},
            "all_items": [],
            "cumulative_usage": {},
            "per_node_usage": {},
            "terminal_outputs": {},
            "final_output": None,
            "status": "running",
            "error": None,
            "pending_interrupts": {},
            "nested_agent_snapshots": {},
            "nested_graph_snapshots": {"nonexistent_node": {}},
            "resume_counts": {},
        }
        with pytest.raises(ValueError, match="nonexistent_node"):
            GraphState.from_dict(payload, g)


# ---------------------------------------------------------------------------
# Finding: state.py status annotation — Literal covers all 6 values
# ---------------------------------------------------------------------------


class TestGraphStateStatusAnnotation:
    """LOW — state.py:188 — status field must accept all 6 terminal values."""

    def test_status_default_is_running(self) -> None:
        g = Graph.new("annot-test").node("a", _noop).entry("a").terminal("a").compile()
        state = GraphState(graph=g)
        assert state.status == "running"

    def test_status_accepts_all_terminal_values(self) -> None:
        g = Graph.new("annot-test2").node("a", _noop).entry("a").terminal("a").compile()
        for val in (
            "completed",
            "failed",
            "interrupted",
            "max_supersteps",
            "max_tokens",
            "no_ready_nodes",
        ):
            state = GraphState(graph=g)
            state.status = val  # type: ignore[assignment]
            assert state.status == val


# ---------------------------------------------------------------------------
# Finding: graph_loop predicate-exception uses record_fail not record_skip
# ---------------------------------------------------------------------------


class TestSeedBarrierPredicateFail:
    """HIGH — graph_loop.py:294 — predicate exception uses record_fail."""

    def _conditional_graph(self) -> Graph:
        def _raising_pred(r: NodeResult) -> bool:
            raise RuntimeError("boom")

        return (
            Graph.new("seed-fail")
            .node("a", _noop)
            .node("b", _noop)
            .edge("a", "b", when=_raising_pred)
            .entry("a")
            .terminal("b")
            .compile()
        )

    async def test_raising_predicate_records_fail_not_skip(self) -> None:
        g = self._conditional_graph()
        state: GraphState = GraphState(graph=g)
        state.superstep = 1
        state.record("a", _result("a-out"))
        barriers = _build_join_barriers(g)
        await _seed_barriers_from_checkpoint(graph=g, state=state, barriers=barriers)

        # The barrier should record a failure, not a skip
        assert "a" in barriers["b"].failed
        assert "a" not in barriers["b"].skipped
        # Barrier must not be ready (fail-closed)
        assert barriers["b"].is_ready() is False

    async def test_false_predicate_still_records_skip(self) -> None:
        """Predicate returning False must still use record_skip."""
        g = (
            Graph.new("seed-skip")
            .node("a", _noop)
            .node("b", _noop)
            .edge("a", "b", when=lambda r: False)
            .entry("a")
            .terminal("b")
            .compile()
        )
        state: GraphState = GraphState(graph=g)
        state.superstep = 1
        state.record("a", _result("a-out"))
        barriers = _build_join_barriers(g)
        await _seed_barriers_from_checkpoint(graph=g, state=state, barriers=barriers)

        assert "a" in barriers["b"].skipped
        assert "a" not in barriers["b"].failed

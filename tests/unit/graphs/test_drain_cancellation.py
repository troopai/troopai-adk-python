"""Feature 2: Graceful drain cancellation.

Tests that:
- CancelMode.DRAIN exists and is a distinct value.
- GraphRunResultStreaming.cancel("drain") sets the drain mode.
- The drain predicate in _run_bsp_loop causes the loop to exit cleanly
  after the current superstep, without scheduling new nodes.
"""

from __future__ import annotations

import pytest

from troopai.adk.run.stream import CancelMode


class TestCancelModeDrain:
    """CancelMode.DRAIN constant exists and has the right value."""

    def test_drain_member_exists(self) -> None:
        assert CancelMode.DRAIN == "drain"

    def test_drain_is_distinct_from_other_modes(self) -> None:
        assert CancelMode.DRAIN != CancelMode.NONE
        assert CancelMode.DRAIN != CancelMode.IMMEDIATE
        assert CancelMode.DRAIN != CancelMode.AFTER_SUPERSTEP
        assert CancelMode.DRAIN != CancelMode.AFTER_TURN


class TestGraphRunResultStreamingDrainCancel:
    """GraphRunResultStreaming.cancel('drain') sets the drain mode without
    draining the queue or cancelling tasks."""

    async def test_cancel_drain_sets_mode(self) -> None:
        from troopai.adk.graphs.result import GraphRunResultStreaming

        r: GraphRunResultStreaming = GraphRunResultStreaming()
        assert r.cancel_mode == CancelMode.NONE
        r.cancel("drain")
        assert r.cancel_mode == CancelMode.DRAIN

    async def test_cancel_drain_does_not_cancel_run_task(self) -> None:
        """drain mode must not cancel the driver task (unlike immediate)."""
        import asyncio

        from troopai.adk.graphs.result import GraphRunResultStreaming

        r: GraphRunResultStreaming = GraphRunResultStreaming()

        async def long_task() -> None:
            await asyncio.sleep(10)

        task = asyncio.get_running_loop().create_task(long_task())
        r.set_run_task(task)
        r.cancel("drain")
        # Task should still be running — drain mode is cooperative, not immediate.
        assert not task.cancelled()
        assert not task.done()
        task.cancel()

    async def test_cancel_drain_does_not_drain_queue(self) -> None:
        """Drain mode must NOT empty the event queue (unlike immediate)."""
        from troopai.adk.graphs.result import GraphRunResultStreaming

        r: GraphRunResultStreaming = GraphRunResultStreaming()
        await r.put_event({"type": "ev"})
        r.cancel("drain")
        # Event must still be in the queue
        assert r._event_queue.qsize() == 1


class TestDrainPredicateInLoop:
    """_run_bsp_loop exits cleanly when drain_cancel() returns True."""

    def _make_drain_predicate(self, *, fire_after_n: int):
        """Returns a predicate that returns True after ``fire_after_n`` calls."""
        count = [0]

        def pred() -> bool:
            count[0] += 1
            return count[0] > fire_after_n

        return pred

    async def test_drain_cancel_fires_after_first_superstep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When drain_cancel() returns True after the first superstep, the loop
        must exit without starting a second superstep.

        We drive _run_bsp_loop directly with a trivial graph to avoid
        spinning up a real LLM.
        """

        from troopai.adk.graphs.graph import Graph
        from troopai.adk.graphs.hooks import HookRegistry
        from troopai.adk.graphs.join import JoinBarrier
        from troopai.adk.graphs.result import GraphRunStatus
        from troopai.adk.graphs.state import GraphState
        from troopai.adk.orchestration.executable import NodeResult
        from troopai.adk.run.graph_loop import _run_bsp_loop

        async def noop_emit(_: object) -> None:
            return None

        graph = (
            Graph.new("drain-test")
            .node("a", lambda: "a")
            .node("b", lambda: "b")
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )

        state = GraphState(graph=graph, thread_id=None)
        barriers = {"b": JoinBarrier(target="b", expected=frozenset({"a"}))}
        registry = HookRegistry()

        call_counts: dict[str, int] = {"a": 0, "b": 0}

        async def fake_node_runner(*, graph, node_id, input, context, config) -> NodeResult:
            call_counts[node_id] = call_counts.get(node_id, 0) + 1
            return NodeResult(output=f"result-{node_id}")

        # drain_cancel fires after the first check (superstep 0 boot, then True)
        drain_calls = [0]

        def drain_cancel() -> bool:
            drain_calls[0] += 1
            # Fire after node "a" has run (first real superstep completes)
            return drain_calls[0] > 1

        status, err = await _run_bsp_loop(
            graph=graph,
            user_prompt="hello",
            context=None,  # type: ignore[arg-type]
            config=None,  # type: ignore[arg-type]
            state=state,
            barriers=barriers,
            registry=registry,
            graph_path=(graph.id,),
            emit=noop_emit,
            node_runner=fake_node_runner,  # type: ignore[arg-type]
            drain_cancel=drain_cancel,
        )
        # Node "a" (entry) ran. Node "b" was NOT scheduled (drain fired).
        assert call_counts.get("a", 0) == 1
        assert call_counts.get("b", 0) == 0
        # Status must be NO_READY_NODES (no terminal fired before drain).
        assert status == GraphRunStatus.NO_READY_NODES
        assert err is None

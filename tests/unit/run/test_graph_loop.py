"""Regression tests for ``run/graph_loop.py`` BSP-driver edge cases.

Covers three confirmed defects:

1. ``_dispatch_node`` must NOT route an ``asyncio.CancelledError`` through
   the per-node error handler. Cooperative cancellation (a fail-fast
   sibling cancel or an immediate streamed cancel) has to propagate so the
   task is genuinely cancelled — never completing with a handler-supplied
   fallback that masks the cancellation.
2. The fail-fast exit must drain the WHOLE ``asyncio.wait`` ``done`` batch
   before cancelling siblings. A peer that succeeded or parked on an
   ``InterruptException`` in the same batch must not be silently dropped
   when it happens to be visited after the errored task.
3. The per-superstep tracing span must be closed on every escape path —
   including a user hook raising mid-superstep — rather than leaking.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from troopai.adk.graphs.config import GraphConfig
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.hooks import GraphHooks
from troopai.adk.graphs.interrupt import request_human_input
from troopai.adk.graphs.result import GraphRunStatus
from troopai.adk.graphs.state import GraphState
from troopai.adk.orchestration.executable import ExecutableInput, NodeResult
from troopai.adk.run.config import RunConfig
from troopai.adk.run.context import RunContext
from troopai.adk.run.graph_loop import _dispatch_node, run_graph_loop
from troopai.adk.run.runner import Runner

# ---------------------------------------------------------------------------
# Finding 1 — CancelledError must not be swallowed by the node error handler.
# ---------------------------------------------------------------------------


class TestDispatchNodeCancellation:
    """``_dispatch_node`` re-raises ``CancelledError`` instead of recovering."""

    async def test_cancel_does_not_invoke_error_handler_or_return_fallback(self) -> None:
        started = asyncio.Event()
        handler_calls: list[BaseException] = []

        def recover(_node_id: str, exc: BaseException) -> NodeResult[Any]:
            # If cancellation were (wrongly) routed here, this fallback would
            # let the task complete "successfully" and swallow the cancel.
            handler_calls.append(exc)
            return NodeResult(output="recovered")

        async def _blocking_runner(
            *,
            graph: Graph[Any],
            node_id: str,
            input: ExecutableInput,
            context: RunContext[Any],
            config: RunConfig,
        ) -> NodeResult[Any]:
            del graph, node_id, input, context, config
            started.set()
            await asyncio.Event().wait()  # blocks until cancelled
            return NodeResult(output="never")  # pragma: no cover

        g: Graph[Any] = (
            Graph.new("cancel-test").node("a", lambda: "a", on_error=recover).entry("a").terminal("a").compile()
        )

        task = asyncio.create_task(
            _dispatch_node(
                graph=g,
                node_id="a",
                prepared_input=ExecutableInput(content=[]),
                context=RunContext(context=None),
                config=RunConfig(),
                state=GraphState(graph=g),
                node_runner=_blocking_runner,
                is_streaming=False,
            )
        )

        await started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # The cancellation must propagate as a genuine cancel.
        assert task.cancelled() is True
        # The error handler must NEVER see a CancelledError.
        assert len(handler_calls) == 0


# ---------------------------------------------------------------------------
# Finding 2 — fail-fast must drain the whole ``done`` batch first.
# ---------------------------------------------------------------------------


class TestFailFastDrainsDoneBatch:
    """A sibling completing in the same batch as an errored node is kept."""

    async def test_sibling_interrupt_in_same_batch_is_not_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Two parallel nodes fan out from a root. In superstep 2 both are
        # ready and both complete in the SAME asyncio.wait batch: one raises
        # (fail-fast trigger), one parks on an interrupt. We force the
        # errored task to be visited FIRST so the buggy ``break`` would drop
        # the interrupt — deterministically reproducing the defect.
        import troopai.adk.run.graph_loop as graph_loop

        def _boom(_text: str) -> str:
            raise RuntimeError("boom")

        def _ask(inp: ExecutableInput, _ctx: Any) -> str:
            return f"ok:{request_human_input(inp, 'approve?', kind='generic')}"

        g: Graph[Any] = (
            Graph.new("drain-test")
            .node("root", lambda: "go")
            .node("boom", _boom)
            .node("ask", _ask)
            .edge("root", "boom")
            .edge("root", "ask")
            .entry("root")
            .terminal("boom")
            .terminal("ask")
            .with_config(GraphConfig(fail_fast=True))
            .compile()
        )

        real_wait = asyncio.wait

        async def ordered_wait(aws: Any, **kwargs: Any) -> Any:
            done, pending = await real_wait(aws, **kwargs)

            # Surface ``done`` as a list with the errored task first so the
            # for-loop visits the failure before the interrupting sibling.
            def _is_error(t: asyncio.Task[Any]) -> bool:
                e = t.exception() if t.done() and not t.cancelled() else None
                return isinstance(e, RuntimeError)

            ordered = sorted(done, key=lambda t: 0 if _is_error(t) else 1)
            return ordered, pending

        monkeypatch.setattr(graph_loop.asyncio, "wait", ordered_wait)

        result = await run_graph_loop(
            graph=g,
            user_prompt="go",
            context=RunContext(context=None),
            config=RunConfig(),
        )

        # The interrupting sibling must have been captured despite the
        # fail-fast error in the same batch. Before the fix the interrupt
        # was dropped and the run reported FAILED with no pending interrupt.
        assert result.status == GraphRunStatus.INTERRUPTED
        assert result.state is not None
        assert "ask" in result.state.pending_interrupts


# ---------------------------------------------------------------------------
# Finding 3 — the superstep span must always be closed (no leak on hook raise).
# ---------------------------------------------------------------------------


class _RaisingHooks(GraphHooks[Any]):
    """Hooks whose ``on_superstep_end`` raises after the span is opened."""

    # propagate so the exception escapes the registry fan-out (best-effort
    # hooks are otherwise logged and swallowed) and exercises the leak path.
    propagate_errors = True

    async def on_superstep_end(
        self,
        context: RunContext[Any],
        state: Any,
        fired_nodes: tuple[str, ...],
        new_items: list[Any],
    ) -> None:
        del context, state, fired_nodes, new_items
        raise RuntimeError("hook-blew-up")


class _TrackingTracer:
    """Tracer recording every span so we can assert each one was finished."""

    def __init__(self) -> None:
        from troopai.adk.tracing.spans import Span

        self._Span = Span
        self.spans: list[Any] = []

    def custom_span(self, data: Any) -> Any:
        span = self._Span(data)
        self.spans.append(span)
        return span

    # The graph loop only uses custom_span; the remaining protocol methods
    # are present for completeness and route through the same recorder.
    def agent_span(self, data: Any) -> Any:
        return self.custom_span(data)

    def function_span(self, data: Any) -> Any:
        return self.custom_span(data)

    def generation_span(self, data: Any) -> Any:
        return self.custom_span(data)

    def response_span(self, data: Any) -> Any:
        return self.custom_span(data)

    def handoff_span(self, data: Any) -> Any:
        return self.custom_span(data)

    def guardrail_span(self, data: Any) -> Any:
        return self.custom_span(data)


def _superstep_spans(tracer: _TrackingTracer) -> list[Any]:
    """Filter recorded spans to the per-superstep ones."""
    from troopai.adk.types.tracing.span_data import CustomSpanData

    return [
        s for s in tracer.spans if isinstance(s.data, CustomSpanData) and s.data.data.get("type") == "graph_superstep"
    ]


class TestSuperstepSpanClosedOnHookError:
    """A hook raising mid-superstep must not leak the open superstep span."""

    async def test_superstep_span_finished_when_hook_raises(self) -> None:
        from troopai.adk.tracing import set_tracer

        tracer = _TrackingTracer()
        set_tracer(tracer)
        try:
            g: Graph[Any] = Graph.new("span-leak-test").node("a", lambda: "done").entry("a").terminal("a").compile()
            result = await Runner.arun_graph(g, "go", hooks=[_RaisingHooks()])
            # The hook raised → the run fails, but the span must still close.
            assert result.status == GraphRunStatus.FAILED

            superstep_spans = _superstep_spans(tracer)
            assert len(superstep_spans) == 1
            # The defect left the span open (_finished False) when the hook
            # raised before the explicit finish() call.
            assert superstep_spans[0]._finished is True
        finally:
            set_tracer(None)

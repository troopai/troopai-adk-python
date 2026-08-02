"""Unit tests for :class:`FlowExecutor`.

Covers basic step sequencing, parallel @flow_start fan-out, AND/OR gate
semantics, max_steps cap, and error policies.
"""

from __future__ import annotations

from pydantic import BaseModel

from troopai.adk.flows import (
    Flow,
    FlowConfig,
    FlowEndEvent,
    FlowEvent,
    FlowStepCachePolicy,
    FlowStepContext,
    FlowTriggerEvent,
    flow_listen,
    flow_router,
    flow_start,
)
from troopai.adk.flows.exceptions import FlowAgentDeferred
from troopai.adk.flows.executor import FlowExecutor


class _State(BaseModel):
    events: list[str] = []


class TestControlFlowSignalsAreNotRetried:
    async def test_flow_agent_deferred_is_not_retried(self) -> None:
        """An internal control-flow signal must NOT be retried by _run_body.

        Regression: _run_body's broad ``except BaseException`` retried every
        non-cancellation exception, so a ``FlowAgentDeferred`` (an inner-agent
        HITL deferral raised from the step body) re-ran the body — and its
        inner agent — ``max_retries + 1`` times: double billing plus a
        corrupted deferral capture. Internal signals now propagate immediately.
        """
        calls = 0

        class F(Flow[_State]):
            @flow_start(max_retries=3)
            async def a(self) -> None:
                nonlocal calls
                calls += 1
                raise FlowAgentDeferred(step_name="a", defer_key="a", run_state_data="{}")

        result = await FlowExecutor(F(_State), config=FlowConfig()).run()

        # The body ran exactly once — the deferral was NOT retried.
        assert calls == 1, f"step body retried (ran {calls}x) on a control-flow signal"
        assert result.status == "deferred"

    async def test_batch_with_deferral_and_error_prefers_deferred(self) -> None:
        """A parallel batch that both defers and errors must stay recoverable.

        Regression: ``_process_batch_results`` early-returned on the first
        errored/rejected sibling, so a concurrently-deferred sibling's captured
        checkpoint was dropped — the HITL deferral became unrecoverable
        (status='failed', checkpoint=None). The deferral now takes precedence;
        the errored sibling is not in ``completed_steps`` and re-runs on resume,
        so nothing is permanently lost.
        """

        class F(Flow[_State]):
            @flow_start
            async def deferring(self) -> None:
                raise FlowAgentDeferred(step_name="deferring", defer_key="d", run_state_data="{}")

            @flow_start
            async def erroring(self) -> None:
                raise RuntimeError("boom")

        result = await FlowExecutor(F(_State), config=FlowConfig(error_policy="halt")).run()

        assert result.status == "deferred"
        assert result.checkpoint is not None
        assert len(result.deferred_steps) >= 1


class TestBasicSequencing:
    async def test_linear_chain(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None:
                self.state.events.append("a")

            @flow_listen(a)
            async def b(self) -> None:
                self.state.events.append("b")

        result = await FlowExecutor(F(_State), config=FlowConfig()).run()
        assert result.status == "completed"
        assert result.completed_steps == ("a", "b")
        assert result.final_state.events == ["a", "b"]

    async def test_multiple_starts_run_in_parallel(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None:
                self.state.events.append("a")

            @flow_start
            async def b(self) -> None:
                self.state.events.append("b")

        result = await FlowExecutor(F(_State), config=FlowConfig()).run()
        assert set(result.completed_steps) == {"a", "b"}
        assert set(result.final_state.events) == {"a", "b"}


class TestAndGate:
    async def test_and_gate_fires_after_both(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None:
                self.state.events.append("a")

            @flow_start
            async def b(self) -> None:
                self.state.events.append("b")

            @flow_listen(a & b)
            async def merged(self) -> None:
                self.state.events.append("merged")

        result = await FlowExecutor(F(_State), config=FlowConfig()).run()
        assert "merged" in result.completed_steps
        assert result.final_state.events[-1] == "merged"


class TestOrGate:
    async def test_or_gate_fires_once_on_first_arrival(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None:
                self.state.events.append("a")

            @flow_start
            async def b(self) -> None:
                self.state.events.append("b")

            @flow_listen(a | b)
            async def either(self) -> None:
                self.state.events.append("either")

        result = await FlowExecutor(F(_State), config=FlowConfig()).run()
        assert result.final_state.events.count("either") == 1


class TestRouter:
    async def test_router_dispatches_on_label(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def kickoff(self) -> None:
                self.state.events.append("kickoff")

            @flow_router(kickoff)
            async def route(self) -> str:
                return "selected"

            @flow_listen("selected")
            async def selected_branch(self) -> None:
                self.state.events.append("selected")

            @flow_listen("other")
            async def other_branch(self) -> None:
                self.state.events.append("other")

        result = await FlowExecutor(F(_State), config=FlowConfig()).run()
        assert "selected" in result.final_state.events
        assert "other" not in result.final_state.events

    async def test_router_returning_empty_string_surfaces_as_failure(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def k(self) -> None: ...

            @flow_router(k)
            async def route(self) -> str:
                return ""

        result = await FlowExecutor(F(_State), config=FlowConfig()).run()
        assert result.status == "failed"
        assert result.error is not None
        assert "non-empty string" in result.error


class TestMaxStepsCap:
    async def test_max_steps_halts(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None: ...

            @flow_listen(a)
            async def b(self) -> None: ...

            @flow_listen(b)
            async def c(self) -> None: ...

        result = await FlowExecutor(F(_State), config=FlowConfig(max_steps=2)).run()
        assert result.status == "halted_max_steps"


class TestErrorPolicy:
    async def test_halt_on_error(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None:
                raise RuntimeError("kaboom")

        result = await FlowExecutor(F(_State), config=FlowConfig(error_policy="halt")).run()
        assert result.status == "failed"
        assert result.error is not None
        assert "kaboom" in result.error

    async def test_route_to_error_handler(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None:
                raise RuntimeError("bad")

            @flow_listen("__error__")
            async def handle(self) -> None:
                self.state.events.append("handled")

        config = FlowConfig(error_policy="route_to_error_handler")
        result = await FlowExecutor(F(_State), config=config).run()
        assert "handled" in result.final_state.events


class TestCheckpointCarriesNonDeferredPending:
    async def test_deferral_checkpoint_keeps_sibling_successor(self) -> None:
        """A deferral checkpoint must carry successors scheduled by a sibling.

        Regression: in a fan-out batch where one branch defers via
        ``requires_approval`` and a sibling branch completes and schedules a
        downstream successor, ``_build_checkpoint`` read ``pending_queue_snapshot``
        — but that snapshot was only assigned AFTER ``_process_batch_results``
        returned, so it was still empty when the deferred terminal (and its
        checkpoint) was built. The sibling's successor was silently dropped from
        ``pending_steps`` and never re-fired on resume. The snapshot is now
        captured inside ``_process_batch_results`` before building the deferred
        result.
        """

        def always(_ctx: FlowStepContext[_State]) -> bool:
            return True

        class F(Flow[_State]):
            @flow_start
            async def completing(self) -> None:
                self.state.events.append("completing")

            @flow_listen("completing")
            async def successor(self) -> None:
                self.state.events.append("successor")

            @flow_start(requires_approval=always)
            async def deferring(self) -> None:
                self.state.events.append("deferring")

        result = await FlowExecutor(F(_State), config=FlowConfig()).run()

        assert result.status == "deferred"
        assert result.checkpoint is not None
        # The deferred step is queued first so it can pick up its decision...
        assert "deferring" in result.checkpoint.pending_steps
        # ...and the sibling's successor must NOT be lost — without it the
        # successor never runs on resume (silent step loss).
        assert "successor" in result.checkpoint.pending_steps


class TestSingleFlowEndEventPerRun:
    async def test_error_plus_deferral_emits_one_flow_end(self) -> None:
        """A batch that both errors and defers must emit exactly one FlowEndEvent.

        Regression: ``_handle_error`` built its own (later-discarded) failed
        terminal via ``_build_result``, which unconditionally emits a
        ``FlowEndEvent``. When a sibling deferred, the deferred terminal emitted
        a second ``FlowEndEvent`` — a streaming consumer saw both
        ``status='failed'`` and ``status='deferred'`` for one run. The terminal
        result is now built once, after the batch loop.
        """
        ends: list[FlowEndEvent] = []

        def sink(event: FlowEvent) -> None:
            if isinstance(event, FlowEndEvent):
                ends.append(event)

        def always(_ctx: FlowStepContext[_State]) -> bool:
            return True

        class F(Flow[_State]):
            @flow_start(requires_approval=always)
            async def deferring(self) -> None:
                self.state.events.append("deferring")

            @flow_start
            async def erroring(self) -> None:
                raise RuntimeError("boom")

        result = await FlowExecutor(F(_State), config=FlowConfig(error_policy="halt"), on_event=sink).run()

        assert result.status == "deferred"
        assert len(ends) == 1
        assert ends[0].status == "deferred"

    async def test_two_errors_in_one_batch_emit_one_flow_end(self) -> None:
        """Two halting errors in one batch must still emit a single FlowEndEvent."""
        ends: list[FlowEndEvent] = []

        def sink(event: FlowEvent) -> None:
            if isinstance(event, FlowEndEvent):
                ends.append(event)

        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None:
                raise RuntimeError("boom-a")

            @flow_start
            async def b(self) -> None:
                raise RuntimeError("boom-b")

        result = await FlowExecutor(F(_State), config=FlowConfig(error_policy="halt"), on_event=sink).run()

        assert result.status == "failed"
        assert len(ends) == 1
        assert ends[0].status == "failed"


class TestCacheDoesNotRouteNonRouterReturn:
    async def test_non_router_string_return_not_cached_as_route_label(self) -> None:
        """A cached non-router step's string return must NOT become a route label.

        Regression: ``_store_cache`` cached any string body return as a route
        label without checking the step's role. On a cache hit, ``_invoke_step``
        replays ``hit.route_label`` directly (bypassing ``_finalize_step``),
        which dispatches to ``@flow_listen("<that string>")`` and emits a route
        event the cache-miss path never produces. Non-router returns are ignored
        on the live path, so they must not be cached as a route label.
        """

        class F(Flow[_State]):
            @flow_start(cache=FlowStepCachePolicy(cache_key_fn=lambda ctx: "k"))
            async def producing(self) -> str:
                self.state.events.append("producing")
                # A legal non-router return value (ignored on the live path).
                return "successor"

            @flow_listen("successor")
            async def successor(self) -> None:
                self.state.events.append("successor-fired")

        executor = FlowExecutor(F(_State), config=FlowConfig())
        result = await executor.run()

        assert result.status == "completed"
        # The non-router return must not have dispatched the listener.
        assert "successor-fired" not in result.final_state.events
        # And the cache entry must store no route label for the non-router step.
        cache = executor.step_caches["producing"]
        (_created_at, _snapshot, route_label) = cache.entries["k"]
        assert route_label is None


class TestRouterLabelMatchingStepName:
    async def test_listener_fires_once_when_label_equals_step_name(self) -> None:
        """A router returning a label equal to a step name must fire the
        listener ONCE, not twice.

        Regression: ``_resolve_next`` combined
        ``_resolve_arrival(completed_step)`` and
        ``_resolve_arrival(route_label)`` without deduplication. When the
        router returned a label equal to an existing step's method name
        (here its own), the same listener landed twice in one batch and
        its body executed twice. The returned fire list is now deduped
        (first occurrence wins); both provenance ``FlowTriggerEvent``s are
        still recorded for the step's ``FlowStepContext``.
        """
        seen_triggers: list[tuple[FlowTriggerEvent, ...]] = []

        def record(ctx: FlowStepContext[_State]) -> bool:
            seen_triggers.append(ctx.triggers)
            return True

        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None: ...

            @flow_router(a)
            async def route(self) -> str:
                # Label equal to the router's own step name — the listener
                # on "route" matches BOTH the completion arrival and the
                # route-label arrival.
                return "route"

            @flow_listen("route", enabled=record)
            async def b(self) -> None:
                self.state.events.append("b")

        result = await FlowExecutor(F(_State), config=FlowConfig()).run()

        assert result.status == "completed"
        # The listener's body executed exactly once.
        assert result.final_state.events.count("b") == 1
        # ...and its single invocation still carries BOTH provenance events.
        assert len(seen_triggers) == 1
        assert {t.kind for t in seen_triggers[0]} == {"step_completion", "route_label"}

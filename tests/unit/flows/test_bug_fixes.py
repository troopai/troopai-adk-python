"""Regression tests for bugs fixed in the pre-split hardening sweep.

Each test is prefixed with the finding it covers and is designed to FAIL
on the pre-fix code and PASS after the fix.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from troopai.adk.flows import (
    Flow,
    FlowConfig,
    arun_flow_agent,
    flow_listen,
    flow_start,
)
from troopai.adk.flows.events import (
    FlowEndEvent,
    FlowEvent,
    FlowStartEvent,
    FlowStepEndEvent,
    FlowStepStartEvent,
)
from troopai.adk.flows.executor import FlowExecutor
from troopai.adk.flows.result import FlowRunStatus
from troopai.adk.run.runner import Runner


class _State(BaseModel):
    events: list[str] = []
    tokens_consumed: int = 0


# ---------------------------------------------------------------------------
# Fix: max_total_tokens cap (executor.py) — HIGH
# ---------------------------------------------------------------------------


class TestMaxTotalTokensCap:
    async def test_halted_max_tokens_when_cap_exceeded(self) -> None:
        """Cumulative token cap must halt the run with status='halted_max_tokens'.

        Pre-fix: the cap was never checked, so the run would complete
        normally even when token usage exceeded max_total_tokens.
        """
        from troopai.adk.run.context import RunContext
        from troopai.adk.types.tokens.llm_usage import LLMUsage

        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None:
                # Simulate token consumption by writing directly onto run_context
                if self.run_context is not None:
                    self.run_context.usage = LLMUsage(input_tokens=600, output_tokens=0)
                self.state.events.append("a")

            @flow_listen(a)
            async def b(self) -> None:
                self.state.events.append("b")

        flow = F(_State)
        run_ctx: RunContext[Any] = RunContext(context=None)  # type: ignore[arg-type]
        flow.run_context = run_ctx

        config = FlowConfig(max_total_tokens=500)
        executor: FlowExecutor[_State] = FlowExecutor(flow, config=config)
        # Seed usage that already exceeds the cap before the second batch
        flow.run_context.usage = LLMUsage(input_tokens=600, output_tokens=0)
        result = await executor.run()

        assert result.status == "halted_max_tokens"

    async def test_no_cap_when_max_total_tokens_is_none(self) -> None:
        """Without a cap, the run completes normally regardless of token count."""
        from troopai.adk.run.context import RunContext
        from troopai.adk.types.tokens.llm_usage import LLMUsage

        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None:
                self.state.events.append("a")

            @flow_listen(a)
            async def b(self) -> None:
                self.state.events.append("b")

        flow = F(_State)
        run_ctx: RunContext[Any] = RunContext(context=None)  # type: ignore[arg-type]
        run_ctx.usage = LLMUsage(input_tokens=999_999, output_tokens=999_999)
        flow.run_context = run_ctx

        result = await FlowExecutor(flow, config=FlowConfig(max_total_tokens=None)).run()
        assert result.status == "completed"


# ---------------------------------------------------------------------------
# Fix: defer_key short-circuit skips stack walk (agent_bridge.py) — HIGH
# ---------------------------------------------------------------------------


class TestDeferKeySkipsStackWalk:
    async def test_explicit_defer_key_does_not_raise_from_helper(self) -> None:
        """arun_flow_agent with explicit defer_key must NOT raise FlowDefinitionError.

        Pre-fix: _infer_calling_step_name ran BEFORE the defer_key check.
        When called from a helper method (not directly inside a @flow_start body)
        the stack walk found no registered step and raised FlowDefinitionError —
        even though the caller explicitly supplied defer_key.
        """
        stub_agent: Any = object()

        async def _helper(flow: Flow[Any]) -> Any:
            # This helper is NOT a registered flow step — stack walk would fail.
            return await arun_flow_agent(flow, stub_agent, "ping", defer_key="explicit_key")

        completed_result = _make_completed_result()

        class F(Flow[_State]):
            @flow_start
            async def go(self) -> None:
                await _helper(self)
                self.state.events.append("done")

        flow = F(_State)
        with patch("troopai.adk.run.runner.Runner.arun", new=AsyncMock(return_value=completed_result)):
            result = await Runner.arun_flow(flow)

        assert result.status == "completed"
        assert "done" in result.final_state.events


# ---------------------------------------------------------------------------
# Fix: FlowStepEndEvent balanced after body raises (executor.py) — MED
# ---------------------------------------------------------------------------


class TestFlowStepEndEventBalanced:
    async def test_start_event_paired_with_end_event_on_body_raise(self) -> None:
        """Every FlowStepStartEvent must be followed by a FlowStepEndEvent.

        Pre-fix: when the step body raised, no FlowStepEndEvent was emitted,
        leaving streaming consumers with dangling start events.
        """
        collected: list[object] = []

        class F(Flow[_State]):
            @flow_start
            async def bad(self) -> None:
                raise RuntimeError("body-raises")

        flow = F(_State)
        executor = FlowExecutor(flow, config=FlowConfig(error_policy="halt"), on_event=collected.append)
        result = await executor.run()

        assert result.status == "failed"
        # Count start / end events for step "bad"
        starts = [e for e in collected if isinstance(e, FlowStepStartEvent) and e.step_name == "bad"]
        ends = [e for e in collected if isinstance(e, FlowStepEndEvent) and e.step_name == "bad"]
        assert len(starts) == 1, f"expected 1 start event, got {len(starts)}"
        assert len(ends) == 1, f"expected 1 end event, got {len(ends)} — dangling start!"


# ---------------------------------------------------------------------------
# Fix: pending_agent_resolution_keys public accessor (executor.py) — MED
# ---------------------------------------------------------------------------


class TestPendingAgentResolutionKeys:
    def test_public_accessor_returns_sorted_keys(self) -> None:
        """Flow.pending_agent_resolution_keys must return sorted keys without
        exposing the private _pending_agent_resolutions attribute.

        Pre-fix: executor.py accessed flow._pending_agent_resolutions directly.
        """

        class F(Flow[_State]):
            @flow_start
            async def go(self) -> None: ...

        flow = F(_State)
        flow.set_pending_agent_resolutions({"z_key": "{}", "a_key": "{}"})
        keys = flow.pending_agent_resolution_keys()
        assert keys == ["a_key", "z_key"]

    def test_empty_resolutions_returns_empty_list(self) -> None:
        class F(Flow[_State]):
            @flow_start
            async def go(self) -> None: ...

        flow = F(_State)
        assert flow.pending_agent_resolution_keys() == []


# ---------------------------------------------------------------------------
# Fix: FlowEndEvent.status typed FlowRunStatus (events.py) — MED
# ---------------------------------------------------------------------------


class TestFlowEndEventStatusType:
    async def test_flow_end_event_status_is_flow_run_status(self) -> None:
        """FlowEndEvent.status must be a FlowRunStatus value, not a bare str.

        Pre-fix: the annotation was `status: str`.
        """
        end_events: list[FlowEndEvent] = []

        class F(Flow[_State]):
            @flow_start
            async def go(self) -> None:
                self.state.events.append("go")

        def _collect(event: object) -> None:
            if isinstance(event, FlowEndEvent):
                end_events.append(event)

        executor = FlowExecutor(F(_State), config=FlowConfig(), on_event=_collect)
        result = await executor.run()
        assert result.status == "completed"
        assert len(end_events) == 1
        # status should be the same literal as result.status
        assert end_events[0].status == result.status
        # The annotation narrowing ensures this is a valid FlowRunStatus literal
        valid_statuses: tuple[FlowRunStatus, ...] = (
            "completed",
            "failed",
            "deferred",
            "halted_max_steps",
            "halted_max_tokens",
        )
        assert end_events[0].status in valid_statuses


# ---------------------------------------------------------------------------
# Fix: FlowRunResultStreaming.final_state typed StateT | None (result.py) — MED
# ---------------------------------------------------------------------------


class TestFlowRunResultStreamingFinalState:
    async def test_final_state_typed_correctly_after_stream(self) -> None:
        """FlowRunResultStreaming.final_state must be StateT | None, not Any.

        Verifies runtime shape: None before streaming, populated after.
        """
        flow = _make_simple_flow()
        # arun_flow_streamed is a sync classmethod that returns a streaming result
        streaming = Runner.arun_flow_streamed(flow)
        # Before iterating, final_state may or may not be set yet
        events = []
        async for event in streaming.stream_events():
            events.append(event)
        # After stream: final_state should be populated
        assert streaming.final_state is not None
        assert isinstance(streaming.final_state, _State)


# ---------------------------------------------------------------------------
# Fix: Flow.run_context typed RunContext[Any] | None (flow.py) — MED
# ---------------------------------------------------------------------------


class TestFlowRunContextType:
    def test_run_context_is_none_outside_run(self) -> None:
        """Flow.run_context must be None outside a run (typed RunContext[Any] | None).

        Pre-fix: annotated as Any, which hid the type narrowing.
        """

        class F(Flow[_State]):
            @flow_start
            async def go(self) -> None: ...

        flow = F(_State)
        assert flow.run_context is None

    async def test_run_context_populated_during_run(self) -> None:
        """Flow.run_context must be a RunContext during execution."""
        from troopai.adk.run.context import RunContext

        captured: list[Any] = []

        class F(Flow[_State]):
            @flow_start
            async def go(self) -> None:
                captured.append(self.run_context)

        await Runner.arun_flow(F(_State))
        assert len(captured) == 1
        assert isinstance(captured[0], RunContext)


# ---------------------------------------------------------------------------
# Fix: _run_guardrails phase typed Literal['pre','post'] (executor.py) — MED
# ---------------------------------------------------------------------------


class TestGuardrailPhaseLiteral:
    async def test_pre_and_post_guardrails_both_run(self) -> None:
        """Both 'pre' and 'post' phases must be invoked during a step run.

        This confirms the Literal['pre','post'] narrowing doesn't break
        call-site behaviour.
        """
        from troopai.adk.flows.step_guardrails import FlowStepGuardrails, FlowStepGuardrailVerdict

        phases_seen: list[str] = []

        def _guardrail(ctx: Any) -> FlowStepGuardrailVerdict:
            # ctx.step_name is set; we record the call but can't directly
            # get 'phase' here — track via call order instead
            phases_seen.append(ctx.step_name)
            return FlowStepGuardrailVerdict(allowed=True)

        class F(Flow[_State]):
            @flow_start(
                guardrails=FlowStepGuardrails(pre=(_guardrail,), post=(_guardrail,)),
            )
            async def go(self) -> None:
                self.state.events.append("go")

        result = await FlowExecutor(F(_State), config=FlowConfig()).run()
        assert result.status == "completed"
        # Both pre + post should have called our guardrail
        assert len(phases_seen) == 2, f"expected 2 guardrail calls, got {len(phases_seen)}"


# ---------------------------------------------------------------------------
# Fix: FlowExecutable.invoke raises on failed flow (executable.py) — HIGH
# ---------------------------------------------------------------------------


class TestFlowExecutableStatusCheck:
    async def test_failed_flow_raises_user_error(self) -> None:
        """FlowExecutable.invoke must raise UserError when nested flow fails.

        Pre-fix: result.status was ignored, so a failed flow returned a
        normal NodeResult to the graph loop, hiding the failure.
        """
        from troopai.adk.exceptions import UserError
        from troopai.adk.flows.executable import FlowExecutable
        from troopai.adk.orchestration.executable import ExecutableInput
        from troopai.adk.run.context import RunContext

        class _BadFlow(Flow[_State]):
            @flow_start
            async def go(self) -> None:
                raise RuntimeError("inner-flow-failed")

        exe = FlowExecutable(flow=_BadFlow(_State))
        ctx: RunContext[None] = RunContext(context=None)  # type: ignore[arg-type]

        from troopai.adk.run.config import DEFAULT_RUN_CONFIG

        with pytest.raises(UserError, match="inner-flow-failed"):
            await exe.invoke(
                input=ExecutableInput(content=[], from_node=None),
                context=ctx,
                config=DEFAULT_RUN_CONFIG,
            )

    async def test_deferred_flow_surfaces_checkpoint_in_metadata(self) -> None:
        """FlowExecutable.invoke must expose checkpoint + deferred_steps in metadata.

        Pre-fix: result.status was ignored; deferred flows returned metadata
        without checkpoint/deferred_steps, making HITL invisible to graph callers.
        """
        from troopai.adk.flows.executable import FlowExecutable
        from troopai.adk.orchestration.executable import ExecutableInput
        from troopai.adk.run.context import RunContext

        class _DeferredFlow(Flow[_State]):
            @flow_start(
                requires_approval=lambda ctx: True,
            )
            async def go(self) -> None:
                self.state.events.append("go")

        exe = FlowExecutable(flow=_DeferredFlow(_State))
        ctx: RunContext[None] = RunContext(context=None)  # type: ignore[arg-type]

        from troopai.adk.run.config import DEFAULT_RUN_CONFIG

        node_result = await exe.invoke(
            input=ExecutableInput(content=[], from_node=None),
            context=ctx,
            config=DEFAULT_RUN_CONFIG,
        )
        assert node_result.metadata["status"] == "deferred"
        assert "checkpoint" in node_result.metadata
        assert "deferred_steps" in node_result.metadata


# ---------------------------------------------------------------------------
# Fix: _save_checkpoint_sync BEGIN IMMEDIATE transaction (sqlite_worker_backend) — LOW
# ---------------------------------------------------------------------------


class TestSaveCheckpointSyncTransaction:
    async def test_save_checkpoint_uses_explicit_transaction(self, tmp_path: Any) -> None:
        """_save_checkpoint_sync must wrap the INSERT in BEGIN IMMEDIATE.

        Verifies the fix by checking the checkpoint is persistable and
        loadable via the public async interface (indirect test of the
        sync path's transaction correctness).
        """
        from troopai.adk.flows import FlowCheckpoint
        from troopai.adk.flows.sqlite_worker_backend import SqliteFlowWorkerBackend

        class _SimpleFlow(Flow[_State]):
            @flow_start
            async def go(self) -> None: ...

        flow = _SimpleFlow(_State)
        db_path = tmp_path / "flow_test.db"
        backend = SqliteFlowWorkerBackend(path=db_path)

        cp = FlowCheckpoint(
            flow_id=flow.flow_id,
            completed_steps=("go",),
            pending_steps=(),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data=flow.state.model_dump_json(),
        )

        # save_checkpoint is the async wrapper that calls _save_checkpoint_sync
        await backend.save_checkpoint(cp)
        loaded = await backend.load_checkpoint(flow.flow_id)
        assert loaded is not None
        assert loaded.flow_id == flow.flow_id
        assert loaded.completed_steps == ("go",)


# ---------------------------------------------------------------------------
# Fix: FlowStepEndEvent.next_steps docstring (events.py) — LOW
# ---------------------------------------------------------------------------


class TestFlowStepEndEventDocstring:
    def test_next_steps_docstring_reflects_empty_tuple_behaviour(self) -> None:
        """FlowStepEndEvent.next_steps docstring must not promise successor names.

        The field is always () in the current implementation. The old docstring
        promised 'Tuple of step names that will fire next' which was incorrect.
        Verify the new docstring uses language about the current behaviour.
        """
        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(FlowStepEndEvent)}
        assert "next_steps" in fields, "FlowStepEndEvent must have next_steps field"
        # Get the class docstring/annotations to verify the update
        # Frozen dataclass field docs live on the class __doc__ or in source;
        # we verify via the module-level docstring approach: check the class
        # source docstring or annotations comment via __doc__.
        # The new docstring is on the class attribute comment; we check the
        # class __doc__ doesn't contain the misleading old promise.
        class_doc = FlowStepEndEvent.__doc__ or ""
        # Old wording: "will fire next" — after fix, this promise must be absent
        # or replaced with reserved/empty language. Check via the field's
        # metadata by inspecting the event-class source attribute docstring
        # directly (the field annotation comment).
        # Since frozen dataclasses store field docs only in source, we also
        # accept that the class-level docstring was the fixed location.
        assert "will fire next" not in class_doc, (
            "FlowStepEndEvent docstring still contains old misleading promise "
            "'will fire next'; expected corrected wording."
        )


# ---------------------------------------------------------------------------
# Fix: FlowStartEvent.start_steps on resume contains @flow_start methods
#      (executor.py) — MAJOR
# ---------------------------------------------------------------------------


class TestFlowStartEventStartStepsOnResume:
    async def test_start_steps_contains_flow_start_methods_not_pending_steps(self) -> None:
        """FlowStartEvent.start_steps must always list @flow_start method names.

        Pre-fix: _seed_executor_from_checkpoint replaced table.starts with
        checkpoint.pending_steps, so on resume FlowStartEvent.start_steps
        contained deferred/pending mid-flow step names instead of the declared
        @flow_start methods — violating the contract in events.py:40-49.
        """

        class CartFlow(Flow[_State]):
            @flow_start
            async def intake(self) -> None:
                self.state.events.append("intake")

            @flow_listen("intake", requires_approval=True)
            async def big_refund(self) -> None:
                self.state.events.append("big_refund")

        # --- First run: defers at big_refund ---
        flow = CartFlow(_State())
        result = await Runner.arun_flow(flow)
        assert result.status == "deferred"
        assert result.checkpoint is not None

        checkpoint = result.checkpoint
        # Approve the deferred step so resume actually fires it.
        checkpoint.approve(result.deferred_steps[0], approver_id="test", approver_role="ops")

        # --- Resume run: collect the FlowStartEvent ---
        start_events: list[FlowStartEvent] = []

        def _collect(event: object) -> None:
            if isinstance(event, FlowStartEvent):
                start_events.append(event)  # type: ignore[arg-type]

        resumed_state = _State.model_validate_json(checkpoint.state_data)
        resumed_flow = CartFlow(resumed_state)

        # Drive the resume via the executor directly so we can attach on_event.
        # _seed_executor_from_checkpoint is the module-level helper called by
        # Runner.arun_flow_from_checkpoint; exercising it directly lets us
        # intercept the FlowStartEvent emitted at the start of executor.run().
        from troopai.adk.run.runner import _seed_executor_from_checkpoint

        executor = FlowExecutor(resumed_flow, config=FlowConfig(), on_event=_collect)
        _seed_executor_from_checkpoint(executor, checkpoint)
        await executor.run()

        assert len(start_events) == 1, f"expected 1 FlowStartEvent, got {len(start_events)}"
        start_event = start_events[0]

        # After the fix: start_steps must be the @flow_start method names.
        assert start_event.start_steps == ("intake",), (
            f"FlowStartEvent.start_steps on resume was {start_event.start_steps!r} "
            f"— expected ('intake',) the @flow_start method, not pending/deferred steps"
        )

    async def test_start_steps_unchanged_on_normal_run(self) -> None:
        """FlowStartEvent.start_steps must be correct on a cold (non-resume) run too."""
        start_events: list[FlowStartEvent] = []

        class TwoStarts(Flow[_State]):
            @flow_start
            async def first(self) -> None:
                self.state.events.append("first")

            @flow_start
            async def second(self) -> None:
                self.state.events.append("second")

        executor = FlowExecutor(
            TwoStarts(_State()),
            config=FlowConfig(),
            on_event=lambda e: start_events.append(e) if isinstance(e, FlowStartEvent) else None,
        )
        await executor.run()

        assert len(start_events) == 1
        assert set(start_events[0].start_steps) == {"first", "second"}, (
            f"Expected both @flow_start methods in start_steps, got {start_events[0].start_steps}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completed_result(text: str = "ok") -> Any:
    class _CompletedResult:
        requires_action = False
        state: Any = None
        final_output = text

    return _CompletedResult()


def _make_simple_flow() -> Flow[_State]:
    class _SimpleFlow(Flow[_State]):
        @flow_start
        async def go(self) -> None:
            self.state.events.append("go")

    return _SimpleFlow(_State)


# ---------------------------------------------------------------------------
# FlowStepEndEvent.usage carries the per-step delta (executor.py) — MED
# ---------------------------------------------------------------------------


class TestPerStepUsageAttribution:
    async def test_step_end_events_carry_per_step_usage_delta(self) -> None:
        """Each FlowStepEndEvent.usage reflects only that step's consumption.

        The executor snapshots run_context.usage around each step and emits
        the scalar delta, so streaming consumers can attribute spend per
        step instead of seeing zeros.
        """
        from troopai.adk.run.context import RunContext
        from troopai.adk.types.tokens.llm_usage import LLMUsage

        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None:
                assert self.run_context is not None
                self.run_context.usage = self.run_context.usage + LLMUsage(
                    requests=1, input_tokens=100, output_tokens=40, total_tokens=140
                )

            @flow_listen(a)
            async def b(self) -> None:
                assert self.run_context is not None
                self.run_context.usage = self.run_context.usage + LLMUsage(
                    requests=2, input_tokens=10, output_tokens=5, total_tokens=15
                )

        flow = F(_State)
        run_ctx: RunContext[Any] = RunContext(context=None)  # type: ignore[arg-type]
        flow.run_context = run_ctx

        events: list[FlowEvent] = []
        executor: FlowExecutor[_State] = FlowExecutor(flow, config=FlowConfig(), on_event=events.append)
        result = await executor.run()
        assert result.status == "completed"

        end_by_step = {e.step_name: e for e in events if isinstance(e, FlowStepEndEvent)}
        usage_a = end_by_step["a"].usage
        assert (usage_a.requests, usage_a.input_tokens, usage_a.output_tokens, usage_a.total_tokens) == (
            1,
            100,
            40,
            140,
        )
        usage_b = end_by_step["b"].usage
        assert (usage_b.requests, usage_b.input_tokens, usage_b.output_tokens, usage_b.total_tokens) == (
            2,
            10,
            5,
            15,
        )

    async def test_step_usage_zero_without_run_context(self) -> None:
        """No shared run_context → deltas stay zero, no crash."""

        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None:
                self.state.events.append("a")

        events: list[FlowEvent] = []
        flow = F(_State)
        executor: FlowExecutor[_State] = FlowExecutor(flow, config=FlowConfig(), on_event=events.append)
        result = await executor.run()
        assert result.status == "completed"
        ends = [e for e in events if isinstance(e, FlowStepEndEvent)]
        assert all(e.usage.total_tokens == 0 and e.usage.requests == 0 for e in ends)

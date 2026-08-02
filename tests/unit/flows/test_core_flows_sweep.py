"""Regression tests for the CORE — flows sweep findings.

Each test targets one finding in ``flows/executor.py`` / ``agent_bridge.py``
/ ``result.py`` / ``executable.py`` and is written to FAIL on the pre-fix
code and PASS after the fix. The finding line references match the sweep
playbook.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field, replace as dc_replace
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from troopai.adk.flows import (
    Flow,
    FlowConfig,
    FlowStepCachePolicy,
    arun_flow_agent,
    flow_listen,
    flow_start,
)
from troopai.adk.flows.executor import FlowExecutor, _snapshot_state, encode_state
from troopai.adk.flows.triggers import FlowTriggerEvent
from troopai.adk.run.runner import Runner


class _State(BaseModel):
    events: list[str] = []


def _make_deferred_agent_result() -> Any:
    """A stub RunResult that reports an agent-level HITL deferral."""

    class _StubState:
        def to_dict(self) -> dict[str, Any]:
            return {"conversation_history": [], "current_agent_name": "stub"}

    class _StubResult:
        requires_action = True
        state = _StubState()
        final_output = None

    return _StubResult()


# ---------------------------------------------------------------------------
# Finding 1: executor.py:271 — run() clobbers restored pending_triggers on resume
# ---------------------------------------------------------------------------


class TestResumePreservesPendingTriggers:
    async def test_resume_does_not_clobber_restored_pending_triggers(self) -> None:
        """On resume, run() must NOT reset restored pending_triggers to empty.

        Pre-fix: ``run()`` did ``pending_triggers[start] = []`` for every
        start; on resume the runner had already restored the triggers into
        those same names (they occupy table.starts), so ctx.triggers was
        always () — gate callables branch differently than on cold start.
        """
        seen: list[tuple[FlowTriggerEvent, ...]] = []

        def _record(ctx: Any) -> bool:
            seen.append(ctx.triggers)
            return True

        class F(Flow[_State]):
            @flow_start
            async def entry(self) -> None: ...

            @flow_listen("entry", enabled=_record)
            async def resumed(self) -> None:
                self.state.events.append("resumed")

        flow = F(_State())
        executor: FlowExecutor[_State] = FlowExecutor(flow, config=FlowConfig())
        # Simulate the runner's checkpoint reseed: table.starts becomes the
        # pending queue and pending_triggers is restored for those steps.
        executor.table = dc_replace(executor.table, starts=("resumed",))
        restored = FlowTriggerEvent(name="entry", source_step="entry", kind="step_completion")
        executor.pending_triggers = {"resumed": [restored]}

        await executor.run()

        assert seen == [(restored,)], f"restored pending_triggers were clobbered on resume: {seen!r}"
        assert "resumed" in flow.state.events


# ---------------------------------------------------------------------------
# Finding 2: executor.py:353 — errored sibling dropped when a batch also defers
# ---------------------------------------------------------------------------


class TestErroredSiblingRequeuedOnDefer:
    async def test_errored_sibling_is_requeued_for_resume(self) -> None:
        """A deferral that preempts an errored sibling must keep it resumable.

        Pre-fix: the deferral won (correct) but the errored step was neither
        in completed_steps nor re-queued, so it vanished from the checkpoint
        and never re-ran on resume — its branch was silently dropped.
        """

        class F(Flow[_State]):
            @flow_start
            async def boom(self) -> None:
                raise RuntimeError("kaboom")

            @flow_start(requires_approval=lambda ctx: True)
            async def hold(self) -> None:
                self.state.events.append("hold")

        flow = F(_State())
        result = await FlowExecutor(flow, config=FlowConfig(error_policy="halt")).run()

        assert result.status == "deferred"
        assert result.checkpoint is not None
        assert "boom" in result.checkpoint.pending_steps, (
            f"errored sibling dropped from resume checkpoint: {result.checkpoint.pending_steps!r}"
        )

    async def test_lone_error_still_fails_without_requeue(self) -> None:
        """A pure error (no sibling deferral) still halts with status='failed'."""

        class F(Flow[_State]):
            @flow_start
            async def boom(self) -> None:
                raise RuntimeError("kaboom")

        result = await FlowExecutor(flow=F(_State()), config=FlowConfig(error_policy="halt")).run()
        assert result.status == "failed"
        assert result.checkpoint is None


# ---------------------------------------------------------------------------
# Finding 3: agent_bridge.py:116 — unregistered defer_key poisons pending_steps
# ---------------------------------------------------------------------------


class TestAgentBridgeUnregisteredDeferKey:
    async def test_unregistered_defer_key_across_async_boundary_raises(self) -> None:
        """A deferral whose step name can't be resolved must raise, not poison.

        Pre-fix: when inference failed across an async boundary, step_name fell
        back to the raw defer_key and was written into checkpoint.pending_steps
        — an unregistered name that makes resume crash. Now it raises a clear
        FlowDefinitionError at deferral time, surfacing as a failed step.
        """
        deferred = _make_deferred_agent_result()
        stub_agent: Any = object()

        class F(Flow[_State]):
            @flow_start
            async def go(self) -> None:
                async def _sibling() -> Any:
                    # Called from a gather sibling → the 'go' step frame is off
                    # the stack, so step-name inference fails.
                    return await arun_flow_agent(self, stub_agent, "ping", defer_key="not_a_registered_step")

                await asyncio.gather(_sibling())

        flow = F(_State())
        with patch("troopai.adk.run.runner.Runner.arun", new=AsyncMock(return_value=deferred)):
            result = await Runner.arun_flow(flow)

        assert result.status == "failed", f"unregistered defer_key silently deferred: {result.status}"
        assert result.error is not None
        assert "not_a_registered_step" in result.error

    async def test_registered_defer_key_across_async_boundary_defers_cleanly(self) -> None:
        """A defer_key that DOES name a registered step still defers normally."""
        deferred = _make_deferred_agent_result()
        stub_agent: Any = object()

        class F(Flow[_State]):
            @flow_start
            async def go(self) -> None:
                async def _sibling() -> Any:
                    return await arun_flow_agent(self, stub_agent, "ping", defer_key="go")

                await asyncio.gather(_sibling())

        flow = F(_State())
        with patch("troopai.adk.run.runner.Runner.arun", new=AsyncMock(return_value=deferred)):
            result = await Runner.arun_flow(flow)

        assert result.status == "deferred"
        assert result.checkpoint is not None
        assert "go" in result.checkpoint.pending_steps


# ---------------------------------------------------------------------------
# Finding 4: executor.py:891 — cache hit wholesale-rebinds flow.state
# ---------------------------------------------------------------------------


class TestCacheHitDoesNotRebindState:
    async def test_cache_hit_preserves_state_object_identity(self) -> None:
        """A cache hit restores fields in place — it must NOT swap the object.

        Pre-fix: ``self.flow.state = _restore_state(snapshot)`` rebound the
        reference; a sibling step running concurrently in the same batch that
        held the old reference lost its writes to the swap.
        """

        @dataclass
        class _MutState:
            a: int = 0
            b: int = 0

        class F(Flow[_MutState]):
            @flow_start(cache=FlowStepCachePolicy(cache_key_fn=lambda ctx: "k"))
            async def cached(self) -> None:
                self.state.a = 1

        flow = F(_MutState())
        executor: FlowExecutor[_MutState] = FlowExecutor(flow, config=FlowConfig())
        step = executor._get_step_descriptor("cached")
        ctx = executor._build_step_context("cached")
        cache = executor._get_step_cache(step, ctx)
        assert cache is not None
        cache.put("k", _snapshot_state(_MutState(a=1, b=0)), None)

        original = flow.state
        hit = executor._lookup_cache(step, ctx, "k")

        assert hit is not None
        assert flow.state is original, (
            "cache hit rebound flow.state (wholesale object replace); a concurrent "
            "sibling holding the same reference would lose its writes"
        )
        assert flow.state.a == 1

    async def test_cache_hit_frozen_state_falls_back_to_rebind(self) -> None:
        """Frozen state cannot be mutated in place; the fallback rebinds (no crash)."""

        @dataclass(frozen=True)
        class _FrozenState:
            a: int = 0

        class F(Flow[_FrozenState]):
            @flow_start(cache=FlowStepCachePolicy(cache_key_fn=lambda ctx: "k"))
            async def cached(self) -> None: ...

        flow = F(_FrozenState())
        executor: FlowExecutor[_FrozenState] = FlowExecutor(flow, config=FlowConfig())
        step = executor._get_step_descriptor("cached")
        ctx = executor._build_step_context("cached")
        cache = executor._get_step_cache(step, ctx)
        assert cache is not None
        cache.put("k", _snapshot_state(_FrozenState(a=9)), None)

        hit = executor._lookup_cache(step, ctx, "k")
        assert hit is not None
        assert flow.state.a == 9


# ---------------------------------------------------------------------------
# Finding 5: executor.py:932 — _store_cache only catches FlowDefinitionError
# ---------------------------------------------------------------------------


class TestCacheWriteSoftFail:
    async def test_cache_write_soft_fails_on_uncopyable_field(self) -> None:
        """A successful body must survive a copy.deepcopy TypeError on cache write.

        Pre-fix: _store_cache only caught FlowDefinitionError, so an
        un-deepcopyable field (a lock) made copy.deepcopy raise TypeError,
        which propagated and failed a step whose body had already succeeded.
        """

        @dataclass
        class _LockState:
            value: int = 0
            lock: Any = field(default_factory=threading.Lock)

        class F(Flow[_LockState]):
            @flow_start(cache=FlowStepCachePolicy(cache_key_fn=lambda ctx: "k"))
            async def step(self) -> None:
                self.state.value = 1

        result = await Runner.arun_flow(F(_LockState()))
        assert result.status == "completed", f"cache-write failure invalidated a successful body: {result.status}"
        assert result.final_state.value == 1


# ---------------------------------------------------------------------------
# Finding 6: executor.py:1375 — encode_state crashes on datetime/set/enum
# ---------------------------------------------------------------------------


class TestEncodeStateCoercesExoticTypes:
    def test_encode_state_coerces_datetime_and_set(self) -> None:
        """encode_state must coerce exotic dataclass values, never raise TypeError."""

        @dataclass
        class _ExoticState:
            when: datetime = field(default_factory=lambda: datetime(2020, 1, 1, tzinfo=UTC))
            tags: set[str] = field(default_factory=lambda: {"x", "y"})

        data = json.loads(encode_state(_ExoticState()))
        assert isinstance(data, dict)
        assert "when" in data
        assert "tags" in data

    async def test_deferral_with_exotic_dataclass_state_does_not_crash(self) -> None:
        """A deferral checkpoint over an exotic dataclass state must not crash run().

        encode_state runs inside _build_checkpoint, PAST the error_policy
        boundary — a bare json.dumps would raise TypeError straight out of
        FlowExecutor.run().
        """

        @dataclass
        class _ExoticState:
            when: datetime = field(default_factory=lambda: datetime(2021, 6, 1, tzinfo=UTC))
            note: str = ""

        class F(Flow[_ExoticState]):
            @flow_start(requires_approval=lambda ctx: True)
            async def go(self) -> None: ...

        result = await FlowExecutor(flow=F(_ExoticState()), config=FlowConfig()).run()
        assert result.status == "deferred"
        assert result.checkpoint is not None
        assert isinstance(json.loads(result.checkpoint.state_data), dict)


# ---------------------------------------------------------------------------
# Finding 7: executor.py:468 — __error__ handler runs with empty triggers
# ---------------------------------------------------------------------------


class TestErrorHandlerTriggerEvent:
    async def test_error_handler_receives_error_trigger_event(self) -> None:
        """The routed __error__ handler must see a trigger naming the failed step.

        Pre-fix: error listeners were queued directly without any
        pending_triggers entry, so ctx.triggers was () with no provenance for
        which step failed.
        """
        seen: list[tuple[FlowTriggerEvent, ...]] = []

        def _record(ctx: Any) -> bool:
            seen.append(ctx.triggers)
            return True

        class F(Flow[_State]):
            @flow_start
            async def boom(self) -> None:
                raise RuntimeError("kaboom")

            @flow_listen("__error__", enabled=_record)
            async def recover(self) -> None:
                self.state.events.append("recovered")

        flow = F(_State())
        result = await FlowExecutor(flow, config=FlowConfig(error_policy="route_to_error_handler")).run()

        assert result.status == "completed"
        assert "recovered" in flow.state.events
        assert len(seen) == 1
        triggers = seen[0]
        assert len(triggers) == 1, f"error handler fired with empty triggers: {triggers!r}"
        assert triggers[0].name == "__error__"
        assert triggers[0].source_step == "boom"


# ---------------------------------------------------------------------------
# Finding 8: result.py:302 — stream_events leaks the producer on early close
# ---------------------------------------------------------------------------


class TestStreamEarlyCloseCancelsProducer:
    async def test_stream_events_cancels_producer_on_early_close(self) -> None:
        """Closing the stream early must cancel the background producer task.

        Pre-fix: stream_events had no finally, so a consumer that stopped
        iterating left the executor-driving producer running (here blocked
        forever on an event that is never set).
        """
        gate = asyncio.Event()

        class F(Flow[_State]):
            @flow_start
            async def a(self) -> None:
                self.state.events.append("a")

            @flow_listen(a)
            async def slow(self) -> None:
                await gate.wait()  # never set → the producer would hang forever

        streaming = Runner.arun_flow_streamed(F(_State()))
        # stream_events is an async generator; its public return type widens to
        # AsyncIterator, which does not expose aclose(). Narrow it so the test
        # can drive an explicit early close.
        gen = cast("AsyncGenerator[Any, None]", streaming.stream_events())
        first = await gen.__anext__()
        assert first is not None
        await gen.aclose()

        task = streaming._producer_task
        assert task is not None
        assert task.done(), "producer task leaked (still running) after early stream close"


# ---------------------------------------------------------------------------
# Finding 9: executable.py:105 — halted_* nested flow passed as success
# ---------------------------------------------------------------------------


class TestFlowExecutableHaltedIsNonSuccess:
    async def test_flow_executable_raises_on_halted_max_steps(self) -> None:
        """A nested flow that halts on a cap must surface as a node failure.

        Pre-fix: only 'failed' raised; 'halted_max_steps' /
        'halted_max_tokens' fell through to a normal NodeResult, routing the
        graph downstream on partial mid-flow state.
        """
        from troopai.adk.exceptions import UserError
        from troopai.adk.flows.executable import FlowExecutable
        from troopai.adk.orchestration.executable import ExecutableInput
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG
        from troopai.adk.run.context import RunContext

        class _TwoStep(Flow[_State]):
            @flow_start
            async def a(self) -> None:
                self.state.events.append("a")

            @flow_listen(a)
            async def b(self) -> None:
                self.state.events.append("b")

        exe: FlowExecutable[None] = FlowExecutable(flow=_TwoStep(_State()), config=FlowConfig(max_steps=1))
        ctx: RunContext[None] = RunContext.make(None)
        with pytest.raises(UserError, match="halted"):
            await exe.invoke(
                input=ExecutableInput(content=[], from_node=None),
                context=ctx,
                config=DEFAULT_RUN_CONFIG,
            )


# ---------------------------------------------------------------------------
# Finding 10: result.py:114 — per_step_usage always empty in the result
# ---------------------------------------------------------------------------


class TestPerStepUsagePopulated:
    async def test_per_step_usage_populated_in_result(self) -> None:
        """_build_result must populate FlowRunResult.per_step_usage.

        Pre-fix: per_step_usage was declared but never populated, so it was
        always an empty dict even though the executor computes per-step
        deltas for the FlowStepEndEvent.
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
                    requests=1, input_tokens=10, output_tokens=5, total_tokens=15
                )

        flow = F(_State())
        run_ctx: RunContext[Any] = RunContext(context=None)  # type: ignore[arg-type]
        flow.run_context = run_ctx
        result = await FlowExecutor(flow, config=FlowConfig()).run()

        assert result.status == "completed"
        assert set(result.per_step_usage.keys()) == {"a", "b"}
        assert result.per_step_usage["a"].total_tokens == 140
        assert result.per_step_usage["b"].total_tokens == 15

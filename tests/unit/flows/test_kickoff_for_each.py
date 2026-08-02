"""Unit tests for :meth:`Runner.arun_flow_for_each` batch helper."""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import BaseModel

from troopai.adk.flows import Flow, FlowConfig, flow_listen, flow_start
from troopai.adk.run.runner import Runner


class _IntState(BaseModel):
    """Minimal pydantic state for batch tests."""

    value: int = 0
    doubled: int = 0


class _DoublingFlow(Flow[_IntState]):
    @flow_start
    async def double(self) -> None:
        self.state.doubled = self.state.value * 2


def _factory(state: _IntState) -> _DoublingFlow:
    return _DoublingFlow(state)


class TestSequential:
    async def test_empty_input_returns_empty_tuple(self) -> None:
        results = await Runner.arun_flow_for_each(_factory, [])
        assert results == ()

    async def test_single_input_runs_to_completion(self) -> None:
        results = await Runner.arun_flow_for_each(_factory, [_IntState(value=3)])
        assert len(results) == 1
        assert results[0].status == "completed"
        assert results[0].final_state.doubled == 6

    async def test_batch_runs_in_input_order(self) -> None:
        states = [_IntState(value=i) for i in range(5)]
        results = await Runner.arun_flow_for_each(_factory, states)
        assert tuple(r.final_state.doubled for r in results) == (0, 2, 4, 6, 8)
        # Each result aligned to its input.
        for inp, out in zip(states, results, strict=True):
            assert out.final_state.value == inp.value


class TestConcurrency:
    async def test_concurrency_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            await Runner.arun_flow_for_each(_factory, [_IntState(value=1)], concurrency=0)

    async def test_concurrency_cap_observed(self) -> None:
        active = {"count": 0, "peak": 0}

        class _ParallelFlow(Flow[_IntState]):
            @flow_start
            async def slow(self) -> None:
                active["count"] += 1
                active["peak"] = max(active["peak"], active["count"])
                await asyncio.sleep(0.05)
                active["count"] -= 1
                self.state.doubled = self.state.value * 2

        def _slow_factory(state: _IntState) -> _ParallelFlow:
            return _ParallelFlow(state)

        states = [_IntState(value=i) for i in range(8)]
        results = await Runner.arun_flow_for_each(
            _slow_factory,
            states,
            concurrency=3,
        )
        assert all(r.status == "completed" for r in results)
        # Semaphore caps active count at the supplied concurrency.
        assert active["peak"] <= 3
        # And actually parallelised something — sequential would never exceed 1.
        assert active["peak"] >= 2

    @pytest.mark.slow
    async def test_concurrency_outpaces_sequential(self) -> None:
        class _SlowFlow(Flow[_IntState]):
            @flow_start
            async def slow(self) -> None:
                await asyncio.sleep(0.05)
                self.state.doubled = self.state.value

        states = [_IntState(value=i) for i in range(6)]

        # Sequential baseline.
        t0 = time.perf_counter()
        await Runner.arun_flow_for_each(lambda s: _SlowFlow(s), states)
        sequential_elapsed = time.perf_counter() - t0

        # Concurrent.
        t1 = time.perf_counter()
        await Runner.arun_flow_for_each(lambda s: _SlowFlow(s), states, concurrency=6)
        concurrent_elapsed = time.perf_counter() - t1

        # 6 items × 50ms each: sequential ≈ 300ms, concurrent ≈ 60ms.
        # Marked @slow so CI runners with tight event-loop scheduling
        # can deselect; the ratio uses 0.4 (less aggressive than the
        # theoretical 0.2) to accommodate CI variance.
        assert concurrent_elapsed < sequential_elapsed * 0.4


class TestErrorIsolation:
    async def test_factory_framework_error_captures_into_failed_result(self) -> None:
        """``TroopAIError`` subclasses from the factory land on per-item result."""
        from troopai.adk.flows.exceptions import FlowDefinitionError

        def _bad_factory(state: _IntState) -> Flow[_IntState]:
            if state.value == 2:
                raise FlowDefinitionError("intentional framework failure")
            return _DoublingFlow(state)

        states = [_IntState(value=i) for i in range(4)]
        results = await Runner.arun_flow_for_each(_bad_factory, states)

        assert results[0].status == "completed"
        assert results[1].status == "completed"
        assert results[2].status == "failed"
        assert results[2].error is not None and "FlowDefinitionError" in results[2].error
        assert results[3].status == "completed"
        # Aligned to inputs; factory-failure flow_id is a unique sentinel.
        assert results[0].final_state.value == 0
        assert results[3].final_state.value == 3
        assert results[2].flow_id.startswith("flow-factory-failed-")

    async def test_factory_programmer_error_propagates(self) -> None:
        """Programmer errors (``RuntimeError``, ``NameError``) propagate; not silently captured."""

        def _buggy_factory(state: _IntState) -> Flow[_IntState]:
            if state.value == 1:
                raise RuntimeError("programmer bug — must surface")
            return _DoublingFlow(state)

        states = [_IntState(value=i) for i in range(3)]
        with pytest.raises(RuntimeError, match="must surface"):
            await Runner.arun_flow_for_each(_buggy_factory, states)

    async def test_step_exception_captured_per_item(self) -> None:
        """Step body exceptions are routed through ``error_policy`` inside the executor.

        Even a plain :class:`RuntimeError` raised from a step body
        surfaces as ``FlowRunResult.status="failed"`` because the
        executor catches it. The batch helper never sees the raw
        exception — only the result.
        """

        class _MaybeFailFlow(Flow[_IntState]):
            @flow_start
            async def step(self) -> None:
                if self.state.value == 1:
                    raise RuntimeError("step failed")
                self.state.doubled = self.state.value * 2

        states = [_IntState(value=i) for i in range(3)]
        results = await Runner.arun_flow_for_each(
            lambda s: _MaybeFailFlow(s),
            states,
        )
        assert results[0].status == "completed"
        assert results[1].status == "failed"
        assert results[2].status == "completed"


class TestConfigForwarding:
    async def test_config_max_steps_applies_per_item(self) -> None:
        # max_steps=1 will be hit during the run for a flow with only one
        # @flow_start step + a listener; assert each item independently
        # gets its own bound rather than the bound running out across
        # the batch.
        class _TwoStepFlow(Flow[_IntState]):
            @flow_start
            async def first(self) -> None:
                self.state.value += 1

            @flow_listen("first")
            async def second(self) -> None:
                self.state.doubled = self.state.value * 2

        results = await Runner.arun_flow_for_each(
            lambda s: _TwoStepFlow(s),
            [_IntState(value=0), _IntState(value=10)],
            config=FlowConfig(max_steps=1),
        )
        # max_steps=1 halts before the second step fires for each item.
        assert all(r.status == "halted_max_steps" for r in results)

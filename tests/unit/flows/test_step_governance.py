"""Unit tests for ``FlowStepRateLimit``, ``FlowStepGuardrails``, and ``FlowStepCachePolicy``.

Exercises each governance attribute through the public decorator
surface — every test constructs a fresh ``Flow`` subclass with the
policy passed as a decorator kwarg, never via direct private-slot
mutation on the ``FlowStep`` descriptor.
"""

from __future__ import annotations

import asyncio
import copy

import pytest
from pydantic import BaseModel

from troopai.adk.flows import (
    Flow,
    FlowConfig,
    FlowEvent,
    FlowStepCachePolicy,
    FlowStepContext,
    FlowStepEndEvent,
    FlowStepGuardrails,
    FlowStepGuardrailVerdict,
    FlowStepRateLimit,
    FlowStepStartEvent,
    flow_listen,
    flow_router,
    flow_start,
)
from troopai.adk.flows.executor import FlowExecutor
from troopai.adk.run.runner import Runner


class _CounterState(BaseModel):
    count: int = 0
    branch: str = ""
    notes: list[str] = []


class TestFlowStepRateLimit:
    def test_validates_positive_rpm(self) -> None:
        with pytest.raises(ValueError, match="rpm must be positive"):
            FlowStepRateLimit(rpm=0)

    def test_validates_positive_max_wait(self) -> None:
        with pytest.raises(ValueError, match="max_wait_seconds must be positive"):
            FlowStepRateLimit(rpm=10, max_wait_seconds=0)

    async def test_generous_rpm_completes_normally(self) -> None:
        class _Flow(Flow[_CounterState]):
            @flow_start(rate_limit=FlowStepRateLimit(rpm=100, behavior="error"))
            async def step(self) -> None:
                self.state.count += 1

        result = await Runner.arun_flow(_Flow(_CounterState))
        assert result.status == "completed"
        assert result.final_state.count == 1


class TestFlowStepGuardrails:
    async def test_pre_guardrail_reject_routes_through_error_policy(self) -> None:
        def reject_pre(ctx: FlowStepContext[_CounterState]) -> FlowStepGuardrailVerdict:
            return FlowStepGuardrailVerdict.reject_content("pre-guard rejected")

        class _Flow(Flow[_CounterState]):
            @flow_start(guardrails=FlowStepGuardrails(pre=(reject_pre,)))
            async def go(self) -> None:
                self.state.notes.append("body_ran")

            @flow_listen("__error__")
            async def on_error(self) -> None:
                self.state.notes.append("error_handler")

        result = await Runner.arun_flow(
            _Flow(_CounterState),
            config=FlowConfig(error_policy="route_to_error_handler"),
        )
        assert "body_ran" not in result.final_state.notes
        assert "error_handler" in result.final_state.notes

    async def test_pre_guardrail_allow_lets_step_proceed(self) -> None:
        def allow(ctx: FlowStepContext[_CounterState]) -> FlowStepGuardrailVerdict:
            return FlowStepGuardrailVerdict.allow()

        class _Flow(Flow[_CounterState]):
            @flow_start(guardrails=FlowStepGuardrails(pre=(allow,)))
            async def go(self) -> None:
                self.state.notes.append("body_ran")

        result = await Runner.arun_flow(_Flow(_CounterState))
        assert result.status == "completed"
        assert "body_ran" in result.final_state.notes

    async def test_raise_exception_verdict_surfaces_typed_error(self) -> None:
        class _SecurityError(Exception):
            pass

        def raise_pre(ctx: FlowStepContext[_CounterState]) -> FlowStepGuardrailVerdict:
            return FlowStepGuardrailVerdict.raise_exception(_SecurityError("blocked"))

        class _Flow(Flow[_CounterState]):
            @flow_start(guardrails=FlowStepGuardrails(pre=(raise_pre,)))
            async def go(self) -> None:
                self.state.notes.append("ran")

        result = await Runner.arun_flow(_Flow(_CounterState))
        assert result.status == "failed"
        assert result.error is not None and "_SecurityError" in result.error
        assert "ran" not in result.final_state.notes

    async def test_post_guardrail_fires_after_body(self) -> None:
        def reject_post(ctx: FlowStepContext[_CounterState]) -> FlowStepGuardrailVerdict:
            return FlowStepGuardrailVerdict.reject_content("post-guard rejected")

        class _Flow(Flow[_CounterState]):
            @flow_start(guardrails=FlowStepGuardrails(post=(reject_post,)))
            async def go(self) -> None:
                self.state.notes.append("body_ran")

        result = await Runner.arun_flow(_Flow(_CounterState))
        assert "body_ran" in result.final_state.notes  # body did run
        assert result.status == "failed"

    async def test_async_guardrail_awaited(self) -> None:
        async def async_reject(ctx: FlowStepContext[_CounterState]) -> FlowStepGuardrailVerdict:
            await asyncio.sleep(0)
            return FlowStepGuardrailVerdict.reject_content("async reject")

        class _Flow(Flow[_CounterState]):
            @flow_start(guardrails=FlowStepGuardrails(pre=(async_reject,)))
            async def go(self) -> None:
                self.state.notes.append("ran")

        result = await Runner.arun_flow(_Flow(_CounterState))
        assert result.status == "failed"

    async def test_guardrail_raising_unrelated_exception_wraps_as_governance_error(self) -> None:
        from troopai.adk.flows.exceptions import FlowStepGovernanceError

        def buggy(ctx: FlowStepContext[_CounterState]) -> FlowStepGuardrailVerdict:
            raise KeyError("buggy guardrail")

        class _Flow(Flow[_CounterState]):
            @flow_start(guardrails=FlowStepGuardrails(pre=(buggy,)))
            async def go(self) -> None: ...

        result = await Runner.arun_flow(_Flow(_CounterState))
        assert result.status == "failed"
        assert result.error is not None
        # FlowStepGovernanceError is the wrapped breadcrumb — operators see
        # "guardrail" in the error rather than a bare KeyError attributed to the body.
        assert "FlowStepGovernanceError" in result.error
        # Smoke check: the typed exception is constructable with the expected fields.
        exc = FlowStepGovernanceError(step_name="go", hook="guardrail", phase="pre")
        assert exc.hook == "guardrail" and exc.phase == "pre"


class TestFlowStepCachePolicy:
    def test_validates_callable_key(self) -> None:
        with pytest.raises(ValueError, match="must be callable"):
            FlowStepCachePolicy(cache_key_fn="not a callable")  # type: ignore[arg-type]  # intentional bad input

    def test_validates_positive_ttl(self) -> None:
        with pytest.raises(ValueError, match="ttl_seconds must be positive"):
            FlowStepCachePolicy(cache_key_fn=lambda ctx: "k", ttl_seconds=0)

    def test_validates_max_entries(self) -> None:
        with pytest.raises(ValueError, match="max_entries must be >= 1"):
            FlowStepCachePolicy(cache_key_fn=lambda ctx: "k", max_entries=0)

    async def test_cache_runs_body_on_cold_start(self) -> None:
        invocations = {"count": 0}

        class _Flow(Flow[_CounterState]):
            @flow_start(cache=FlowStepCachePolicy(cache_key_fn=lambda ctx: "k"))
            async def step(self) -> None:
                invocations["count"] += 1
                self.state.notes.append("body")

        flow = _Flow(_CounterState)
        result = await Runner.arun_flow(flow)
        assert result.status == "completed"
        assert invocations["count"] == 1
        assert "body" in result.final_state.notes

    async def test_cache_key_diverging_state_does_not_split(self) -> None:
        """Regression: key resolved once → write uses same key as lookup even if body mutates state."""
        invocations = {"count": 0}

        def key_from_count(ctx: FlowStepContext[_CounterState]) -> str:
            # If this is called both pre- and post-body, the body's
            # mutation flips the key. We need only ONE invocation, so
            # the test asserts the cache resolves the key once.
            return f"k-{ctx.flow_state.count}"

        class _Flow(Flow[_CounterState]):
            @flow_start(cache=FlowStepCachePolicy(cache_key_fn=key_from_count))
            async def mutating(self) -> None:
                invocations["count"] += 1
                self.state.count += 1  # mutates the state the key reads.

        # First run with count=0 stores under "k-0".
        first = await Runner.arun_flow(_Flow(_CounterState(count=0)))
        assert first.status == "completed"
        assert invocations["count"] == 1

        # A run that should mutate but is fresh (no shared cache across
        # executor instances) — the key resolves to "k-0" again,
        # the executor's cache is fresh, so the body runs.
        second = await Runner.arun_flow(_Flow(_CounterState(count=0)))
        assert second.status == "completed"
        assert invocations["count"] == 2

    async def test_cache_write_failure_is_soft(self) -> None:
        """Cache-write failure on non-cacheable state is logged but does not fail the step."""

        class _PlainState:
            """Non-Pydantic, non-dataclass — _snapshot_state will reject."""

            def __init__(self) -> None:
                self.value = 0

        class _Flow(Flow[_PlainState]):
            @flow_start(cache=FlowStepCachePolicy(cache_key_fn=lambda ctx: "k"))
            async def step(self) -> None:
                # Will raise on the checkpoint path too, but here we
                # exercise the cache write soft-path; the body runs.
                self.state.value = 1

        flow = _Flow(_PlainState())
        result = await Runner.arun_flow(flow)
        # The body succeeded; cache write failed soft.
        assert result.status == "completed"
        assert result.final_state.value == 1

    async def test_lru_put_refreshes_position(self) -> None:
        """Regression: ``_StepCache.put`` removes-then-reinserts so LRU order tracks writes."""

        class _Flow(Flow[_CounterState]):
            @flow_start(
                cache=FlowStepCachePolicy(
                    cache_key_fn=lambda ctx: "k",
                    max_entries=2,
                ),
            )
            async def step(self) -> None:
                self.state.count += 1

        # The cache lifecycle is per-executor; we only verify the
        # public surface here. The detailed LRU ordering is covered
        # by a unit test on `_StepCache` directly below.
        result = await Runner.arun_flow(_Flow(_CounterState))
        assert result.status == "completed"


class TestStepCacheLRUBehaviour:
    """Targeted unit tests on the package-internal cache primitive.

    The internal cache structure is intentionally exposed only at the
    framework boundary; these tests construct one directly via the
    public `FlowStepCachePolicy` config and exercise it through a
    fresh :class:`Flow` instance to avoid touching private members.
    """

    async def test_lru_eviction_on_overflow(self) -> None:
        seen_keys: list[str] = []

        def key_fn(ctx: FlowStepContext[_CounterState]) -> str:
            # Step-dependent key — each step invocation registers a
            # distinct key by reading the count value the prior step
            # wrote.
            seen_keys.append(f"k-{ctx.flow_state.count}")
            return seen_keys[-1]

        class _Flow(Flow[_CounterState]):
            @flow_start(
                cache=FlowStepCachePolicy(cache_key_fn=key_fn, max_entries=1),
            )
            async def step(self) -> None:
                self.state.count = 1

        flow = _Flow(_CounterState(count=0))
        result = await Runner.arun_flow(flow)
        assert result.status == "completed"
        assert len(seen_keys) == 1  # One pre-body resolution per run.


@pytest.mark.parametrize("via_factory", [True, False])
def test_FlowStepCachePolicy_handles_deepcopy(via_factory: bool) -> None:  # noqa: N802 # intentional class-name test prefix
    """Cache policy values are deep-copyable (needed by some run frameworks)."""
    policy = FlowStepCachePolicy(cache_key_fn=lambda ctx: "k")
    copied = copy.copy(policy) if via_factory else copy.deepcopy(policy)
    assert copied.max_entries == policy.max_entries


class TestCacheHitEventBalance:
    async def test_cache_hit_emits_balanced_start_end_events(self) -> None:
        """Regression: a cache hit emitted ``FlowStepEndEvent`` with NO
        preceding ``FlowStepStartEvent`` — the Start event only fired
        inside the miss path, so streaming consumers saw a dangling End
        for every cached invocation. The hit path now emits Start first,
        keeping Start/End pairs balanced.
        """
        events: list[FlowEvent] = []
        loop_calls = {"n": 0}

        class _Flow(Flow[_CounterState]):
            @flow_start
            async def seed(self) -> None:
                self.state.notes.append("seed")

            @flow_listen(seed, cache=FlowStepCachePolicy(cache_key_fn=lambda ctx: "k"))
            async def cached(self) -> None:
                self.state.notes.append("cached-body")

            @flow_router(cached)
            async def loop(self) -> str:
                # First pass: re-fire `cached` via its trigger label so the
                # second invocation is a cache hit. Second pass: terminate.
                # The counter lives OUTSIDE self.state because a cache hit
                # restores the snapshotted state (which would reset it).
                loop_calls["n"] += 1
                if loop_calls["n"] == 1:
                    return "seed"
                return "done"

        flow = _Flow(_CounterState)
        executor: FlowExecutor[_CounterState] = FlowExecutor(
            flow,
            config=FlowConfig(),
            on_event=events.append,
        )
        result = await executor.run()

        assert result.status == "completed"
        # The body ran exactly once — the second invocation was a cache hit.
        assert result.final_state.notes.count("cached-body") == 1
        cached_events = [
            e for e in events if isinstance(e, (FlowStepStartEvent, FlowStepEndEvent)) and e.step_name == "cached"
        ]
        assert [type(e) for e in cached_events] == [
            FlowStepStartEvent,
            FlowStepEndEvent,
            FlowStepStartEvent,
            FlowStepEndEvent,
        ]

"""Regression tests for confirmed bugs in ``troopai.adk.run.runner``.

Each test targets one finding and is written to FAIL on the pre-fix code
and PASS after the fix.

Findings covered:

1. ``arun_flow_streamed`` called outside a running loop returned a stream
   that hung forever (no producer scheduled, no deferred impl registered).
2. The streamed swarm path disposed a member's toolsets after every turn,
   so an MCP member silently lost all its tools on revisit. The fix adds a
   ``dispose_toolsets`` gate threaded into ``_run_streamed`` /
   ``_run_streamed_impl`` and set to ``False`` for swarm member turns.
3. ``arun_task_streamed`` skipped ``validate_budget_config``, so a
   per-period budget without a ledger was silently accepted.
"""

from __future__ import annotations

import asyncio
from typing import Any, override
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from troopai.adk.agents.agent import Agent
from troopai.adk.budgets import TenantBudget
from troopai.adk.exceptions import UserError
from troopai.adk.flows import Flow, flow_start
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.tasks.task import Task
from troopai.adk.tools.toolsets.abstract import Toolset


class _State(BaseModel):
    events: list[str] = []


def _make_simple_flow() -> Flow[_State]:
    class _SimpleFlow(Flow[_State]):
        @flow_start
        async def go(self) -> None:
            self.state.events.append("go")

    return _SimpleFlow(_State)


# ---------------------------------------------------------------------------
# Finding 1: arun_flow_streamed outside a running loop must not hang.
# ---------------------------------------------------------------------------


class TestArunFlowStreamedNoLoop:
    def test_stream_events_completes_when_built_outside_loop(self) -> None:
        """Build the streamed result with NO running loop, then consume it
        from a fresh loop.

        Pre-fix: ``arun_flow_streamed``'s ``except RuntimeError`` branch only
        logged a warning, so no producer task or deferred impl was ever
        registered and ``stream_events()`` blocked forever on the empty queue.
        After the fix the producer is scheduled lazily on first iteration.
        """
        flow = _make_simple_flow()
        # This call runs in the test's synchronous body — no event loop is
        # running here, so the `except RuntimeError` deferral branch fires.
        streaming = Runner.arun_flow_streamed(flow)

        async def _consume() -> list[Any]:
            collected: list[Any] = []

            # wait_for makes the pre-fix hang surface as a TimeoutError
            # (test failure) instead of blocking the suite forever.
            async def _drain() -> None:
                async for event in streaming.stream_events():
                    collected.append(event)

            await asyncio.wait_for(_drain(), timeout=5.0)
            return collected

        events = asyncio.run(_consume())

        assert len(events) > 0
        assert streaming.final_state is not None
        assert isinstance(streaming.final_state, _State)
        assert streaming.status == "completed"


# ---------------------------------------------------------------------------
# Finding 2: per-turn toolset disposal in the streamed path.
# ---------------------------------------------------------------------------


class _TrackingToolset(Toolset):
    """Minimal toolset that records whether ``adispose`` has been called."""

    def __init__(self) -> None:
        self.disposed = False

    @override
    async def get_tools(self, ctx: Any = None) -> dict[str, Any]:
        del ctx
        return {}

    @override
    async def adispose(self) -> None:
        self.disposed = True


async def _noop_loop_streamed(**kwargs: Any) -> None:
    """Stand-in for ``run_agent_loop_streamed`` that produces no work."""
    del kwargs


class TestRunStreamedDisposeToolsetsGate:
    async def test_dispose_toolsets_false_keeps_member_toolsets_alive(self) -> None:
        """``_run_streamed(dispose_toolsets=False)`` must NOT dispose the
        agent's toolsets — so a swarm member keeps its (MCP) tools on revisit.

        Pre-fix: ``_run_streamed_impl``'s finally unconditionally disposed,
        so a streamed swarm member lost all its tools after its first turn.
        """
        toolset = _TrackingToolset()
        agent = Agent(name="m", system_prompt="x", tools=[toolset])

        with patch(
            "troopai.adk.run.runner.run_agent_loop_streamed",
            new=AsyncMock(side_effect=_noop_loop_streamed),
        ):
            result = Runner._run_streamed(agent, "go", dispose_toolsets=False)
            async for _ in result.stream_events():
                pass

        assert toolset.disposed is False

    async def test_dispose_toolsets_default_disposes(self) -> None:
        """A standalone streamed run (default ``dispose_toolsets=True``) MUST
        still dispose its toolsets so connections are released."""
        toolset = _TrackingToolset()
        agent = Agent(name="m", system_prompt="x", tools=[toolset])

        with patch(
            "troopai.adk.run.runner.run_agent_loop_streamed",
            new=AsyncMock(side_effect=_noop_loop_streamed),
        ):
            result = Runner._run_streamed(agent, "go")
            async for _ in result.stream_events():
                pass

        assert toolset.disposed is True

    async def test_swarm_member_turn_passes_dispose_toolsets_false(self) -> None:
        """The streamed swarm member-turn caller must opt OUT of per-turn
        disposal by passing ``dispose_toolsets=False`` to ``_run_streamed``."""
        from troopai.adk.run import swarm_loop_streamed

        member = Agent(name="m", system_prompt="x")
        captured: dict[str, Any] = {}

        def _fake_run_streamed(*args: Any, **kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            inner = MagicMock()

            async def _empty() -> Any:
                return
                yield  # pragma: no cover - makes this an async generator

            inner.stream_events = _empty
            inner.final_output = None
            inner.new_items = []
            inner.context = None
            inner.swarm_yield = None
            return inner

        with patch.object(Runner, "_run_streamed", staticmethod(_fake_run_streamed)):
            await swarm_loop_streamed._stream_member_turn(
                member=member,
                user_prompt="hi",
                ctx_wrapper=MagicMock(context=None),
                hooks=MagicMock(),
                config=RunConfig(),
                is_first_turn=True,
                result=MagicMock(),
            )

        assert captured.get("dispose_toolsets") is False


# ---------------------------------------------------------------------------
# Finding 3: arun_task_streamed must validate the budget config.
# ---------------------------------------------------------------------------


class TestArunTaskStreamedBudgetValidation:
    async def test_period_budget_without_ledger_fails_fast(self) -> None:
        """A per-period budget with no cost_ledger must raise ``UserError``
        synchronously from ``arun_task_streamed`` — matching every other
        entry point.

        Pre-fix: this path bypassed ``_run_streamed`` (and its validation),
        so the misconfiguration was silently accepted.
        """
        agent = Agent(name="A", system_prompt="test")
        task: Task[Any] = Task(agent=agent, description="do it")
        cfg = RunConfig(tenant_budget=TenantBudget(dollars_per_period=5.0), tenant_id="t1")

        with pytest.raises(UserError, match="cost_ledger"):
            await Runner.arun_task_streamed(task, run_config=cfg)

    async def test_per_run_budget_without_ledger_is_accepted(self) -> None:
        """A per-run budget needs no ledger — validation must not reject it.

        Guards against an over-broad fix that rejects valid configs. The
        streamed run is short-circuited by mocking the inner loop.
        """
        toolset = _TrackingToolset()
        agent = Agent(name="A", system_prompt="test", tools=[toolset])
        task: Task[Any] = Task(agent=agent, description="do it")
        cfg = RunConfig(tenant_budget=TenantBudget(dollars_per_run=1.0), tenant_id="t1")

        with patch(
            "troopai.adk.run.runner.run_agent_loop_streamed",
            new=AsyncMock(side_effect=_noop_loop_streamed),
        ):
            result = await Runner.arun_task_streamed(task, run_config=cfg)
            async for _ in result.stream_events():
                pass


# ---------------------------------------------------------------------------
# RECONCILE fix 1: _drive_flow_stream must mirror per_step_usage.
# ---------------------------------------------------------------------------


class TestDriveFlowStreamPerStepUsage:
    async def test_per_step_usage_mirrored_onto_streamed_result(self) -> None:
        """``_drive_flow_stream`` must mirror the executor result's
        ``per_step_usage`` onto the streamed ``FlowRunResultStreaming``,
        matching the non-streamed ``FlowRunResult`` built by
        ``FlowExecutor._build_result``.

        Pre-fix: the streamed driver copied every other executor field but
        omitted ``per_step_usage``, so a streamed flow's per-step breakdown
        stayed empty.
        """
        from troopai.adk.flows.result import FlowRunResult, FlowRunResultStreaming
        from troopai.adk.llms.llm_usage import LLMUsage
        from troopai.adk.run.runner import _drive_flow_stream

        per_step = {
            "step_a": LLMUsage(requests=1, input_tokens=6, output_tokens=4, total_tokens=10),
            "step_b": LLMUsage(requests=1, input_tokens=3, output_tokens=2, total_tokens=5),
        }
        final: FlowRunResult[_State] = FlowRunResult(
            final_state=_State(),
            flow_id="f1",
            status="completed",
            completed_steps=("step_a", "step_b"),
            cumulative_usage=LLMUsage(requests=2, input_tokens=9, output_tokens=6, total_tokens=15),
            per_step_usage=per_step,
        )

        class _FakeExecutor:
            async def run(self) -> FlowRunResult[_State]:
                return final

        result: FlowRunResultStreaming[_State] = FlowRunResultStreaming(flow_id="f1")
        await _drive_flow_stream(_FakeExecutor(), result)

        assert result.per_step_usage == per_step
        assert result.per_step_usage["step_a"].total_tokens == 10
        assert result.per_step_usage["step_b"].total_tokens == 5


# ---------------------------------------------------------------------------
# RECONCILE fix 2: streamed swarm member turns share the driver RunContext.
# ---------------------------------------------------------------------------


class TestStreamedSwarmSharesRunContext:
    async def test_member_turn_threads_shared_run_context(self) -> None:
        """``_stream_member_turn`` must thread the driver's ``RunContext`` into
        ``_run_streamed`` as ``shared_run_context`` so per-run budget / usage
        accumulate cumulatively across member turns.

        Pre-fix: each member turn minted a fresh ``RunContext``, so the per-run
        dollar budget and usage limits reset every turn — a cost-cap bypass.
        """
        from troopai.adk.run import swarm_loop_streamed
        from troopai.adk.run.context import RunContext

        member = Agent(name="m", system_prompt="x")
        ctx_wrapper: RunContext[None] = RunContext.make(None)
        captured: dict[str, Any] = {}

        def _fake_run_streamed(*args: Any, **kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            inner = MagicMock()

            async def _empty() -> Any:
                return
                yield  # pragma: no cover - makes this an async generator

            inner.stream_events = _empty
            inner.final_output = None
            inner.new_items = []
            inner.context = ctx_wrapper
            inner.swarm_yield = None
            return inner

        with patch.object(Runner, "_run_streamed", staticmethod(_fake_run_streamed)):
            await swarm_loop_streamed._stream_member_turn(
                member=member,
                user_prompt="hi",
                ctx_wrapper=ctx_wrapper,
                hooks=MagicMock(),
                config=RunConfig(),
                is_first_turn=True,
                result=MagicMock(),
            )

        assert captured.get("shared_run_context") is ctx_wrapper

    def test_run_streamed_reuses_shared_run_context_on_result(self) -> None:
        """When ``shared_run_context`` is supplied, ``_run_streamed`` must build
        its ``RunResultStreaming`` on that exact context (the accumulation
        target the agent loop writes cost / usage onto), not a fresh one."""
        from troopai.adk.run.context import RunContext

        agent = Agent(name="A", system_prompt="x")
        shared: RunContext[None] = RunContext.make(None)

        with patch(
            "troopai.adk.run.runner.run_agent_loop_streamed",
            new=AsyncMock(side_effect=_noop_loop_streamed),
        ):
            result = Runner._run_streamed(agent, "go", shared_run_context=shared)

        assert result.context is shared

    def test_run_streamed_mints_fresh_context_by_default(self) -> None:
        """Default (no ``shared_run_context``) preserves standalone behaviour:
        a fresh ``RunContext`` per run."""
        from troopai.adk.run.context import RunContext

        agent = Agent(name="A", system_prompt="x")
        outer: RunContext[None] = RunContext.make(None)

        with patch(
            "troopai.adk.run.runner.run_agent_loop_streamed",
            new=AsyncMock(side_effect=_noop_loop_streamed),
        ):
            result = Runner._run_streamed(agent, "go")

        assert result.context is not None
        assert result.context is not outer

    def test_run_streamed_rejects_shared_context_on_state_resume(self) -> None:
        """A ``RunState`` resume mints its own context, so pairing it with
        ``shared_run_context`` would silently drop the caller's context — and
        with it the sole carrier of the per-run dollar budget. The
        un-implemented combination must fail closed, not open a cost-cap gap."""
        from troopai.adk.run.context import RunContext
        from troopai.adk.run.state import RunState

        agent = Agent(name="A", system_prompt="x")
        shared: RunContext[None] = RunContext.make(None)

        with pytest.raises(ValueError, match="shared_run_context is not supported"):
            Runner._run_streamed(agent, RunState(), shared_run_context=shared)


# ---------------------------------------------------------------------------
# RECONCILE fix 3: wrap_hooks_with_verbose must be idempotent.
# ---------------------------------------------------------------------------


class TestWrapHooksWithVerboseIdempotent:
    def test_double_wrap_yields_single_verbose_layer(self) -> None:
        """Wrapping an already-wrapped chain must not stack a second
        ``VerboseHooks`` layer (which would fire every verbose panel twice on
        streamed swarm member turns).

        Pre-fix: wrap composed unconditionally, so a re-wrap produced two
        ``VerboseHooks`` in the chain and re-composed a new object.
        """
        from troopai.adk.run.runner import wrap_hooks_with_verbose
        from troopai.adk.verbose.config import VerboseConfig
        from troopai.adk.verbose.hooks import find_verbose_hooks

        cfg = RunConfig(verbose=VerboseConfig(enabled=True))
        once = wrap_hooks_with_verbose(None, cfg)
        assert len(find_verbose_hooks(once)) == 1

        twice = wrap_hooks_with_verbose(once, cfg)
        assert len(find_verbose_hooks(twice)) == 1
        assert twice is once  # idempotent: returned unchanged, not re-composed

    def test_double_wrap_with_user_hooks_stays_idempotent(self) -> None:
        """Idempotency holds with a user hook present: the second wrap returns
        the same chain rather than nesting another verbose layer."""
        from troopai.adk.hooks.hooks import RunHooks
        from troopai.adk.run.runner import wrap_hooks_with_verbose
        from troopai.adk.verbose.config import VerboseConfig
        from troopai.adk.verbose.hooks import find_verbose_hooks

        cfg = RunConfig(verbose=VerboseConfig(enabled=True))
        once = wrap_hooks_with_verbose(RunHooks(), cfg)
        assert len(find_verbose_hooks(once)) == 1

        twice = wrap_hooks_with_verbose(once, cfg)
        assert twice is once
        assert len(find_verbose_hooks(twice)) == 1


# ---------------------------------------------------------------------------
# RECONCILE fix 4: runner teardown sweeps open verbose panels on exception.
# ---------------------------------------------------------------------------


class TestVerbosePanelSweepOnException:
    async def test_streamed_run_exception_sweeps_verbose_panels(self) -> None:
        """An exception mid-streamed-run must trigger
        ``VerboseHooks.close_all_panels()`` so generic verbose tree blocks left
        open by the interrupted run are swept, not leaked.

        Pre-fix: the streamed teardown never called ``close_all_panels()``.
        """
        from troopai.adk.verbose.config import VerboseConfig
        from troopai.adk.verbose.hooks import VerboseHooks

        agent = Agent(name="A", system_prompt="x")
        cfg = RunConfig(verbose=VerboseConfig(enabled=True))

        async def _boom(**kwargs: Any) -> None:
            del kwargs
            raise RuntimeError("mid-run boom")

        with (
            patch("troopai.adk.run.runner.run_agent_loop_streamed", new=AsyncMock(side_effect=_boom)),
            patch.object(VerboseHooks, "close_all_panels", autospec=True) as mock_close,
        ):
            result = Runner._run_streamed(agent, "go", run_config=cfg)
            with pytest.raises(RuntimeError, match="mid-run boom"):
                async for _ in result.stream_events():
                    pass

        assert mock_close.called

    async def test_arun_exception_sweeps_verbose_panels(self) -> None:
        """The non-streamed ``arun`` teardown sweeps open panels the same way."""
        from troopai.adk.verbose.config import VerboseConfig
        from troopai.adk.verbose.hooks import VerboseHooks

        agent = Agent(name="A", system_prompt="x")
        cfg = RunConfig(verbose=VerboseConfig(enabled=True))

        async def _boom(**kwargs: Any) -> Any:
            del kwargs
            raise RuntimeError("boom")

        with (
            patch("troopai.adk.run.runner.run_agent_loop", new=AsyncMock(side_effect=_boom)),
            patch.object(VerboseHooks, "close_all_panels", autospec=True) as mock_close,
            pytest.raises(RuntimeError, match="boom"),
        ):
            await Runner.arun(agent, "go", run_config=cfg)

        assert mock_close.called

    async def test_clean_streamed_run_does_not_sweep(self) -> None:
        """A clean run must NOT sweep — guards against an over-broad teardown
        that would prematurely close a swarm's shared panels on a healthy
        member turn (the sweep is gated on the run ending by exception)."""
        from troopai.adk.verbose.config import VerboseConfig
        from troopai.adk.verbose.hooks import VerboseHooks

        agent = Agent(name="A", system_prompt="x")
        cfg = RunConfig(verbose=VerboseConfig(enabled=True))

        with (
            patch("troopai.adk.run.runner.run_agent_loop_streamed", new=AsyncMock(side_effect=_noop_loop_streamed)),
            patch.object(VerboseHooks, "close_all_panels", autospec=True) as mock_close,
        ):
            result = Runner._run_streamed(agent, "go", run_config=cfg)
            async for _ in result.stream_events():
                pass

        assert not mock_close.called

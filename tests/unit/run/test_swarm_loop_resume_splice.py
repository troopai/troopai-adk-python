"""Unit tests for ``run_swarm_loop``'s step-7 deep-resume splice.

Exercises the dispatch logic that picks between ``run_agent_loop`` and
the two deep-resume helpers (``run_resumed_nested_turn`` /
``run_resumed_hitl_turn``) based on the combination of parked state in
the loaded :class:`SwarmState` and the presence of a ``SwarmResume``
payload threaded through from
:meth:`Runner.arun_swarm_from_checkpoint`.

The inner runners are mocked so the loop exits cleanly after a single
iteration via :class:`MaxTurnsTermination`; the assertions cover which
callable was awaited rather than the full per-turn semantics.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs.interrupt import (
    Interrupt,
    NestedAgentInterrupt,
)
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.config import DEFAULT_RUN_CONFIG
from troopai.adk.run.context import RunContext
from troopai.adk.run.state import RunState
from troopai.adk.run.swarm_loop import run_swarm_loop
from troopai.adk.swarms.interrupt import SwarmResume
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.state import SwarmState
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.tools.deferred_tool import DeferredToolCall, DeferredToolRequests
from troopai.adk.types.run.run_result import RunResult


def _make_swarm(member_name: str = "approver") -> Swarm:
    """Build a single-member swarm capped at one turn for splice tests."""
    member = Agent(name=member_name, system_prompt="x")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(1),
    )


def _stub_run_result(member: Agent[Any], ctx: RunContext[Any]) -> RunResult[Any]:
    """A minimal RunResult that lets the loop's step-8 advance cleanly."""
    return RunResult(
        final_output=None,
        user_prompt="",
        new_items=[],
        context=ctx,
        last_agent=member,
        swarm_yield=None,
    )


def _parked_nested_state(swarm: Swarm, member_name: str) -> SwarmState[Any]:
    """SwarmState with both an interrupt and a nested-agent snapshot parked."""
    state: SwarmState[Any] = SwarmState(
        swarm=swarm,
        current_agent=swarm.entry,
        current_agent_name=member_name,
    )
    state.total_turns = 0
    state.pending_interrupts[member_name] = NestedAgentInterrupt(
        node_id=member_name,
        agent_name=member_name,
        tool_call_ids=("c1",),
        kind="tool_approval",
        question="Approve?",
    )
    deferred_call = DeferredToolCall(
        tool_call_id="c1",
        tool_name="risky_tool",
        tool_arguments={"x": 1},
        raw_arguments='{"x": 1}',
    )
    state.nested_agent_snapshots[member_name] = RunState(
        current_agent_name=member_name,
        turn_count=1,
        deferred_tool_requests=DeferredToolRequests(approvals=[deferred_call]),
    )
    return state


def _parked_hitl_state(swarm: Swarm, member_name: str) -> SwarmState[Any]:
    """SwarmState with only a HITL Interrupt parked (no nested snapshot)."""
    state: SwarmState[Any] = SwarmState(
        swarm=swarm,
        current_agent=swarm.entry,
        current_agent_name=member_name,
    )
    state.total_turns = 0
    state.pending_interrupts[member_name] = Interrupt(
        node_id=member_name,
        question="Approve?",
        kind="tool_approval",
    )
    return state


class TestSwarmLoopResumeSplice:
    async def test_fresh_run_dispatches_to_run_agent_loop(self) -> None:
        """No parked state + no swarm_resume → fresh-turn path."""
        sw = _make_swarm()
        member = sw.entry
        ctx: RunContext[None] = RunContext.make(None)

        run_agent_loop_mock = AsyncMock(return_value=_stub_run_result(member, ctx))
        nested_mock = AsyncMock()
        hitl_mock = AsyncMock()

        with (
            patch("troopai.adk.run.swarm_loop.run_agent_loop", new=run_agent_loop_mock),
            patch("troopai.adk.run.swarm_loop.run_resumed_nested_turn", new=nested_mock),
            patch("troopai.adk.run.swarm_loop.run_resumed_hitl_turn", new=hitl_mock),
        ):
            result = await run_swarm_loop(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                initial_state=None,
                swarm_resume=None,
            )

        assert run_agent_loop_mock.await_count == 1
        assert nested_mock.await_count == 0
        assert hitl_mock.await_count == 0
        assert result.stop_reason.kind == "max_turns"

    async def test_loaded_state_no_resume_takes_clear_and_restart(self) -> None:
        """Parked interrupt + no swarm_resume → clear, then fresh-turn path.

        The clear-and-restart path drops parked state before re-entering
        the loop so the single-iteration run dispatches to
        ``run_agent_loop`` exactly as a fresh run would.
        """
        sw = _make_swarm()
        member = sw.entry
        ctx: RunContext[None] = RunContext.make(None)
        parked = _parked_nested_state(sw, member.name)

        run_agent_loop_mock = AsyncMock(return_value=_stub_run_result(member, ctx))
        nested_mock = AsyncMock()
        hitl_mock = AsyncMock()

        with (
            patch("troopai.adk.run.swarm_loop.run_agent_loop", new=run_agent_loop_mock),
            patch("troopai.adk.run.swarm_loop.run_resumed_nested_turn", new=nested_mock),
            patch("troopai.adk.run.swarm_loop.run_resumed_hitl_turn", new=hitl_mock),
        ):
            await run_swarm_loop(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                initial_state=parked,
                swarm_resume=None,
            )

        # The clear-and-restart guard dropped both parked entries.
        assert member.name not in parked.pending_interrupts
        assert member.name not in parked.nested_agent_snapshots
        # And the loop took the default fresh-turn branch.
        assert run_agent_loop_mock.await_count == 1
        assert nested_mock.await_count == 0
        assert hitl_mock.await_count == 0

    async def test_parked_nested_with_resume_dispatches_to_nested_helper(self) -> None:
        """Interrupt + snapshot + swarm_resume → nested-defer helper."""
        sw = _make_swarm()
        member = sw.entry
        ctx: RunContext[None] = RunContext.make(None)
        parked = _parked_nested_state(sw, member.name)

        run_agent_loop_mock = AsyncMock()
        nested_mock = AsyncMock(return_value=_stub_run_result(member, ctx))
        hitl_mock = AsyncMock()

        with (
            patch("troopai.adk.run.swarm_loop.run_agent_loop", new=run_agent_loop_mock),
            patch("troopai.adk.run.swarm_loop.run_resumed_nested_turn", new=nested_mock),
            patch("troopai.adk.run.swarm_loop.run_resumed_hitl_turn", new=hitl_mock),
        ):
            await run_swarm_loop(
                swarm=sw,
                user_prompt="",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                initial_state=parked,
                swarm_resume=SwarmResume(replies={member.name: "ignored-by-mock"}),
            )

        # Routed to the nested-defer helper exactly once.
        assert nested_mock.await_count == 1
        assert run_agent_loop_mock.await_count == 0
        assert hitl_mock.await_count == 0

        assert nested_mock.await_args is not None
        call_kwargs = nested_mock.await_args.kwargs
        assert call_kwargs["member"] is member
        assert call_kwargs["state"] is parked
        assert call_kwargs["ctx_wrapper"] is ctx

    async def test_parked_hitl_with_resume_dispatches_to_hitl_helper(self) -> None:
        """Interrupt + no snapshot + swarm_resume → HITL-pure helper."""
        sw = _make_swarm()
        member = sw.entry
        ctx: RunContext[None] = RunContext.make(None)
        parked = _parked_hitl_state(sw, member.name)

        run_agent_loop_mock = AsyncMock()
        nested_mock = AsyncMock()
        hitl_mock = AsyncMock(return_value=_stub_run_result(member, ctx))

        with (
            patch("troopai.adk.run.swarm_loop.run_agent_loop", new=run_agent_loop_mock),
            patch("troopai.adk.run.swarm_loop.run_resumed_nested_turn", new=nested_mock),
            patch("troopai.adk.run.swarm_loop.run_resumed_hitl_turn", new=hitl_mock),
        ):
            await run_swarm_loop(
                swarm=sw,
                user_prompt="",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                initial_state=parked,
                swarm_resume=SwarmResume(replies={member.name: "approved"}),
            )

        # Routed to the HITL-pure helper exactly once.
        assert hitl_mock.await_count == 1
        assert run_agent_loop_mock.await_count == 0
        assert nested_mock.await_count == 0

        assert hitl_mock.await_args is not None
        call_kwargs = hitl_mock.await_args.kwargs
        assert call_kwargs["member"] is member
        assert call_kwargs["state"] is parked
        assert call_kwargs["ctx_wrapper"] is ctx
        # The HITL helper needs the per-turn message inputs from step 4/5.
        assert "turn_messages" in call_kwargs
        assert "max_turns" in call_kwargs


class TestSwarmLoopTurnSpanLifecycle:
    async def test_swarm_loop_opens_turn_span_per_iteration(self) -> None:
        """Each loop iteration that runs a member turn opens swarm_turn_span."""
        from troopai.adk.tracing.spans import NoOpSpan
        from troopai.adk.types.tracing.span_data import SwarmTurnSpanData

        sw = _make_swarm()
        ctx: RunContext[None] = RunContext.make(None)

        async def _fake_run_agent_loop(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            return RunResult(
                final_output="done",
                user_prompt="",
                new_items=[],
                context=ctx,
                last_agent=sw.entry,
            )

        captured_calls: list[tuple[str, int, str]] = []

        def _track_swarm_turn_span(
            *, swarm_id: str, index: int, member: str, **rest: Any
        ) -> NoOpSpan[SwarmTurnSpanData]:
            del rest
            captured_calls.append((swarm_id, index, member))
            return NoOpSpan(SwarmTurnSpanData(swarm_id=swarm_id, index=index, member=member))

        with (
            patch(
                "troopai.adk.run.swarm_loop.run_agent_loop",
                new=AsyncMock(side_effect=_fake_run_agent_loop),
            ),
            patch(
                "troopai.adk.run.swarm_loop.swarm_turn_span",
                side_effect=_track_swarm_turn_span,
            ),
        ):
            await run_swarm_loop(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                swarm_id="abc-123",
            )

        assert len(captured_calls) == 1
        assert captured_calls[0] == ("abc-123", 1, sw.entry.name)


class TestSwarmLoopTurnSpanStamping:
    """Cover the enabled-tracing payload-mutation path through ``_stamp_turn_span``.

    ``TestSwarmLoopTurnSpanLifecycle`` patches the factory with a
    ``NoOpSpan`` whose ``data`` is a ``SwarmTurnSpanData`` typed
    payload — not the ``CustomSpanData`` envelope the production
    factory returns when tracing is enabled. That leaves the
    ``cast(CustomSpanData, span.data).data[...]`` mutation block
    uncovered. These tests substitute a span whose ``data`` is a
    real ``CustomSpanData`` envelope so the stamp helper's dict
    mutations are observable.
    """

    async def test_enabled_tracing_stamps_success_status_and_duration(self) -> None:
        from dataclasses import replace as dataclass_replace

        from troopai.adk.tracing.spans import Span
        from troopai.adk.types.tracing.span_data import CustomSpanData, SwarmTurnSpanData

        sw = _make_swarm()
        ctx: RunContext[None] = RunContext.make(None)
        cfg = dataclass_replace(DEFAULT_RUN_CONFIG, tracing_enabled=True)

        async def _fake_run_agent_loop(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            return RunResult(
                final_output="done",
                user_prompt="",
                new_items=[],
                context=ctx,
                last_agent=sw.entry,
            )

        captured_spans: list[Span[Any]] = []

        def _enabled_turn_span(*, swarm_id: str, index: int, member: str, **rest: Any) -> Span[Any]:
            del rest
            envelope = CustomSpanData(
                name=f"swarm.turn.{index}",
                data=SwarmTurnSpanData(
                    swarm_id=swarm_id,
                    index=index,
                    member=member,
                ).export(),
            )
            span: Span[CustomSpanData] = Span(envelope)
            captured_spans.append(span)
            return span

        with (
            patch(
                "troopai.adk.run.swarm_loop.run_agent_loop",
                new=AsyncMock(side_effect=_fake_run_agent_loop),
            ),
            patch(
                "troopai.adk.run.swarm_loop.swarm_turn_span",
                side_effect=_enabled_turn_span,
            ),
        ):
            await run_swarm_loop(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=cfg,
                swarm_id="abc-stamp",
            )

        assert len(captured_spans) == 1
        envelope = captured_spans[0].data
        assert isinstance(envelope, CustomSpanData)
        payload = envelope.data
        # The success branch in step 10 set turn_status="success",
        # which the finally relayed into _stamp_turn_span.
        assert payload["status"] == "success"
        # Wall-clock spans complete in microseconds, so duration_ms
        # is integer-truncated to zero — assert the floor only.
        assert isinstance(payload["duration_ms"], int)
        assert payload["duration_ms"] >= 0

    async def test_disabled_tracing_skips_payload_mutation(self) -> None:
        """tracing_enabled=False must NOT touch envelope.data dict keys.

        Regression for the SwarmTurnSpanData-vs-CustomSpanData mismatch:
        with tracing off the NoOpSpan.data is a SwarmTurnSpanData
        dataclass, not the dict-shaped envelope, so the legacy
        ``cast(CustomSpanData, span.data).data[...]`` site would raise.
        ``_stamp_turn_span`` must gate that mutation on the flag.
        """
        from troopai.adk.tracing.spans import NoOpSpan
        from troopai.adk.types.tracing.span_data import SwarmTurnSpanData

        sw = _make_swarm()
        ctx: RunContext[None] = RunContext.make(None)

        async def _fake_run_agent_loop(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            return RunResult(
                final_output="done",
                user_prompt="",
                new_items=[],
                context=ctx,
                last_agent=sw.entry,
            )

        captured_spans: list[NoOpSpan[SwarmTurnSpanData]] = []

        def _noop_turn_span(*, swarm_id: str, index: int, member: str, **rest: Any) -> NoOpSpan[SwarmTurnSpanData]:
            del rest
            span = NoOpSpan(SwarmTurnSpanData(swarm_id=swarm_id, index=index, member=member))
            captured_spans.append(span)
            return span

        with (
            patch(
                "troopai.adk.run.swarm_loop.run_agent_loop",
                new=AsyncMock(side_effect=_fake_run_agent_loop),
            ),
            patch(
                "troopai.adk.run.swarm_loop.swarm_turn_span",
                side_effect=_noop_turn_span,
            ),
        ):
            # tracing_enabled defaults to False on DEFAULT_RUN_CONFIG.
            await run_swarm_loop(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                swarm_id="abc-disabled",
            )

        assert len(captured_spans) == 1
        # With tracing disabled the typed payload is preserved
        # untouched — no status / duration_ms was written, because
        # the helper short-circuits before the dict-cast block.
        payload = captured_spans[0].data
        assert isinstance(payload, SwarmTurnSpanData)
        assert payload.status is None
        assert payload.duration_ms is None

"""Unit tests for run_swarm_loop_streamed.

Tests stub _stream_member_turn so the loop body's event seams
are testable in isolation. The stub is replaced by the real
Runner._run_streamed passthrough when wiring the inner stream.
"""

from __future__ import annotations

import asyncio
from typing import Any, override
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.config import DEFAULT_RUN_CONFIG
from troopai.adk.run.context import RunContext
from troopai.adk.run.swarm_loop_streamed import run_swarm_loop_streamed
from troopai.adk.swarms.events import (
    SwarmDoneEvent,
    SwarmStartEvent,
    SwarmTurnEndEvent,
    SwarmTurnStartEvent,
)
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.result import SwarmRunResultStreaming
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.swarms.yield_signal import SwarmDone
from troopai.adk.types.run.run_result import RunResult


def _make_swarm(*, max_turns: int = 1) -> Swarm[Any]:
    member = Agent(name="approver", system_prompt="x")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(max_turns),
    )


def _make_deferral(member_name: str = "approver", tool_call_id: str = "c1") -> Any:
    """Build an AgentToolDeferral with one deferred tool approval.

    Kept module-level so the suspend-path test stays under R4's
    60-line cap; the fixture is reusable for any future
    nested-defer test that needs a parked deferral.
    """
    from troopai.adk.exceptions import AgentToolDeferral
    from troopai.adk.run.state import RunState
    from troopai.adk.tools.deferred_tool import (
        DeferredToolCall,
        DeferredToolRequests,
    )

    deferred_call = DeferredToolCall(
        tool_call_id=tool_call_id,
        tool_name="risky",
        tool_arguments={},
        raw_arguments="{}",
    )
    parked_run_state = RunState(
        current_agent_name=member_name,
        turn_count=1,
        deferred_tool_requests=DeferredToolRequests(approvals=[deferred_call]),
    )
    return AgentToolDeferral(
        agent_name=member_name,
        deferred_requests=DeferredToolRequests(approvals=[deferred_call]),
        state=parked_run_state,
    )


class TestRunSwarmLoopStreamedHappyPath:
    async def test_single_turn_emits_start_turnstart_turnend_done(self) -> None:
        sw = _make_swarm(max_turns=1)
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(
            user_prompt="go",
        )

        async def _fake_stream_member_turn(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            return RunResult(
                final_output="done",
                user_prompt="",
                new_items=[],
                context=ctx,
                last_agent=sw.entry,
            )

        with patch(
            "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
            new=AsyncMock(side_effect=_fake_stream_member_turn),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
                swarm_id="abc-123",
            )

        # Set a no-op driver task and drain events.
        result.set_run_task(asyncio.get_running_loop().create_task(asyncio.sleep(0)))

        events: list[Any] = []
        async for ev in result.stream_events():
            events.append(ev)

        types = [type(ev).__name__ for ev in events]
        assert types == [
            "SwarmStartEvent",
            "SwarmTurnStartEvent",
            "SwarmTurnEndEvent",
            "SwarmDoneEvent",
        ]
        assert isinstance(events[0], SwarmStartEvent)
        assert events[0].entry_agent == "approver"
        assert events[0].member_names == ("approver",)
        assert isinstance(events[1], SwarmTurnStartEvent)
        assert events[1].agent == "approver"
        assert events[1].turn == 1
        assert isinstance(events[2], SwarmTurnEndEvent)
        assert events[2].agent == "approver"
        assert isinstance(events[3], SwarmDoneEvent)

    async def test_swarm_done_resolves_last_yield_final_output(self) -> None:
        """A SwarmDone yield stores the RESOLVED final_output on state.last_yield.

        Regression: the streamed loop stored the raw SwarmDone (whose
        final_output is typically None — swarm_done is emitted before the
        terminal string), diverging from the sync loop which stores
        replace(signal, final_output=resolved).
        """
        sw = _make_swarm(max_turns=1)
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(user_prompt="go")

        async def _fake_stream_member_turn(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            return RunResult(
                final_output="resolved-text",
                user_prompt="",
                new_items=[],
                context=ctx,
                last_agent=sw.entry,
                swarm_yield=SwarmDone(reason="done", final_output=None),
            )

        with patch(
            "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
            new=AsyncMock(side_effect=_fake_stream_member_turn),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
                swarm_id="abc-123",
            )

        state = result.state
        assert state is not None
        assert isinstance(state.last_yield, SwarmDone)
        # Resolved from the terminal turn's output, not the raw None.
        assert state.last_yield.final_output == "resolved-text"
        assert result.final_output == "resolved-text"

    async def test_two_turn_run_emits_two_turn_pairs(self) -> None:
        sw = _make_swarm(max_turns=2)
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(
            user_prompt="go",
        )

        async def _fake_stream_member_turn(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            return RunResult(
                final_output="ok",
                user_prompt="",
                new_items=[],
                context=ctx,
                last_agent=sw.entry,
            )

        with patch(
            "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
            new=AsyncMock(side_effect=_fake_stream_member_turn),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
                swarm_id="abc-123",
            )

        result.set_run_task(asyncio.get_running_loop().create_task(asyncio.sleep(0)))

        events: list[Any] = []
        async for ev in result.stream_events():
            events.append(ev)

        types = [type(ev).__name__ for ev in events]
        assert types == [
            "SwarmStartEvent",
            "SwarmTurnStartEvent",
            "SwarmTurnEndEvent",
            "SwarmTurnStartEvent",
            "SwarmTurnEndEvent",
            "SwarmDoneEvent",
        ]
        assert events[1].turn == 1
        assert events[3].turn == 2


class TestRunSwarmLoopStreamedPerAgentPassthrough:
    async def test_inner_agent_events_appear_between_turn_start_and_end(self) -> None:
        """Per-agent events from Runner._run_streamed land between
        SwarmTurnStartEvent and SwarmTurnEndEvent."""
        from troopai.adk.run.stream import RawResponseStreamEvent, RunResultStreaming

        sw = _make_swarm(max_turns=1)
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(
            user_prompt="go",
        )

        # Synthesize inner stream events.
        inner_event_a = RawResponseStreamEvent(data="tok-a")
        inner_event_b = RawResponseStreamEvent(data="tok-b")

        fake_inner = RunResultStreaming(
            current_agent=sw.entry,
            user_prompt="",
            final_output="ok",
        )
        # Pre-populate the inner stream's queue + complete.
        await fake_inner.put_event(inner_event_a)
        await fake_inner.put_event(inner_event_b)
        await fake_inner.complete()
        fake_inner.new_items = []

        with patch(
            "troopai.adk.run.runner.Runner._run_streamed",
            return_value=fake_inner,
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
                swarm_id="abc-123",
            )

        result.set_run_task(asyncio.get_running_loop().create_task(asyncio.sleep(0)))

        events: list[Any] = []
        async for ev in result.stream_events():
            events.append(ev)

        types = [type(ev).__name__ for ev in events]
        # Locate the inner events in the sequence.
        assert types[0] == "SwarmStartEvent"
        assert types[1] == "SwarmTurnStartEvent"
        assert events[2] is inner_event_a
        assert events[3] is inner_event_b
        assert types[-2] == "SwarmTurnEndEvent"
        assert types[-1] == "SwarmDoneEvent"


class TestRunSwarmLoopStreamedSuspendPaths:
    async def test_interrupt_exception_emits_turn_interrupt_then_done(self) -> None:
        """Pure HITL raises InterruptException; stream replaces SwarmTurnEndEvent
        with SwarmTurnInterruptEvent and populates result.interrupts."""
        from troopai.adk.graphs.interrupt import Interrupt, InterruptException

        sw = _make_swarm(max_turns=3)
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(
            user_prompt="go",
        )
        interrupt = Interrupt(
            node_id="approver",
            question="Approve?",
            kind="tool_approval",
        )

        async def _raise_interrupt(**kwargs: Any) -> Any:
            del kwargs
            raise InterruptException(interrupt)

        with patch(
            "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
            new=AsyncMock(side_effect=_raise_interrupt),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
                swarm_id="abc-123",
            )

        result.set_run_task(asyncio.get_running_loop().create_task(asyncio.sleep(0)))

        events: list[Any] = []
        async for ev in result.stream_events():
            events.append(ev)

        types = [type(ev).__name__ for ev in events]
        assert types == [
            "SwarmStartEvent",
            "SwarmTurnStartEvent",
            "SwarmTurnInterruptEvent",
            "SwarmDoneEvent",
        ]
        # SwarmTurnEndEvent is REPLACED by SwarmTurnInterruptEvent for
        # the interrupted turn.
        assert "SwarmTurnEndEvent" not in types
        # The interrupt event carries the parked Interrupt.
        assert events[2].interrupt is interrupt
        # Result fields are populated so the caller can checkpoint + resume.
        assert result.stop_reason is not None
        assert result.stop_reason.kind == "interrupted"
        assert result.interrupts == (interrupt,)

    async def test_agent_tool_deferral_lifts_to_nested_agent_interrupt(self) -> None:
        """Nested-agent defer emits SwarmTurnInterruptEvent carrying a
        NestedAgentInterrupt and parks nested_agent_snapshots."""
        from troopai.adk.graphs.interrupt import NestedAgentInterrupt

        sw = _make_swarm(max_turns=3)
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(
            user_prompt="go",
        )
        deferral = _make_deferral()

        async def _raise_deferral(**kwargs: Any) -> Any:
            del kwargs
            raise deferral

        with patch(
            "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
            new=AsyncMock(side_effect=_raise_deferral),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
                swarm_id="abc-123",
            )

        result.set_run_task(asyncio.get_running_loop().create_task(asyncio.sleep(0)))

        events: list[Any] = []
        async for ev in result.stream_events():
            events.append(ev)

        types = [type(ev).__name__ for ev in events]
        assert types == [
            "SwarmStartEvent",
            "SwarmTurnStartEvent",
            "SwarmTurnInterruptEvent",
            "SwarmDoneEvent",
        ]
        assert isinstance(events[2].interrupt, NestedAgentInterrupt)
        assert events[2].interrupt.tool_call_ids == ("c1",)
        assert result.state is not None
        assert "approver" in result.state.nested_agent_snapshots


class TestRunSwarmLoopStreamedResumeThroughStream:
    async def test_nested_defer_resume_emits_turnstart_turnend(self) -> None:
        """The deep-resume splice from swarm_resume.py fires inside the streamed
        loop when initial_state has a parked nested-defer snapshot and
        swarm_resume carries the matching NestedAgentReply."""
        from troopai.adk.graphs.interrupt import (
            NestedAgentApproval,
            NestedAgentInterrupt,
            NestedAgentReply,
        )
        from troopai.adk.run.state import RunState
        from troopai.adk.swarms.interrupt import SwarmResume
        from troopai.adk.swarms.state import SwarmState
        from troopai.adk.tools.deferred_tool import (
            DeferredToolCall,
            DeferredToolRequests,
        )
        from troopai.adk.types.run.run_result import RunResult

        sw = _make_swarm(max_turns=3)
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(
            user_prompt="",
        )

        # Build a state parked on a nested-agent-defer interrupt for
        # the entry member.
        state: SwarmState[Any] = SwarmState(
            swarm=sw,
            current_agent=sw.entry,
            current_agent_name=sw.entry.name,
        )
        state.swarm_id = "abc-123"
        deferred_call = DeferredToolCall(
            tool_call_id="c1",
            tool_name="risky",
            tool_arguments={},
            raw_arguments="{}",
        )
        state.nested_agent_snapshots[sw.entry.name] = RunState(
            current_agent_name=sw.entry.name,
            turn_count=1,
            deferred_tool_requests=DeferredToolRequests(approvals=[deferred_call]),
        )
        state.pending_interrupts[sw.entry.name] = NestedAgentInterrupt(
            node_id=sw.entry.name,
            agent_name=sw.entry.name,
            tool_call_ids=("c1",),
            question="Approve?",
        )

        # Mock the resume helper to return a clean RunResult so the
        # splice path completes through to a SwarmTurnEndEvent.
        resumed_run_result = RunResult(
            final_output="done",
            user_prompt="",
            new_items=[],
            context=ctx,
            last_agent=sw.entry,
        )

        with patch(
            "troopai.adk.run.swarm_loop_streamed.run_resumed_nested_turn",
            new=AsyncMock(return_value=resumed_run_result),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
                initial_state=state,
                swarm_resume=SwarmResume(
                    replies={
                        sw.entry.name: NestedAgentReply(
                            decisions=(NestedAgentApproval(tool_call_id="c1"),),
                        )
                    }
                ),
                swarm_id="abc-123",
            )

        result.set_run_task(asyncio.get_running_loop().create_task(asyncio.sleep(0)))

        events: list[Any] = []
        async for ev in result.stream_events():
            events.append(ev)

        types = [type(ev).__name__ for ev in events]
        # The resumed turn emits a TurnStart + TurnEnd (no inner-agent
        # passthrough — the resume helper uses the non-streamed
        # Runner.arun internally for the resumed agent run).
        assert "SwarmStartEvent" in types
        assert "SwarmTurnStartEvent" in types
        assert "SwarmTurnEndEvent" in types
        assert "SwarmDoneEvent" in types
        # No TurnInterruptEvent — the resume completed cleanly.
        assert "SwarmTurnInterruptEvent" not in types


class TestStreamedUsageAccumulation:
    async def test_member_turn_usage_accumulates_into_state_and_context(self) -> None:
        """Streamed member usage must accumulate onto state + context + per-member.

        Regression: each member turn ran on a FRESH RunContext (via
        Runner._run_streamed), so state.cumulative_usage / ctx_wrapper.usage
        never updated — the max_total_tokens cap and TokenBudgetTermination
        were dead in streaming and per_member_usage stayed empty. The driver
        now accumulates each turn's usage from the returned inner context.
        """
        from troopai.adk.llms.llm_usage import LLMUsage

        sw = _make_swarm(max_turns=1)
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(user_prompt="go")

        async def _fake_turn(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            # A DISTINCT inner context carrying this turn's usage — mirrors the
            # real _stream_member_turn returning inner_streaming.context.
            inner_ctx: RunContext[None] = RunContext.make(None)
            inner_ctx.usage = LLMUsage(requests=1, input_tokens=30, output_tokens=20, total_tokens=50)
            return RunResult(
                final_output="done",
                user_prompt="",
                new_items=[],
                context=inner_ctx,
                last_agent=sw.entry,
            )

        with patch(
            "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
            new=AsyncMock(side_effect=_fake_turn),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
                swarm_id="u-1",
            )

        assert result.state is not None
        assert result.state.cumulative_usage.total_tokens == 50
        assert ctx.usage.total_tokens == 50
        assert result.per_member_usage.get("approver") is not None
        assert result.per_member_usage["approver"].total_tokens == 50


class TestStreamedResumeUsageAccumulation:
    async def test_nested_resume_turn_usage_reaches_state_and_per_member(self) -> None:
        """A resumed turn's tokens must reach cumulative + per-member usage.

        Regression: resume helpers fold their tokens directly into
        ``ctx_wrapper.usage`` and return a RunResult carrying ``ctx_wrapper``
        itself. The old identity guard skipped the accumulation block on that
        path, so ``state.cumulative_usage`` and ``per_member_usage`` silently
        dropped every resume-turn's tokens — under-counting the
        ``max_total_tokens`` guard and the returned per-member breakdown. The
        snapshot/delta on ``ctx_wrapper.usage`` now captures resume turns too.
        """
        from troopai.adk.graphs.interrupt import (
            NestedAgentApproval,
            NestedAgentInterrupt,
            NestedAgentReply,
        )
        from troopai.adk.llms.llm_usage import LLMUsage
        from troopai.adk.run.state import RunState
        from troopai.adk.swarms.interrupt import SwarmResume
        from troopai.adk.swarms.state import SwarmState
        from troopai.adk.tools.deferred_tool import (
            DeferredToolCall,
            DeferredToolRequests,
        )

        # Single turn: the patched resume helper does not pop the parked
        # entries (the real one does), so a multi-turn swarm would re-fire
        # the splice every turn and multiply the usage. Bound to one turn so
        # the assertion isolates the resume-path accumulation precisely.
        sw = _make_swarm(max_turns=1)
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(user_prompt="")

        state: SwarmState[Any] = SwarmState(
            swarm=sw,
            current_agent=sw.entry,
            current_agent_name=sw.entry.name,
        )
        state.swarm_id = "ru-1"
        deferred_call = DeferredToolCall(
            tool_call_id="c1",
            tool_name="risky",
            tool_arguments={},
            raw_arguments="{}",
        )
        state.nested_agent_snapshots[sw.entry.name] = RunState(
            current_agent_name=sw.entry.name,
            turn_count=1,
            deferred_tool_requests=DeferredToolRequests(approvals=[deferred_call]),
        )
        state.pending_interrupts[sw.entry.name] = NestedAgentInterrupt(
            node_id=sw.entry.name,
            agent_name=sw.entry.name,
            tool_call_ids=("c1",),
            question="Approve?",
        )

        async def _fake_resumed_nested(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            # Mirror run_resumed_nested_turn: fold the resumed run's tokens
            # straight into the shared context and return ctx_wrapper itself.
            ctx.usage = ctx.usage + LLMUsage(requests=1, input_tokens=40, output_tokens=25, total_tokens=65)
            return RunResult(
                final_output="done",
                user_prompt="",
                new_items=[],
                context=ctx,
                last_agent=sw.entry,
            )

        with patch(
            "troopai.adk.run.swarm_loop_streamed.run_resumed_nested_turn",
            new=AsyncMock(side_effect=_fake_resumed_nested),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
                initial_state=state,
                swarm_resume=SwarmResume(
                    replies={
                        sw.entry.name: NestedAgentReply(
                            decisions=(NestedAgentApproval(tool_call_id="c1"),),
                        )
                    }
                ),
                swarm_id="ru-1",
            )

        assert result.state is not None
        # Cumulative + per-member must include the resumed turn's 65 tokens.
        assert result.state.cumulative_usage.total_tokens == 65
        assert result.per_member_usage.get("approver") is not None
        assert result.per_member_usage["approver"].total_tokens == 65


class TestStreamedOutOfRosterHandoff:
    async def test_out_of_roster_handoff_increments_count_and_records_yield(self) -> None:
        """An out-of-roster handoff must bump handoff_count + record the yield.

        Regression: the streamed loop omitted ``state.handoff_count += 1`` and
        ``policy.record_yield`` on the out-of-roster branch, so a member that
        deterministically re-emits the same unknown target never tripped the
        ``max_handoffs`` hard guard and diverged from the sync loop's policy
        bookkeeping. The branch now mirrors the sync loop.
        """
        from troopai.adk.swarms.config import SwarmConfig
        from troopai.adk.swarms.yield_signal import SwarmHandoff

        recorded: list[SwarmHandoff] = []

        class _RecordingPolicy(RoundRobinPolicy):
            @override
            def record_yield(self, state: Any, signal: Any) -> None:
                del state
                if isinstance(signal, SwarmHandoff):
                    recorded.append(signal)

        member = Agent(name="approver", system_prompt="x")
        sw: Swarm[Any] = Swarm(
            members=(member,),
            entry=member,
            policy=_RecordingPolicy(),
            # High backstop so ONLY the max_handoffs guard can stop the loop
            # once handoff_count advances; before the fix it would never trip
            # and would instead run to this 50-turn termination.
            termination=MaxTurnsTermination(50),
            config=SwarmConfig(max_handoffs=2),
        )
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(user_prompt="go")

        async def _fake_turn(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            return RunResult(
                final_output=None,
                user_prompt="",
                new_items=[],
                context=RunContext.make(None),
                last_agent=member,
                swarm_yield=SwarmHandoff(target="nonexistent", message="hi"),
            )

        with patch(
            "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
            new=AsyncMock(side_effect=_fake_turn),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
                swarm_id="or-1",
            )

        assert result.stop_reason is not None
        assert result.stop_reason.kind == "max_handoffs"
        assert result.handoff_count == 2
        # record_yield fired for every out-of-roster handoff.
        assert len(recorded) == 2
        assert all(s.target == "nonexistent" for s in recorded)


class TestStreamedPolicyError:
    async def test_select_next_raise_terminates_with_policy_error(self) -> None:
        """A policy that raises in select_next must terminate gracefully.

        Regression: select_next was awaited uncaught in the streamed driver,
        so a raising policy surfaced only as a bare set_exception (no
        policy_error stop reason). The sync loop converts it to a
        policy_error StopReason; the streamed driver now matches.
        """

        class _RaisingPolicy(RoundRobinPolicy):
            @override
            async def select_next(self, state: Any, ctx: Any) -> Any:
                del state, ctx
                raise RuntimeError("policy boom")

        member = Agent(name="approver", system_prompt="x")
        sw: Swarm[Any] = Swarm(
            members=(member,),
            entry=member,
            policy=_RaisingPolicy(),
            termination=MaxTurnsTermination(3),
        )
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(user_prompt="go")

        async def _fake_turn(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            # No swarm_yield → the loop falls to the policy select_next branch.
            return RunResult(
                final_output=None,
                user_prompt="",
                new_items=[],
                context=RunContext.make(None),
                last_agent=member,
            )

        with patch(
            "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
            new=AsyncMock(side_effect=_fake_turn),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
                swarm_id="p-1",
            )

        assert result.stop_reason is not None
        assert result.stop_reason.kind == "policy_error"


class TestStreamedHitlResumeTurnInput:
    async def test_resume_builds_turn_messages_and_tools(self) -> None:
        """Streamed HITL resume must pass real turn input, not [].

        Regression: the resume branch passed turn_messages=[] / extra_tools=
        None, so the resumed member ran with no messages and no swarm tools.
        The driver now builds the turn input (the sync loop's Step 4-5) before
        resuming — inject_system_prompt alone guarantees a non-empty list.
        """
        from troopai.adk.graphs.interrupt import Interrupt
        from troopai.adk.swarms.interrupt import SwarmResume
        from troopai.adk.swarms.state import SwarmState

        member = Agent(name="approver", system_prompt="x")
        sw: Swarm[Any] = Swarm(
            members=(member,),
            entry=member,
            policy=RoundRobinPolicy(),
            termination=MaxTurnsTermination(1),
        )
        state: SwarmState[Any] = SwarmState(swarm=sw, current_agent=member, current_agent_name=member.name)
        state.pending_interrupts[member.name] = Interrupt(
            node_id=member.name, question="Approve?", kind="tool_approval"
        )
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(user_prompt="go")

        captured: dict[str, Any] = {}

        async def _fake_resumed_hitl(**kwargs: Any) -> RunResult[Any]:
            captured["turn_messages"] = kwargs.get("turn_messages")
            return RunResult(
                final_output="done",
                user_prompt="",
                new_items=[],
                context=RunContext.make(None),
                last_agent=member,
            )

        with patch(
            "troopai.adk.run.swarm_loop_streamed.run_resumed_hitl_turn",
            new=AsyncMock(side_effect=_fake_resumed_hitl),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
                initial_state=state,
                swarm_resume=SwarmResume(replies={member.name: "approved"}),
                swarm_id="h-1",
            )

        assert captured.get("turn_messages") is not None
        assert len(captured["turn_messages"]) > 0


class _FakeInnerStream:
    """Minimal stand-in for the inner ``RunResultStreaming`` that
    ``_stream_member_turn`` drains. Yields no events; carries only the
    fields the swarm loop reads back off a completed member turn.
    """

    def __init__(self, *, context: Any, final_output: Any = "ok") -> None:
        self.context = context
        self.final_output = final_output
        self.new_items: list[Any] = []
        self.swarm_yield: Any = None

    async def stream_events(self) -> Any:
        return
        yield  # pragma: no cover - makes this an async generator


class TestStreamedPerRunBudgetAccumulation:
    async def test_per_run_cost_accumulates_across_streamed_member_turns(self) -> None:
        """Per-run cost / usage must accumulate on the shared driver context
        across streamed member turns, so the per-run dollar budget and usage
        limits accrue cumulatively (matching the sync swarm).

        Regression: each member turn ran on a FRESH RunContext (via
        ``Runner._run_streamed``), so ``ctx_wrapper.cost_usd`` reset to 0 each
        turn — the per-run dollar budget never accumulated and a multi-turn
        swarm could silently blow past it (a cost-cap bypass of the
        cost-conservative invariant). The driver now threads its shared
        ``ctx_wrapper`` into every member turn.
        """
        from troopai.adk.llms.llm_usage import LLMUsage

        sw = _make_swarm(max_turns=2)
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(user_prompt="go")
        seen: list[Any] = []

        def _fake_run_streamed(*args: Any, **kwargs: Any) -> Any:
            shared = kwargs.get("shared_run_context")
            seen.append(shared)
            # A real member turn accrues cost + usage onto the context it runs
            # on; the fix makes that the shared driver context.
            assert shared is not None
            shared.cost_usd += 0.10
            shared.usage = shared.usage + LLMUsage(requests=1, input_tokens=6, output_tokens=4, total_tokens=10)
            return _FakeInnerStream(context=shared)

        with patch(
            "troopai.adk.run.runner.Runner._run_streamed",
            side_effect=_fake_run_streamed,
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
                swarm_id="budget-1",
            )

        # Two member turns ran on the SAME shared context → cumulative totals.
        assert len(seen) == 2
        assert all(s is ctx for s in seen)
        assert ctx.cost_usd == pytest.approx(0.20)
        assert ctx.usage.total_tokens == 20

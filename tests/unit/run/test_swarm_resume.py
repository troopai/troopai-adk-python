"""Unit tests for the swarm deep-resume helpers.

Covers ``run_resumed_nested_turn`` (nested-agent-defer path with
``AgentExecutable.resume_from_snapshot``) and ``run_resumed_hitl_turn``
(HITL-pure path with reply seeded onto :class:`RunContext`). Both
helpers are tested in isolation by mocking the inner dependency
(``AgentExecutable.resume_from_snapshot`` and ``run_agent_loop``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs.interrupt import (
    Interrupt,
    InterruptException,
    NestedAgentApproval,
    NestedAgentInterrupt,
    NestedAgentReply,
    NestedAgentResumeError,
)
from troopai.adk.llms.llm_usage import LLMUsage
from troopai.adk.orchestration.executable import NodeResult
from troopai.adk.run.context import RunContext
from troopai.adk.run.state import RunState
from troopai.adk.run.swarm_resume import run_resumed_hitl_turn, run_resumed_nested_turn
from troopai.adk.swarms.interrupt import SwarmResume
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.state import SwarmState
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.tools.deferred_tool import DeferredToolCall, DeferredToolRequests
from troopai.adk.types.items.items import RunItem


def _make_swarm(name: str = "approver") -> Swarm:
    member = Agent(name=name, system_prompt="x")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(3),
    )


def _make_state_with_parked_nested_defer(swarm: Swarm, member_name: str) -> SwarmState[Any]:
    """Build a SwarmState parked on a nested-agent-defer interrupt."""
    state: SwarmState[Any] = SwarmState(
        swarm=swarm,
        current_agent=swarm.entry,
        current_agent_name=member_name,
    )
    deferred_call = DeferredToolCall(
        tool_call_id="c1",
        tool_name="risky_tool",
        tool_arguments={"x": 1},
        raw_arguments='{"x": 1}',
    )
    parked_run_state = RunState(
        current_agent_name=member_name,
        turn_count=1,
        deferred_tool_requests=DeferredToolRequests(approvals=[deferred_call]),
    )
    state.nested_agent_snapshots[member_name] = parked_run_state
    state.pending_interrupts[member_name] = NestedAgentInterrupt(
        node_id=member_name,
        agent_name=member_name,
        tool_call_ids=("c1",),
        kind="tool_approval",
        question="Approve?",
    )
    return state


def _make_state_with_parked_hitl(swarm: Swarm, member_name: str) -> SwarmState[Any]:
    """Build a SwarmState parked on a pure-HITL Interrupt (no snapshot)."""
    state: SwarmState[Any] = SwarmState(
        swarm=swarm,
        current_agent=swarm.entry,
        current_agent_name=member_name,
    )
    state.pending_interrupts[member_name] = Interrupt(
        node_id=member_name,
        question="Approve?",
        kind="tool_approval",
    )
    return state


def _make_node_result(
    new_items: list[RunItem] | None = None,
    final_output: Any = "resumed output",
    usage: LLMUsage | None = None,
) -> NodeResult[Any]:
    return NodeResult(
        output=final_output,
        new_items=list(new_items) if new_items is not None else [],
        usage=usage or LLMUsage(requests=1, total_tokens=42, input_tokens=30, output_tokens=12),
        final_text=final_output if isinstance(final_output, str) else None,
        metadata={"adapter": "agent", "agent_name": "approver", "last_agent_name": "approver"},
    )


# ─────────────────────────────────────────────────────────────────────
# run_resumed_nested_turn
# ─────────────────────────────────────────────────────────────────────


class TestRunResumedNestedTurnValidation:
    async def test_missing_reply_raises_and_preserves_snapshot(self) -> None:
        sw = _make_swarm()
        member = sw.entry
        state = _make_state_with_parked_nested_defer(sw, member.name)
        ctx: RunContext[None] = RunContext.make(None)
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG

        with pytest.raises(ValueError, match="no reply provided for parked member 'approver'.*nested-agent-defer"):
            await run_resumed_nested_turn(
                member=member,
                swarm_resume=SwarmResume(),
                state=state,
                ctx_wrapper=ctx,
                config=DEFAULT_RUN_CONFIG,
            )
        # Snapshot untouched — caller can retry with corrected payload.
        assert member.name in state.nested_agent_snapshots
        assert member.name in state.pending_interrupts

    async def test_wrong_type_reply_raises_and_preserves_snapshot(self) -> None:
        sw = _make_swarm()
        member = sw.entry
        state = _make_state_with_parked_nested_defer(sw, member.name)
        ctx: RunContext[None] = RunContext.make(None)
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG

        with pytest.raises(ValueError, match="must be NestedAgentReply.*got str"):
            await run_resumed_nested_turn(
                member=member,
                swarm_resume=SwarmResume(replies={member.name: "approved"}),
                state=state,
                ctx_wrapper=ctx,
                config=DEFAULT_RUN_CONFIG,
            )
        assert member.name in state.nested_agent_snapshots

    async def test_resume_error_redeposits_snapshot_for_retry(self) -> None:
        """NestedAgentResumeError from invalid tool_call_id must NOT orphan the snapshot.

        The docstring promises the parked snapshot is left untouched so a
        corrected retry succeeds. Without the try/except re-deposit in
        run_resumed_nested_turn, the snapshot would be popped and then
        orphaned when resume_from_snapshot raises before depositing
        anything back. This test asserts the contract.
        """
        sw = _make_swarm()
        member = sw.entry
        state = _make_state_with_parked_nested_defer(sw, member.name)
        original_snapshot = state.nested_agent_snapshots[member.name]
        original_interrupt = state.pending_interrupts[member.name]
        ctx: RunContext[None] = RunContext.make(None)
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG

        reply = NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="c1"),))

        with (
            patch(
                "troopai.adk.run.swarm_resume.AgentExecutable.resume_from_snapshot",
                new=AsyncMock(side_effect=NestedAgentResumeError(node_id=member.name, detail="bad tool_call_id")),
            ),
            pytest.raises(NestedAgentResumeError),
        ):
            await run_resumed_nested_turn(
                member=member,
                swarm_resume=SwarmResume(replies={member.name: reply}),
                state=state,
                ctx_wrapper=ctx,
                config=DEFAULT_RUN_CONFIG,
            )

        # Snapshot + parked interrupt have been restored — a corrected
        # retry can re-enter the splice and find the parked state intact.
        assert state.nested_agent_snapshots[member.name] is original_snapshot
        assert state.pending_interrupts[member.name] is original_interrupt


class TestRunResumedNestedTurnHappyPath:
    async def test_pops_state_and_returns_run_result(self) -> None:
        sw = _make_swarm()
        member = sw.entry
        state = _make_state_with_parked_nested_defer(sw, member.name)
        ctx: RunContext[None] = RunContext.make(None)
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG

        node_result = _make_node_result(final_output="done")
        reply = NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="c1"),))

        with patch(
            "troopai.adk.run.swarm_resume.AgentExecutable.resume_from_snapshot",
            new=AsyncMock(return_value=node_result),
        ) as mock_resume:
            result = await run_resumed_nested_turn(
                member=member,
                swarm_resume=SwarmResume(replies={member.name: reply}),
                state=state,
                ctx_wrapper=ctx,
                config=DEFAULT_RUN_CONFIG,
            )

        # State has been consumed.
        assert member.name not in state.nested_agent_snapshots
        assert member.name not in state.pending_interrupts

        # Helper called resume_from_snapshot with the right arguments.
        mock_resume.assert_awaited_once()
        assert mock_resume.await_args is not None
        call_kwargs = mock_resume.await_args.kwargs
        assert call_kwargs["reply"] is reply
        assert call_kwargs["node_id"] == member.name
        assert call_kwargs["nested_agent_snapshots"] is state.nested_agent_snapshots

        # Result shape matches the synthesized RunResult contract.
        assert result.final_output == "done"
        assert result.last_agent is member
        assert result.swarm_yield is None

    async def test_usage_fold_advances_ctx_wrapper(self) -> None:
        """Load-bearing: resume tokens must flow into ctx_wrapper.usage."""
        sw = _make_swarm()
        member = sw.entry
        state = _make_state_with_parked_nested_defer(sw, member.name)
        ctx: RunContext[None] = RunContext.make(None)
        ctx.usage = LLMUsage(requests=2, total_tokens=100, input_tokens=80, output_tokens=20)
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG

        inner_usage = LLMUsage(requests=1, total_tokens=42, input_tokens=30, output_tokens=12)
        node_result = _make_node_result(usage=inner_usage)
        reply = NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="c1"),))

        with patch(
            "troopai.adk.run.swarm_resume.AgentExecutable.resume_from_snapshot",
            new=AsyncMock(return_value=node_result),
        ):
            await run_resumed_nested_turn(
                member=member,
                swarm_resume=SwarmResume(replies={member.name: reply}),
                state=state,
                ctx_wrapper=ctx,
                config=DEFAULT_RUN_CONFIG,
            )

        # ctx_wrapper.usage advanced by inner_usage (the swarm loop's
        # _snapshot_usage delta computation then attributes this to the
        # member).
        assert ctx.usage.requests == 3
        assert ctx.usage.total_tokens == 142
        assert ctx.usage.input_tokens == 110
        assert ctx.usage.output_tokens == 32


class TestRunResumedNestedTurnReDeferral:
    async def test_propagates_interrupt_exception_on_re_defer(self) -> None:
        """resume_from_snapshot raising propagates out without catch."""
        sw = _make_swarm()
        member = sw.entry
        state = _make_state_with_parked_nested_defer(sw, member.name)
        ctx: RunContext[None] = RunContext.make(None)
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG

        fresh_interrupt = NestedAgentInterrupt(
            node_id=member.name,
            agent_name=member.name,
            tool_call_ids=("c2",),
            kind="tool_approval",
            question="Approve again?",
        )
        reply = NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="c1"),))

        with (
            patch(
                "troopai.adk.run.swarm_resume.AgentExecutable.resume_from_snapshot",
                new=AsyncMock(side_effect=InterruptException(fresh_interrupt)),
            ),
            pytest.raises(InterruptException) as exc_info,
        ):
            await run_resumed_nested_turn(
                member=member,
                swarm_resume=SwarmResume(replies={member.name: reply}),
                state=state,
                ctx_wrapper=ctx,
                config=DEFAULT_RUN_CONFIG,
            )
        assert exc_info.value.interrupt is fresh_interrupt


# ─────────────────────────────────────────────────────────────────────
# run_resumed_hitl_turn
# ─────────────────────────────────────────────────────────────────────


class TestRunResumedHitlTurnValidation:
    async def test_missing_reply_key_raises_and_preserves_interrupt(self) -> None:
        sw = _make_swarm()
        member = sw.entry
        state = _make_state_with_parked_hitl(sw, member.name)
        ctx: RunContext[None] = RunContext.make(None)
        from troopai.adk.hooks.hooks import RunHooks
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG

        with pytest.raises(ValueError, match="no reply provided for parked member 'approver'.*pure-HITL"):
            await run_resumed_hitl_turn(
                member=member,
                swarm_resume=SwarmResume(),
                state=state,
                ctx_wrapper=ctx,
                config=DEFAULT_RUN_CONFIG,
                hooks=RunHooks(),
                user_prompt="go",
                is_first_turn=False,
                turn_messages=[],
                extra_tools=None,
                swarm_tool_names=None,
                max_turns=3,
            )
        assert member.name in state.pending_interrupts


class TestRunResumedHitlTurnHappyPath:
    async def test_seeds_reply_then_fires_run_agent_loop(self) -> None:
        sw = _make_swarm()
        member = sw.entry
        state = _make_state_with_parked_hitl(sw, member.name)
        ctx: RunContext[None] = RunContext.make(None)
        from troopai.adk.hooks.hooks import RunHooks
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG
        from troopai.adk.types.run.run_result import RunResult

        captured_has_reply: list[bool] = []

        async def _fake_run_agent_loop(**kwargs: Any) -> RunResult[Any]:
            # Inside the agent loop, the seeded reply must be present so
            # request_human_input_in_swarm can consume it.
            captured_has_reply.append(ctx.has_swarm_resume_reply())
            return RunResult(
                final_output="continued",
                user_prompt="",
                new_items=[],
                context=ctx,
                last_agent=member,
            )

        with patch("troopai.adk.run.swarm_resume.run_agent_loop", new=AsyncMock(side_effect=_fake_run_agent_loop)):
            result = await run_resumed_hitl_turn(
                member=member,
                swarm_resume=SwarmResume(replies={member.name: "approved"}),
                state=state,
                ctx_wrapper=ctx,
                config=DEFAULT_RUN_CONFIG,
                hooks=RunHooks(),
                user_prompt="go",
                is_first_turn=False,
                turn_messages=[],
                extra_tools=None,
                swarm_tool_names=None,
                max_turns=3,
            )

        assert captured_has_reply == [True]
        # Parked interrupt has been consumed.
        assert member.name not in state.pending_interrupts
        # Slot cleared after the run.
        assert ctx.has_swarm_resume_reply() is False
        assert result.final_output == "continued"

    async def test_explicit_none_reply_is_valid(self) -> None:
        """An abstain reply (None) must seed correctly, not be 'missing'."""
        sw = _make_swarm()
        member = sw.entry
        state = _make_state_with_parked_hitl(sw, member.name)
        ctx: RunContext[None] = RunContext.make(None)
        from troopai.adk.hooks.hooks import RunHooks
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG
        from troopai.adk.types.run.run_result import RunResult

        captured: list[Any] = []

        async def _fake_run_agent_loop(**kwargs: Any) -> RunResult[Any]:
            captured.append(ctx.consume_swarm_resume_reply())
            return RunResult(final_output="ok", user_prompt="", new_items=[], context=ctx, last_agent=member)

        with patch("troopai.adk.run.swarm_resume.run_agent_loop", new=AsyncMock(side_effect=_fake_run_agent_loop)):
            await run_resumed_hitl_turn(
                member=member,
                swarm_resume=SwarmResume(replies={member.name: None}),
                state=state,
                ctx_wrapper=ctx,
                config=DEFAULT_RUN_CONFIG,
                hooks=RunHooks(),
                user_prompt="go",
                is_first_turn=False,
                turn_messages=[],
                extra_tools=None,
                swarm_tool_names=None,
                max_turns=3,
            )

        assert captured == [None]

    async def test_slot_cleared_even_if_inner_loop_raises(self) -> None:
        """Defensive: a tool that doesn't consume the reply leaves no leak."""
        sw = _make_swarm()
        member = sw.entry
        state = _make_state_with_parked_hitl(sw, member.name)
        ctx: RunContext[None] = RunContext.make(None)
        from troopai.adk.hooks.hooks import RunHooks
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG

        async def _raise(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise InterruptException(Interrupt(node_id=member.name, question="again", kind="generic"))

        with (
            patch("troopai.adk.run.swarm_resume.run_agent_loop", new=AsyncMock(side_effect=_raise)),
            pytest.raises(InterruptException),
        ):
            await run_resumed_hitl_turn(
                member=member,
                swarm_resume=SwarmResume(replies={member.name: "yes"}),
                state=state,
                ctx_wrapper=ctx,
                config=DEFAULT_RUN_CONFIG,
                hooks=RunHooks(),
                user_prompt="go",
                is_first_turn=False,
                turn_messages=[],
                extra_tools=None,
                swarm_tool_names=None,
                max_turns=3,
            )

        assert ctx.has_swarm_resume_reply() is False


class TestRunResumedNestedTurnResumeCountsBump:
    async def test_first_resume_sets_count_to_one(self) -> None:
        sw = _make_swarm()
        member = sw.entry
        state = _make_state_with_parked_nested_defer(sw, member.name)
        ctx: RunContext[None] = RunContext.make(None)
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG

        reply = NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="c1"),))
        with patch(
            "troopai.adk.run.swarm_resume.AgentExecutable.resume_from_snapshot",
            new=AsyncMock(return_value=_make_node_result()),
        ):
            await run_resumed_nested_turn(
                member=member,
                swarm_resume=SwarmResume(replies={member.name: reply}),
                state=state,
                ctx_wrapper=ctx,
                config=DEFAULT_RUN_CONFIG,
            )
        assert state.resume_counts[member.name] == 1

    async def test_subsequent_resume_increments_count(self) -> None:
        sw = _make_swarm()
        member = sw.entry
        state = _make_state_with_parked_nested_defer(sw, member.name)
        state.resume_counts[member.name] = 2  # prior resume cycles
        ctx: RunContext[None] = RunContext.make(None)
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG

        reply = NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="c1"),))
        with patch(
            "troopai.adk.run.swarm_resume.AgentExecutable.resume_from_snapshot",
            new=AsyncMock(return_value=_make_node_result()),
        ):
            await run_resumed_nested_turn(
                member=member,
                swarm_resume=SwarmResume(replies={member.name: reply}),
                state=state,
                ctx_wrapper=ctx,
                config=DEFAULT_RUN_CONFIG,
            )
        assert state.resume_counts[member.name] == 3


class TestRunResumedHitlTurnResumeCountsBump:
    async def test_first_resume_sets_count_to_one(self) -> None:
        sw = _make_swarm()
        member = sw.entry
        state = _make_state_with_parked_hitl(sw, member.name)
        ctx: RunContext[None] = RunContext.make(None)
        from troopai.adk.hooks.hooks import RunHooks
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG
        from troopai.adk.types.run.run_result import RunResult

        async def _fake_run_agent_loop(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            return RunResult(final_output="ok", user_prompt="", new_items=[], context=ctx, last_agent=member)

        with patch(
            "troopai.adk.run.swarm_resume.run_agent_loop",
            new=AsyncMock(side_effect=_fake_run_agent_loop),
        ):
            await run_resumed_hitl_turn(
                member=member,
                swarm_resume=SwarmResume(replies={member.name: "yes"}),
                state=state,
                ctx_wrapper=ctx,
                config=DEFAULT_RUN_CONFIG,
                hooks=RunHooks(),
                user_prompt="go",
                is_first_turn=False,
                turn_messages=[],
                extra_tools=None,
                swarm_tool_names=None,
                max_turns=3,
            )
        assert state.resume_counts[member.name] == 1

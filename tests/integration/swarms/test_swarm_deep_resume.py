"""End-to-end integration tests for swarm deep resume.

Exercises the full ``Runner.arun_swarm`` → suspend → checkpoint →
``Runner.arun_swarm_from_checkpoint`` → continue lifecycle for both
resume paths (nested-agent-defer and pure HITL).

Tests mock ``run_agent_loop`` at the swarm-loop import site (or, for
the resume path, mock the deeper ``Runner.arun`` inside
``AgentExecutable.resume_from_snapshot``) so the swarm driver itself
is the unit under test — not a real LLM.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.exceptions import AgentToolDeferral
from troopai.adk.graphs.interrupt import (
    Interrupt,
    InterruptException,
    NestedAgentApproval,
    NestedAgentReply,
)
from troopai.adk.run.context import RunContext
from troopai.adk.run.runner import Runner
from troopai.adk.run.state import RunState
from troopai.adk.swarms.checkpointer import SwarmCheckpoint
from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer
from troopai.adk.swarms.interrupt import SwarmResume
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.tools.deferred_tool import DeferredToolCall, DeferredToolRequests
from troopai.adk.types.run.run_result import RunResult


def _make_swarm(name: str = "approver") -> Swarm:
    member = Agent(name=name, system_prompt="x")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(3),
    )


def _build_run_result(member: Agent[Any], ctx: RunContext[Any] | None = None) -> RunResult[Any]:
    return RunResult(
        final_output="done",
        user_prompt="",
        new_items=[],
        context=ctx if ctx is not None else RunContext.make(None),
        last_agent=member,
    )


class TestSwarmDeepResumeNestedDefer:
    """End-to-end nested-agent-defer suspend → approve → continue."""

    async def test_happy_path_defer_then_approve_completes(self) -> None:
        """A swarm whose member tool defers parks, then resumes cleanly."""
        sw = _make_swarm()
        member = sw.entry
        deferred_call = DeferredToolCall(
            tool_call_id="c1",
            tool_name="risky",
            tool_arguments={},
            raw_arguments="{}",
        )
        parked_state = RunState(
            current_agent_name=member.name,
            turn_count=1,
            deferred_tool_requests=DeferredToolRequests(approvals=[deferred_call]),
        )
        deferral = AgentToolDeferral(
            agent_name=member.name,
            deferred_requests=DeferredToolRequests(approvals=[deferred_call]),
            state=parked_state,
        )

        # First arun_swarm: member defers — swarm parks.
        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(side_effect=deferral),
        ):
            first = await Runner.arun_swarm(sw, "go")
        assert first.stop_reason.kind == "interrupted"
        assert first.state is not None
        assert member.name in first.state.nested_agent_snapshots

        # Save checkpoint (relies on auto-save-on-interrupt OR manual).
        cp = InMemorySwarmCheckpointer(thread_id="thr-1")
        await cp.save(
            SwarmCheckpoint(
                thread_id="thr-1",
                state=dict(first.state.to_dict()),
                turn=first.state.total_turns,
            )
        )

        # Resume: stub the inner Runner.arun that AgentExecutable.
        # resume_from_snapshot calls. The swarm loop's next-turn
        # iteration after the splice consumes the parked entries then
        # exits via MaxTurnsTermination — also stub run_agent_loop so
        # the post-resume iteration doesn't reach LiteLLM.
        resumed_result = _build_run_result(member)

        with (
            patch(
                "troopai.adk.run.runner.Runner.arun",
                new=AsyncMock(return_value=resumed_result),
            ),
            patch(
                "troopai.adk.run.swarm_loop.run_agent_loop",
                new=AsyncMock(return_value=resumed_result),
            ),
        ):
            second = await Runner.arun_swarm_from_checkpoint(
                sw,
                checkpointer=cp,
                thread_id="thr-1",
                resume=SwarmResume(
                    replies={member.name: NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="c1"),))},
                ),
            )

        # Successful resume: not interrupted, parked state cleared.
        assert second.stop_reason.kind != "interrupted"
        assert second.state is not None
        assert member.name not in second.state.nested_agent_snapshots
        assert member.name not in second.state.pending_interrupts

    async def test_missing_reply_propagates_value_error(self) -> None:
        """Resume without a matching reply raises ValueError, snapshot preserved."""
        sw = _make_swarm()
        member = sw.entry
        deferred_call = DeferredToolCall(
            tool_call_id="c1",
            tool_name="risky",
            tool_arguments={},
            raw_arguments="{}",
        )
        deferral = AgentToolDeferral(
            agent_name=member.name,
            deferred_requests=DeferredToolRequests(approvals=[deferred_call]),
            state=RunState(
                current_agent_name=member.name,
                turn_count=1,
                deferred_tool_requests=DeferredToolRequests(approvals=[deferred_call]),
            ),
        )

        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(side_effect=deferral),
        ):
            first = await Runner.arun_swarm(sw, "go")

        cp = InMemorySwarmCheckpointer(thread_id="thr-2")
        assert first.state is not None
        await cp.save(
            SwarmCheckpoint(
                thread_id="thr-2",
                state=dict(first.state.to_dict()),
                turn=first.state.total_turns,
            )
        )

        with pytest.raises(ValueError, match="no reply provided for parked member 'approver'"):
            await Runner.arun_swarm_from_checkpoint(
                sw,
                checkpointer=cp,
                thread_id="thr-2",
                resume=SwarmResume(),
            )

    async def test_wrong_type_reply_propagates_value_error(self) -> None:
        """Reply that is not a NestedAgentReply raises ValueError."""
        sw = _make_swarm()
        member = sw.entry
        deferred_call = DeferredToolCall(
            tool_call_id="c1",
            tool_name="risky",
            tool_arguments={},
            raw_arguments="{}",
        )
        deferral = AgentToolDeferral(
            agent_name=member.name,
            deferred_requests=DeferredToolRequests(approvals=[deferred_call]),
            state=RunState(
                current_agent_name=member.name,
                turn_count=1,
                deferred_tool_requests=DeferredToolRequests(approvals=[deferred_call]),
            ),
        )

        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(side_effect=deferral),
        ):
            first = await Runner.arun_swarm(sw, "go")

        cp = InMemorySwarmCheckpointer(thread_id="thr-3")
        assert first.state is not None
        await cp.save(
            SwarmCheckpoint(
                thread_id="thr-3",
                state=dict(first.state.to_dict()),
                turn=first.state.total_turns,
            )
        )

        with pytest.raises(ValueError, match="must be NestedAgentReply.*got str"):
            await Runner.arun_swarm_from_checkpoint(
                sw,
                checkpointer=cp,
                thread_id="thr-3",
                resume=SwarmResume(replies={member.name: "approved"}),
            )


class TestSwarmDeepResumeHitlPure:
    """End-to-end pure-HITL suspend → approve → continue."""

    async def test_happy_path_hitl_interrupt_then_reply_completes(self) -> None:
        """A swarm member raising InterruptException resumes via context-seeded reply."""
        sw = _make_swarm()
        member = sw.entry
        interrupt = Interrupt(
            node_id=member.name,
            question="Approve?",
            kind="tool_approval",
        )

        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(side_effect=InterruptException(interrupt)),
        ):
            first = await Runner.arun_swarm(sw, "go")
        assert first.stop_reason.kind == "interrupted"
        assert first.state is not None
        assert member.name in first.state.pending_interrupts
        # No snapshot for pure HITL.
        assert member.name not in first.state.nested_agent_snapshots

        cp = InMemorySwarmCheckpointer(thread_id="thr-hitl-1")
        await cp.save(
            SwarmCheckpoint(
                thread_id="thr-hitl-1",
                state=dict(first.state.to_dict()),
                turn=first.state.total_turns,
            )
        )

        # Resume: stub run_agent_loop so on the resumed call we verify
        # the seeded reply made it onto the run context and produce a
        # clean RunResult.
        captured_replies: list[Any] = []

        async def _fake_run_agent_loop(**kwargs: Any) -> RunResult[Any]:
            ctx_wrapper = kwargs["ctx_wrapper"]
            if ctx_wrapper.has_swarm_resume_reply():
                captured_replies.append(ctx_wrapper.consume_swarm_resume_reply())
            return _build_run_result(member, ctx_wrapper)

        with (
            patch(
                "troopai.adk.run.swarm_loop.run_agent_loop",
                new=AsyncMock(side_effect=_fake_run_agent_loop),
            ),
            patch(
                "troopai.adk.run.swarm_resume.run_agent_loop",
                new=AsyncMock(side_effect=_fake_run_agent_loop),
            ),
        ):
            second = await Runner.arun_swarm_from_checkpoint(
                sw,
                checkpointer=cp,
                thread_id="thr-hitl-1",
                resume=SwarmResume(replies={member.name: "approved"}),
            )

        assert captured_replies == ["approved"]
        assert second.stop_reason.kind != "interrupted"
        assert second.state is not None
        assert member.name not in second.state.pending_interrupts

    async def test_explicit_none_reply_is_valid_abstain(self) -> None:
        """A seeded None reply must NOT be treated as 'no reply'."""
        sw = _make_swarm()
        member = sw.entry
        interrupt = Interrupt(node_id=member.name, question="Decide?", kind="generic")

        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(side_effect=InterruptException(interrupt)),
        ):
            first = await Runner.arun_swarm(sw, "go")

        cp = InMemorySwarmCheckpointer(thread_id="thr-hitl-2")
        assert first.state is not None
        await cp.save(
            SwarmCheckpoint(
                thread_id="thr-hitl-2",
                state=dict(first.state.to_dict()),
                turn=first.state.total_turns,
            )
        )

        captured: list[Any] = []

        async def _fake_run_agent_loop(**kwargs: Any) -> RunResult[Any]:
            ctx_wrapper = kwargs["ctx_wrapper"]
            if ctx_wrapper.has_swarm_resume_reply():
                captured.append(ctx_wrapper.consume_swarm_resume_reply())
            return _build_run_result(member, ctx_wrapper)

        with (
            patch(
                "troopai.adk.run.swarm_loop.run_agent_loop",
                new=AsyncMock(side_effect=_fake_run_agent_loop),
            ),
            patch(
                "troopai.adk.run.swarm_resume.run_agent_loop",
                new=AsyncMock(side_effect=_fake_run_agent_loop),
            ),
        ):
            await Runner.arun_swarm_from_checkpoint(
                sw,
                checkpointer=cp,
                thread_id="thr-hitl-2",
                resume=SwarmResume(replies={member.name: None}),
            )

        assert captured == [None]

    async def test_re_park_when_tool_raises_again(self) -> None:
        """A second-stage HITL fires another InterruptException and re-parks."""
        sw = _make_swarm()
        member = sw.entry
        interrupt_1 = Interrupt(node_id=member.name, question="Step 1?", kind="generic")
        interrupt_2 = Interrupt(node_id=member.name, question="Step 2?", kind="generic")

        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(side_effect=InterruptException(interrupt_1)),
        ):
            first = await Runner.arun_swarm(sw, "go")

        cp = InMemorySwarmCheckpointer(thread_id="thr-hitl-3")
        assert first.state is not None
        await cp.save(
            SwarmCheckpoint(
                thread_id="thr-hitl-3",
                state=dict(first.state.to_dict()),
                turn=first.state.total_turns,
            )
        )

        with (
            patch(
                "troopai.adk.run.swarm_loop.run_agent_loop",
                new=AsyncMock(side_effect=InterruptException(interrupt_2)),
            ),
            patch(
                "troopai.adk.run.swarm_resume.run_agent_loop",
                new=AsyncMock(side_effect=InterruptException(interrupt_2)),
            ),
        ):
            second = await Runner.arun_swarm_from_checkpoint(
                sw,
                checkpointer=cp,
                thread_id="thr-hitl-3",
                resume=SwarmResume(replies={member.name: "yes"}),
            )

        # The re-fired tool raised again — swarm re-parks under the
        # same member key with the fresh interrupt.
        assert second.stop_reason.kind == "interrupted"
        assert second.state is not None
        assert second.state.pending_interrupts[member.name].question == "Step 2?"


# Note: end-to-end auto-save-on-interrupt is covered by the unit test
# tests/unit/swarms/test_checkpointer.py::TestAutoSaveOnInterrupt added
# in T5 (which exercises the auto-save hook's on_swarm_turn_interrupt
# directly). A full integration test wiring HookRegistry through
# Swarm.hooks is blocked by HookRegistry not being a SwarmHooks
# subclass — the Swarm.hooks type accepts SwarmHooks[TContext] | None
# only. Tracking as a follow-up: either make HookRegistry implement
# the SwarmHooks protocol or add a Swarm.hook_registry slot.

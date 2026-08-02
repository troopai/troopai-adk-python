"""Swarm HITL suspend cycle — InterruptException + AgentToolDeferral lift.

Mirror of ``tests/integration/graphs/test_graph_hitl.py`` for the swarms
substrate. Verifies that when a member turn raises a cooperative-pause
exception, the swarm loop catches, parks state, and returns a clean
``SwarmRunResult`` with ``stop_reason.kind == "interrupted"`` rather
than propagating the exception to the caller.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from troopai.adk.agents.agent import Agent
from troopai.adk.exceptions import AgentToolDeferral
from troopai.adk.graphs.interrupt import (
    Interrupt,
    InterruptException,
    NestedAgentInterrupt,
)
from troopai.adk.run.runner import Runner
from troopai.adk.run.state import RunState
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.tools.deferred_tool import DeferredToolCall, DeferredToolRequests


def _swarm_with_member(name: str = "ask") -> Swarm:
    member = Agent(name=name, system_prompt="x")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(3),
    )


class TestSwarmInterruptExceptionSuspend:
    async def test_hitl_interrupt_parks_state_and_returns_interrupted(self) -> None:
        """A member tool calling request_human_input pauses the swarm cleanly."""
        sw = _swarm_with_member()
        interrupt = Interrupt(
            node_id="ask",
            question="approve?",
            kind="tool_approval",
        )

        async def _raise_hitl(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise InterruptException(interrupt)

        with patch("troopai.adk.run.swarm_loop.run_agent_loop", new=AsyncMock(side_effect=_raise_hitl)):
            result = await Runner.arun_swarm(sw, "go")

        assert result.stop_reason.kind == "interrupted"
        assert result.last_agent is not None
        assert result.last_agent.name == "ask"
        assert len(result.interrupts) == 1
        assert result.interrupts[0].question == "approve?"
        assert result.state is not None
        assert "ask" in result.state.pending_interrupts
        assert result.state.status == "interrupted"


class TestSwarmNestedAgentDeferralLift:
    async def test_tool_deferral_lifts_to_nested_agent_interrupt(self) -> None:
        """A member whose tool defers parks a NestedAgentInterrupt + RunState."""
        sw = _swarm_with_member()
        deferred_call = DeferredToolCall(
            tool_call_id="c1",
            tool_name="approve_me",
            tool_arguments={},
            raw_arguments="{}",
        )
        deferred_requests = DeferredToolRequests(approvals=[deferred_call])
        parked_run_state = RunState(
            current_agent_name="ask",
            turn_count=1,
            deferred_tool_requests=deferred_requests,
        )
        deferral = AgentToolDeferral(
            agent_name="ask",
            deferred_requests=deferred_requests,
            state=parked_run_state,
        )

        async def _raise_deferral(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise deferral

        with patch("troopai.adk.run.swarm_loop.run_agent_loop", new=AsyncMock(side_effect=_raise_deferral)):
            result = await Runner.arun_swarm(sw, "go")

        assert result.stop_reason.kind == "interrupted"
        assert len(result.interrupts) == 1
        iv = result.interrupts[0]
        assert isinstance(iv, NestedAgentInterrupt)
        assert iv.agent_name == "ask"
        assert iv.tool_call_ids == ("c1",)
        # Parked RunState is on the state's nested_agent_snapshots.
        assert result.state is not None
        assert "ask" in result.state.nested_agent_snapshots
        assert result.state.nested_agent_snapshots["ask"].current_agent_name == "ask"
        assert result.state.status == "interrupted"


class TestSwarmResumeFromCheckpoint:
    async def test_resume_clears_pending_interrupts_and_continues(self) -> None:
        """arun_swarm_from_checkpoint loads parked state and re-runs cleanly."""
        from troopai.adk.run.context import RunContext
        from troopai.adk.swarms.checkpointer import SwarmCheckpoint
        from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer
        from troopai.adk.swarms.interrupt import SwarmResume
        from troopai.adk.swarms.state import SwarmState

        sw = _swarm_with_member()

        # Hand-build a parked state and persist it under a known thread_id.
        parked = SwarmState(
            swarm=sw,
            current_agent=sw.members[0],
            current_agent_name="ask",
        )
        parked.total_turns = 1
        parked.pending_interrupts["ask"] = Interrupt(
            node_id="ask",
            question="approve?",
            kind="tool_approval",
        )
        parked.status = "interrupted"

        cp = InMemorySwarmCheckpointer(thread_id="thr-resume")
        await cp.save(SwarmCheckpoint(thread_id="thr-resume", state=dict(parked.to_dict()), turn=1))

        # Stub run_agent_loop so the resumed swarm advances one clean turn
        # and then handoffs/terminates via MaxTurnsTermination(3).
        from troopai.adk.types.run.run_result import RunResult

        stub_result = RunResult[Any](
            final_output=None,
            user_prompt="",
            new_items=[],
            context=RunContext(context=None),
            last_agent=sw.members[0],
            swarm_yield=None,
        )

        async def _stub_turn(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            return stub_result

        # The resume-with-payload path routes through ``run_resumed_hitl_turn``,
        # which calls ``run_agent_loop`` through the ``swarm_resume`` module's
        # own import binding — stub both call sites so the resumed turn and
        # the subsequent policy-driven turn complete without a real LLM call.
        with (
            patch("troopai.adk.run.swarm_loop.run_agent_loop", new=AsyncMock(side_effect=_stub_turn)),
            patch("troopai.adk.run.swarm_resume.run_agent_loop", new=AsyncMock(side_effect=_stub_turn)),
        ):
            result = await Runner.arun_swarm_from_checkpoint(
                sw,
                checkpointer=cp,
                thread_id="thr-resume",
                resume=SwarmResume(replies={"ask": "approved"}),
            )

        # After resume, the loaded state's pending_interrupts are cleared.
        assert result.state is not None
        assert "ask" not in result.state.pending_interrupts
        # total_turns carried over from the checkpoint (was 1) and at least
        # one further turn fired during resume.
        assert result.state.total_turns >= 2

    async def test_resume_unknown_thread_id_raises(self) -> None:
        from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer
        from troopai.adk.swarms.interrupt import SwarmResume

        sw = _swarm_with_member()
        cp = InMemorySwarmCheckpointer()
        import pytest

        with pytest.raises(ValueError, match="no checkpoint"):
            await Runner.arun_swarm_from_checkpoint(
                sw,
                checkpointer=cp,
                thread_id="does-not-exist",
                resume=SwarmResume(),
            )


class TestSwarmTurnInterruptHookFires:
    async def test_on_swarm_turn_interrupt_called_with_member_and_interrupt(self) -> None:
        """Custom SwarmHooks subclass receives the interrupt at parking time."""
        from typing import override

        from troopai.adk.run.context import RunContext
        from troopai.adk.swarms.hooks import SwarmHooks
        from troopai.adk.swarms.state import SwarmState

        class _Recorder(SwarmHooks[Any]):
            calls: list[tuple[str, Interrupt]] = []

            @override
            async def on_swarm_turn_interrupt(
                self,
                context: RunContext[Any],
                state: SwarmState[Any],
                member_name: str,
                interrupt: Interrupt,
            ) -> None:
                del context, state
                _Recorder.calls.append((member_name, interrupt))

        _Recorder.calls = []
        member = Agent(name="ask", system_prompt="x")
        sw = Swarm(
            members=(member,),
            entry=member,
            policy=RoundRobinPolicy(),
            termination=MaxTurnsTermination(3),
            hooks=_Recorder(),
        )
        interrupt = Interrupt(node_id="ask", question="approve?", kind="tool_approval")

        async def _raise_hitl(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise InterruptException(interrupt)

        with patch("troopai.adk.run.swarm_loop.run_agent_loop", new=AsyncMock(side_effect=_raise_hitl)):
            await Runner.arun_swarm(sw, "go")

        assert len(_Recorder.calls) == 1
        member_name, recorded_iv = _Recorder.calls[0]
        assert member_name == "ask"
        assert recorded_iv.question == "approve?"

"""End-to-end streaming tests for Runner.arun_swarm_streamed.

Three scenarios:
- Happy multi-turn path: events flow through stream_events() with the
  full swarm-level sequence.
- Suspend path: a SwarmTurnInterruptEvent arrives before the closing
  SwarmDoneEvent.
- Resume-through-stream: a parked checkpoint resumed via the streamed
  entry produces the documented event sequence; the same swarm_id
  flows through both invocations' result.state.swarm_id.

Inner agent execution is mocked at Runner._run_streamed (for the
happy path) or at _stream_member_turn (for the suspend path) so the
tests don't reach LiteLLM.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs.interrupt import Interrupt, InterruptException
from troopai.adk.run.context import RunContext
from troopai.adk.run.runner import Runner
from troopai.adk.swarms.checkpointer import SwarmCheckpoint
from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer
from troopai.adk.swarms.interrupt import SwarmResume
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.state import SwarmState
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.types.run.run_result import RunResult


def _make_swarm(*, max_turns: int = 1) -> Swarm[Any]:
    member = Agent(name="approver", system_prompt="x")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(max_turns),
    )


class TestSwarmStreamedHappyPath:
    async def test_single_turn_emits_full_swarm_event_sequence(self) -> None:
        sw = _make_swarm(max_turns=1)
        ctx: RunContext[None] = RunContext.make(None)

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
            result = await Runner.arun_swarm_streamed(sw, "go")
            events: list[Any] = []
            async for ev in result.stream_events():
                events.append(ev)

        # Pin the swarm-level event sequence.
        swarm_event_types = [type(ev).__name__ for ev in events if type(ev).__name__.startswith("Swarm")]
        assert swarm_event_types == [
            "SwarmStartEvent",
            "SwarmTurnStartEvent",
            "SwarmTurnEndEvent",
            "SwarmDoneEvent",
        ]
        # Terminal fields populated.
        assert result.stop_reason is not None
        assert result.state is not None
        assert result.state.total_turns == 1


class TestSwarmStreamedSuspendPath:
    async def test_interrupt_emits_turn_interrupt_event_replacing_turn_end(
        self,
    ) -> None:
        sw = _make_swarm(max_turns=2)
        interrupt = Interrupt(
            node_id="approver",
            question="Approve?",
            kind="generic",
        )

        async def _raise_interrupt(**kwargs: Any) -> Any:
            del kwargs
            raise InterruptException(interrupt)

        with patch(
            "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
            new=AsyncMock(side_effect=_raise_interrupt),
        ):
            result = await Runner.arun_swarm_streamed(sw, "go")
            events: list[Any] = []
            async for ev in result.stream_events():
                events.append(ev)

        swarm_event_types = [type(ev).__name__ for ev in events if type(ev).__name__.startswith("Swarm")]
        assert "SwarmTurnInterruptEvent" in swarm_event_types
        # TurnInterrupt replaces TurnEnd — no TurnEnd this turn.
        assert "SwarmTurnEndEvent" not in swarm_event_types
        # Result fields are populated for the caller.
        assert result.interrupts == (interrupt,)
        assert result.stop_reason is not None
        assert result.stop_reason.kind == "interrupted"


class TestSwarmStreamedResumeCycle:
    async def test_resume_through_stream_continues_run_and_shares_swarm_id(
        self,
    ) -> None:
        """Suspend via streamed, persist, resume via streamed — verify the
        same troopai.swarm.id flows through both invocations'
        result.state.swarm_id."""
        sw = _make_swarm(max_turns=2)
        ctx: RunContext[None] = RunContext.make(None)
        interrupt = Interrupt(
            node_id="approver",
            question="Approve?",
            kind="generic",
        )

        async def _raise_interrupt(**kwargs: Any) -> Any:
            del kwargs
            raise InterruptException(interrupt)

        with patch(
            "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
            new=AsyncMock(side_effect=_raise_interrupt),
        ):
            first_result = await Runner.arun_swarm_streamed(sw, "go")
            async for _ in first_result.stream_events():
                pass

        assert first_result.state is not None
        original_swarm_id = first_result.state.swarm_id
        assert original_swarm_id is not None

        # Save the parked state.
        cp = InMemorySwarmCheckpointer(thread_id="thr-stream")
        await cp.save(
            SwarmCheckpoint(
                thread_id="thr-stream",
                state=dict(first_result.state.to_dict()),
                turn=first_result.state.total_turns,
            )
        )

        # Resume via the streamed entry.
        loaded = await cp.load("thr-stream", sw)
        assert loaded is not None
        loaded_state = SwarmState.from_dict(
            loaded.state,  # type: ignore[arg-type]  # checkpoint.state is dict[str, Any]; from_dict expects SwarmStateDict
            sw,
        )

        async def _fake_resumed_turn(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            return RunResult(
                final_output="resumed-ok",
                user_prompt="",
                new_items=[],
                context=ctx,
                last_agent=sw.entry,
            )

        # The splice takes the HITL-pure branch (InterruptException without
        # an AgentToolDeferral), which calls run_resumed_hitl_turn. That
        # helper imports run_agent_loop at its own module level, so both
        # bindings must be patched for a hermetic test.
        with (
            patch(
                "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
                new=AsyncMock(side_effect=_fake_resumed_turn),
            ),
            patch(
                "troopai.adk.run.swarm_resume.run_agent_loop",
                new=AsyncMock(side_effect=_fake_resumed_turn),
            ),
        ):
            second = await Runner.arun_swarm_streamed(
                sw,
                "",
                initial_state=loaded_state,
                resume=SwarmResume(replies={"approver": "yes"}),
            )
            async for _ in second.stream_events():
                pass

        # Same swarm_id flows through both invocations.
        assert second.state is not None
        assert second.state.swarm_id == original_swarm_id

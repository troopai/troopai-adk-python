"""Regression tests for swarm-driver sweep fixes (sync + streamed parity).

Each test targets a specific drift between ``run_swarm_loop`` (sync) and
``run_swarm_loop_streamed`` (streamed):

- The sync driver ran member turns with NO input/output guardrails while the
  streamed driver did (via ``Runner._run_streamed``).
- Both interrupt/deferral handlers returned before folding the partial turn's
  usage into cumulative + per-member totals.
- ``SwarmState.status`` never advanced past ``"running"`` to
  ``"completed"`` / ``"failed"``.
- A first-turn interrupt resumed with an empty SCOPED body instead of the
  opening prompt.
- The streamed driver left steps 8-10 outside its turn try, leaking the span
  on a step-8/9/10 failure.
"""

from __future__ import annotations

import asyncio
from typing import Any, override
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentGuardrails,
    AgentInputGuardrail,
)
from troopai.adk.exceptions import AgentInputGuardrailTripwireTriggered
from troopai.adk.graphs.interrupt import Interrupt, InterruptException
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.llms.llm_usage import LLMUsage
from troopai.adk.run.config import DEFAULT_RUN_CONFIG
from troopai.adk.run.context import RunContext
from troopai.adk.run.swarm_loop import run_swarm_loop
from troopai.adk.run.swarm_loop_streamed import run_swarm_loop_streamed
from troopai.adk.swarms.interrupt import SwarmResume
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.result import SwarmRunResultStreaming
from troopai.adk.swarms.state import SwarmState
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.types.run.run_result import RunResult


def _make_swarm(member: Agent[Any] | None = None, *, max_turns: int = 1) -> Swarm[Any]:
    m = member if member is not None else Agent(name="m", system_prompt="x")
    return Swarm(
        members=(m,),
        entry=m,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(max_turns),
    )


def _stub_result(member: Agent[Any], ctx: RunContext[Any]) -> RunResult[Any]:
    return RunResult(
        final_output=None,
        user_prompt="",
        new_items=[],
        context=ctx,
        last_agent=member,
        swarm_yield=None,
    )


class TestSyncGuardrails:
    async def test_member_input_guardrail_tripwire_aborts_sync_swarm(self) -> None:
        async def _trip(_data: Any) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=True)

        member = Agent(
            name="m",
            system_prompt="x",
            guardrails=AgentGuardrails(input=[AgentInputGuardrail(guardrail_function=_trip, name="block")]),
        )
        sw = _make_swarm(member)
        ctx: RunContext[None] = RunContext.make(None)

        # run_agent_loop is stubbed so that pre-fix (no guardrails run) the swarm
        # completes cleanly and the tripwire is never raised.
        with (
            patch("troopai.adk.run.swarm_loop.run_agent_loop", new=AsyncMock(return_value=_stub_result(member, ctx))),
            pytest.raises(AgentInputGuardrailTripwireTriggered),
        ):
            await run_swarm_loop(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
            )


class TestInterruptUsageFold:
    async def test_sync_interrupt_folds_partial_usage(self) -> None:
        member = Agent(name="m", system_prompt="x")
        sw = _make_swarm(member)
        ctx: RunContext[None] = RunContext.make(None)

        async def _spend_then_interrupt(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            ctx.usage = ctx.usage + LLMUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
            raise InterruptException(Interrupt(node_id="m", question="?", kind="tool_approval"))

        with patch("troopai.adk.run.swarm_loop.run_agent_loop", new=AsyncMock(side_effect=_spend_then_interrupt)):
            result = await run_swarm_loop(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
            )

        assert result.stop_reason.kind == "interrupted"
        assert result.per_member_usage["m"].total_tokens == 15
        assert result.state is not None
        assert result.state.cumulative_usage.total_tokens == 15

    async def test_streamed_interrupt_folds_partial_usage(self) -> None:
        member = Agent(name="m", system_prompt="x")
        sw = _make_swarm(member)
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(user_prompt="go")

        async def _spend_then_interrupt(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            ctx.usage = ctx.usage + LLMUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
            raise InterruptException(Interrupt(node_id="m", question="?", kind="tool_approval"))

        with patch(
            "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
            new=AsyncMock(side_effect=_spend_then_interrupt),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
            )

        assert result.per_member_usage["m"].total_tokens == 15
        assert result.state is not None
        assert result.state.cumulative_usage.total_tokens == 15


class TestSwarmStatus:
    async def test_sync_completed_status(self) -> None:
        member = Agent(name="m", system_prompt="x")
        sw = _make_swarm(member)
        ctx: RunContext[None] = RunContext.make(None)

        with patch("troopai.adk.run.swarm_loop.run_agent_loop", new=AsyncMock(return_value=_stub_result(member, ctx))):
            result = await run_swarm_loop(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
            )

        assert result.state is not None
        assert result.state.status == "completed"

    async def test_streamed_completed_status(self) -> None:
        member = Agent(name="m", system_prompt="x")
        sw = _make_swarm(member)
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(user_prompt="go")

        with patch(
            "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
            new=AsyncMock(return_value=_stub_result(member, ctx)),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
            )

        assert result.state is not None
        assert result.state.status == "completed"


class TestFirstTurnResumeMessages:
    async def test_first_turn_interrupt_resume_rebuilds_initial_messages(self) -> None:
        member = Agent(name="m", system_prompt="x")
        sw = _make_swarm(member)
        ctx: RunContext[None] = RunContext.make(None)

        state: SwarmState[Any] = SwarmState(swarm=sw, current_agent=member, current_agent_name="m")
        state.total_turns = 0  # turn 1 never completed
        state.pending_interrupts["m"] = Interrupt(node_id="m", question="?", kind="tool_approval")

        build_mock = AsyncMock(return_value=[{"role": "user", "content": "go"}])
        prepare_mock = AsyncMock(return_value=[])
        hitl_mock = AsyncMock(return_value=_stub_result(member, ctx))

        with (
            patch("troopai.adk.run.swarm_loop.build_initial_messages", new=build_mock),
            patch("troopai.adk.run.swarm_loop.prepare_turn_input", new=prepare_mock),
            patch("troopai.adk.run.swarm_loop.run_resumed_hitl_turn", new=hitl_mock),
        ):
            await run_swarm_loop(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                initial_state=state,
                swarm_resume=SwarmResume(replies={"m": "approved"}),
            )

        # The resumed first turn must rebuild from the opening prompt, not from
        # the (empty) SCOPED body.
        assert build_mock.await_count == 1
        assert prepare_mock.await_count == 0


class TestStreamedSpanNoLeak:
    async def test_out_of_roster_error_stamps_turn_span(self) -> None:
        # A policy that returns an agent outside the roster raises the
        # out-of-roster ValueError in step 9 — outside the old try/finally.
        foreign = Agent(name="ghost", system_prompt="x")
        member = Agent(name="m", system_prompt="x")

        class _ForeignPolicy(RoundRobinPolicy):
            @override
            async def select_next(self, state: Any, context: Any) -> Any:
                del state, context
                return foreign

        sw: Swarm[Any] = Swarm(
            members=(member,),
            entry=member,
            policy=_ForeignPolicy(),
            termination=MaxTurnsTermination(5),
        )
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(user_prompt="go")

        span = MagicMock()
        with (
            patch(
                "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
                new=AsyncMock(return_value=_stub_result(member, ctx)),
            ),
            patch("troopai.adk.run.swarm_loop_streamed.swarm_turn_span", return_value=span),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
            )

        # Span must be closed even though step 9 raised, and the run marked failed.
        assert span.finish.called
        assert result.state is not None
        assert result.state.status == "failed"

    async def test_streamed_result_surfaces_out_of_roster_exception(self) -> None:
        foreign = Agent(name="ghost", system_prompt="x")
        member = Agent(name="m", system_prompt="x")

        class _ForeignPolicy(RoundRobinPolicy):
            @override
            async def select_next(self, state: Any, context: Any) -> Any:
                del state, context
                return foreign

        sw: Swarm[Any] = Swarm(
            members=(member,),
            entry=member,
            policy=_ForeignPolicy(),
            termination=MaxTurnsTermination(5),
        )
        ctx: RunContext[None] = RunContext.make(None)
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(user_prompt="go")

        with patch(
            "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
            new=AsyncMock(return_value=_stub_result(member, ctx)),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=result,
            )
            result.set_run_task(asyncio.get_running_loop().create_task(asyncio.sleep(0)))
            with pytest.raises(ValueError, match="not in Swarm.members"):
                async for _ev in result.stream_events():
                    pass

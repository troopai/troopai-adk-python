"""End-to-end swarm run tests with a mocked LLM.

Covers:
- ``RoundRobinPolicy`` advances deterministically until ``swarm_done``
  is called.
- ``LLMHandoffPolicy`` dispatches via a ``transfer_to_<name>`` tool call
  emitted by the LLM.
- ``SwarmState.total_turns`` / ``handoff_count`` / ``cumulative_usage``
  are populated from actual turn execution.

Mocking strategy: patch ``troopai.adk.run.loop.call_llm`` to return
scripted ``LLMResponse`` sequences, plus stub the guardrail pipeline
(which is not under test here).
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.runner import Runner
from troopai.adk.swarms.policy import LLMHandoffPolicy, RoundRobinPolicy
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import (
    ExplicitDoneTermination,
    MaxTurnsTermination,
)
from troopai.adk.swarms.yield_signal import SWARM_DONE_TOOL_NAME
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
    LLMResponseText,
)


def _tool_call_response(name: str, args: dict[str, Any], call_id: str) -> LLMResponse:
    return LLMResponse(
        response_id=f"resp-{call_id}",
        model="fake",
        response=[
            LLMResponseFunctionToolCall(
                call_id=call_id,
                name=name,
                arguments=json.dumps(args),
            )
        ],
    )


def _text_response(text: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        response_id=f"resp-{call_id}",
        model="fake",
        response=[LLMResponseText(text=text)],
    )


def _agent(name: str) -> Agent:
    return Agent(name=name, system_prompt=f"You are {name}.")


async def _run_with_mocked_llm(
    swarm: Swarm,
    prompt: str,
    responses: list[LLMResponse],
):
    """Run a swarm with a scripted sequence of LLM responses."""
    queue = iter(responses)

    async def fake_call_llm(*_a: Any, **_kw: Any) -> LLMResponse:
        try:
            return next(queue)
        except StopIteration:
            # Default tail: a text response so the loop resolves cleanly
            return _text_response("done", call_id="tail")

    with (
        patch(
            "troopai.adk.run.loop.call_llm",
            new=AsyncMock(side_effect=fake_call_llm),
        ),
        patch(
            "troopai.adk.run.runner.run_blocking_input_guardrails",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "troopai.adk.run.runner.run_parallel_input_guardrails",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "troopai.adk.run.runner.run_output_guardrails",
            new=AsyncMock(return_value=[]),
        ),
    ):
        return await Runner.arun_swarm(swarm, prompt)


class TestRoundRobinSwarm:
    @pytest.mark.asyncio
    async def test_single_agent_calls_swarm_done(self) -> None:
        a = _agent("only")
        swarm = Swarm(
            members=(a,),
            entry=a,
            policy=RoundRobinPolicy(),
            termination=ExplicitDoneTermination() | MaxTurnsTermination(5),
        )

        responses = [
            _tool_call_response(
                SWARM_DONE_TOOL_NAME,
                {"reason": "task finished"},
                call_id="d1",
            ),
        ]
        result = await _run_with_mocked_llm(swarm, "go", responses)

        assert result.stop_reason.kind == "explicit_done"
        assert result.stop_reason.detail == "task finished"
        assert result.total_turns >= 1
        assert result.last_agent is a

    @pytest.mark.asyncio
    async def test_max_turns_termination_without_done(self) -> None:
        a, b = _agent("a"), _agent("b")
        swarm = Swarm(
            members=(a, b),
            entry=a,
            policy=RoundRobinPolicy(),
            termination=MaxTurnsTermination(2),
        )

        # Each turn just emits a plain text message — no swarm_done.
        # MaxTurnsTermination fires when total_turns reaches 2.
        responses = [
            _text_response("a speaks", call_id="t1"),
            _text_response("b speaks", call_id="t2"),
        ]
        result = await _run_with_mocked_llm(swarm, "go", responses)

        assert result.stop_reason.kind == "max_turns"
        assert result.total_turns == 2


class TestLLMHandoffSwarm:
    @pytest.mark.asyncio
    async def test_llm_transfer_then_done(self) -> None:
        author, reviewer = _agent("author"), _agent("reviewer")
        swarm = Swarm(
            members=(author, reviewer),
            entry=author,
            policy=LLMHandoffPolicy(),
            termination=ExplicitDoneTermination() | MaxTurnsTermination(8),
        )

        responses = [
            # Author hands off to reviewer.
            _tool_call_response(
                "transfer_to_reviewer",
                {"message": "please review"},
                call_id="h1",
            ),
            # Reviewer marks the swarm done.
            _tool_call_response(
                SWARM_DONE_TOOL_NAME,
                {"reason": "approved"},
                call_id="d1",
            ),
        ]
        result = await _run_with_mocked_llm(swarm, "write it", responses)

        assert result.stop_reason.kind == "explicit_done"
        assert result.handoff_count >= 1
        assert result.last_agent is reviewer


class TestPolicyDrivenTransitionCarriesPrompt:
    """Regression for the SCOPED empty-messages bug.

    When a policy (RoundRobin / StructuredRouting / Custom) routes to a
    different agent without emitting a ``SwarmHandoff`` yield, the target
    agent's first turn would arrive at the provider with ``messages=[]``
    because SCOPED (the default) only reads per-agent scratch + the
    addressed handoff payload. The driver now synthesizes a
    ``SwarmHandoff`` carrying the initial user prompt so the target has
    something to answer.
    """

    @pytest.mark.asyncio
    async def test_round_robin_second_agent_sees_user_prompt(self) -> None:
        a, b = _agent("a"), _agent("b")
        swarm = Swarm(
            members=(a, b),
            entry=a,
            policy=RoundRobinPolicy(),
            termination=MaxTurnsTermination(2),
        )

        seen_messages: list[list[Any]] = []

        async def fake_call_llm(*_args: Any, **kwargs: Any) -> LLMResponse:
            messages = kwargs.get("messages")
            if messages is None and len(_args) >= 2:
                messages = _args[1]
            seen_messages.append(list(messages) if messages is not None else [])
            return _text_response("ack", call_id=f"t{len(seen_messages)}")

        with (
            patch(
                "troopai.adk.run.loop.call_llm",
                new=AsyncMock(side_effect=fake_call_llm),
            ),
            patch(
                "troopai.adk.run.runner.run_blocking_input_guardrails",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "troopai.adk.run.runner.run_parallel_input_guardrails",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "troopai.adk.run.runner.run_output_guardrails",
                new=AsyncMock(return_value=[]),
            ),
        ):
            result = await Runner.arun_swarm(swarm, "hello world")

        assert len(seen_messages) >= 2
        turn_two = seen_messages[1]
        assert len(turn_two) > 0, "Turn 2 (agent b) received no messages — SCOPED leak regression"
        has_user_prompt = any(
            isinstance(m, dict) and m.get("role") == "user" and "hello world" in str(m.get("content", ""))
            for m in turn_two
        )
        assert has_user_prompt, "Turn 2 should carry the initial user prompt via the synthesized SwarmHandoff"
        # The synthesized SwarmHandoff MUST bump handoff_count so
        # SwarmHooks + RunResult observers see policy-driven transitions
        # identically to explicit LLM-emitted transfer_to_<name> calls.
        # (Exact count depends on how many post-turn rotations fire
        # before termination — the >=1 check is the SCOPED-fix
        # invariant; precise counting is covered elsewhere.)
        assert result.handoff_count >= 1, f"Expected at least 1 synthesized handoff, got {result.handoff_count}"

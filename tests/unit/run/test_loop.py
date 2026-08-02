"""Tests for ``run/loop.py`` driver edge cases.

Focused regression coverage for the streaming driver's post-loop
max-turns salvage check in ``run_agent_loop_streamed``. The driver must
only raise ``MaxTurnsExceeded`` on a *genuine* turn exhaustion — never
when the streamed run terminated deliberately via a HITL interruption,
a swarm yield, or a cancellation that happened to coincide with the
turn ceiling. The non-streaming driver already returns those terminal
results cleanly; this pins the streaming path to the same contract.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.exceptions.exceptions import MaxTurnsExceeded
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.run.stream import RunResultStreaming
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
)


def _tool_call_response(tool_name: str, call_id: str = "call_0") -> LLMResponse:
    """Build an LLMResponse that calls ``tool_name`` with no text output."""
    return LLMResponse(
        response_id=f"resp-{call_id}",
        model="fake",
        response=[
            LLMResponseFunctionToolCall(
                call_id=call_id,
                name=tool_name,
                arguments='{"value": "x"}',
            )
        ],
    )


def _make_agent(*, requires_approval: bool) -> Agent:
    """Agent with a single tool; optionally gated behind HITL approval."""

    async def _echo_invoker(_ctx: Any, _raw_args: str) -> str:
        return "echoed"

    echo = FunctionTool(
        name="echo",
        description="Echo back the input.",
        schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        on_invoke=_echo_invoker,
        requires_approval=requires_approval,
    )
    return Agent(
        name="test-agent",
        system_prompt="You are a test agent.",
        tools=[echo],
    )


async def _run_streamed(
    agent: Agent,
    *,
    max_turns: int,
    run_config: RunConfig,
) -> RunResultStreaming:
    """Drive ``Runner.arun(stream=True)`` with the LLM call mocked.

    The streamed background task that invokes ``call_llm_streamed`` is
    scheduled only once ``stream_events()`` is iterated, so the patch
    must remain active while the stream drains.
    """
    call_count = {"n": 0}

    async def fake_call_llm_streamed(*_args: Any, **_kwargs: Any) -> LLMResponse:
        call_count["n"] += 1
        return _tool_call_response("echo", call_id=f"call_{call_count['n']}")

    with (
        patch(
            "troopai.adk.run.loop.call_llm_streamed",
            new=AsyncMock(side_effect=fake_call_llm_streamed),
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
        streaming: RunResultStreaming = await Runner.arun(
            agent,
            "go",
            max_turns=max_turns,
            run_config=run_config,
            stream=True,
        )
        async for _ in streaming.stream_events():
            pass
        return streaming


class TestStreamedMaxTurnsSalvageGuard:
    async def test_hitl_interruption_at_turn_ceiling_is_not_max_turns_error(self) -> None:
        """A HITL deferral on the last allowed streamed turn must surface
        ``deferred_requests`` instead of raising ``MaxTurnsExceeded``.

        With ``max_turns=1`` the single turn defers the approval-gated
        tool. The block returns ``kind="final"`` with
        ``deferred_requests`` set and ``final_output=None`` while
        ``current_turn == max_turns``. Before the fix the driver's
        post-loop check saw ``current_turn >= max_turns and final_output
        is None`` and raised, silently dropping the HITL state.
        """
        agent = _make_agent(requires_approval=True)
        config = RunConfig()  # on_max_turns=None — exhaustion would raise

        streaming = await _run_streamed(agent, max_turns=1, run_config=config)

        assert streaming.deferred_requests is not None
        assert streaming.requires_action is True
        assert streaming.final_output is None

    async def test_genuine_turn_exhaustion_still_raises(self) -> None:
        """A real turn exhaustion (no deferral/yield/cancel) must still
        raise ``MaxTurnsExceeded`` so the salvage path is preserved.
        """
        agent = _make_agent(requires_approval=False)
        config = RunConfig()

        with pytest.raises(MaxTurnsExceeded, match="Agent loop exceeded 2 turns"):
            await _run_streamed(agent, max_turns=2, run_config=config)

    async def test_empty_final_output_at_ceiling_returns_cleanly(self) -> None:
        """A legitimate empty final output (``final_output is None``) produced
        on the last allowed streamed turn must return cleanly, NOT raise
        ``MaxTurnsExceeded``.

        The agent has no tools and the model returns an empty response, so the
        block resolves ``NextStepFinalOutput(output=None)`` while
        ``current_turn == max_turns``. Before the fix the driver could not tell
        this legitimate ``None`` output apart from genuine turn exhaustion and
        raised; now the block raises exhaustion itself, so a real final output
        (even ``None``) always returns.
        """
        agent = Agent(name="empty-agent", system_prompt="You are a test agent.")

        async def fake_empty(*_args: Any, **_kwargs: Any) -> LLMResponse:
            return LLMResponse(response_id="resp-empty", model="fake", response=[])

        with (
            patch("troopai.adk.run.loop.call_llm_streamed", new=AsyncMock(side_effect=fake_empty)),
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", new=AsyncMock(return_value=[])),
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", new=AsyncMock(return_value=[])),
            patch("troopai.adk.run.runner.run_output_guardrails", new=AsyncMock(return_value=[])),
        ):
            streaming: RunResultStreaming = await Runner.arun(
                agent,
                "go",
                max_turns=1,
                run_config=RunConfig(),
                stream=True,
            )
            async for _ in streaming.stream_events():
                pass

        assert streaming.final_output is None
        assert streaming.deferred_requests is None

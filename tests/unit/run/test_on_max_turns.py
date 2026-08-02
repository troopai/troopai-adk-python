"""Tests for ``RunConfig.on_max_turns`` salvage handler.

Covers both the non-streaming ``run_agent_loop`` path (line 501) and
the streaming ``run_agent_loop_streamed`` path (line 794), confirming
that:

1. A ``None`` handler falls through to the usual ``MaxTurnsExceeded``.
2. A handler returning a string short-circuits with that string as
   ``final_output``.
3. A handler returning ``None`` falls through to ``MaxTurnsExceeded``.
4. The swarm-level ``max_total_turns`` cap is **not** routed through
   the handler — it represents a runaway workflow, not a per-agent
   budget, and always raises.
5. The handler receives the current agent and the exhausted turn
   count.
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
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
)


def _tool_call_response(call_id: str = "call_0") -> LLMResponse:
    """Build an LLMResponse that forces the loop to run another turn.

    A response with a function_call part and no text output means the
    turn resolution layer will emit ``NextStepRunAgain`` (once the tool
    output is appended), so the loop advances to the next turn. By
    feeding this indefinitely we force ``max_turns`` to trip.
    """
    return LLMResponse(
        response_id=f"resp-{call_id}",
        model="fake",
        response=[
            LLMResponseFunctionToolCall(
                call_id=call_id,
                name="echo",
                arguments='{"value": "x"}',
            )
        ],
    )


def _make_agent_with_tool() -> Agent:
    """Agent with one simple tool so the loop has something to execute."""
    from troopai.adk.tools.function_tool import FunctionTool

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
    )
    return Agent(
        name="test-agent",
        system_prompt="You are a test agent.",
        tools=[echo],
    )


async def _patched_arun(
    agent: Agent,
    prompt: str,
    *,
    max_turns: int,
    run_config: RunConfig,
    stream: bool = False,
) -> Any:
    """Run ``Runner.arun`` with call_llm, call_llm_streamed, and guardrails stubbed.

    In streaming mode, the background task that actually invokes
    ``call_llm_streamed`` is only scheduled when ``Runner.arun`` returns;
    the patches must stay in effect while we drain ``stream_events()``,
    otherwise the task runs with the real litellm-backed LLM.
    """
    call_count = {"n": 0}

    async def fake_call_llm(*_args: Any, **_kwargs: Any) -> LLMResponse:
        call_count["n"] += 1
        return _tool_call_response(call_id=f"call_{call_count['n']}")

    async def fake_call_llm_streamed(*_args: Any, **_kwargs: Any) -> LLMResponse:
        call_count["n"] += 1
        return _tool_call_response(call_id=f"call_{call_count['n']}")

    with (
        patch(
            "troopai.adk.run.loop.call_llm",
            new=AsyncMock(side_effect=fake_call_llm),
        ),
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
        if stream:
            streaming: RunResultStreaming = await Runner.arun(
                agent,
                prompt,
                max_turns=max_turns,
                run_config=run_config,
                stream=True,
            )
            # Drain the stream INSIDE the patch block so the background
            # task's call_llm_streamed invocations hit the mock.
            async for _ in streaming.stream_events():
                pass
            return streaming
        return await Runner.arun(
            agent,
            prompt,
            max_turns=max_turns,
            run_config=run_config,
            stream=False,
        )


# ── Non-streaming path ───────────────────────────────────────────────


class TestOnMaxTurnsNonStreaming:
    @pytest.mark.asyncio
    async def test_no_handler_raises(self) -> None:
        """Default behavior (handler=None) still raises MaxTurnsExceeded."""
        agent = _make_agent_with_tool()
        config = RunConfig()  # on_max_turns=None

        with pytest.raises(MaxTurnsExceeded, match="Agent loop exceeded 2 turns"):
            await _patched_arun(agent, "go", max_turns=2, run_config=config)

    @pytest.mark.asyncio
    async def test_string_return_salvages_final_output(self) -> None:
        """A non-None string return becomes ``result.final_output``."""
        agent = _make_agent_with_tool()

        captured: dict[str, Any] = {}

        async def handler(which_agent: Agent, turns: int) -> str:
            captured["agent"] = which_agent
            captured["turns"] = turns
            return f"salvaged after {turns} turns"

        config = RunConfig(on_max_turns=handler)
        result = await _patched_arun(agent, "go", max_turns=2, run_config=config)

        assert result.final_output == "salvaged after 2 turns"
        assert captured["agent"].name == "test-agent"
        assert captured["turns"] == 2

    @pytest.mark.asyncio
    async def test_none_return_falls_through_to_raise(self) -> None:
        """A handler returning None does NOT swallow the exception."""
        agent = _make_agent_with_tool()

        async def handler(_agent: Agent, _turns: int) -> Any:
            return None

        config = RunConfig(on_max_turns=handler)

        with pytest.raises(MaxTurnsExceeded):
            await _patched_arun(agent, "go", max_turns=2, run_config=config)

    @pytest.mark.asyncio
    async def test_max_total_turns_bypasses_handler(self) -> None:
        """Swarm-level ``max_total_turns`` must always raise.

        ``max_total_turns`` represents a runaway workflow, not a
        per-agent budget. The handler is explicitly NOT wired into
        that branch.
        """
        agent = _make_agent_with_tool()

        salvage_calls = {"n": 0}

        async def handler(_agent: Agent, _turns: int) -> str:
            salvage_calls["n"] += 1
            return "should not be seen"

        # max_total_turns=1 trips before per-agent max_turns=10.
        config = RunConfig(on_max_turns=handler, max_total_turns=1)

        with pytest.raises(MaxTurnsExceeded, match="Cross-agent total turns"):
            await _patched_arun(agent, "go", max_turns=10, run_config=config)

        assert salvage_calls["n"] == 0  # handler was never invoked


# ── Streaming path ───────────────────────────────────────────────────


class TestOnMaxTurnsStreaming:
    @pytest.mark.asyncio
    async def test_no_handler_raises_streaming(self) -> None:
        """Streaming default behavior still raises MaxTurnsExceeded."""
        agent = _make_agent_with_tool()
        config = RunConfig()

        with pytest.raises(MaxTurnsExceeded, match="Agent loop exceeded 2 turns"):
            await _patched_arun(agent, "go", max_turns=2, run_config=config, stream=True)

    @pytest.mark.asyncio
    async def test_string_return_salvages_final_output_streaming(self) -> None:
        """A non-None string return becomes ``result.final_output`` in streaming."""
        agent = _make_agent_with_tool()

        captured: dict[str, Any] = {}

        async def handler(which_agent: Agent, turns: int) -> str:
            captured["agent"] = which_agent
            captured["turns"] = turns
            return f"stream-salvaged after {turns} turns"

        config = RunConfig(on_max_turns=handler)

        streaming = await _patched_arun(agent, "go", max_turns=2, run_config=config, stream=True)

        assert streaming.final_output == "stream-salvaged after 2 turns"
        assert captured["agent"].name == "test-agent"
        assert captured["turns"] == 2

    @pytest.mark.asyncio
    async def test_none_return_falls_through_to_raise_streaming(self) -> None:
        """Streaming: None return does NOT swallow MaxTurnsExceeded."""
        agent = _make_agent_with_tool()

        async def handler(_agent: Agent, _turns: int) -> Any:
            return None

        config = RunConfig(on_max_turns=handler)

        with pytest.raises(MaxTurnsExceeded):
            await _patched_arun(agent, "go", max_turns=2, run_config=config, stream=True)

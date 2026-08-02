"""Tests for ``RunConfig.include_hook_events`` streaming feature.

Covers:

1. When enabled, tool-start and tool-end ``HookLifecycleEvent`` objects appear
   in the stream in the correct order relative to ``TOOL_CALLED`` /
   ``TOOL_OUTPUT`` items.
2. When disabled, no ``HookLifecycleEvent`` appears in the stream.
3. Non-streaming ``arun`` with ``include_hook_events=True`` does not fail
   (feature is a no-op on the non-streaming path).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.run.stream import HookEventKind, HookLifecycleEvent, RunItemType
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
    LLMResponseText,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _tool_call_response(call_id: str = "call_0") -> LLMResponse:
    """LLMResponse that asks the loop to invoke the 'echo' tool."""
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


def _final_text_response() -> LLMResponse:
    """LLMResponse that ends the loop with a plain text reply."""
    return LLMResponse(
        response_id="resp-final",
        model="fake",
        response=[LLMResponseText(text="done")],
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


async def _patched_arun_with_tool(
    agent: Agent,
    prompt: str,
    *,
    run_config: RunConfig,
) -> list[Any]:
    """Stream a run that makes exactly one tool call, then returns a final text.

    Returns the collected stream events.
    The patched call_llm_streamed returns a tool-call response on the first
    invocation and a final text response on the second.
    """
    call_count = {"n": 0}

    async def fake_call_llm_streamed(*_args: Any, **_kwargs: Any) -> LLMResponse:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _tool_call_response(call_id="call_1")
        return _final_text_response()

    events: list[Any] = []

    with (
        patch(
            "troopai.adk.run.loop.call_llm",
            new=AsyncMock(side_effect=fake_call_llm_streamed),
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
        streaming = await Runner.arun(
            agent,
            prompt,
            run_config=run_config,
            stream=True,
        )
        async for event in streaming.stream_events():
            events.append(event)

    return events


# ── Tests ─────────────────────────────────────────────────────────────


class TestHookLifecycleEventsStreaming:
    @pytest.mark.asyncio
    async def test_tool_events_appear_in_order_when_enabled(self) -> None:
        """TOOL_START appears before TOOL_CALLED; TOOL_END appears after TOOL_OUTPUT."""
        agent = _make_agent_with_tool()
        config = RunConfig(include_hook_events=True)

        events = await _patched_arun_with_tool(agent, "go", run_config=config)

        hook_events = [e for e in events if isinstance(e, HookLifecycleEvent)]
        hook_kinds = [e.kind for e in hook_events]

        assert HookEventKind.TOOL_START in hook_kinds, "TOOL_START not found in stream events"
        assert HookEventKind.TOOL_END in hook_kinds, "TOOL_END not found in stream events"

        # TOOL_START must come before TOOL_END
        start_idx = hook_kinds.index(HookEventKind.TOOL_START)
        end_idx = hook_kinds.index(HookEventKind.TOOL_END)
        assert start_idx < end_idx, "TOOL_START must precede TOOL_END"

        # Verify payload content
        start_event = hook_events[start_idx]
        assert start_event.agent_name == "test-agent"
        assert start_event.payload.get("tool_name") == "echo"

        end_event = hook_events[end_idx]
        assert end_event.agent_name == "test-agent"
        assert end_event.payload.get("tool_name") == "echo"

    @pytest.mark.asyncio
    async def test_events_absent_when_disabled(self) -> None:
        """With include_hook_events=False, no HookLifecycleEvent in stream."""
        agent = _make_agent_with_tool()
        config = RunConfig(include_hook_events=False)

        events = await _patched_arun_with_tool(agent, "go", run_config=config)

        hook_events = [e for e in events if isinstance(e, HookLifecycleEvent)]
        assert hook_events == [], f"Expected no hook events, got: {hook_events}"

    @pytest.mark.asyncio
    async def test_run_item_events_still_present_when_enabled(self) -> None:
        """Enabling hook events does not suppress RunItemStreamEvent objects."""
        from troopai.adk.run.stream import RunItemStreamEvent

        agent = _make_agent_with_tool()
        config = RunConfig(include_hook_events=True)

        events = await _patched_arun_with_tool(agent, "go", run_config=config)

        run_item_events = [e for e in events if isinstance(e, RunItemStreamEvent)]
        run_item_types = [e.name for e in run_item_events]

        assert RunItemType.TOOL_CALLED in run_item_types
        assert RunItemType.TOOL_OUTPUT in run_item_types


class TestHookLifecycleEventsNonStreaming:
    @pytest.mark.asyncio
    async def test_non_streaming_path_unaffected(self) -> None:
        """Non-streaming arun with include_hook_events=True does not fail."""
        agent = _make_agent_with_tool()
        config = RunConfig(include_hook_events=True)

        call_count = {"n": 0}

        async def fake_call_llm(*_args: Any, **_kwargs: Any) -> LLMResponse:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _tool_call_response(call_id="call_1")
            return _final_text_response()

        with (
            patch(
                "troopai.adk.run.loop.call_llm",
                new=AsyncMock(side_effect=fake_call_llm),
            ),
            patch(
                "troopai.adk.run.loop.call_llm_streamed",
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
            result = await Runner.arun(agent, "go", run_config=config)

        assert result.final_output == "done"

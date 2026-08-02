"""Regression tests for ``run/`` turn-flow fixes.

Covers four corrections to the shared agent loop and turn-resolution helpers:

- ``turn_resolution.resolve_tool_results_step`` must pass a multimodal
  (list-of-parts) tool output through UNCHANGED, never ``str()`` it into a
  ``"[{'type': 'input_text', ...}]"`` repr string.
- The non-streaming block must stamp the run-CUMULATIVE turn number onto a
  HITL ``RunState`` (``turn_offset + turn``), not the block-local turn, so a
  post-handoff resume computes the correct remaining-turn budget. The
  streaming block already passes the cumulative ``result.current_turn``.
- ``context_end`` (the temporal-slicing boundary between an agent's inherited
  context and its own output) must shift by a per-turn rebind's net
  message-length change so a later handoff split never slices at a stale
  index after compaction / history-processor / input-filter rebinds.
- A streamed HITL interruption must NOT re-emit ``TOOL_OUTPUT`` for results
  the tool executor already emitted — exactly one per completed call.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from troopai.adk.agents.agent import Agent
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.config import CallModelData, ModelInputData, RunConfig
from troopai.adk.run.context import RunContext
from troopai.adk.run.runner import Runner
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.output.function_tool_call_result import FunctionToolCallResult
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
)


def _echo_tool(name: str = "echo", *, requires_approval: bool = False) -> FunctionTool:
    async def _invoke(_ctx: Any, _raw_args: str) -> str:
        return "echoed"

    return FunctionTool(
        name=name,
        description="Echo back the input.",
        schema={"type": "object", "properties": {"value": {"type": "string"}}},
        on_invoke=_invoke,
        requires_approval=requires_approval,
    )


def _tool_call(name: str, call_id: str) -> LLMResponseFunctionToolCall:
    return LLMResponseFunctionToolCall(call_id=call_id, name=name, arguments='{"value": "x"}')


# ---------------------------------------------------------------------------
# Multimodal tool output must survive resolve_tool_results_step unchanged
# ---------------------------------------------------------------------------
class TestMultimodalToolOutputPreserved:
    async def test_normal_continuation_preserves_list_output(self) -> None:
        """A list-of-parts tool output must be appended to history verbatim,
        not stringified into a repr — otherwise the provider converter can no
        longer emit the image/text blocks."""
        from troopai.adk.run.next_step import NextStepRunAgain
        from troopai.adk.run.turn_resolution import resolve_tool_results_step

        multimodal: list[Any] = [
            {"type": "input_text", "text": "chart description"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]
        tr = FunctionToolCallResult(call_id="call_1", output=multimodal)
        agent = Agent(name="a", system_prompt="s", tools=[_echo_tool()])
        ctx: RunContext[Any] = RunContext(context=None)
        messages: list[Any] = []
        new_items: list[Any] = []

        step = await resolve_tool_results_step(
            agent,
            [tr],
            None,
            [_tool_call("echo", "call_1")],
            messages,
            new_items,
            "prompt",
            None,
            ctx,
            1,
        )

        assert isinstance(step, NextStepRunAgain)
        assert len(messages) == 1
        # The multimodal list must round-trip byte-for-byte — never str()'d.
        assert messages[0]["output"] == multimodal
        assert isinstance(messages[0]["output"], list)

    async def test_string_output_still_passes_through(self) -> None:
        """A plain-string output is unaffected by dropping the ``str()`` call."""
        from troopai.adk.run.turn_resolution import resolve_tool_results_step

        tr = FunctionToolCallResult(call_id="call_1", output="plain text")
        agent = Agent(name="a", system_prompt="s", tools=[_echo_tool()])
        ctx: RunContext[Any] = RunContext(context=None)
        messages: list[Any] = []

        await resolve_tool_results_step(
            agent, [tr], None, [_tool_call("echo", "call_1")], messages, [], "prompt", None, ctx, 1
        )

        assert messages[0]["output"] == "plain text"


# ---------------------------------------------------------------------------
# HITL RunState must carry the run-cumulative turn count, not block-local turn
# ---------------------------------------------------------------------------
class TestCumulativeTurnCountOnInterruption:
    async def test_run_agent_block_stamps_turn_offset_plus_turn(self) -> None:
        """With a non-zero ``turn_offset`` (a block reached after a handoff),
        a first-block-turn HITL deferral must record ``turn_offset + 1`` on the
        ``RunState`` so a resume does not over-grant the remaining-turn budget."""
        from troopai.adk.run.loop import run_agent_block

        agent = Agent(name="specialist", system_prompt="s", tools=[_echo_tool(requires_approval=True)])
        ctx: RunContext[Any] = RunContext(context=None)

        async def fake_call_llm(*_args: Any, **_kwargs: Any) -> LLMResponse:
            return LLMResponse(response_id="r", model="test", response=[_tool_call("echo", "call_0")])

        with patch("troopai.adk.run.loop.call_llm", new=AsyncMock(side_effect=fake_call_llm)):
            outcome = await run_agent_block(
                agent=agent,
                messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "go"}],
                context_end=2,
                user_prompt="go",
                context=ctx,
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                max_turns=10,
                config=RunConfig(),
                new_items=[],
                tool_failure_counts={},
                initial_tool_choice_override=None,
                extra_tools=None,
                swarm_tool_names=None,
                ctx_mgr=None,
                jit_directives=None,
                skill_tool_map={},
                activated_skills=set(),
                turn_offset=3,
                starting_total_turns=3,
            )

        assert outcome.result is not None
        assert outcome.result.state is not None
        # turn_offset (3) + block-local turn (1) — NOT the block-local turn (1).
        assert outcome.result.state.turn_count == 4


# ---------------------------------------------------------------------------
# context_end must track per-turn rebind length changes
# ---------------------------------------------------------------------------
class TestContextEndTracksRebind:
    def test_adjust_context_end_shifts_by_delta_and_clamps(self) -> None:
        from troopai.adk.run.loop import _adjust_context_end

        # No length change → unchanged (the default, no-rebind hot path).
        assert _adjust_context_end(2, 4, 4) == 2
        # Shrunk by one → boundary moves down one.
        assert _adjust_context_end(2, 4, 3) == 1
        # Grew by two → boundary moves up two.
        assert _adjust_context_end(2, 4, 6) == 4
        # Over-shrink clamps to zero, never negative.
        assert _adjust_context_end(1, 5, 2) == 0
        # Never exceeds the new length.
        assert _adjust_context_end(10, 4, 3) == 3

    async def test_loop_shifts_context_end_after_input_filter_rebind(self) -> None:
        """A ``call_model_input_filter`` that drops the leading context message
        shrinks ``messages`` by one; the loop must hand the step resolvers the
        adjusted boundary (``2 - 1 == 1``), not the stale block-entry ``2``."""
        from troopai.adk.run import turn_resolution
        from troopai.adk.run.loop import run_agent_loop

        captured: dict[str, int] = {}

        async def capture_structured(
            _agent: Any, _response: Any, _messages: Any, _new_items: Any, context_end: int, *_rest: Any
        ) -> None:
            captured["context_end"] = context_end
            return None

        def shrink_filter(payload: CallModelData[Any]) -> ModelInputData:
            # Drop the first (system/context) message every turn.
            return ModelInputData(input=payload.model_data.input[1:])

        async def fake_call_llm(*_args: Any, **_kwargs: Any) -> LLMResponse:
            return LLMResponse(response_id="r", model="test", response=[])

        agent = Agent(name="a", system_prompt="s")
        ctx: RunContext[Any] = RunContext(context=None)
        config = RunConfig(call_model_input_filter=shrink_filter)

        with (
            patch("troopai.adk.run.loop.call_llm", new=AsyncMock(side_effect=fake_call_llm)),
            patch.object(turn_resolution, "resolve_structured_output_step", new=capture_structured),
        ):
            await run_agent_loop(
                agent=agent,
                user_prompt="go",
                context=ctx,
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                max_turns=5,
                config=config,
            )

        # Block entry context_end == 2 ([system, user]); the filter dropped one
        # message, so the resolver must see 1.
        assert captured["context_end"] == 1


# ---------------------------------------------------------------------------
# Streamed HITL interruption must not duplicate TOOL_OUTPUT events
# ---------------------------------------------------------------------------
class TestStreamedInterruptionNoDuplicateToolOutput:
    async def test_completed_tool_emits_single_tool_output(self) -> None:
        """One tool completes and a sibling defers in the same batch. The
        completed tool's ``TOOL_OUTPUT`` is emitted once by the executor; the
        interruption handler must not re-emit it."""
        from troopai.adk.run.stream import RunItemStreamEvent, RunItemType

        agent = Agent(
            name="hitl-agent",
            system_prompt="s",
            tools=[_echo_tool("auto_tool"), _echo_tool("gated_tool", requires_approval=True)],
        )

        async def fake_stream(*_args: Any, **_kwargs: Any) -> LLMResponse:
            return LLMResponse(
                response_id="resp-1",
                model="fake",
                response=[_tool_call("auto_tool", "call_a"), _tool_call("gated_tool", "call_b")],
            )

        tool_output_events = 0
        with (
            patch("troopai.adk.run.loop.call_llm_streamed", new=AsyncMock(side_effect=fake_stream)),
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", new=AsyncMock(return_value=[])),
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", new=AsyncMock(return_value=[])),
            patch("troopai.adk.run.runner.run_output_guardrails", new=AsyncMock(return_value=[])),
        ):
            streaming = await Runner.arun(agent, "go", max_turns=3, run_config=RunConfig(), stream=True)
            async for event in streaming.stream_events():
                if isinstance(event, RunItemStreamEvent) and event.name == RunItemType.TOOL_OUTPUT:
                    tool_output_events += 1

        assert streaming.deferred_requests is not None
        # Exactly one completed tool → exactly one TOOL_OUTPUT (no duplicate).
        assert tool_output_events == 1

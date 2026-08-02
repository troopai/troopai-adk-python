"""Unit tests for :class:`TaskPipelineState` + persistence + resume.

Covered:

- ``TaskOutput.to_dict / from_dict`` round-trip preserves the
  scalar-shaped fields (task_id, task_name, error, skipped,
  streaming_placeholder, metadata) and the usage scalar counts.
- ``TaskPipelineState.to_json / from_json`` round-trip preserves
  pipeline_id, resume_index, slots, completed_task_ids.
- ``TaskPipelineState.__post_init__`` rejects invalid resume_index
  values (negative or exceeding slot count).
- ``Runner.arun_task_pipeline_from_state`` resumes from the recorded
  slot, returns recorded slots verbatim, executes the remaining
  tasks, and aggregates usage across both halves.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.tasks import (
    Task,
    TaskOutput,
    TaskPipeline,
    TaskPipelineState,
)
from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText
from troopai.adk.types.tokens.llm_usage import LLMUsage


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        response_id="resp-x",
        model="fake",
        response=[LLMResponseText(text=text)],
    )


class TestTaskOutputSerialization:
    def test_round_trip_preserves_basic_fields(self) -> None:
        original = TaskOutput(
            task_id="abc-123",
            task_name="my-task",
            final_output="hello world",
            usage=LLMUsage(requests=2, input_tokens=10, output_tokens=20, total_tokens=30),
            skipped=False,
            error=None,
            streaming_placeholder=False,
            metadata={"tenant": "demo"},
        )

        rehydrated = TaskOutput.from_dict(original.to_dict())

        assert rehydrated.task_id == "abc-123"
        assert rehydrated.task_name == "my-task"
        assert rehydrated.final_output == "hello world"
        assert rehydrated.skipped is False
        assert rehydrated.error is None
        assert rehydrated.streaming_placeholder is False
        assert rehydrated.metadata == {"tenant": "demo"}
        assert rehydrated.usage is not None
        assert rehydrated.usage.total_tokens == 30
        assert rehydrated.usage.requests == 2
        # `new_items` is not serialized; rehydrate to empty.
        assert rehydrated.new_items == ()

    def test_round_trip_with_no_usage(self) -> None:
        original = TaskOutput(
            task_id="abc",
            task_name="t",
            skipped=True,
        )
        rehydrated = TaskOutput.from_dict(original.to_dict())
        assert rehydrated.skipped is True
        assert rehydrated.usage is None

    def test_round_trip_preserves_streaming_placeholder_flag(self) -> None:
        placeholder = TaskOutput(
            task_id="abc",
            task_name="t",
            streaming_placeholder=True,
        )
        rehydrated = TaskOutput.from_dict(placeholder.to_dict())
        assert rehydrated.streaming_placeholder is True


class TestTaskPipelineStateSerialization:
    def test_round_trip_via_json_preserves_slots(self) -> None:
        slot0 = TaskOutput(
            task_id="t0",
            task_name="task-0",
            final_output="output-0",
            usage=LLMUsage(requests=1, input_tokens=5, output_tokens=10, total_tokens=15),
        )
        slot1 = TaskOutput(task_id="t1", task_name="task-1", skipped=True)

        state = TaskPipelineState(
            pipeline_id="pipe-001",
            slots=(slot0, slot1),
            resume_index=2,
        )

        rehydrated = TaskPipelineState.from_json(state.to_json())

        assert rehydrated.pipeline_id == "pipe-001"
        assert rehydrated.resume_index == 2
        assert len(rehydrated.slots) == 2
        assert rehydrated.slots[0].task_id == "t0"
        assert rehydrated.slots[0].final_output == "output-0"
        assert rehydrated.slots[1].skipped is True

    def test_negative_resume_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            TaskPipelineState(pipeline_id="p", slots=(), resume_index=-1)

    def test_resume_index_exceeding_slots_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            TaskPipelineState(
                pipeline_id="p",
                slots=(TaskOutput(task_id="t", task_name="t"),),
                resume_index=5,
            )


class TestResumeEndToEnd:
    @pytest.mark.integration
    async def test_resume_continues_from_recorded_slot(self) -> None:
        agent = Agent(name="a", system_prompt="x")

        # Three-task pipeline. State records the first task as
        # already complete; resume_index=1.
        pipeline = TaskPipeline(
            tasks=(
                Task(description="t1", agent=agent, name="task-1"),
                Task(description="t2", agent=agent, name="task-2"),
                Task(description="t3", agent=agent, name="task-3"),
            ),
        )
        state = TaskPipelineState(
            pipeline_id="resume-test",
            slots=(
                TaskOutput(
                    task_id="recorded-t1",
                    task_name="task-1",
                    final_output="prior-output",
                    usage=LLMUsage(
                        requests=1,
                        input_tokens=10,
                        output_tokens=20,
                        total_tokens=30,
                    ),
                ),
            ),
            resume_index=1,
        )

        async def fake_call_llm(*_args: Any, **kwargs: Any) -> LLMResponse:
            messages = kwargs.get("messages") or _args[1]
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return _text_response(f"answered-{msg.get('content')}")
            return _text_response("unknown")

        from troopai.adk.run.runner import Runner

        with (
            patch("troopai.adk.run.loop.call_llm", new=AsyncMock(side_effect=fake_call_llm)),
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
            result = await Runner.arun_task_pipeline_from_state(pipeline, state)

        # Three slots in the result: first is the recorded one,
        # last two are freshly executed.
        assert len(result.task_outputs) == 3
        assert result.task_outputs[0].task_id == "recorded-t1"
        assert result.task_outputs[0].final_output == "prior-output"
        assert result.task_outputs[1].final_output == "answered-t2"
        assert result.task_outputs[2].final_output == "answered-t3"
        # final_output is the last non-skipped task's final output.
        assert result.final_output == "answered-t3"
        # Cumulative usage includes the recorded slot's 30 tokens
        # plus whatever the two new tasks consumed.
        assert result.context is not None
        assert result.context.usage.total_tokens >= 30

    @pytest.mark.integration
    async def test_resume_with_short_pipeline_rejected(self) -> None:
        agent = Agent(name="a", system_prompt="x")
        # Originating pipeline had 5 tasks; resuming side only knows
        # about 1. The Runner detects the mismatch.
        pipeline = TaskPipeline(tasks=(Task(description="t1", agent=agent),))
        state = TaskPipelineState(
            pipeline_id="bad",
            slots=tuple(TaskOutput(task_id=f"t{i}", task_name=f"t{i}") for i in range(5)),
            resume_index=5,
        )

        from troopai.adk.run.runner import Runner

        with pytest.raises(ValueError, match="exceeds the reconstructed pipeline length"):
            await Runner.arun_task_pipeline_from_state(pipeline, state)

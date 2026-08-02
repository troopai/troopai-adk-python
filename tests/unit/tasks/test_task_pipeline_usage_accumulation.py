"""Unit tests for pipeline-level :class:`LLMUsage` accumulation.

:meth:`Runner.arun_task_pipeline` MUST sum each non-skipped task's
``TaskOutput.usage`` into the shared
``TaskPipelineResult.context.usage``. Skipped tasks contribute zero.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from troopai.adk.agents.agent import Agent
from troopai.adk.run.context import RunContext
from troopai.adk.tasks import Task, TaskPipeline
from troopai.adk.types.run.run_result import RunResult
from troopai.adk.types.tokens.llm_usage import LLMUsage


def _agent() -> Agent:
    return Agent(name="t", system_prompt="x")


def _run_result_with_usage(text: str, total_tokens: int) -> RunResult:
    ctx: RunContext[Any] = RunContext.make(None)
    ctx.usage = LLMUsage(requests=1, input_tokens=10, output_tokens=10, total_tokens=total_tokens)
    return RunResult(
        final_output=text,
        user_prompt="ignored",
        new_items=[],
        context=ctx,
    )


async def test_pipeline_usage_sums_across_tasks() -> None:
    agent = _agent()
    t1 = Task(description="a", agent=agent)
    t2 = Task(description="b", agent=agent)
    t3 = Task(description="c", agent=agent)

    sequence = iter([10, 20, 30])

    async def fake_arun(*_args: Any, **_kwargs: Any) -> RunResult:
        return _run_result_with_usage("ok", next(sequence))

    from troopai.adk.run.runner import Runner

    with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake_arun)):
        result = await Runner.arun_task_pipeline(TaskPipeline(tasks=(t1, t2, t3)))

    assert result.context is not None
    assert result.context.usage.total_tokens == 60
    # Per-task usages remain on the individual TaskOutputs.
    assert result.task_outputs[0].usage is not None
    assert result.task_outputs[0].usage.total_tokens == 10
    assert result.task_outputs[2].usage is not None
    assert result.task_outputs[2].usage.total_tokens == 30


async def test_skipped_task_contributes_zero_usage() -> None:
    agent = _agent()
    t1 = Task(description="a", agent=agent)
    t2 = Task(description="b", agent=agent, skip_if=lambda _p: True)
    t3 = Task(description="c", agent=agent)

    sequence = iter([15, 25])  # only t1 and t3 actually run

    async def fake_arun(*_args: Any, **_kwargs: Any) -> RunResult:
        return _run_result_with_usage("ok", next(sequence))

    from troopai.adk.run.runner import Runner

    with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake_arun)):
        result = await Runner.arun_task_pipeline(TaskPipeline(tasks=(t1, t2, t3)))

    assert result.context is not None
    assert result.context.usage.total_tokens == 40
    # Skipped task's TaskOutput has no usage.
    assert result.task_outputs[1].skipped is True
    assert result.task_outputs[1].usage is None

"""Regression tests for runner-level cancellation and streamed-pipeline fixes.

Covered:

- _run_one_task_in_level catches asyncio.CancelledError and returns an
  error TaskOutput (siblings survive; no unhandled propagation through
  asyncio.gather).
- arun_task_streamed's run_impl catches CancelledError, calls
  set_exception, then re-raises so the asyncio machinery sees the
  cancellation.
- _stream_task_pipeline_impl yields (index, None) before breaking when
  skip_if raises (error-stop notification) so the consumer sees which
  index halted the pipeline.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.context import RunContext
from troopai.adk.tasks import Task, TaskOutput, TaskPipeline
from troopai.adk.types.run.run_result import RunResult


def _agent() -> Agent:
    return Agent(name="t", system_prompt="x")


def _run_result(final: str) -> RunResult:
    return RunResult(
        final_output=final,
        user_prompt="ignored",
        new_items=[],
        context=RunContext.make(None),
    )


class TestRunOneLevelCancelledError:
    """_run_one_task_in_level must propagate CancelledError, not swallow it."""

    async def test_cancelled_error_propagates_not_swallowed(self) -> None:
        """When a task's arun_task raises CancelledError in a DAG level, the
        cancellation must propagate — NOT be converted to an error TaskOutput.

        Swallowing it into a slot would defeat external cancellation of the whole
        pipeline (a user Ctrl-C / pipeline timeout cancels the level's gather, and
        every task would otherwise quietly become an error slot and the DAG keep
        running). Genuine task *failures* (Exceptions) still become error slots so
        siblings finish — only CancelledError is special-cased to propagate, per
        the asyncio cancellation contract."""
        from troopai.adk.run.runner import Runner

        agent = _agent()
        # Both t1 and t2 depend on root; they run concurrently in level-1.
        root = Task(description="root", agent=agent, task_id="root")
        t1 = Task(description="fast", agent=agent, task_id="t1", depends_on=("root",))
        t2 = Task(description="slow", agent=agent, task_id="t2", depends_on=("root",))

        async def fake_arun_task(task: Task, **kwargs: Any) -> TaskOutput:
            if task.task_id == "t1":
                raise asyncio.CancelledError("simulated cancel")
            return TaskOutput(
                task_id=task.task_id or "unknown",
                task_name=task.description,
                final_output=f"{task.task_id}-ok",
            )

        pipeline = TaskPipeline(tasks=(root, t1, t2))

        with (
            patch.object(Runner, "arun_task", new=AsyncMock(side_effect=fake_arun_task)),
            pytest.raises(asyncio.CancelledError),
        ):
            await Runner.arun_task_pipeline(pipeline)


class TestStreamedTaskCancelledError:
    """run_impl in arun_task_streamed propagates CancelledError to the stream."""

    async def test_cancelled_error_sets_exception_on_result(self) -> None:
        """CancelledError inside run_impl must call set_exception so that
        stream_events() surfaces the cancellation rather than completing
        cleanly. Pre-fix: except Exception gap silently ate the cancellation."""
        from troopai.adk.run.runner import Runner

        agent = _agent()
        task = Task(description="do it", agent=agent)

        # Simulate CancelledError inside the streamed LLM call path
        # (the innermost async boundary inside run_impl).
        async def fake_streamed(*args: Any, **kwargs: Any) -> Any:
            raise asyncio.CancelledError("simulated mid-run cancel")

        with (
            patch("troopai.adk.run.loop.call_llm_streamed", new=AsyncMock(side_effect=fake_streamed)),
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
            result = await Runner.arun_task_streamed(task)

            # Consume the stream INSIDE the patch scope. arun_task_streamed only
            # *schedules* run_impl as a background task, so the patched
            # call_llm_streamed executes later, while stream_events() drives that
            # task. Consuming outside the `with` lets the patches tear down first,
            # so the real LLM path runs instead — the test then passes on an
            # unrelated error (e.g. a missing API key) or fails outright depending
            # on the environment and event-loop scheduling.
            raised: asyncio.CancelledError | None = None
            try:
                async for _ in result.stream_events():
                    pass
            except asyncio.CancelledError as exc:
                raised = exc

        # run_impl's CancelledError handler must call set_exception, so
        # stream_events() re-raises the cancellation rather than completing
        # cleanly. Assert the exact type so the test cannot pass on an unrelated
        # error again.
        assert isinstance(raised, asyncio.CancelledError), (
            f"stream_events() must re-raise the CancelledError from run_impl; got {raised!r}"
        )


class TestStreamedPipelineErrorNotification:
    """_stream_task_pipeline_impl yields (index, None) before halting on
    skip_if error so consumers see which task caused the stop."""

    async def test_skip_if_error_yields_index_before_halt(self) -> None:
        """When skip_if raises, the generator must yield (index, None)
        before breaking. Pre-fix: the break happened with no yield, so
        the consumer received no notification of the halted task index."""
        from troopai.adk.run.runner import Runner

        agent = _agent()

        def exploding_skip(prior: Any) -> bool:
            raise RuntimeError("skip_if exploded")

        t1 = Task(description="first", agent=agent)
        t2 = Task(description="second", agent=agent, skip_if=exploding_skip)

        pipeline = TaskPipeline(tasks=(t1, t2))

        async def fake_arun_task_streamed(task: Task, **kwargs: Any) -> Any:
            # Return a minimal mock RunResultStreaming.
            mock = MagicMock()
            mock.stream_events = AsyncMock(return_value=aiter([]))
            return mock

        with patch.object(Runner, "arun_task_streamed", new=AsyncMock(side_effect=fake_arun_task_streamed)):
            yielded: list[tuple[int, Any]] = []
            async for item in Runner.arun_task_pipeline_streamed(pipeline):
                yielded.append(item)

        # t1 should have yielded (0, <stream>).
        # t2 halted by skip_if error — must still yield (1, None) as a
        # stop notification (not silently drop).
        assert len(yielded) >= 2
        # The last yield must be (1, None) — the error-stop sentinel.
        assert yielded[-1] == (1, None)


def aiter(items: list[Any]) -> Any:
    """Return an async iterator over items (helper for tests)."""

    async def _gen() -> Any:
        for item in items:
            yield item

    return _gen()

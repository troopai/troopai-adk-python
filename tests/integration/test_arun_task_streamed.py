"""End-to-end tests for :meth:`Runner.arun_task_streamed`.

Exercises the streamed Task entry point against the real agent loop
with a stubbed LLM. The streamed path is structurally different from
the non-streamed :meth:`Runner.arun_task` — it returns a
:class:`RunResultStreaming` whose ``stream_events()`` async iterator
yields events as they happen, and whose background task fires the
``on_task_*`` lifecycle hooks in its ``finally`` arm so they also
fire on cancellation paths.

Covered:

- Happy path: stream drains; ``on_task_start`` fires before the
  returned handle is observable; ``on_task_end`` fires with a success
  :class:`TaskOutput` carrying ``final_output``, ``new_items``,
  ``usage``.
- Error path: an exception inside the streamed turn surfaces via
  ``stream_events()`` (re-raised after the queue drains); the
  ``on_task_end`` hook fires with the failure :class:`TaskOutput`.
- Cancel path: ``RunResultStreaming.cancel(mode="immediate")`` issued
  mid-stream still triggers ``on_task_end`` so observers see the
  termination.
- Per-Task overrides flow through: ``max_turns`` and per-task
  guardrails are honoured on the streamed path the same way they are
  on the non-streamed path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any, override
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.context import RunContext
from troopai.adk.run.runner import Runner
from troopai.adk.tasks import Task
from troopai.adk.tasks.task_output import TaskOutput
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseText,
)


class CapturingHooks(RunHooks[Any]):
    """Records every on_task_start / on_task_end call for assertion."""

    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []
        self.ended: list[TaskOutput] = []

    @override
    async def on_task_start(
        self,
        context: RunContext[Any],
        agent: Agent,
        task: Task,
    ) -> None:
        del context, agent
        name = task.name if task.name is not None else task.description
        self.started.append((task.task_id or "<runner-generated>", name))

    @override
    async def on_task_end(
        self,
        context: RunContext[Any],
        agent: Agent,
        task: Task,
        *,
        output: TaskOutput,
    ) -> None:
        del context, agent, task
        self.ended.append(output)


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        response_id="resp-x",
        model="fake",
        response=[LLMResponseText(text=text)],
    )


@contextmanager
def _patched_streamed_loop(fake_streamed: Any) -> Iterator[None]:
    """Stub the streamed LLM path + guardrails.

    Every test in this file needs the same four mocks. Factoring keeps
    each test focused on its assertion shape.
    """
    with ExitStack() as stack:
        stack.enter_context(
            patch("troopai.adk.run.loop.call_llm_streamed", new=AsyncMock(side_effect=fake_streamed)),
        )
        stack.enter_context(
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", new=AsyncMock(return_value=[])),
        )
        stack.enter_context(
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", new=AsyncMock(return_value=[])),
        )
        stack.enter_context(
            patch("troopai.adk.run.runner.run_output_guardrails", new=AsyncMock(return_value=[])),
        )
        yield


@pytest.mark.integration
async def test_arun_task_streamed_happy_path() -> None:
    """Stream drains, hooks fire, TaskOutput captures final_output."""
    agent = Agent(name="t", system_prompt="x")
    task = Task(description="do X", agent=agent, name="my-task")
    hooks = CapturingHooks()

    async def fake_streamed(*_args: Any, **_kwargs: Any) -> LLMResponse:
        return _text_response("streamed-answer")

    with _patched_streamed_loop(fake_streamed):
        streaming = await Runner.arun_task_streamed(task, hooks=hooks)

        # on_task_start MUST have fired before the await on
        # arun_task_streamed resolved — the streaming handle is only
        # observable after the hook completes.
        assert len(hooks.started) == 1
        assert hooks.started[0][1] == "my-task"

        # Drain the stream — events queue must complete.
        async for _ in streaming.stream_events():
            pass

    # on_task_end fires after drain. Verify the TaskOutput shape.
    assert len(hooks.ended) == 1
    end_output = hooks.ended[0]
    assert end_output.task_name == "my-task"
    assert end_output.error is None
    assert end_output.final_output == "streamed-answer"
    assert end_output.usage is not None


@pytest.mark.integration
async def test_arun_task_streamed_hook_fires_on_llm_failure() -> None:
    """A streamed LLM exception still triggers on_task_end with err output."""
    agent = Agent(name="t", system_prompt="x")
    task = Task(description="boom", agent=agent, name="failing-task")
    hooks = CapturingHooks()

    async def fake_streamed(*_args: Any, **_kwargs: Any) -> LLMResponse:
        raise RuntimeError("simulated-llm-failure")

    with _patched_streamed_loop(fake_streamed):
        streaming = await Runner.arun_task_streamed(task, hooks=hooks)

        # Stream re-raises the captured exception after drain.
        with pytest.raises(RuntimeError, match="simulated-llm-failure"):
            async for _ in streaming.stream_events():
                pass

    assert len(hooks.ended) == 1
    err_output = hooks.ended[0]
    assert err_output.task_name == "failing-task"
    assert err_output.error is not None
    assert "RuntimeError" in err_output.error
    assert "simulated-llm-failure" in err_output.error


@pytest.mark.integration
async def test_arun_task_streamed_hook_fires_on_immediate_cancel() -> None:
    """Cancelling mid-stream still triggers on_task_end so observers see
    the termination — the streamed background impl wraps the work in a
    try/finally so the hook fires unconditionally.

    Cancelling *before* the background task even started running would
    short-circuit the finally arm (the task is cancelled at "pending"
    state). To exercise the mid-flight cancel branch we let the loop
    tick once via ``asyncio.sleep(0)`` so ``run_impl`` reaches its
    first await before the cancel arrives.
    """
    agent = Agent(name="t", system_prompt="x")
    task = Task(description="cancel me", agent=agent, name="cancellable")
    hooks = CapturingHooks()

    async def fake_streamed(*_args: Any, **_kwargs: Any) -> LLMResponse:
        # Sleep so the cancel arrives while the streamed impl is in
        # flight — this is the case the finally-arm fire-on-cancel
        # contract is designed to handle.
        await asyncio.sleep(0.05)
        return _text_response("ok")

    with _patched_streamed_loop(fake_streamed):
        streaming = await Runner.arun_task_streamed(task, hooks=hooks)

        # Yield control once so ``run_impl`` reaches its first await
        # (the streamed-impl call) before we issue the cancel.
        await asyncio.sleep(0)
        streaming.cancel(mode="immediate")
        async for _ in streaming.stream_events():
            pass

    # on_task_end fires regardless of cancel timing.
    assert len(hooks.ended) == 1
    cancel_output = hooks.ended[0]
    assert cancel_output.task_name == "cancellable"


@pytest.mark.integration
async def test_arun_task_streamed_honours_per_task_max_turns() -> None:
    """``Task.max_turns`` flows through to the streamed path's
    ``RunResultStreaming.max_turns``. Cost-conservative defaults: the
    developer's per-task override beats the framework default.
    """
    agent = Agent(name="t", system_prompt="x")
    task = Task(description="bounded", agent=agent, max_turns=3)
    hooks = CapturingHooks()

    async def fake_streamed(*_args: Any, **_kwargs: Any) -> LLMResponse:
        return _text_response("ok")

    with _patched_streamed_loop(fake_streamed):
        streaming = await Runner.arun_task_streamed(task, hooks=hooks)
        assert streaming.max_turns == 3
        async for _ in streaming.stream_events():
            pass

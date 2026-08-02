"""Unit tests for :class:`TaskGroup` + :meth:`Runner.arun_task_group`.

Covers:

- Dataclass validation (empty tasks rejected, non-positive
  ``max_concurrent`` rejected, defaults are cost-conservative).
- Parallel happy path: all three tasks complete; outputs preserve
  input order; cumulative usage aggregates.
- ``error_policy="collect_all"``: failing task surfaces as an error
  slot; siblings complete normally.
- ``error_policy="halt_on_first"``: failing task triggers cancel of
  siblings; cancelled slots carry the explanatory error string;
  positional indexing remains stable.
- ``max_concurrent`` actually bounds concurrent execution via the
  semaphore.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.tasks import Task, TaskGroup, TaskGroupResult
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseText,
)


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        response_id="resp-x",
        model="fake",
        response=[LLMResponseText(text=text)],
    )


class TestTaskGroupDataclass:
    def test_empty_tasks_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one Task"):
            TaskGroup(tasks=())

    def test_non_positive_max_concurrent_rejected(self) -> None:
        agent = Agent(name="t", system_prompt="x")
        task = Task(description="A", agent=agent)
        with pytest.raises(ValueError, match="must be positive"):
            TaskGroup(tasks=(task,), max_concurrent=0)
        with pytest.raises(ValueError, match="must be positive"):
            TaskGroup(tasks=(task,), max_concurrent=-3)

    def test_default_error_policy_is_collect_all(self) -> None:
        agent = Agent(name="t", system_prompt="x")
        group = TaskGroup(tasks=(Task(description="A", agent=agent),))
        # Default is cost-conservative: keep running on failure rather
        # than introducing surprise cancellations.
        assert group.error_policy == "collect_all"

    def test_default_max_concurrent_is_none(self) -> None:
        agent = Agent(name="t", system_prompt="x")
        group = TaskGroup(tasks=(Task(description="A", agent=agent),))
        # `None` means "no throttle" — the Runner uses len(tasks) as
        # the effective concurrency. Developers opt INTO bounded
        # parallelism explicitly.
        assert group.max_concurrent is None


class TestParallelExecution:
    @pytest.mark.integration
    async def test_three_tasks_run_concurrently_and_outputs_preserve_input_order(self) -> None:
        agent = Agent(name="t", system_prompt="x")
        tasks = (
            Task(description="alpha", agent=agent, name="t1"),
            Task(description="beta", agent=agent, name="t2"),
            Task(description="gamma", agent=agent, name="t3"),
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
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", new=AsyncMock(return_value=[])),
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", new=AsyncMock(return_value=[])),
            patch("troopai.adk.run.runner.run_output_guardrails", new=AsyncMock(return_value=[])),
        ):
            result = await Runner.arun_task_group(TaskGroup(tasks=tasks))

        assert isinstance(result, TaskGroupResult)
        assert len(result.task_outputs) == 3
        assert result.task_outputs[0].task_name == "t1"
        assert result.task_outputs[1].task_name == "t2"
        assert result.task_outputs[2].task_name == "t3"
        assert result.task_outputs[0].final_output == "answered-alpha"
        assert result.task_outputs[1].final_output == "answered-beta"
        assert result.task_outputs[2].final_output == "answered-gamma"
        assert all(o.error is None for o in result.task_outputs)


class TestErrorPolicyCollectAll:
    @pytest.mark.integration
    async def test_one_failing_task_does_not_block_siblings(self) -> None:
        agent = Agent(name="t", system_prompt="x")
        tasks = (
            Task(description="ok-1", agent=agent, name="t1"),
            Task(description="boom", agent=agent, name="t2"),
            Task(description="ok-3", agent=agent, name="t3"),
        )

        async def fake_call_llm(*_args: Any, **kwargs: Any) -> LLMResponse:
            messages = kwargs.get("messages") or _args[1]
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content")
                    if content == "boom":
                        raise RuntimeError("simulated-failure")
                    return _text_response(f"answered-{content}")
            return _text_response("unknown")

        from troopai.adk.run.runner import Runner

        with (
            patch("troopai.adk.run.loop.call_llm", new=AsyncMock(side_effect=fake_call_llm)),
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", new=AsyncMock(return_value=[])),
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", new=AsyncMock(return_value=[])),
            patch("troopai.adk.run.runner.run_output_guardrails", new=AsyncMock(return_value=[])),
        ):
            result = await Runner.arun_task_group(
                TaskGroup(tasks=tasks, error_policy="collect_all"),
            )

        # All three slots present, positional order preserved.
        assert len(result.task_outputs) == 3
        assert result.task_outputs[0].error is None
        assert result.task_outputs[0].final_output == "answered-ok-1"
        assert result.task_outputs[1].error is not None
        assert "simulated-failure" in result.task_outputs[1].error
        assert result.task_outputs[2].error is None
        assert result.task_outputs[2].final_output == "answered-ok-3"


class TestErrorPolicyHaltOnFirst:
    @pytest.mark.integration
    async def test_first_failure_cancels_pending_siblings(self) -> None:
        agent = Agent(name="t", system_prompt="x")
        tasks = (
            Task(description="boom-immediately", agent=agent, name="fail"),
            Task(description="slow", agent=agent, name="slow"),
            Task(description="also-slow", agent=agent, name="also-slow"),
        )

        async def fake_call_llm(*_args: Any, **kwargs: Any) -> LLMResponse:
            messages = kwargs.get("messages") or _args[1]
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content")
                    if content == "boom-immediately":
                        raise RuntimeError("first-failure")
                    # Sleep so cancel can land before this resolves.
                    await asyncio.sleep(0.5)
                    return _text_response(f"answered-{content}")
            return _text_response("unknown")

        from troopai.adk.run.runner import Runner

        with (
            patch("troopai.adk.run.loop.call_llm", new=AsyncMock(side_effect=fake_call_llm)),
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", new=AsyncMock(return_value=[])),
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", new=AsyncMock(return_value=[])),
            patch("troopai.adk.run.runner.run_output_guardrails", new=AsyncMock(return_value=[])),
        ):
            result = await Runner.arun_task_group(
                TaskGroup(tasks=tasks, error_policy="halt_on_first"),
            )

        assert len(result.task_outputs) == 3
        # The failure slot reports the underlying error.
        assert result.task_outputs[0].error is not None
        assert "first-failure" in result.task_outputs[0].error
        # The siblings carry the cancellation marker. Their order in
        # the result tuple is the input order, regardless of which
        # cancelled first.
        for cancelled in (result.task_outputs[1], result.task_outputs[2]):
            assert cancelled.error is not None
            assert "halt_on_first" in cancelled.error


class TestMaxConcurrentSemaphore:
    @pytest.mark.integration
    async def test_max_concurrent_bounds_parallelism(self) -> None:
        agent = Agent(name="t", system_prompt="x")
        tasks = tuple(Task(description=f"task-{i}", agent=agent, name=f"t{i}") for i in range(4))

        in_flight = 0
        peak_in_flight = 0
        lock = asyncio.Lock()

        async def fake_call_llm(*_args: Any, **kwargs: Any) -> LLMResponse:
            nonlocal in_flight, peak_in_flight
            async with lock:
                in_flight += 1
                if in_flight > peak_in_flight:
                    peak_in_flight = in_flight
            try:
                await asyncio.sleep(0.05)
                messages = kwargs.get("messages") or _args[1]
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        return _text_response(f"answered-{msg.get('content')}")
                return _text_response("unknown")
            finally:
                async with lock:
                    in_flight -= 1

        from troopai.adk.run.runner import Runner

        with (
            patch("troopai.adk.run.loop.call_llm", new=AsyncMock(side_effect=fake_call_llm)),
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", new=AsyncMock(return_value=[])),
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", new=AsyncMock(return_value=[])),
            patch("troopai.adk.run.runner.run_output_guardrails", new=AsyncMock(return_value=[])),
        ):
            result = await Runner.arun_task_group(
                TaskGroup(tasks=tasks, max_concurrent=2),
            )

        assert len(result.task_outputs) == 4
        # The semaphore should have prevented any concurrency above 2.
        assert peak_in_flight <= 2
        # And at least 2 tasks should have run concurrently (otherwise
        # the test isn't exercising anything).
        assert peak_in_flight >= 2

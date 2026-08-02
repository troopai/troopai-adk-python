"""Unit tests for :meth:`RunHooks.on_task_start` / :meth:`on_task_end`.

The lifecycle hooks MUST fire exactly once per :meth:`Runner.arun_task`
invocation — start before inner arun, end after (in both success and
exception paths).
"""

from __future__ import annotations

from typing import Any, override
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.context import RunContext
from troopai.adk.tasks import Task
from troopai.adk.tasks.task_output import TaskOutput
from troopai.adk.types.run.run_result import RunResult


class CapturingHooks(RunHooks[Any]):
    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []  # (task_id, task_name)
        self.ended: list[tuple[str, str, bool, str | None]] = []
        # (task_id, task_name, success, error)

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
        self.ended.append((output.task_id, output.task_name, output.error is None, output.error))


async def test_hooks_fire_on_success() -> None:
    agent = Agent(name="t", system_prompt="x")
    task = Task(description="do X", agent=agent, name="my-task")
    hooks = CapturingHooks()

    async def fake_arun(*_a: Any, **_kw: Any) -> RunResult:
        return RunResult(
            final_output="ok",
            user_prompt="ignored",
            new_items=[],
            context=RunContext.make(None),
        )

    from troopai.adk.run.runner import Runner

    with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake_arun)):
        output = await Runner.arun_task(task, hooks=hooks)

    assert len(hooks.started) == 1
    assert hooks.started[0][1] == "my-task"
    assert len(hooks.ended) == 1
    end_id, end_name, success, error = hooks.ended[0]
    assert end_name == "my-task"
    assert success is True
    assert error is None
    assert output.task_id == end_id


async def test_hooks_fire_on_exception() -> None:
    agent = Agent(name="t", system_prompt="x")
    task = Task(description="do X", agent=agent, name="my-task")
    hooks = CapturingHooks()

    async def fake_arun(*_a: Any, **_kw: Any) -> RunResult:
        raise RuntimeError("boom")

    from troopai.adk.run.runner import Runner

    with (
        patch.object(Runner, "arun", new=AsyncMock(side_effect=fake_arun)),
        pytest.raises(RuntimeError, match="boom"),
    ):
        await Runner.arun_task(task, hooks=hooks)

    # on_task_end MUST fire even on exception, with error captured.
    assert len(hooks.started) == 1
    assert len(hooks.ended) == 1
    _, end_name, success, error = hooks.ended[0]
    assert end_name == "my-task"
    assert success is False
    assert error is not None
    assert "RuntimeError: boom" in error


async def test_explicit_task_id_flows_to_hooks() -> None:
    agent = Agent(name="t", system_prompt="x")
    task = Task(description="do X", agent=agent, task_id="custom-id", name="custom")
    hooks = CapturingHooks()

    async def fake_arun(*_a: Any, **_kw: Any) -> RunResult:
        return RunResult(
            final_output="ok",
            user_prompt="ignored",
            new_items=[],
            context=RunContext.make(None),
        )

    from troopai.adk.run.runner import Runner

    with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake_arun)):
        output = await Runner.arun_task(task, hooks=hooks)

    assert output.task_id == "custom-id"
    assert hooks.ended[0][0] == "custom-id"

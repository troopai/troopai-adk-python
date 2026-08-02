"""Unit tests for :attr:`Task.skip_if` semantics under
:meth:`Runner.arun_task_pipeline`.

Skipped tasks MUST remain in :attr:`TaskPipelineResult.task_outputs`
with ``skipped=True`` so positional indexing is stable. The
``chain_inputs`` formatter receives every prior output, including
skipped ones.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from troopai.adk.agents.agent import Agent
from troopai.adk.run.context import RunContext
from troopai.adk.tasks import Task, TaskPipeline
from troopai.adk.tasks.task_output import TaskOutput
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


async def test_skip_if_true_inserts_skipped_slot() -> None:
    """When skip_if returns True, a TaskOutput(skipped=True) slot
    is inserted and the underlying arun is NEVER called."""
    agent = _agent()
    t1 = Task(description="first", agent=agent)
    t2 = Task(
        description="second",
        agent=agent,
        skip_if=lambda prior: True,  # always skip
    )
    t3 = Task(description="third", agent=agent)

    call_count = {"n": 0}

    async def fake_arun(*_args: Any, **_kwargs: Any) -> RunResult:
        call_count["n"] += 1
        return _run_result(f"out_{call_count['n']}")

    from troopai.adk.run.runner import Runner

    with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake_arun)):
        result = await Runner.arun_task_pipeline(
            TaskPipeline(tasks=(t1, t2, t3)),
        )

    assert len(result.task_outputs) == 3
    assert result.task_outputs[0].skipped is False
    assert result.task_outputs[1].skipped is True
    assert result.task_outputs[1].final_output is None
    assert result.task_outputs[1].new_items == ()
    assert result.task_outputs[2].skipped is False
    # arun was called only for the two non-skipped tasks.
    assert call_count["n"] == 2


async def test_skip_if_receives_prior_outputs_in_order() -> None:
    """skip_if MUST receive a tuple of prior TaskOutputs in
    pipeline order."""
    agent = _agent()
    captured: list[tuple[TaskOutput, ...]] = []

    def capture_and_skip(prior: Any) -> bool:
        captured.append(tuple(prior))
        return False

    t1 = Task(description="A", agent=agent)
    t2 = Task(description="B", agent=agent, skip_if=capture_and_skip)
    t3 = Task(description="C", agent=agent, skip_if=capture_and_skip)

    from troopai.adk.run.runner import Runner

    counter = {"n": 0}

    async def fake_arun(*_args: Any, **_kwargs: Any) -> RunResult:
        counter["n"] += 1
        return _run_result(f"r{counter['n']}")

    with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake_arun)):
        await Runner.arun_task_pipeline(TaskPipeline(tasks=(t1, t2, t3)))

    # t2.skip_if received prior outputs = (t1_output,)
    # t3.skip_if received prior outputs = (t1_output, t2_output)
    assert len(captured) == 2
    assert len(captured[0]) == 1
    assert len(captured[1]) == 2
    assert captured[1][0].final_output == "r1"
    assert captured[1][1].final_output == "r2"


async def test_skipped_slot_preserves_positional_indexing() -> None:
    """A skipped task remains in ``TaskPipelineResult.task_outputs``
    in its original position so downstream consumers (skip_if
    predicates, audit logs, indexing) see a stable layout — slots are
    never silently dropped."""
    agent = _agent()
    t1 = Task(description="A", agent=agent)
    t2 = Task(description="B", agent=agent, skip_if=lambda _prior: True)
    t3 = Task(description="C", agent=agent)

    counter = {"n": 0}

    async def fake_arun(*_args: Any, **_kwargs: Any) -> RunResult:
        counter["n"] += 1
        return _run_result(f"r{counter['n']}")

    from troopai.adk.run.runner import Runner

    with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake_arun)):
        result = await Runner.arun_task_pipeline(TaskPipeline(tasks=(t1, t2, t3)))

    # All three slots present, in input order.
    assert len(result.task_outputs) == 3
    assert result.task_outputs[0].skipped is False
    assert result.task_outputs[1].skipped is True
    assert result.task_outputs[2].skipped is False
    # The skipped slot's task_name still reflects task t2 (not collapsed).
    assert result.task_outputs[1].task_name == "B"

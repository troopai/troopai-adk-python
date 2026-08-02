"""Unit tests for :attr:`Task.depends_on` declarative DAG ordering.

Covered:

- A pipeline with no ``depends_on`` keeps sequential-by-declaration
  semantics (the cost-conservative default path).
- A pipeline with ``depends_on`` executes in topological order.
- Independent same-depth tasks run concurrently.
- ``TaskPipeline.topological_levels()`` returns deterministic
  depth-grouped tuples.
- Resume via ``TaskPipelineState.completed_task_ids`` skips already-
  finished tasks and re-runs only the rest.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.context import RunContext
from troopai.adk.tasks import Task, TaskOutput, TaskPipeline, TaskPipelineState
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


class TestTopologicalLevels:
    def test_no_depends_on_returns_single_level_in_order(self) -> None:
        agent = _agent()
        t0 = Task(description="0", agent=agent, task_id="t0")
        t1 = Task(description="1", agent=agent, task_id="t1")
        pipeline = TaskPipeline(tasks=(t0, t1))
        levels = pipeline.topological_levels()
        assert len(levels) == 1
        assert tuple(t.task_id for t in levels[0]) == ("t0", "t1")

    def test_linear_chain_yields_one_per_level(self) -> None:
        agent = _agent()
        a = Task(description="a", agent=agent, task_id="a")
        b = Task(description="b", agent=agent, task_id="b", depends_on=("a",))
        c = Task(description="c", agent=agent, task_id="c", depends_on=("b",))
        pipeline = TaskPipeline(tasks=(a, b, c))
        levels = pipeline.topological_levels()
        assert tuple(tuple(t.task_id for t in lvl) for lvl in levels) == (("a",), ("b",), ("c",))

    def test_diamond_yields_two_at_middle_level(self) -> None:
        # a -> b; a -> c; b,c -> d
        agent = _agent()
        a = Task(description="a", agent=agent, task_id="a")
        b = Task(description="b", agent=agent, task_id="b", depends_on=("a",))
        c = Task(description="c", agent=agent, task_id="c", depends_on=("a",))
        d = Task(description="d", agent=agent, task_id="d", depends_on=("b", "c"))
        pipeline = TaskPipeline(tasks=(a, b, c, d))
        levels = pipeline.topological_levels()
        assert tuple(tuple(t.task_id for t in lvl) for lvl in levels) == (
            ("a",),
            ("b", "c"),
            ("d",),
        )


class TestPipelineRunDAG:
    async def test_diamond_runs_in_topological_order(self) -> None:
        agent = _agent()
        order: list[str] = []

        a = Task(description="a", agent=agent, task_id="a")
        b = Task(description="b", agent=agent, task_id="b", depends_on=("a",))
        c = Task(description="c", agent=agent, task_id="c", depends_on=("a",))
        d = Task(description="d", agent=agent, task_id="d", depends_on=("b", "c"))
        pipeline = TaskPipeline(tasks=(a, b, c, d))

        from troopai.adk.run.runner import Runner

        # AsyncMock(side_effect=...) treats the patched arun as unbound;
        # signature is (agent, prompt, **kwargs).
        async def fake(_agent: Any, prompt: Any, **_kwargs: Any) -> RunResult:
            order.append(prompt)
            await asyncio.sleep(0.01)
            return _run_result("ok")

        with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake)):
            result = await Runner.arun_task_pipeline(pipeline)

        assert len(result.task_outputs) == 4
        # Topological constraint: 'a' completes before 'b' and 'c' starts;
        # 'b' and 'c' both complete before 'd' starts.
        assert order[0] == "a"
        assert order[-1] == "d"
        assert set(order[1:3]) == {"b", "c"}

    def test_no_id_task_in_dag_pipeline_rejected(self) -> None:
        """A no-task_id task in a pipeline where any task uses depends_on
        must be rejected at construction time.

        Previously, a no-id task would silently run as the appended final DAG
        level under a generated UUID. This was ambiguous: the task cannot be
        addressed via ``completed_task_ids`` on resume, and its position in the
        final level is undefined when tasks are reordered. The definition error
        forces developers to always assign stable IDs in DAG pipelines.
        """
        from troopai.adk.tasks.topology import TaskPipelineDefinitionError

        agent = _agent()
        a = Task(description="a", agent=agent, task_id="a")
        b = Task(description="b", agent=agent, task_id="b", depends_on=("a",))
        # No task_id, no depends_on — was silently accepted before this fix.
        independent = Task(description="independent", agent=agent)

        with pytest.raises(TaskPipelineDefinitionError, match="task_id"):
            TaskPipeline(tasks=(a, b, independent))

    async def test_same_level_tasks_run_concurrently(self) -> None:
        agent = _agent()
        active = {"count": 0, "peak": 0}

        async def fake(_agent: Any, _prompt: Any, **_kwargs: Any) -> RunResult:
            active["count"] += 1
            active["peak"] = max(active["peak"], active["count"])
            await asyncio.sleep(0.02)
            active["count"] -= 1
            return _run_result("ok")

        a = Task(description="a", agent=agent, task_id="a")
        b1 = Task(description="b1", agent=agent, task_id="b1", depends_on=("a",))
        b2 = Task(description="b2", agent=agent, task_id="b2", depends_on=("a",))
        b3 = Task(description="b3", agent=agent, task_id="b3", depends_on=("a",))

        from troopai.adk.run.runner import Runner

        with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake)):
            await Runner.arun_task_pipeline(TaskPipeline(tasks=(a, b1, b2, b3)))

        assert active["peak"] >= 2  # at least two of b1/b2/b3 ran concurrently


class TestDAGResume:
    async def test_completed_task_ids_skips_those_tasks(self) -> None:
        agent = _agent()
        called: list[str] = []

        async def fake(_agent: Any, prompt: Any, **_kwargs: Any) -> RunResult:
            called.append(prompt)
            return _run_result("ok")

        a = Task(description="a", agent=agent, task_id="a")
        b = Task(description="b", agent=agent, task_id="b", depends_on=("a",))
        c = Task(description="c", agent=agent, task_id="c", depends_on=("a",))
        pipeline = TaskPipeline(tasks=(a, b, c))

        # Pretend "a" already completed before the checkpoint.
        prior_slot = TaskOutput(task_id="a", task_name="a", final_output="ok")
        state = TaskPipelineState(
            pipeline_id="p",
            slots=(prior_slot,),
            resume_index=1,
            completed_task_ids=("a",),
        )

        from troopai.adk.run.runner import Runner

        with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake)):
            result = await Runner.arun_task_pipeline_from_state(pipeline, state)

        # 'a' did NOT re-run; only 'b' and 'c'.
        assert "a" not in called
        assert set(called) == {"b", "c"}
        assert len(result.task_outputs) == 3

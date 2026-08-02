"""Negative-path tests for :class:`TaskPipeline` DAG validation.

Covers the four failure modes :meth:`TaskPipeline.topological_levels`
detects at construction time: missing ``task_id`` on a task with
``depends_on``, duplicate ``task_id``s, references to unknown IDs,
and cycles.
"""

from __future__ import annotations

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.tasks import Task, TaskPipeline, TaskPipelineDefinitionError


def _agent() -> Agent:
    return Agent(name="t", system_prompt="x")


class TestDAGDefinitionErrors:
    def test_missing_task_id_with_depends_on_rejected(self) -> None:
        # Tasks that declare depends_on MUST have an explicit task_id
        # so the dependency resolver can name them.
        agent = _agent()
        a = Task(description="a", agent=agent, task_id="a")
        # The downstream task has depends_on but no task_id of its own.
        with pytest.raises(TaskPipelineDefinitionError, match="depends_on"):
            TaskPipeline(
                tasks=(
                    a,
                    Task(description="b", agent=agent, depends_on=("a",)),
                ),
            )

    def test_duplicate_task_id_rejected(self) -> None:
        agent = _agent()
        a1 = Task(description="a1", agent=agent, task_id="a")
        a2 = Task(description="a2", agent=agent, task_id="a")  # collision
        b = Task(description="b", agent=agent, task_id="b", depends_on=("a",))
        with pytest.raises(TaskPipelineDefinitionError, match="duplicate task_id"):
            TaskPipeline(tasks=(a1, a2, b))

    def test_unknown_dependency_id_rejected(self) -> None:
        agent = _agent()
        a = Task(description="a", agent=agent, task_id="a")
        b = Task(description="b", agent=agent, task_id="b", depends_on=("ghost",))
        with pytest.raises(TaskPipelineDefinitionError, match="ghost"):
            TaskPipeline(tasks=(a, b))

    def test_simple_cycle_rejected(self) -> None:
        # a depends on b; b depends on a — cycle.
        agent = _agent()
        a = Task(description="a", agent=agent, task_id="a", depends_on=("b",))
        b = Task(description="b", agent=agent, task_id="b", depends_on=("a",))
        with pytest.raises(TaskPipelineDefinitionError, match="cycle"):
            TaskPipeline(tasks=(a, b))

    def test_self_cycle_rejected(self) -> None:
        # A task can't depend on itself.
        agent = _agent()
        a = Task(description="a", agent=agent, task_id="a", depends_on=("a",))
        with pytest.raises(TaskPipelineDefinitionError, match="cycle"):
            TaskPipeline(tasks=(a,))

    def test_three_cycle_rejected(self) -> None:
        agent = _agent()
        a = Task(description="a", agent=agent, task_id="a", depends_on=("c",))
        b = Task(description="b", agent=agent, task_id="b", depends_on=("a",))
        c = Task(description="c", agent=agent, task_id="c", depends_on=("b",))
        with pytest.raises(TaskPipelineDefinitionError, match="cycle"):
            TaskPipeline(tasks=(a, b, c))

    def test_no_id_leaf_in_dag_pipeline_rejected(self) -> None:
        """A no-task_id task in a pipeline where any task uses depends_on
        must be rejected. A no-id leaf cannot be addressed via
        completed_task_ids on resume and its position is ambiguous.
        """
        agent = _agent()
        a = Task(description="a", agent=agent, task_id="a")
        b = Task(description="b", agent=agent, task_id="b", depends_on=("a",))
        # This task has no task_id but lives in a DAG pipeline.
        leaf = Task(description="leaf", agent=agent)
        with pytest.raises(TaskPipelineDefinitionError, match="task_id"):
            TaskPipeline(tasks=(a, b, leaf))

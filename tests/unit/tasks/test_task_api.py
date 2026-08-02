"""Unit tests for the tasks-module developer-facing API surface.

Covers the human-readable ``__repr__`` one-liners on ``Task``,
``TaskDependency``, ``TaskPipeline`` / ``TaskPipelineResult``,
``TaskGroup`` / ``TaskGroupResult``, ``TaskOutput``, and
``TaskPipelineState``, plus the package-level export contract of
:mod:`troopai.adk.tasks`. Reprs are one-liners: full descriptions,
prompts, and item trails never leak into them; previews are capped at
60 chars with newlines stripped and a ``…`` ellipsis.
"""

from __future__ import annotations

import troopai.adk.tasks as tasks_pkg
from troopai.adk.agents.agent import Agent
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.tasks import (
    ErrorPolicy,
    Task,
    TaskDependency,
    TaskGroup,
    TaskGroupResult,
    TaskOutput,
    TaskPipeline,
    TaskPipelineDefinitionError,
    TaskPipelineResult,
    TaskPipelineState,
)
from troopai.adk.tasks.task_filters import forward_final_output


def _agent(name: str = "researcher") -> Agent:
    return Agent(name=name, system_prompt="noop")


def _task(description: str = "do X", **kwargs: object) -> Task:
    return Task(description=description, agent=_agent(), **kwargs)  # type: ignore[arg-type]


class TestPackageExports:
    def test_all_exports(self) -> None:
        """The package export surface is exactly the 12 documented names."""
        assert set(tasks_pkg.__all__) == {
            "ErrorPolicy",
            "Task",
            "TaskDependency",
            "TaskGroup",
            "TaskGroupResult",
            "TaskInputData",
            "TaskInputFilter",
            "TaskOutput",
            "TaskPipeline",
            "TaskPipelineDefinitionError",
            "TaskPipelineResult",
            "TaskPipelineState",
        }

    def test_all_exports_resolve(self) -> None:
        """Every name in ``__all__`` resolves to a module attribute."""
        for name in tasks_pkg.__all__:
            assert getattr(tasks_pkg, name) is not None

    def test_top_level_parity_exports(self) -> None:
        """``ErrorPolicy`` / ``TaskPipelineDefinitionError`` are importable from ``troopai.adk``."""
        import troopai.adk as adk

        assert adk.ErrorPolicy is ErrorPolicy
        assert adk.TaskPipelineDefinitionError is TaskPipelineDefinitionError
        assert "ErrorPolicy" in adk.__all__
        assert "TaskPipelineDefinitionError" in adk.__all__

    def test_definition_error_is_user_error(self) -> None:
        from troopai.adk.exceptions.exceptions import UserError

        assert issubclass(TaskPipelineDefinitionError, UserError)


class TestTaskRepr:
    def test_named_task(self) -> None:
        task = _task("Summarize the notes.", name="facts")
        assert repr(task) == "Task(name='facts', agent='researcher')"

    def test_unnamed_task_shows_description_preview(self) -> None:
        task = _task("Summarize the notes.")
        assert repr(task) == "Task(description='Summarize the notes.', agent='researcher')"

    def test_description_preview_capped_at_60_chars(self) -> None:
        task = _task("x" * 100)
        assert repr(task) == f"Task(description='{'x' * 59}…', agent='researcher')"

    def test_description_preview_strips_newlines(self) -> None:
        task = _task("line one\nline two")
        assert repr(task) == "Task(description='line one line two', agent='researcher')"

    def test_agent_without_name_falls_back_to_class_name(self) -> None:
        swarm = Swarm(
            members=(_agent("a"), _agent("b")),
            entry=_agent("a"),
            policy=RoundRobinPolicy(),
            termination=MaxTurnsTermination(2),
        )
        task = Task(description="do X", agent=swarm)
        assert repr(task) == "Task(description='do X', agent=Swarm)"

    def test_depends_on_shows_count(self) -> None:
        upstream = _task("a", task_id="a")
        task = _task("b", task_id="b", depends_on=(upstream, "c"))
        assert repr(task) == "Task(description='b', agent='researcher', depends_on=2)"

    def test_skip_if_shown_when_set(self) -> None:
        task = _task("b", skip_if=lambda prior: False)
        assert repr(task) == "Task(description='b', agent='researcher', skip_if=True)"


class TestTaskDependencyRepr:
    def test_task_id_preferred(self) -> None:
        upstream = _task("d", task_id="facts", name="Facts")
        assert repr(TaskDependency(task=upstream)) == "TaskDependency(task='facts')"

    def test_name_when_no_task_id(self) -> None:
        upstream = _task("d", name="Facts")
        assert repr(TaskDependency(task=upstream)) == "TaskDependency(task='Facts')"

    def test_description_preview_when_no_id_or_name(self) -> None:
        upstream = _task("d")
        assert repr(TaskDependency(task=upstream)) == "TaskDependency(task='d')"

    def test_string_reference(self) -> None:
        assert repr(TaskDependency(task="facts")) == "TaskDependency(task='facts')"

    def test_input_filter_shown_by_function_name(self) -> None:
        dep = TaskDependency(task="facts", input_filter=forward_final_output)
        assert repr(dep) == "TaskDependency(task='facts', input_filter=forward_final_output)"

    def test_input_filter_omitted_when_none(self) -> None:
        assert "input_filter" not in repr(TaskDependency(task="facts"))

    def test_input_filter_non_function_callable_falls_back_to_class_name(self) -> None:
        class _Filter:
            def __call__(self, data: object) -> object:
                return data

        dep = TaskDependency(task="facts", input_filter=_Filter())  # type: ignore[arg-type]
        assert repr(dep) == "TaskDependency(task='facts', input_filter=_Filter)"


class TestTaskPipelineRepr:
    def test_sequential_pipeline(self) -> None:
        pipeline = TaskPipeline(tasks=(_task("a"), _task("b"), _task("c"), _task("d")))
        assert repr(pipeline) == "TaskPipeline(tasks=4)"

    def test_dag_flag_when_any_task_declares_depends_on(self) -> None:
        a = _task("a", task_id="a")
        b = _task("b", task_id="b", depends_on=("a",))
        pipeline = TaskPipeline(tasks=(a, b))
        assert repr(pipeline) == "TaskPipeline(tasks=2, dag=True)"

    def test_empty_depends_on_is_not_a_dag(self) -> None:
        pipeline = TaskPipeline(tasks=(_task("a", depends_on=()),))
        assert repr(pipeline) == "TaskPipeline(tasks=1)"


class TestTaskPipelineResultRepr:
    def test_counts_and_final_output_preview(self) -> None:
        outputs = (
            TaskOutput(task_id="a", task_name="a"),
            TaskOutput(task_id="b", task_name="b", skipped=True),
            TaskOutput(task_id="c", task_name="c", error="ValueError: boom"),
            TaskOutput(task_id="d", task_name="d", final_output="done"),
        )
        result = TaskPipelineResult(task_outputs=outputs, final_output="done")
        assert repr(result) == "TaskPipelineResult(tasks=4, skipped=1, errors=1, final_output='done')"

    def test_final_output_omitted_when_none(self) -> None:
        result = TaskPipelineResult(task_outputs=(TaskOutput(task_id="a", task_name="a"),))
        assert repr(result) == "TaskPipelineResult(tasks=1, skipped=0, errors=0)"

    def test_final_output_preview_capped(self) -> None:
        result = TaskPipelineResult(
            task_outputs=(TaskOutput(task_id="a", task_name="a", final_output="y" * 100),),
            final_output="y" * 100,
        )
        assert repr(result) == (f"TaskPipelineResult(tasks=1, skipped=0, errors=0, final_output='{'y' * 59}…')")


class TestTaskGroupRepr:
    def test_defaults(self) -> None:
        group = TaskGroup(tasks=(_task("a"), _task("b"), _task("c")))
        assert repr(group) == "TaskGroup(tasks=3, error_policy='collect_all')"

    def test_max_concurrent_shown_when_set(self) -> None:
        group = TaskGroup(tasks=(_task("a"),), error_policy="halt_on_first", max_concurrent=2)
        assert repr(group) == "TaskGroup(tasks=1, error_policy='halt_on_first', max_concurrent=2)"

    def test_max_concurrent_omitted_when_none(self) -> None:
        assert "max_concurrent" not in repr(TaskGroup(tasks=(_task("a"),)))


class TestTaskGroupResultRepr:
    def test_counts(self) -> None:
        outputs = (
            TaskOutput(task_id="a", task_name="a", final_output="ok"),
            TaskOutput(task_id="b", task_name="b", error="ValueError: boom"),
        )
        assert repr(TaskGroupResult(task_outputs=outputs)) == "TaskGroupResult(tasks=2, errors=1)"

    def test_no_errors(self) -> None:
        outputs = (TaskOutput(task_id="a", task_name="a", final_output="ok"),)
        assert repr(TaskGroupResult(task_outputs=outputs)) == "TaskGroupResult(tasks=1, errors=0)"


class TestTaskOutputRepr:
    def test_minimal(self) -> None:
        output = TaskOutput(task_id="t1", task_name="one")
        assert repr(output) == "TaskOutput(task_id='t1', task_name='one')"

    def test_final_output_preview(self) -> None:
        output = TaskOutput(task_id="t1", task_name="one", final_output="answer")
        assert repr(output) == "TaskOutput(task_id='t1', task_name='one', final_output='answer')"

    def test_final_output_preview_capped(self) -> None:
        output = TaskOutput(task_id="t1", task_name="one", final_output="y" * 100)
        assert repr(output) == f"TaskOutput(task_id='t1', task_name='one', final_output='{'y' * 59}…')"

    def test_skipped_shown_when_true(self) -> None:
        output = TaskOutput(task_id="t1", task_name="one", skipped=True)
        assert repr(output) == "TaskOutput(task_id='t1', task_name='one', skipped=True)"

    def test_error_shown_as_exception_class_name(self) -> None:
        output = TaskOutput(task_id="t1", task_name="one", error="ValueError: boom")
        assert repr(output) == "TaskOutput(task_id='t1', task_name='one', error=ValueError)"

    def test_error_without_colon_capped(self) -> None:
        output = TaskOutput(task_id="t1", task_name="one", error="e" * 100)
        assert repr(output) == f"TaskOutput(task_id='t1', task_name='one', error={'e' * 59}…)"


class TestTaskPipelineStateRepr:
    def test_with_completed_ids(self) -> None:
        slots = (
            TaskOutput(task_id="a", task_name="a"),
            TaskOutput(task_id="b", task_name="b"),
        )
        state = TaskPipelineState(
            pipeline_id="run-1",
            slots=slots,
            resume_index=2,
            completed_task_ids=("a", "b"),
        )
        assert repr(state) == "TaskPipelineState(pipeline_id='run-1', slots=2, resume_index=2, completed=2)"

    def test_defaults(self) -> None:
        state = TaskPipelineState(pipeline_id="run-1", slots=(), resume_index=0)
        assert repr(state) == "TaskPipelineState(pipeline_id='run-1', slots=0, resume_index=0, completed=0)"

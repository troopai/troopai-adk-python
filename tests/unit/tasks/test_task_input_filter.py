"""Unit tests for :attr:`Task.input_filter` upstream-output forwarding.

Covered:

- ``Task.input_filter=None`` keeps ``description`` verbatim (default).
- ``Task.input_filter=forward_final_output`` injects the upstream's
  final output into the downstream prompt.
- A custom filter has access to upstream ``output`` and ``items`` and
  can shape ``forwarded`` arbitrarily via ``TaskInputData.clone``.
- ``depends_on`` accepts both :class:`Task` instances and ``task_id``
  strings, mixed freely.
- ``TaskInputData.clone`` only accepts ``forwarded`` as a named param
  (no **kwargs; audit fields task_id/output/items are read-only).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.context import RunContext
from troopai.adk.tasks import Task, TaskDependency, TaskInputData, TaskPipeline
from troopai.adk.tasks.task_filters import forward_final_output, keep_last_n
from troopai.adk.types.items.items import UserItem
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


class TestNoFilterKeepsDescription:
    async def test_no_input_filter_passes_description_verbatim(self) -> None:
        agent = _agent()
        captured: list[str] = []

        async def fake(_agent: Any, prompt: Any, **_kwargs: Any) -> RunResult:
            captured.append(prompt)
            return _run_result("ok")

        a = Task(description="A", agent=agent, task_id="a")
        b = Task(description="B description", agent=agent, task_id="b", depends_on=(a,))

        from troopai.adk.run.runner import Runner

        with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake)):
            await Runner.arun_task_pipeline(TaskPipeline(tasks=(a, b)))

        assert captured[1] == "B description"


class TestBuiltinForwardFinalOutput:
    async def test_forward_final_output_injects_upstream_answer(self) -> None:
        agent = _agent()
        captured: list[str] = []

        async def fake(_agent: Any, prompt: Any, **_kwargs: Any) -> RunResult:
            captured.append(prompt)
            return _run_result("UPSTREAM_ANSWER" if prompt == "A" else "ok")

        a = Task(description="A", agent=agent, task_id="a")
        b = Task(
            description="B description",
            agent=agent,
            task_id="b",
            depends_on=(TaskDependency(task=a, input_filter=forward_final_output),),
        )

        from troopai.adk.run.runner import Runner

        with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake)):
            await Runner.arun_task_pipeline(TaskPipeline(tasks=(a, b)))

        downstream_prompt = captured[1]
        # The downstream prompt is a list of LLMInputContentItem messages
        # when input_filter forwards content. Forwarded items precede the
        # description.
        assert isinstance(downstream_prompt, list)
        contents = [msg["content"] for msg in downstream_prompt]
        assert "UPSTREAM_ANSWER" in contents
        assert "B description" in contents
        assert contents.index("UPSTREAM_ANSWER") < contents.index("B description")


class TestCustomFilter:
    async def test_filter_has_full_taskinputdata_access(self) -> None:
        agent = _agent()
        captured: list[str] = []
        seen_data: list[TaskInputData] = []

        async def fake(_agent: Any, prompt: Any, **_kwargs: Any) -> RunResult:
            captured.append(prompt)
            return _run_result("UPSTREAM_OUT")

        def custom_filter(data: TaskInputData) -> TaskInputData:
            seen_data.append(data)
            # Wrap the upstream output with a custom prefix.
            text = f"[from {data.task_id}] {data.output.final_output}"
            return data.clone(forwarded=(UserItem(raw={"role": "user", "content": text}),))

        a = Task(description="A", agent=agent, task_id="a")
        b = Task(
            description="B description",
            agent=agent,
            task_id="b",
            depends_on=(TaskDependency(task=a, input_filter=custom_filter),),
        )

        from troopai.adk.run.runner import Runner

        with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake)):
            await Runner.arun_task_pipeline(TaskPipeline(tasks=(a, b)))

        assert len(seen_data) == 1
        assert seen_data[0].task_id == "a"
        assert seen_data[0].output.final_output == "UPSTREAM_OUT"
        downstream_prompt = captured[1]
        assert isinstance(downstream_prompt, list)
        contents = [msg["content"] for msg in downstream_prompt]
        assert "[from a] UPSTREAM_OUT" in contents
        assert "B description" in contents


class TestKeepLastN:
    def test_keep_last_n_negative_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="keep_last_n"):
            keep_last_n(-1)

    def test_keep_last_n_returns_callable(self) -> None:
        filt = keep_last_n(3)
        assert callable(filt)


class TestDependsOnAcceptsTaskRefs:
    def test_depends_on_with_task_instance(self) -> None:
        agent = _agent()
        a = Task(description="a", agent=agent, task_id="a")
        b = Task(description="b", agent=agent, task_id="b", depends_on=(a,))
        pipeline = TaskPipeline(tasks=(a, b))
        levels = pipeline.topological_levels()
        assert tuple(tuple(t.task_id for t in lvl) for lvl in levels) == (("a",), ("b",))

    def test_depends_on_mixes_task_and_string(self) -> None:
        agent = _agent()
        a = Task(description="a", agent=agent, task_id="a")
        b = Task(description="b", agent=agent, task_id="b", depends_on=(a, "a"))
        pipeline = TaskPipeline(tasks=(a, b))
        levels = pipeline.topological_levels()
        assert levels[1] == (b,)


class TestTaskInputDataClone:
    """Regression tests for the explicit-forwarded-only clone API."""

    def _make_input_data(self) -> TaskInputData:
        from troopai.adk.tasks.task_output import TaskOutput

        output = TaskOutput(task_id="up1", task_name="upstream-1", final_output="ans")
        return TaskInputData(task_id="up1", output=output, items=())

    def test_clone_without_args_copies_unchanged(self) -> None:
        data = self._make_input_data()
        cloned = data.clone()
        assert cloned.task_id == data.task_id
        assert cloned.output is data.output
        assert cloned.items == ()
        assert cloned.forwarded is None

    def test_clone_sets_forwarded(self) -> None:
        data = self._make_input_data()
        item = UserItem(raw={"role": "user", "content": "hello"})
        cloned = data.clone(forwarded=(item,))
        assert cloned.forwarded == (item,)
        # Audit fields unchanged.
        assert cloned.task_id == data.task_id
        assert cloned.output is data.output

    def test_clone_with_none_forwarded_clears_it(self) -> None:
        item = UserItem(raw={"role": "user", "content": "hi"})
        from troopai.adk.tasks.task_output import TaskOutput

        output = TaskOutput(task_id="x", task_name="x")
        data = TaskInputData(task_id="x", output=output, items=(), forwarded=(item,))
        cloned = data.clone(forwarded=None)
        assert cloned.forwarded is None

    def test_clone_does_not_accept_task_id_override(self) -> None:
        """task_id is an audit field and must not be overridable via clone."""
        data = self._make_input_data()
        with pytest.raises(TypeError):
            data.clone(task_id="different")  # type: ignore[call-arg]

    def test_clone_does_not_accept_output_override(self) -> None:
        """output is an audit field and must not be overridable via clone."""
        data = self._make_input_data()
        from troopai.adk.tasks.task_output import TaskOutput

        other_output = TaskOutput(task_id="y", task_name="y")
        with pytest.raises(TypeError):
            data.clone(output=other_output)  # type: ignore[call-arg]

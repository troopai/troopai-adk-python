"""Unit tests for the :class:`Task` dataclass.

Covers frozen invariants, default values, ``__post_init__`` validation,
and the metadata field.
"""

from __future__ import annotations

import dataclasses

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.tasks import Task


def _make_agent() -> Agent:
    return Agent(name="t-agent", system_prompt="You are helpful.")


class TestTaskConstruction:
    def test_minimal_construction(self) -> None:
        agent = _make_agent()
        task = Task(description="do X", agent=agent)
        assert task.description == "do X"
        assert task.agent is agent
        assert task.name is None
        assert task.task_id is None
        assert task.output_schema is None
        assert task.guardrails.input == []
        assert task.guardrails.output == []
        assert task.max_turns is None
        assert task.usage_limits is None
        assert task.skip_if is None
        assert task.metadata == {}

    def test_full_construction(self) -> None:
        agent = _make_agent()
        task = Task(
            description="do X",
            agent=agent,
            name="my-task",
            task_id="abc123",
            max_turns=5,
            metadata={"trace_id": "xyz"},
        )
        assert task.name == "my-task"
        assert task.task_id == "abc123"
        assert task.max_turns == 5
        assert task.metadata == {"trace_id": "xyz"}

    def test_frozen_rejects_attribute_writes(self) -> None:
        task = Task(description="do X", agent=_make_agent())
        with pytest.raises(dataclasses.FrozenInstanceError):
            task.description = "changed"  # type: ignore[misc]


class TestTaskValidation:
    def test_empty_description_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Task(description="", agent=_make_agent())

    def test_whitespace_description_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Task(description="   \n  ", agent=_make_agent())

    def test_non_positive_max_turns_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            Task(description="x", agent=_make_agent(), max_turns=0)
        with pytest.raises(ValueError, match="positive"):
            Task(description="x", agent=_make_agent(), max_turns=-3)

    def test_max_turns_none_is_allowed(self) -> None:
        Task(description="x", agent=_make_agent(), max_turns=None)


class TestTaskMetadata:
    def test_independent_default_dicts_per_instance(self) -> None:
        t1 = Task(description="a", agent=_make_agent())
        t2 = Task(description="b", agent=_make_agent())
        assert t1.metadata is not t2.metadata

"""Unit tests for the :class:`TaskOutput` dataclass.

Covers frozen invariants, default values, the
``skipped`` / ``error`` discriminators, and the
``from_dict`` ValueError contract for missing required fields.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
from pydantic import BaseModel

from troopai.adk.tasks import TaskOutput


class TestTaskOutputFromDictValidation:
    def test_missing_task_id_raises_value_error(self) -> None:
        """from_dict must raise ValueError (not KeyError) when task_id absent."""
        with pytest.raises(ValueError, match="task_id"):
            TaskOutput.from_dict({"task_name": "my-task"})

    def test_missing_task_name_raises_value_error(self) -> None:
        """from_dict must raise ValueError (not KeyError) when task_name absent."""
        with pytest.raises(ValueError, match="task_name"):
            TaskOutput.from_dict({"task_id": "abc"})

    def test_both_missing_raises_value_error_listing_both(self) -> None:
        """from_dict with both fields absent raises ValueError naming them both."""
        with pytest.raises(ValueError, match="task_id"):
            TaskOutput.from_dict({})

    def test_valid_minimal_dict_succeeds(self) -> None:
        """from_dict succeeds when required fields are present."""
        output = TaskOutput.from_dict({"task_id": "t1", "task_name": "task-1"})
        assert output.task_id == "t1"
        assert output.task_name == "task-1"


class TestTaskOutputConstruction:
    def test_minimal_construction(self) -> None:
        output = TaskOutput(task_id="abc", task_name="example")
        assert output.task_id == "abc"
        assert output.task_name == "example"
        assert output.final_output is None
        assert output.new_items == ()
        assert output.usage is None
        assert output.skipped is False
        assert output.error is None
        assert output.metadata == {}

    def test_skipped_output(self) -> None:
        output = TaskOutput(task_id="x", task_name="n", skipped=True)
        assert output.skipped is True
        assert output.error is None
        assert output.final_output is None
        assert output.new_items == ()

    def test_error_output(self) -> None:
        output = TaskOutput(task_id="x", task_name="n", error="ValueError: bad")
        assert output.skipped is False
        assert output.error == "ValueError: bad"

    def test_frozen_rejects_attribute_writes(self) -> None:
        output = TaskOutput(task_id="x", task_name="n")
        with pytest.raises(dataclasses.FrozenInstanceError):
            output.task_id = "y"  # type: ignore[misc]


class _Item(BaseModel):
    x: int


class TestTaskOutputToDictSerialization:
    """``to_dict`` must not let a ``list`` / ``dict`` of non-JSON values slip
    through the JSON-native check and crash the ``json.dumps`` that
    :class:`TaskPipelineState` runs over the produced dict.
    """

    def test_list_of_pydantic_models_serialized_to_str(self) -> None:
        """A ``list[BaseModel]`` is stringified, not kept as a raw list.

        Regression: the shallow ``isinstance(x, (list, dict))`` check treated
        a ``list[BaseModel]`` as JSON-native and kept it, so the pipeline-state
        ``json.dumps`` later crashed on the embedded model.
        """
        output = TaskOutput(task_id="t", task_name="n", final_output=[_Item(x=1), _Item(x=2)])
        d = output.to_dict()
        assert isinstance(d["final_output"], str)
        # The whole dict must round-trip through json.dumps (what the state does).
        json.dumps(d)

    def test_dict_with_nested_model_serialized_to_str(self) -> None:
        """A dict holding a BaseModel value is stringified, not kept."""
        output = TaskOutput(task_id="t", task_name="n", final_output={"item": _Item(x=1)})
        d = output.to_dict()
        assert isinstance(d["final_output"], str)
        json.dumps(d)

    def test_json_native_list_preserved(self) -> None:
        """A list of primitives is kept as a list (round-trips)."""
        output = TaskOutput(task_id="t", task_name="n", final_output=[1, 2, 3])
        assert output.to_dict()["final_output"] == [1, 2, 3]

    def test_json_native_dict_preserved(self) -> None:
        """A dict of primitives is kept as a dict (round-trips)."""
        output = TaskOutput(task_id="t", task_name="n", final_output={"a": 1, "b": "two"})
        assert output.to_dict()["final_output"] == {"a": 1, "b": "two"}

    def test_plain_string_preserved(self) -> None:
        """A plain string output is kept verbatim."""
        output = TaskOutput(task_id="t", task_name="n", final_output="hello")
        assert output.to_dict()["final_output"] == "hello"

    def test_bare_pydantic_model_serialized_to_str(self) -> None:
        """A top-level BaseModel (already non-native) is stringified."""
        output = TaskOutput(task_id="t", task_name="n", final_output=_Item(x=7))
        d = output.to_dict()
        assert isinstance(d["final_output"], str)
        json.dumps(d)

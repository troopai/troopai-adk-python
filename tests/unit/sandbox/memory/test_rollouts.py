"""Tests for ``troopai.adk.sandbox.memory.rollouts``."""

from __future__ import annotations

import asyncio
import json

import pytest

from troopai.adk.sandbox.memory.interface import RolloutTerminalMetadata
from troopai.adk.sandbox.memory.rollouts import (
    build_rollout_payload,
    dump_rollout_json,
    terminal_metadata_for_exception,
    terminal_metadata_for_result,
)


def test_dump_rollout_json_is_newline_framed() -> None:
    line = dump_rollout_json({"rollout_id": "r1", "x": 1})
    assert line.endswith("\n")
    assert json.loads(line) == {"rollout_id": "r1", "x": 1}


def test_dump_rollout_json_uses_structured_fallback_for_non_native_objects() -> None:
    class Custom:
        def __repr__(self) -> str:
            return "Custom()"

    line = dump_rollout_json({"value": Custom()})
    parsed = json.loads(line)
    # The fallback emits a STRUCTURED sentinel (not a bare repr()) so downstream
    # tooling can detect and filter the unserializable case instead of receiving
    # debug strings inside MEMORY.md.
    assert parsed["value"] == {
        "__unserializable_type__": "Custom",
        "repr": "Custom()",
    }


def test_build_rollout_payload_minimum_fields() -> None:
    payload = build_rollout_payload(
        rollout_id="r1",
        inputs="hi",
        new_items=[{"type": "msg"}],
        final_output="ok",
    )
    assert payload["rollout_id"] == "r1"
    assert payload["input"] == "hi"
    assert "terminal_metadata" in payload
    assert "updated_at" in payload
    assert payload["final_output"] == "ok"


def test_build_rollout_payload_with_explicit_metadata() -> None:
    metadata = RolloutTerminalMetadata(
        terminal_state="failed",
        exception_type="ValueError",
        exception_message="bad input",
        has_final_output=False,
    )
    payload = build_rollout_payload(
        rollout_id="r1",
        inputs=None,
        new_items=[],
        final_output=None,
        terminal_metadata=metadata,
    )
    assert payload["terminal_metadata"]["terminal_state"] == "failed"
    assert payload["terminal_metadata"]["exception_type"] == "ValueError"


def test_terminal_metadata_for_result_completed() -> None:
    metadata = terminal_metadata_for_result(has_final_output=True)
    assert metadata.terminal_state == "completed"


def test_terminal_metadata_for_result_max_turns() -> None:
    metadata = terminal_metadata_for_result(has_final_output=False, max_turns_exceeded=True)
    assert metadata.terminal_state == "max_turns_exceeded"


def test_terminal_metadata_for_result_interrupted() -> None:
    metadata = terminal_metadata_for_result(has_final_output=False, interrupted=True)
    assert metadata.terminal_state == "interrupted"


def test_terminal_metadata_for_exception_value_error() -> None:
    metadata = terminal_metadata_for_exception(ValueError("nope"))
    assert metadata.terminal_state == "failed"
    assert metadata.exception_type == "ValueError"
    assert metadata.exception_message == "nope"


def test_terminal_metadata_for_exception_cancelled() -> None:
    metadata = terminal_metadata_for_exception(asyncio.CancelledError())
    assert metadata.terminal_state == "cancelled"


def test_terminal_metadata_for_exception_max_turns() -> None:
    class MaxTurnsExceeded(Exception):
        pass

    metadata = terminal_metadata_for_exception(MaxTurnsExceeded("too many"))
    assert metadata.terminal_state == "max_turns_exceeded"


def test_terminal_metadata_for_exception_guardrail() -> None:
    class InputGuardrailTripwireTriggered(Exception):
        pass

    metadata = terminal_metadata_for_exception(InputGuardrailTripwireTriggered("blocked"))
    assert metadata.terminal_state == "guardrail_tripped"


@pytest.mark.parametrize(
    "exc, expected",
    [
        (ValueError(""), "failed"),
        (RuntimeError("x"), "failed"),
    ],
)
def test_terminal_metadata_for_exception_default(exc: BaseException, expected: str) -> None:
    metadata = terminal_metadata_for_exception(exc)
    assert metadata.terminal_state == expected

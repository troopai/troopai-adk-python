"""Tests for ``troopai.adk.sandbox.session.events``."""

from __future__ import annotations

import json
import uuid

from troopai.adk.sandbox.session import (
    ErrorCode,
    EventPayloadPolicy,
    OpName,
    SandboxSessionFinishEvent,
    SandboxSessionStartEvent,
    event_to_json_line,
    validate_sandbox_session_event,
)

SESSION_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _start_event(**overrides: object) -> SandboxSessionStartEvent:
    base: dict[str, object] = {
        "session_id": SESSION_ID,
        "seq": 1,
        "op": OpName.RUN,
        "span_id": "span-1",
    }
    base.update(overrides)
    return SandboxSessionStartEvent.model_validate(base)


def _finish_event(**overrides: object) -> SandboxSessionFinishEvent:
    base: dict[str, object] = {
        "session_id": SESSION_ID,
        "seq": 2,
        "op": OpName.RUN,
        "span_id": "span-1",
        "ok": True,
        "duration_ms": 10.0,
    }
    base.update(overrides)
    return SandboxSessionFinishEvent.model_validate(base)


def test_start_event_discriminator() -> None:
    event = _start_event()
    assert event.phase == "start"


def test_finish_event_ok_defaults() -> None:
    event = _finish_event()
    assert event.phase == "finish"
    assert event.ok is True
    assert event.error_code is None


def test_finish_event_with_error_classification() -> None:
    event = _finish_event(
        ok=False, error_code=ErrorCode.NON_ZERO_EXIT, error_type="ExecNonZeroError", error_message="rc=2"
    )
    assert event.error_code == ErrorCode.NON_ZERO_EXIT
    assert event.error_type == "ExecNonZeroError"


def test_event_round_trip_through_json_line() -> None:
    event = _start_event(data={"argv": ["ls", "/"]})
    line = event_to_json_line(event)
    parsed = json.loads(line)
    rebuilt = validate_sandbox_session_event(parsed)
    assert isinstance(rebuilt, SandboxSessionStartEvent)
    assert rebuilt.phase == "start"
    assert rebuilt.data["argv"] == ["ls", "/"]


def test_validate_sandbox_session_event_discriminates_phase() -> None:
    finish = _finish_event(seq=3)
    rebuilt = validate_sandbox_session_event(json.loads(event_to_json_line(finish)))
    assert isinstance(rebuilt, SandboxSessionFinishEvent)


def test_event_payload_policy_defaults() -> None:
    policy = EventPayloadPolicy()
    assert policy.include_exec_output is False
    assert policy.include_write_len is True
    assert policy.max_stdout_chars == 8_000
    assert policy.max_stderr_chars == 8_000


def test_stdout_bytes_excluded_from_serialization() -> None:
    event = _finish_event(stdout_bytes=b"secret-binary-payload")
    line = event_to_json_line(event)
    parsed = json.loads(line)
    assert "stdout_bytes" not in parsed
    assert "stderr_bytes" not in parsed


def test_payload_policy_drives_decoded_stdout_via_safe_decode_helper() -> None:
    """Event builders use ``safe_decode_with_max_chars`` to honor the policy ceiling."""
    from troopai.adk.sandbox.session import safe_decode_with_max_chars

    raw = b"alpha\nbeta\ngamma\n" * 500
    policy = EventPayloadPolicy()
    decoded = safe_decode_with_max_chars(raw, max_chars=policy.max_stdout_chars)
    assert len(decoded) <= policy.max_stdout_chars + 1  # +1 for the trailing ellipsis
    assert decoded.endswith("…")

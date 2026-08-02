from __future__ import annotations

from datetime import UTC, datetime

from troopai.adk.audit.event import AuditEvent, hash_payload


def test_hash_payload_is_stable_and_order_independent() -> None:
    a = hash_payload({"x": 1, "y": 2})
    b = hash_payload({"y": 2, "x": 1})
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_hash_payload_handles_unserializable() -> None:
    class Weird:
        def __repr__(self) -> str:
            raise RuntimeError("nope")

    assert hash_payload(Weird()) == "<unhashable>"


def test_hash_payload_falls_back_to_str_when_json_fails() -> None:
    # A circular reference makes json.dumps raise, but str() still renders
    # it ("[[...]]") — exercising the json-fails / str-succeeds branch.
    circular: list[object] = []
    circular.append(circular)
    result = hash_payload(circular)
    assert result != "<unhashable>"
    assert len(result) == 64


def test_audit_event_fields() -> None:
    event = AuditEvent(
        tenant_id="t1",
        agent_name="agent",
        tool_name="search",
        tool_call_id="call_1",
        args_hash=hash_payload({"q": "hi"}),
        result_hash=None,
        outcome="denied",
        timestamp=datetime.now(UTC),
    )
    assert event.tenant_id == "t1"
    assert event.outcome == "denied"
    assert event.result_hash is None

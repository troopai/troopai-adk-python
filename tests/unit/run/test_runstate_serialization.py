"""Tests for ``RunState`` JSON serialization and structured approvals.

Covers:

- ``to_json()`` / ``from_json()`` round-trip
- ``to_json()`` emits no version key; ``from_json()`` tolerantly
  ignores an unrecognised key in an older persisted payload
- ``approve(approver_id=..., reason=...)`` records ``ApprovalMetadata``
- ``reject(message=..., approver_id=..., reason=...)`` records both
  the model-visible ``message`` AND internal audit metadata
- Approval metadata survives ``to_json()`` / ``from_json()`` round-trip
- Empty-metadata calls keep ``approval_metadata`` empty
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from troopai.adk.run.state import (
    ApprovalMetadata,
    RunState,
)
from troopai.adk.tools.deferred_tool import (
    DeferredToolCall,
    DeferredToolRequests,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_deferred(call_id: str, name: str = "delete_user") -> DeferredToolCall:
    return DeferredToolCall(
        tool_call_id=call_id,
        tool_name=name,
        tool_arguments={"user_id": "u-1"},
        raw_arguments='{"user_id": "u-1"}',
        request_time=datetime(2026, 4, 14, 12, 0, 0),
    )


def _make_state_with_deferred() -> RunState:
    d1 = _make_deferred("call-1")
    d2 = _make_deferred("call-2")
    return RunState(
        original_user_prompt="Delete user u-1 and user u-2",
        current_agent_name="admin-agent",
        turn_count=2,
        deferred_tool_requests=DeferredToolRequests(approvals=[d1, d2]),
    )


# ── `to_json()` / `from_json()` round-trip ───────────────────────────


def test_to_json_emits_no_version_key() -> None:
    """``to_json`` is a plain wrapper over ``to_dict`` — no envelope,
    no version stamp of any kind."""
    state = RunState(original_user_prompt="hello")
    payload = json.loads(state.to_json())
    assert "_schema_version" not in payload
    assert payload == state.to_dict()


def test_round_trip_preserves_core_fields() -> None:
    state = _make_state_with_deferred()
    blob = state.to_json()
    restored = RunState.from_json(blob)

    assert restored.original_user_prompt == state.original_user_prompt
    assert restored.current_agent_name == "admin-agent"
    assert restored.turn_count == 2
    assert len(restored.deferred_tool_requests.approvals) == 2
    assert restored.deferred_tool_requests.approvals[0].tool_call_id == "call-1"
    assert restored.deferred_tool_requests.approvals[1].tool_call_id == "call-2"


def test_from_json_ignores_unrecognised_keys_in_older_payload() -> None:
    """An older persisted blob may carry keys this build does not know
    (e.g. a stale ``_schema_version`` stamp written by a prior build).
    ``from_json`` MUST load it cleanly — ``from_dict`` reads each field
    by name and ignores extras — NOT raise."""
    state = _make_state_with_deferred()
    payload = json.loads(state.to_json())
    payload["_schema_version"] = 1  # stale key an older build wrote
    payload["some_future_unknown_field"] = {"ignored": True}
    blob = json.dumps(payload)

    restored = RunState.from_json(blob)

    assert restored.original_user_prompt == state.original_user_prompt
    assert len(restored.deferred_tool_requests.approvals) == 2


def test_from_json_rejects_invalid_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        RunState.from_json("not json at all {[}")


# ── Structured approval metadata ─────────────────────────────────────


def test_approve_without_metadata_keeps_dict_empty() -> None:
    state = _make_state_with_deferred()
    d1 = state.deferred_tool_requests.approvals[0]

    state.approve(d1)

    assert d1 in state.approved_tools
    assert len(state.approval_metadata) == 0  # no audit trail recorded


def test_approve_with_metadata_records_approver() -> None:
    state = _make_state_with_deferred()
    d1 = state.deferred_tool_requests.approvals[0]

    state.approve(d1, approver_id="alice@example.com", reason="signed change order")

    assert d1 in state.approved_tools
    assert "call-1" in state.approval_metadata
    meta = state.approval_metadata["call-1"]
    assert meta.approver_id == "alice@example.com"
    assert meta.reason == "signed change order"
    assert isinstance(meta.timestamp, datetime)


def test_reject_separates_model_message_from_audit_reason() -> None:
    """`message` is the model-visible explanation.
    `reason` is the internal audit log entry. They MUST NOT be
    conflated — the model sees `message`, the compliance system
    sees `reason`."""
    state = _make_state_with_deferred()
    d1 = state.deferred_tool_requests.approvals[0]

    state.reject(
        d1,
        message="Not authorized for this user id",
        approver_id="bob@example.com",
        reason="policy violation: write to tier-0 resource",
    )

    # Model-visible message lands on rejected_tools.
    assert len(state.rejected_tools) == 1
    tool, message = state.rejected_tools[0]
    assert tool is d1
    assert message == "Not authorized for this user id"

    # Audit metadata lands on approval_metadata, indexed by call id.
    assert "call-1" in state.approval_metadata
    meta = state.approval_metadata["call-1"]
    assert meta.approver_id == "bob@example.com"
    assert meta.reason == "policy violation: write to tier-0 resource"


def test_approval_metadata_survives_json_round_trip() -> None:
    state = _make_state_with_deferred()
    d1 = state.deferred_tool_requests.approvals[0]
    d2 = state.deferred_tool_requests.approvals[1]
    state.approve(d1, approver_id="alice", reason="ok")
    state.reject(d2, message="no", approver_id="bob", reason="bad")

    restored = RunState.from_json(state.to_json())

    assert len(restored.approval_metadata) == 2

    meta_1 = restored.approval_metadata["call-1"]
    assert meta_1.approver_id == "alice"
    assert meta_1.reason == "ok"

    meta_2 = restored.approval_metadata["call-2"]
    assert meta_2.approver_id == "bob"
    assert meta_2.reason == "bad"

    # Round-trip preserves the approval/rejection lists too.
    assert len(restored.approved_tools) == 1
    assert restored.approved_tools[0].tool_call_id == "call-1"
    assert len(restored.rejected_tools) == 1
    restored_tool, restored_msg = restored.rejected_tools[0]
    assert restored_tool.tool_call_id == "call-2"
    assert restored_msg == "no"


# ── reject() idempotency ─────────────────────────────────────────────


def test_reject_is_idempotent_on_second_call() -> None:
    """reject() called twice on the same DeferredToolCall must update the
    message without duplicating the entry — the LLM must see exactly one
    function_call_output per call_id on resumption."""
    state = _make_state_with_deferred()
    d1 = state.deferred_tool_requests.approvals[0]

    state.reject(d1, message="first rejection")
    state.reject(d1, message="updated rejection")  # second call — must not duplicate

    assert len(state.rejected_tools) == 1, "double reject must not duplicate entries"
    _tool, msg = state.rejected_tools[0]
    assert msg == "updated rejection", "second reject must update the message"
    assert _tool.tool_call_id == "call-1"


def test_reject_dedup_by_tool_call_id_not_object_identity() -> None:
    """Dedup must be keyed on tool_call_id (not object identity) so that
    a freshly-reconstructed DeferredToolCall with the same ID deduplicates."""
    d1a = _make_deferred("call-x", "my_tool")
    d1b = _make_deferred("call-x", "my_tool")  # different object, same id
    state = RunState(
        original_user_prompt="test",
        deferred_tool_requests=DeferredToolRequests(approvals=[d1a]),
    )

    state.reject(d1a, message="v1")
    state.reject(d1b, message="v2")

    assert len(state.rejected_tools) == 1, "same call_id must not produce two entries"
    _, msg = state.rejected_tools[0]
    assert msg == "v2"


# ── `ApprovalMetadata` dataclass round-trip ──────────────────────────


def test_approval_metadata_to_from_dict() -> None:
    original = ApprovalMetadata(
        approver_id="svc-token-abc",
        reason="automated CI approval",
    )
    data = original.to_dict()
    restored = ApprovalMetadata.from_dict(data)

    assert restored.approver_id == "svc-token-abc"
    assert restored.reason == "automated CI approval"
    assert restored.timestamp == original.timestamp


def test_approval_metadata_from_dict_missing_timestamp() -> None:
    """Hand-authored or older payloads may omit ``timestamp``.
    ``from_dict`` should fall back to `now` rather than crashing."""
    restored = ApprovalMetadata.from_dict(
        {
            "approver_id": "alice",
            "reason": "ok",
        }
    )
    assert restored.approver_id == "alice"
    assert isinstance(restored.timestamp, datetime)


# ── `to_dict()` / `from_dict()` round-trip ───────────────────────────


def test_to_dict_and_to_json_emit_no_version_key() -> None:
    """Neither ``to_dict`` nor ``to_json`` emits any ``_schema_version``
    stamp; ``from_dict`` reconstructs from the bare dict."""
    state = _make_state_with_deferred()
    state.approve(state.deferred_tool_requests.approvals[0], approver_id="alice")

    raw = state.to_dict()
    assert "_schema_version" not in raw
    assert "_schema_version" not in json.loads(state.to_json())

    restored = RunState.from_dict(raw)
    assert len(restored.approved_tools) == 1
    assert restored.approval_metadata["call-1"].approver_id == "alice"

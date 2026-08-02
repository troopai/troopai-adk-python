"""Regression tests for ``troopai.adk.run.state`` (``RunState``).

Covers three confirmed defects:

- ``from_dict`` must not coerce a plain string prompt that merely happens
  to be valid JSON ("42", "null", "true", '{"k": 1}') into a non-string
  value. Only a serialized ``list`` prompt may be recovered.
- ``to_json()`` must serialize nested HITL deferrals whose nested approval
  requests carry raw ``datetime`` ``request_time`` fields.
- ``provide_result`` must not append a duplicate external result for the
  same ``call_id`` (mirrors ``approve()`` / ``reject()`` dedup).
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from troopai.adk.run.state import RunState
from troopai.adk.tools.deferred_tool import (
    DeferredToolCall,
    DeferredToolCallMetadata,
    DeferredToolRequests,
    NestedDeferredToolRequests,
)


def _make_deferred(call_id: str, name: str = "mcp_tool") -> DeferredToolCall:
    return DeferredToolCall(
        tool_call_id=call_id,
        tool_name=name,
        tool_arguments={"q": "x"},
        raw_arguments='{"q": "x"}',
        request_time=datetime(2026, 4, 14, 12, 0, 0),
    )


# ── Finding 1: string prompts that are valid JSON must not be corrupted ──


class TestStringPromptNotCoerced:
    @pytest.mark.parametrize(
        "prompt",
        ["42", "null", "true", "false", '{"k": 1}', "3.14", '"quoted"'],
    )
    def test_json_like_string_prompt_round_trips_unchanged(self, prompt: str) -> None:
        """A plain ``str`` prompt that is itself valid JSON must survive the
        to_dict/from_dict round-trip as the SAME string, not a coerced
        int/None/bool/dict."""
        state = RunState(original_user_prompt=prompt)
        restored = RunState.from_dict(state.to_dict())

        assert isinstance(restored.original_user_prompt, str)
        assert restored.original_user_prompt == prompt

    def test_json_like_string_prompt_via_json_round_trip(self) -> None:
        state = RunState(original_user_prompt="null")
        restored = RunState.from_json(state.to_json())

        assert restored.original_user_prompt == "null"

    def test_plain_string_prompt_round_trips(self) -> None:
        state = RunState(original_user_prompt="Delete user u-1")
        restored = RunState.from_dict(state.to_dict())

        assert restored.original_user_prompt == "Delete user u-1"

    def test_list_prompt_is_still_recovered(self) -> None:
        """The legitimate list-form prompt (json.dumps'd by to_dict) must
        still be recovered as a list."""
        list_prompt = [
            {"type": "text", "text": "describe this"},
            {"type": "text", "text": "in detail"},
        ]
        state = RunState(original_user_prompt=list_prompt)
        restored = RunState.from_dict(state.to_dict())

        assert isinstance(restored.original_user_prompt, list)
        assert restored.original_user_prompt == list_prompt


# ── Finding 2: nested HITL deferrals must serialize to JSON ──────────────


class TestNestedDeferralSerialization:
    def test_to_json_handles_nested_deferred_request_datetime(self) -> None:
        """A DeferredToolCall whose metadata carries nested approval
        requests (each with a raw datetime request_time) must serialize
        without raising TypeError."""
        nested_call = DeferredToolCall(
            tool_call_id="nested_1",
            tool_name="search",
            tool_arguments={},
            raw_arguments="{}",
            request_time=datetime(2026, 6, 1, 9, 30, 0),
        )
        parent = DeferredToolCall(
            tool_call_id="parent_1",
            tool_name="researcher_subagent",
            tool_arguments={},
            raw_arguments="{}",
            request_time=datetime(2026, 6, 1, 9, 0, 0),
            metadata=DeferredToolCallMetadata(
                nested_agent=True,
                nested_agent_name="Researcher",
                nested_state={"turn_count": 1},
                nested_deferred_requests=NestedDeferredToolRequests(approvals=[nested_call]),
            ),
        )
        state = RunState(deferred_tool_requests=DeferredToolRequests(approvals=[parent]))

        # Must not raise "Object of type datetime is not JSON serializable".
        payload = state.to_json()

        # And it must round-trip back to a recoverable nested structure.
        restored = RunState.from_json(payload)
        restored_parent = restored.deferred_tool_requests.approvals[0]
        assert restored_parent.metadata is not None
        nested = restored_parent.metadata.nested_deferred_requests
        assert nested is not None
        assert len(nested.approvals) == 1
        assert nested.approvals[0].tool_call_id == "nested_1"
        assert nested.approvals[0].request_time == datetime(2026, 6, 1, 9, 30, 0)

    def test_metadata_to_dict_is_json_serializable(self) -> None:
        """DeferredToolCallMetadata.to_dict() output must be directly
        json.dumps-able even with nested datetimes."""
        metadata = DeferredToolCallMetadata(
            nested_agent=True,
            nested_agent_name="Sub",
            nested_deferred_requests=NestedDeferredToolRequests(
                approvals=[_make_deferred("n1")],
            ),
        )
        # Must not raise.
        dumped = json.dumps(metadata.to_dict())
        assert "n1" in dumped


# ── Finding 3: provide_result must dedup by call_id ─────────────────────


class TestProvideResultDedup:
    def test_duplicate_provide_result_replaces_not_appends(self) -> None:
        call = _make_deferred("ext_1")
        state = RunState(deferred_tool_requests=DeferredToolRequests(calls=[call]))

        state.provide_result(call, "first")
        state.provide_result(call, "second")

        # Exactly one result for this call_id, last-write-wins.
        matching = [r for r in state.external_results if r.call_id == "ext_1"]
        assert len(matching) == 1
        assert matching[0].output == "second"

    def test_duplicate_provide_result_does_not_emit_two_outputs(self) -> None:
        """No two ExternalToolCallResult entries may share a call_id —
        otherwise resumption emits two function_call_output items for the
        same id, a malformed exchange."""
        call = _make_deferred("dup")
        state = RunState(deferred_tool_requests=DeferredToolRequests(calls=[call]))

        state.provide_result(call, "a")
        state.provide_result(call, "b")
        state.provide_result(call, "c")

        call_ids = [r.call_id for r in state.external_results]
        assert call_ids.count("dup") == 1

    def test_distinct_calls_still_both_kept(self) -> None:
        call_a = _make_deferred("a")
        call_b = _make_deferred("b")
        state = RunState(deferred_tool_requests=DeferredToolRequests(calls=[call_a, call_b]))

        state.provide_result(call_a, "ra")
        state.provide_result(call_b, "rb")

        results = {r.call_id: r.output for r in state.external_results}
        assert results == {"a": "ra", "b": "rb"}


# ── Finding 4: approve()/reject() must cross-remove (last decision wins) ──


class TestApproveRejectCrossRemoval:
    """A flipped decision must supersede the prior one.

    Without cross-removal a call ends up in BOTH ``approved_tools`` and
    ``rejected_tools``, so resumption executes the tool AND appends a
    rejection message under the same ``call_id`` — a double, contradictory
    function_call_output exchange.
    """

    def test_approve_then_reject_leaves_only_rejection(self) -> None:
        call = _make_deferred("tc_1")
        state = RunState(deferred_tool_requests=DeferredToolRequests(approvals=[call]))

        state.approve(call)
        state.reject(call, "changed my mind")

        assert all(t.tool_call_id != "tc_1" for t in state.approved_tools)
        assert [(t.tool_call_id, m) for t, m in state.rejected_tools] == [("tc_1", "changed my mind")]

    def test_reject_then_approve_leaves_only_approval(self) -> None:
        call = _make_deferred("tc_1")
        state = RunState(deferred_tool_requests=DeferredToolRequests(approvals=[call]))

        state.reject(call, "no")
        state.approve(call)

        assert all(t.tool_call_id != "tc_1" for t, _ in state.rejected_tools)
        assert [t.tool_call_id for t in state.approved_tools] == ["tc_1"]

    def test_to_deferred_results_has_no_duplicate_after_flip(self) -> None:
        """to_deferred_results() must never yield two verdicts for one call_id."""
        call = _make_deferred("tc_dup")
        state = RunState(deferred_tool_requests=DeferredToolRequests(approvals=[call]))

        state.approve(call)
        state.reject(call, "denied")

        approvals = state.to_deferred_results().approvals
        matching = [a for a in approvals if a.tool_call_id == "tc_dup"]
        assert len(matching) == 1
        # Last decision wins: the rejection.
        assert matching[0].approved is False
        assert matching[0].message == "denied"

    def test_reject_then_approve_deferred_results_is_approved(self) -> None:
        call = _make_deferred("tc_dup")
        state = RunState(deferred_tool_requests=DeferredToolRequests(approvals=[call]))

        state.reject(call, "denied")
        state.approve(call)

        approvals = state.to_deferred_results().approvals
        matching = [a for a in approvals if a.tool_call_id == "tc_dup"]
        assert len(matching) == 1
        assert matching[0].approved is True

    def test_flip_clears_stale_audit_metadata(self) -> None:
        """A flip with no new audit info must drop the superseded decision's
        metadata rather than misattribute it to the new verdict."""
        call = _make_deferred("tc_1")
        state = RunState(deferred_tool_requests=DeferredToolRequests(approvals=[call]))

        state.reject(call, "no", approver_id="bob", reason="policy")
        assert "tc_1" in state.approval_metadata

        state.approve(call)  # no approver_id/reason
        assert "tc_1" not in state.approval_metadata

    def test_flip_records_new_audit_metadata(self) -> None:
        call = _make_deferred("tc_1")
        state = RunState(deferred_tool_requests=DeferredToolRequests(approvals=[call]))

        state.approve(call, approver_id="alice", reason="ok")
        state.reject(call, "no", approver_id="bob", reason="policy")

        meta = state.approval_metadata["tc_1"]
        assert meta.approver_id == "bob"
        assert meta.reason == "policy"

    def test_distinct_calls_unaffected_by_cross_removal(self) -> None:
        call_a = _make_deferred("a")
        call_b = _make_deferred("b")
        state = RunState(deferred_tool_requests=DeferredToolRequests(approvals=[call_a, call_b]))

        state.approve(call_a)
        state.reject(call_b, "no")

        assert [t.tool_call_id for t in state.approved_tools] == ["a"]
        assert [t.tool_call_id for t, _ in state.rejected_tools] == ["b"]

    def test_reapprove_preserves_same_decision_metadata(self) -> None:
        """Re-affirming the SAME decision (no flip) must keep its audit."""
        call = _make_deferred("tc_1")
        state = RunState(deferred_tool_requests=DeferredToolRequests(approvals=[call]))

        state.approve(call, approver_id="alice", reason="ok")
        state.approve(call)  # idempotent re-approval, no new audit

        assert state.approval_metadata["tc_1"].approver_id == "alice"
        assert [t.tool_call_id for t in state.approved_tools] == ["tc_1"]

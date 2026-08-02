"""Tests for deferred tool type serialization roundtrips.

Covers:
- Bug 2: ExternalToolCallResult.from_dict() field name mismatch
- Bug 3: NestedDeferredToolRequests.from_dict() request_time=None
"""

from datetime import datetime

import pytest

from troopai.adk.tools.deferred_tool import (
    DeferredToolCall,
    DeferredToolCallMetadata,
    ExternalToolCallResult,
    NestedDeferredToolRequests,
)

# ── ExternalToolCallResult roundtrip (Bug 2) ─────────────────────────


class TestExternalToolCallResultSerialization:
    def test_roundtrip(self) -> None:
        """to_dict() -> from_dict() preserves all fields."""
        original = ExternalToolCallResult(
            call_id="tc_123",
            output="search results here",
        )
        data = original.to_dict()
        restored = ExternalToolCallResult.from_dict(data)

        assert restored.call_id == "tc_123"
        assert restored.output == "search results here"

    def test_roundtrip_none_output(self) -> None:
        """output=None roundtrips correctly."""
        original = ExternalToolCallResult(call_id="tc_456", output=None)
        data = original.to_dict()
        restored = ExternalToolCallResult.from_dict(data)

        assert restored.call_id == "tc_456"
        assert restored.output is None

    def test_from_dict_uses_correct_keys(self) -> None:
        """from_dict() reads 'call_id' and 'output', not 'tool_call_id'/'result'."""
        data = {"call_id": "abc", "output": "some result"}
        result = ExternalToolCallResult.from_dict(data)
        assert result.call_id == "abc"
        assert result.output == "some result"

    def test_from_dict_wrong_keys_raises(self) -> None:
        """Old-style keys ('tool_call_id', 'result') raise KeyError."""
        data = {"tool_call_id": "abc", "result": "some result"}
        with pytest.raises(KeyError):
            ExternalToolCallResult.from_dict(data)


# ── NestedDeferredToolRequests roundtrip (Bug 3) ────────────────────


class TestNestedDeferredToolRequestsSerialization:
    def test_request_time_absent_defaults_to_now(self) -> None:
        """When request_time is missing from dict, defaults to datetime.now()."""
        data = {
            "approvals": [
                {
                    "tool_call_id": "tc_1",
                    "tool_name": "search",
                    "tool_arguments": {"q": "test"},
                    "raw_arguments": '{"q": "test"}',
                    # No request_time key
                }
            ]
        }
        result = NestedDeferredToolRequests.from_dict(data)
        approval = result.approvals[0]

        assert approval.tool_call_id == "tc_1"
        assert isinstance(approval.request_time, datetime)
        # Should be very recent (within 5 seconds)
        delta = (datetime.now() - approval.request_time).total_seconds()
        assert delta < 5

    def test_request_time_present_is_preserved(self) -> None:
        """When request_time is in the dict, it's preserved."""
        fixed_time = datetime(2025, 6, 15, 12, 0, 0)
        data = {
            "approvals": [
                {
                    "tool_call_id": "tc_2",
                    "tool_name": "delete",
                    "tool_arguments": {},
                    "raw_arguments": "{}",
                    "request_time": fixed_time.isoformat(),
                }
            ]
        }
        result = NestedDeferredToolRequests.from_dict(data)
        assert result.approvals[0].request_time == fixed_time

    def test_isoformat_does_not_crash(self) -> None:
        """request_time.isoformat() works after from_dict() (Bug 3 regression)."""
        data = {
            "approvals": [
                {
                    "tool_call_id": "tc_3",
                    "tool_name": "action",
                    "tool_arguments": {},
                    "raw_arguments": "{}",
                }
            ]
        }
        result = NestedDeferredToolRequests.from_dict(data)
        # This would crash with AttributeError if request_time was None
        iso_str = result.approvals[0].request_time.isoformat()
        assert isinstance(iso_str, str)


# ── DeferredToolCallMetadata roundtrip ───────────────────────────────


class TestDeferredToolCallMetadataRoundtrip:
    def test_roundtrip_with_nested_requests(self) -> None:
        """Full metadata roundtrip preserves nested structure."""
        original = DeferredToolCallMetadata(
            nested_agent=True,
            nested_agent_name="Researcher",
            nested_state={"history": [], "context": None},
            nested_deferred_requests=NestedDeferredToolRequests(
                approvals=[
                    DeferredToolCall(
                        tool_call_id="inner_tc",
                        tool_name="web_search",
                        tool_arguments={"q": "test"},
                        raw_arguments='{"q": "test"}',
                    )
                ]
            ),
        )
        data = original.to_dict()
        restored = DeferredToolCallMetadata.from_dict(data)

        assert restored.nested_agent is True
        assert restored.nested_agent_name == "Researcher"
        assert restored.nested_state == {"history": [], "context": None}
        assert len(restored.nested_deferred_requests.approvals) == 1
        assert restored.nested_deferred_requests.approvals[0].tool_call_id == "inner_tc"

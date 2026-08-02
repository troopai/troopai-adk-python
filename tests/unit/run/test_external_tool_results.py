"""Tests for external tool call result flow via RunState.provide_result()."""

from troopai.adk.run.state import RunState
from troopai.adk.tools.deferred_tool import (
    DeferredToolCall,
    DeferredToolRequests,
)


class TestExternalToolResults:
    def test_provide_result_stores_external_result(self) -> None:
        call = DeferredToolCall(
            tool_call_id="ext_1",
            tool_name="mcp_search",
            tool_arguments={"q": "test"},
            raw_arguments='{"q": "test"}',
        )
        state = RunState(
            deferred_tool_requests=DeferredToolRequests(calls=[call]),
        )

        state.provide_result(call, "search results here")

        assert len(state.external_results) == 1
        assert state.external_results[0].call_id == "ext_1"
        assert state.external_results[0].output == "search results here"

    def test_provide_result_removes_from_calls(self) -> None:
        call = DeferredToolCall(
            tool_call_id="ext_1",
            tool_name="mcp_tool",
            tool_arguments={},
            raw_arguments="{}",
        )
        state = RunState(
            deferred_tool_requests=DeferredToolRequests(calls=[call]),
        )

        state.provide_result(call, "done")
        assert len(state.deferred_tool_requests.calls) == 0

    def test_multiple_external_results(self) -> None:
        call_a = DeferredToolCall(
            tool_call_id="a",
            tool_name="tool_a",
            tool_arguments={},
            raw_arguments="{}",
        )
        call_b = DeferredToolCall(
            tool_call_id="b",
            tool_name="tool_b",
            tool_arguments={},
            raw_arguments="{}",
        )
        state = RunState(
            deferred_tool_requests=DeferredToolRequests(calls=[call_a, call_b]),
        )

        state.provide_result(call_a, "result_a")
        state.provide_result(call_b, "result_b")

        assert len(state.external_results) == 2
        results = {r.call_id: r.output for r in state.external_results}
        assert results == {"a": "result_a", "b": "result_b"}

    def test_external_results_empty_by_default(self) -> None:
        state = RunState()
        assert state.external_results == []

    def test_external_results_survive_serialization_round_trip(self) -> None:
        """external_results must be preserved across to_json / from_json."""
        call = DeferredToolCall(
            tool_call_id="ext_rt",
            tool_name="mcp_fetch",
            tool_arguments={"url": "https://example.com"},
            raw_arguments='{"url": "https://example.com"}',
        )
        state = RunState(
            original_user_prompt="fetch https://example.com",
            deferred_tool_requests=DeferredToolRequests(calls=[call]),
        )
        state.provide_result(call, {"status": 200, "body": "hello"})

        payload = state.to_json()
        restored = RunState.from_json(payload)

        assert len(restored.external_results) == 1, "external_results must survive serialization round-trip"
        assert restored.external_results[0].call_id == "ext_rt"
        # Output is stored as-is (not re-parsed from str) so compare str form
        assert restored.external_results[0].output == {"status": 200, "body": "hello"}

    def test_external_results_absent_in_legacy_payload_defaults_to_empty(self) -> None:
        """from_json on a payload without 'external_results' key must not raise
        and must restore _external_results as an empty list (tolerant load)."""
        import json

        state = RunState(original_user_prompt="ping")
        raw = json.loads(state.to_json())
        # Simulate a payload from a build before the key was added
        raw.pop("external_results", None)
        restored = RunState.from_json(json.dumps(raw))

        assert restored.external_results == []

    def test_mixed_approvals_and_external(self) -> None:
        approval_call = DeferredToolCall(
            tool_call_id="approve_1",
            tool_name="delete_user",
            tool_arguments={},
            raw_arguments="{}",
        )
        ext_call = DeferredToolCall(
            tool_call_id="ext_1",
            tool_name="mcp_search",
            tool_arguments={},
            raw_arguments="{}",
        )
        state = RunState(
            deferred_tool_requests=DeferredToolRequests(
                approvals=[approval_call],
                calls=[ext_call],
            ),
        )

        state.approve(approval_call)
        state.provide_result(ext_call, "external result")

        assert len(state.approved_tools) == 1
        assert len(state.external_results) == 1
        assert state.deferred_tool_requests.is_empty()

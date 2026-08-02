"""Tests for :mod:`troopai.adk.workflows.temporal.workflow`.

Covers:
- ``HumanReply`` frozen dataclass fields and defaults.
- ``ToolApprovalDecision`` frozen dataclass fields and defaults.
- ``TroopAIWorkflow`` exposes the expected signal, query, and update methods.
- ``consume_replies`` drains and clears the pending-replies queue.
- ``consume_approval`` pops a stored decision and returns ``None`` on miss.
"""

from __future__ import annotations

import pytest

temporalio = pytest.importorskip("temporalio")

from troopai.adk.workflows.temporal.workflow import (
    HumanReply,
    ToolApprovalDecision,
    TroopAIWorkflow,
)


class TestHumanReplyDataclass:
    def test_human_reply_dataclass(self) -> None:
        """``HumanReply`` stores all fields and defaults metadata to an empty dict."""
        reply = HumanReply(node_id="node-1", value="approve")

        assert reply.node_id == "node-1"
        assert reply.value == "approve"
        assert reply.metadata == {}

    def test_human_reply_with_metadata(self) -> None:
        """``HumanReply`` stores provided metadata without modification."""
        reply = HumanReply(node_id="node-2", value="reject", metadata={"source": "ui"})

        assert reply.metadata == {"source": "ui"}

    def test_human_reply_is_frozen(self) -> None:
        """``HumanReply`` is immutable — attribute assignment raises ``FrozenInstanceError``."""
        from dataclasses import FrozenInstanceError

        reply = HumanReply(node_id="n", value="v")

        with pytest.raises(FrozenInstanceError):
            reply.value = "changed"  # type: ignore[misc]


class TestToolApprovalDecisionDataclass:
    def test_tool_approval_decision_dataclass(self) -> None:
        """``ToolApprovalDecision`` stores required fields and defaults optional ones."""
        decision = ToolApprovalDecision(call_id="call-42", approved=True)

        assert decision.call_id == "call-42"
        assert decision.approved is True
        assert decision.reason == ""
        assert decision.message == ""

    def test_tool_approval_decision_with_rejection(self) -> None:
        """``ToolApprovalDecision`` stores reason and message on rejection."""
        decision = ToolApprovalDecision(
            call_id="call-99",
            approved=False,
            reason="policy violation",
            message="Tool call denied by policy.",
        )

        assert decision.approved is False
        assert decision.reason == "policy violation"
        assert decision.message == "Tool call denied by policy."

    def test_tool_approval_decision_is_frozen(self) -> None:
        """``ToolApprovalDecision`` is immutable — attribute assignment raises ``FrozenInstanceError``."""
        from dataclasses import FrozenInstanceError

        decision = ToolApprovalDecision(call_id="c", approved=True)

        with pytest.raises(FrozenInstanceError):
            decision.approved = False  # type: ignore[misc]


class TestTroopAIWorkflowHasSignalMethods:
    def test_troopai_workflow_has_signal_methods(self) -> None:
        """``TroopAIWorkflow`` exposes the expected signal, query, and update methods."""
        assert callable(TroopAIWorkflow.send_human_reply)
        assert callable(TroopAIWorkflow.get_state)
        assert callable(TroopAIWorkflow.approve_tool_call)

    def test_troopai_workflow_has_run_method(self) -> None:
        """``TroopAIWorkflow`` has an async ``run`` method."""
        import inspect

        assert callable(TroopAIWorkflow.run)
        assert inspect.iscoroutinefunction(TroopAIWorkflow.run)

    def test_troopai_workflow_default_agents_empty(self) -> None:
        """``TroopAIWorkflow.__troopai_agents__`` defaults to an empty sequence."""
        assert len(TroopAIWorkflow.__troopai_agents__) == 0


class TestConsumeRepliesClears:
    def test_consume_replies_clears(self) -> None:
        """``consume_replies`` returns queued replies and empties the queue."""
        wf = TroopAIWorkflow()
        wf._pending_replies.append(HumanReply(node_id="n1", value="yes"))
        wf._pending_replies.append(HumanReply(node_id="n2", value="no"))

        result = wf.consume_replies()

        assert len(result) == 2
        assert result[0].node_id == "n1"
        assert result[1].node_id == "n2"
        assert len(wf._pending_replies) == 0

    def test_consume_replies_empty_returns_empty_list(self) -> None:
        """``consume_replies`` returns an empty list when no replies are queued."""
        wf = TroopAIWorkflow()

        result = wf.consume_replies()

        assert result == []


class TestGetStateCancellationReason:
    """``get_state()`` includes a ``"cancellation_reason"`` key.

    ``workflow.in_workflow()`` returns ``False`` outside a Temporal worker, so
    the production code path that calls ``workflow.cancellation_reason()`` is
    not reached in unit tests.  The guard in ``get_state()`` falls through to
    ``setdefault("cancellation_reason", None)``, which means:

    - Tests can verify the key is always present in the snapshot.
    - The path that calls ``workflow.cancellation_reason()`` is verified by
      mocking ``workflow.in_workflow()`` and ``workflow.cancellation_reason()``.
    """

    def test_get_state_contains_cancellation_reason_key(self) -> None:
        """``get_state()`` always contains ``"cancellation_reason"``."""
        wf = TroopAIWorkflow()

        state = wf.get_state()

        assert "cancellation_reason" in state

    def test_get_state_cancellation_reason_none_outside_workflow(self) -> None:
        """Outside a workflow runtime ``"cancellation_reason"`` is ``None``."""
        wf = TroopAIWorkflow()

        state = wf.get_state()

        assert state["cancellation_reason"] is None

    def test_get_state_preserves_existing_keys(self) -> None:
        """``get_state()`` returns user-set state alongside ``"cancellation_reason"``."""
        wf = TroopAIWorkflow()
        wf.update_state({"step": "embedding", "progress": 42})

        state = wf.get_state()

        assert state["step"] == "embedding"
        assert state["progress"] == 42
        assert "cancellation_reason" in state

    def test_get_state_cancellation_reason_in_workflow_context(self) -> None:
        """When ``in_workflow()`` is ``True``, ``cancellation_reason()`` is called.

        This test mocks the temporalio workflow module so the unit test does
        not require a live Temporal server.  The mock simulates a workflow that
        received a cancellation request with the reason ``"user requested"``.
        """
        from unittest.mock import patch

        import temporalio.workflow as _tw

        wf = TroopAIWorkflow()
        wf.update_state({"phase": "running"})

        with (
            patch.object(_tw, "in_workflow", return_value=True),
            patch.object(_tw, "cancellation_reason", return_value="user requested"),
        ):
            state = wf.get_state()

        assert state["cancellation_reason"] == "user requested"
        assert state["phase"] == "running"

    def test_get_state_cancellation_reason_none_when_not_cancelled(self) -> None:
        """When ``in_workflow()`` is ``True`` and no cancellation was sent, reason is ``None``."""
        from unittest.mock import patch

        import temporalio.workflow as _tw

        wf = TroopAIWorkflow()

        with (
            patch.object(_tw, "in_workflow", return_value=True),
            patch.object(_tw, "cancellation_reason", return_value=None),
        ):
            state = wf.get_state()

        assert state["cancellation_reason"] is None


class TestConsumeApprovalPops:
    def test_consume_approval_pops(self) -> None:
        """``consume_approval`` returns the stored decision and removes it."""
        wf = TroopAIWorkflow()
        decision = ToolApprovalDecision(call_id="call-1", approved=True)
        wf._approval_decisions["call-1"] = decision

        result = wf.consume_approval("call-1")

        assert result is decision
        assert "call-1" not in wf._approval_decisions

    def test_consume_approval_returns_none_on_miss(self) -> None:
        """``consume_approval`` returns ``None`` when the call_id is not present."""
        wf = TroopAIWorkflow()

        result = wf.consume_approval("nonexistent-call")

        assert result is None

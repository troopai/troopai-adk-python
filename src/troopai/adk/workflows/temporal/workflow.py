"""Base workflow class and HITL payload types for TroopAI Temporal workflows.

Provides :class:`TroopAIWorkflow` — a :func:`~temporalio.workflow.defn`-decorated
base class that wires up the Human-In-The-Loop (HITL) protocol.  Concrete
workflow implementations subclass it and override :meth:`~TroopAIWorkflow.run`.

Also exposes two frozen dataclasses used as HITL signal/update payloads:

- :class:`HumanReply` — carries a human's response to an interrupted agent node.
- :class:`ToolApprovalDecision` — carries an approval or rejection for a deferred
  tool call.

References:
    Temporal Python SDK workflow docs:
    https://docs.temporal.io/develop/python/core-application#develop-workflows
    Temporal signals and queries:
    https://docs.temporal.io/develop/python/message-passing
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from dataclasses import field
from typing import Any

logger = logging.getLogger(__name__)


# ==================================================================
# HITL payload types
# ==================================================================


@dataclasses.dataclass(frozen=True, kw_only=True)
class HumanReply:
    """Payload for a human response sent to an interrupted agent node.

    Attributes:
        node_id: Identifier of the interrupted node or agent awaiting the reply.
        value: The human's response content.
        metadata: Optional key-value metadata attached to the reply.
    """

    node_id: str
    """Identifier of the interrupted node or agent awaiting this reply."""

    value: str
    """The human's response content."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional key-value metadata attached to the reply."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class ToolApprovalDecision:
    """Payload for a human approval or rejection of a deferred tool call.

    Attributes:
        call_id: Identifier of the deferred tool call.
        approved: ``True`` if the tool call is approved; ``False`` to reject.
        reason: Internal audit reason (not shown to the LLM).
        message: Optional message shown to the LLM when the call is rejected.
    """

    call_id: str
    """Identifier of the deferred tool call."""

    approved: bool
    """``True`` if the tool call is approved; ``False`` to reject it."""

    reason: str = ""
    """Internal audit reason (not forwarded to the LLM)."""

    message: str = ""
    """Optional message shown to the LLM when *approved* is ``False``."""


# ==================================================================
# Base workflow class (registered only when temporalio is available)
# ==================================================================

try:
    from temporalio import workflow

    @workflow.defn
    class TroopAIWorkflow:
        """Base Temporal workflow that implements the HITL protocol.

        Subclasses override :meth:`run` and call the helper methods below to
        exchange HITL messages with external actors.

        The class attribute :attr:`__troopai_agents__` is a sequence of agent
        instances that the workflow orchestrates.  Subclasses populate it at
        class definition time.

        Attributes:
            __troopai_agents__: Agent instances orchestrated by this workflow.
                Defaults to an empty tuple; subclasses override at class level.
        """

        __troopai_agents__: Sequence[Any] = ()

        def __init__(self) -> None:
            self._pending_replies: list[HumanReply] = []
            self._approval_decisions: dict[str, ToolApprovalDecision] = {}
            self._current_state: dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Signal handlers
        # ------------------------------------------------------------------

        @workflow.signal
        def send_human_reply(self, reply: HumanReply) -> None:
            """Signal: receive a human reply for an interrupted node.

            Appends *reply* to the internal queue.  Callers drain the queue via
            :meth:`consume_replies`.

            Args:
                reply: The human response to enqueue.
            """
            self._pending_replies.append(reply)
            workflow.logger.info("Received human reply for node_id=%r", reply.node_id)

        # ------------------------------------------------------------------
        # Query handlers
        # ------------------------------------------------------------------

        @workflow.query
        def get_state(self) -> dict[str, Any]:
            """Query: return the current workflow state snapshot.

            The returned dict includes all key-value pairs set via
            :meth:`update_state` plus a ``"cancellation_reason"`` key whose
            value is the reason string supplied with the most recent external
            cancellation request, or ``None`` when the workflow has not been
            cancelled.

            ``"cancellation_reason"`` is sourced from
            ``temporalio.workflow.cancellation_reason()``, which returns a
            non-``None`` value (including the empty string) only after the
            Temporal server has delivered an explicit cancellation request.
            It is not set for inner ``asyncio`` task cancels or cache eviction.

            Returns:
                A copy of the internal state dict augmented with
                ``"cancellation_reason"``.
            """
            snapshot = dict(self._current_state)
            # Guard: cancellation_reason() requires an active workflow runtime
            # (raises RuntimeError outside one). in_workflow() is the
            # recommended guard from the temporalio SDK (_context.py).
            if workflow.in_workflow():
                snapshot["cancellation_reason"] = workflow.cancellation_reason()
            else:
                # The framework owns this key in both branches — a value
                # written via update_state must not shadow the runtime signal.
                snapshot["cancellation_reason"] = None
            return snapshot

        # ------------------------------------------------------------------
        # Update handlers
        # ------------------------------------------------------------------

        @workflow.update
        def approve_tool_call(self, decision: ToolApprovalDecision) -> None:
            """Update: record an approval or rejection for a deferred tool call.

            Stores *decision* keyed by :attr:`~ToolApprovalDecision.call_id`.
            Callers retrieve the decision via :meth:`consume_approval`.

            Args:
                decision: The approval or rejection decision to store.
            """
            self._approval_decisions[decision.call_id] = decision
            workflow.logger.info(
                "Received tool approval decision: call_id=%r approved=%r",
                decision.call_id,
                decision.approved,
            )

        # ------------------------------------------------------------------
        # Internal helpers (called by subclass run() implementations)
        # ------------------------------------------------------------------

        def consume_replies(self) -> list[HumanReply]:
            """Return and clear all pending human replies.

            Returns:
                The list of replies received since the last call.  The internal
                queue is emptied before returning.
            """
            replies = list(self._pending_replies)
            self._pending_replies.clear()
            return replies

        def consume_approval(self, call_id: str) -> ToolApprovalDecision | None:
            """Pop and return the approval decision for *call_id*, if present.

            Args:
                call_id: The tool call identifier to look up.

            Returns:
                The stored :class:`ToolApprovalDecision`, or ``None`` if no
                decision has been recorded for *call_id*.
            """
            return self._approval_decisions.pop(call_id, None)

        def update_state(self, state: dict[str, Any]) -> None:
            """Merge *state* into the current workflow state snapshot.

            Args:
                state: Key-value pairs to merge into :attr:`_current_state`.
            """
            self._current_state.update(state)

        # ------------------------------------------------------------------
        # Run entrypoint (subclasses override)
        # ------------------------------------------------------------------

        @workflow.run
        async def run(self, input: Any) -> Any:
            """Workflow entrypoint — subclasses MUST override this method.

            Args:
                input: Workflow input as supplied by the Temporal client.

            Raises:
                NotImplementedError: Always — subclasses must provide an
                    implementation.
            """
            raise NotImplementedError(f"{type(self).__name__} must implement run()")

except ImportError:
    pass

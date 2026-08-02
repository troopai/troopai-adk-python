"""Run state for resuming interrupted agent execution.

This module provides the RunState class which enables serialization and
resumption of agent runs, particularly for Human-in-the-Loop (HITL) workflows
where execution may be paused for human approval.

Example:
    # First run - may be interrupted
    result = await Runner.arun(agent, "Delete user 123")

    # Save state for later (compact JSON transport)
    await db.save_pending_approval(result.state.to_json())

    # Later: Load and resume
    state = RunState.from_json(await db.get_pending_approval(id))
    state.approve(
        state.deferred_requests.approvals[0],
        approver_id="alice@example.com",
        reason="authorized destructive op",
    )
    result = await Runner.arun(agent, state)
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from troopai.adk.run.types import UserPrompt
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.items.items import RunItem

from troopai.adk.tools.deferred_tool import (
    DeferredToolCall,
    DeferredToolCallMetadata,
    DeferredToolRequests,
    DeferredToolResults,
    ExternalToolCallResult,
    ToolApprovalResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ApprovalMetadata:
    """Audit metadata for a human approval decision.

    Captured when ``RunState.approve()`` or ``.reject()`` is called with
    an ``approver_id`` / ``reason``. Indexed by ``tool_call_id`` on the
    containing ``RunState``; the raw ``approved_tools`` /
    ``rejected_tools`` lists remain the resumption-driver fields and
    this structured audit metadata is stored separately.

    Attributes:
        approver_id: Opaque identifier for the human/service that made
            the decision — email, user id, api key fingerprint, etc.
            ``None`` when not supplied.
        reason: Free-form rationale. Shown to the model only if the
            caller separately passes a rejection ``message``.
        timestamp: When the decision was recorded. Defaults to "now".
    """

    approver_id: str | None = None
    """Opaque identifier for the approver (email, user id, etc.)."""

    reason: str | None = None
    """Free-form rationale for the decision."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When the decision was recorded."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "approver_id": self.approver_id,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalMetadata:
        """Deserialize from a JSON-friendly dict."""
        raw_ts = data.get("timestamp")
        ts = datetime.fromisoformat(raw_ts) if raw_ts is not None else datetime.now()
        return cls(
            approver_id=data.get("approver_id"),
            reason=data.get("reason"),
            timestamp=ts,
        )


@dataclass
class RunState:
    """Serializable state for resuming interrupted agent runs.

    This class captures all necessary state to resume an agent run that was
    interrupted for human approval or external tool execution. It supports
    serialization to/from dictionaries for persistence.

    Attributes:
        conversation_history: The full conversation history up to the
            interruption (Layer 3 RunItems).
        context: User-provided context data.
        deferred_tool_requests: Tools that were deferred for approval
            or external execution.
        approved_tools: Tools that have been approved for execution on
            resume.
        rejected_tools: Tools that have been rejected with optional
            model-visible messages.
        approval_metadata: Audit metadata for each approval decision,
            keyed by ``tool_call_id``. Populated only when
            ``approver_id`` or ``reason`` are supplied to
            :meth:`approve` / :meth:`reject`.
        original_user_prompt: The original input that started this
            run.
        current_agent_name: Name of the agent currently executing.
        turn_count: Current turn count in the agent loop.
        metadata: Additional metadata for the run.

    Example:
        # Create from interrupted run
        state = RunState(
            conversation_history=messages,
            context={"user_id": "123"},
            deferred_requests=deferred_requests,
        )

        # Approve a tool
        state.approve(deferred_requests.approvals[0])

        # Or reject it
        state.reject(deferred_requests.approvals[0], "Not authorized")

        # Serialize for storage
        state_dict = state.to_dict()

        # Deserialize later
        state = RunState.from_dict(state_dict)
    """

    conversation_history: list[RunItem] = field(default_factory=list)
    """The full conversation history up to the interruption (Layer 3 RunItems)."""

    context: Any = None
    """User-provided context data."""

    deferred_tool_requests: DeferredToolRequests = field(default_factory=DeferredToolRequests)
    """Tools that were deferred for approval/execution."""

    approved_tools: list[DeferredToolCall] = field(default_factory=list)
    """Tools that have been approved for execution on resume."""

    rejected_tools: list[tuple[DeferredToolCall, str | None]] = field(default_factory=list)
    """Tools that have been rejected with optional messages."""

    approval_metadata: dict[str, ApprovalMetadata] = field(default_factory=dict)
    """Audit metadata for each approval decision, keyed by ``tool_call_id``.

    Populated opportunistically: ``approve()`` / ``reject()`` only write
    here when ``approver_id`` or ``reason`` are supplied. Callers that
    do not need audit metadata see an empty dict — the existing
    ``approved_tools`` / ``rejected_tools`` lists still drive resumption.
    """

    _external_results: list[ExternalToolCallResult] = field(default_factory=list, init=False, repr=False)
    """Externally-provided tool results for resumption (internal).

    ``init=False`` keeps this off the public constructor — callers MUST
    use ``provide_result()`` to append entries, never the
    constructor. External code MUST NOT touch the backing list
    directly; the ``external_results`` property is read-only (returns
    a shallow copy) and the sanctioned write path is
    ``provide_result()``.
    """

    original_user_prompt: UserPrompt = ""
    """The original input that started this run."""

    current_agent_name: str | None = None
    """Name of the agent currently executing."""

    turn_count: int = 0
    """Current turn count in the agent loop."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the run."""

    def to_input_list(self) -> list[LLMInputContentItem]:
        """Convert conversation history to input list for continued conversation.

        Converts the Layer 3 RunItems stored in ``conversation_history``
        back to Layer 1 params, suitable for passing as input to
        a subsequent ``Runner.arun()`` call.
        """
        from troopai.adk.types.items.items import ItemHelpers

        return ItemHelpers.run_items_to_params(self.conversation_history)

    def approve(
        self,
        tool_call: DeferredToolCall,
        *,
        approver_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Mark a tool call as approved for execution on resume.

        Args:
            tool_call: The deferred tool call to approve.
            approver_id: Optional opaque identifier for the human or
                service that approved this call (email, user id,
                service token fingerprint, etc.). Stored on
                ``approval_metadata`` for audit purposes.
            reason: Optional free-form rationale for the approval.
                Stored on ``approval_metadata``. Not shown to the
                model on resume.
        """
        call_id = tool_call.tool_call_id
        # Remove from deferred_requests if present
        if tool_call in self.deferred_tool_requests.approvals:
            self.deferred_tool_requests.approvals.remove(tool_call)

        # Supersede a prior rejection of the same call so the latest
        # decision wins. Without this cross-removal the call lands in BOTH
        # approved_tools and rejected_tools, and resume emits two
        # function_call_output items — the tool then executes AND is
        # reported rejected under one call_id, a double, contradictory
        # exchange. Drop the stale rejection audit alongside it.
        if any(t.tool_call_id == call_id for t, _ in self.rejected_tools):
            self.rejected_tools = [(t, m) for t, m in self.rejected_tools if t.tool_call_id != call_id]
            self.approval_metadata.pop(call_id, None)

        # Add to approved list if not already there
        if tool_call not in self.approved_tools:
            self.approved_tools.append(tool_call)

        if approver_id is not None or reason is not None:
            self.approval_metadata[call_id] = ApprovalMetadata(
                approver_id=approver_id,
                reason=reason,
            )

    def reject(
        self,
        tool_call: DeferredToolCall,
        message: str | None = None,
        *,
        approver_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Mark a tool call as rejected with an optional message.

        Args:
            tool_call: The deferred tool call to reject.
            message: Optional rejection message to show the model on
                resumption. This is the model-visible explanation.
            approver_id: Optional opaque identifier for the human or
                service that rejected this call. Stored on
                ``approval_metadata`` for audit purposes; not shown
                to the model.
            reason: Optional internal rationale for the rejection.
                Stored on ``approval_metadata``; not shown to the
                model. Use ``message`` if you want the model to see it.
        """
        call_id = tool_call.tool_call_id
        # Remove from deferred_requests if present
        if tool_call in self.deferred_tool_requests.approvals:
            self.deferred_tool_requests.approvals.remove(tool_call)

        # Supersede a prior approval of the same call so the latest decision
        # wins (mirrors approve()'s cross-removal). Otherwise the call stays
        # in approved_tools and resume executes it despite the rejection,
        # emitting two function_call_output items for one call_id. Drop the
        # stale approval audit alongside it.
        if any(t.tool_call_id == call_id for t in self.approved_tools):
            self.approved_tools = [t for t in self.approved_tools if t.tool_call_id != call_id]
            self.approval_metadata.pop(call_id, None)

        # Guard against duplicate rejections (mirrors approve()'s dedup).
        # If the same tool_call is rejected again (e.g. to update the message),
        # replace the existing entry so the resumed LLM never sees two
        # function_call_output items for the same call_id.
        self.rejected_tools = [(t, m) for t, m in self.rejected_tools if t.tool_call_id != call_id]
        self.rejected_tools.append((tool_call, message))

        if approver_id is not None or reason is not None:
            self.approval_metadata[call_id] = ApprovalMetadata(
                approver_id=approver_id,
                reason=reason,
            )

    def provide_result(self, tool_call: DeferredToolCall, result: Any) -> None:
        """Provide an externally-produced result for a deferred tool call.

        Use this when a tool call needs to be executed outside the agent
        runtime (e.g., an MCP tool run by an external service). The
        result is fed back to the LLM on resumption.

        Args:
            tool_call: The deferred tool call to provide a result for.
            result: The output produced by external execution.
        """
        # Remove from deferred_requests.calls if present
        if tool_call in self.deferred_tool_requests.calls:
            self.deferred_tool_requests.calls.remove(tool_call)

        # Guard against duplicate results (mirrors approve()/reject()'s dedup).
        # A repeated provide_result for the same call_id (retry, idempotency,
        # two workers racing) must replace the earlier entry so the resumed
        # LLM never sees two function_call_output items for the same call_id.
        self._external_results = [r for r in self._external_results if r.call_id != tool_call.tool_call_id]
        self._external_results.append(
            ExternalToolCallResult(
                call_id=tool_call.tool_call_id,
                output=result,
            )
        )

    @property
    def external_results(self) -> list[ExternalToolCallResult]:
        """Externally-provided tool results for resumption.

        Returns a shallow copy so callers cannot mutate the internal
        list via ``state.external_results.append(...)`` and bypass the
        sanctioned ``provide_result()`` write path. The audit
        contract for HITL approval/rejection lives on that method, so
        the mutable reference would silently undermine it.
        """
        return list(self._external_results)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the run state to a dictionary.

        Returns:
            A dictionary representation of the state that can be JSON serialized.
        """
        from troopai.adk.types.items.items import ItemHelpers

        return {
            "conversation_history": ItemHelpers.run_items_to_params(self.conversation_history),
            "context": self._serialize_context(self.context),
            "deferred_requests": self._serialize_deferred_requests(self.deferred_tool_requests),
            "approved_tools": [self._serialize_deferred_tool(t) for t in self.approved_tools],
            "rejected_tools": [
                {
                    "tool": self._serialize_deferred_tool(t),
                    "message": m,
                }
                for t, m in self.rejected_tools
            ],
            "approval_metadata": {call_id: meta.to_dict() for call_id, meta in self.approval_metadata.items()},
            "original_user_prompt": self.original_user_prompt
            if isinstance(self.original_user_prompt, str)
            else json.dumps(self.original_user_prompt),
            "current_agent_name": self.current_agent_name,
            "turn_count": self.turn_count,
            "metadata": self.metadata,
            "external_results": [r.to_dict() for r in self._external_results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        """Deserialize a run state from a dictionary.

        Args:
            data: The dictionary containing the serialized state.

        Returns:
            A RunState instance reconstructed from the dictionary.
        """
        # Recover a list-form prompt. ``to_dict`` stores a ``str`` prompt
        # verbatim and ``json.dumps``-es the only other ``UserPrompt`` shape
        # (``list[LLMInputContentItem]``) into a JSON array string. Only a
        # parse that yields a ``list`` is a serialized list-prompt; a string
        # prompt that merely happens to be valid JSON ("42", "null", "true",
        # '{"k": 1}') must be kept as-is, not coerced to int/None/bool/dict.
        original_user_prompt = data.get("original_user_prompt", "")
        if isinstance(original_user_prompt, str):
            parsed: Any = None
            with contextlib.suppress(json.JSONDecodeError):
                parsed = json.loads(original_user_prompt)
            if isinstance(parsed, list):
                original_user_prompt = parsed

        from troopai.adk.types.items.items import ItemHelpers

        state = cls(
            conversation_history=list(ItemHelpers.messages_to_run_items(data.get("conversation_history", []))),
            context=data.get("context"),
            deferred_tool_requests=cls._deserialize_deferred_requests(data.get("deferred_requests", {})),
            approved_tools=[cls._deserialize_deferred_tool(t) for t in data.get("approved_tools", [])],
            rejected_tools=[
                (
                    cls._deserialize_deferred_tool(item["tool"]),
                    item.get("message"),
                )
                for item in data.get("rejected_tools", [])
            ],
            approval_metadata={
                call_id: ApprovalMetadata.from_dict(meta) for call_id, meta in data.get("approval_metadata", {}).items()
            },
            original_user_prompt=original_user_prompt,
            current_agent_name=data.get("current_agent_name"),
            turn_count=data.get("turn_count", 0),
            metadata=data.get("metadata", {}),
        )
        # Restore external results — the field is init=False so it cannot be
        # set via the constructor; assign directly after construction.
        state._external_results = [ExternalToolCallResult.from_dict(r) for r in data.get("external_results", [])]
        return state

    def to_json(self) -> str:
        """Serialize the run state to a JSON string.

        Thin wrapper over ``to_dict()``. Fields added in later builds
        carry safe defaults and ``from_dict`` reads every field with a
        default, so a string produced by an earlier build loads back
        through ``from_json()`` without special handling.

        Returns:
            A JSON string suitable for persistence or transport.

        Raises:
            TypeError: If any field in the state is not JSON serializable
                (e.g. a ``context`` object that did not override
                ``_serialize_context``).
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, data: str) -> RunState:
        """Deserialize a run state from a JSON string.

        Tolerant: ``from_dict`` reads each field via ``dict.get`` with
        a safe default, so a payload from an earlier build (missing
        later-added keys) loads cleanly and any extra keys it does not
        recognise are ignored.

        Args:
            data: A JSON string previously produced by ``to_json()``.

        Returns:
            A reconstructed ``RunState``.

        Raises:
            json.JSONDecodeError: If ``data`` is not valid JSON.
        """
        logger.debug("RunState.from_json: deserialising %d-char payload", len(data))
        return cls.from_dict(json.loads(data))

    def to_deferred_results(self) -> DeferredToolResults:
        """Convert approved/rejected tools to DeferredToolResults.

        Returns:
            DeferredToolResults containing all approval decisions.
        """
        approvals = []

        # Add approved tools
        for tool in self.approved_tools:
            approvals.append(
                ToolApprovalResult(
                    tool_call_id=tool.tool_call_id,
                    approved=True,
                    message=None,
                )
            )

        # Add rejected tools
        for tool, message in self.rejected_tools:
            approvals.append(
                ToolApprovalResult(
                    tool_call_id=tool.tool_call_id,
                    approved=False,
                    message=message,
                )
            )

        return DeferredToolResults(approvals=approvals, call_results=[])

    @staticmethod
    def _serialize_context(context: Any) -> Any:
        """Serialize context data.

        Override this method if your context needs special serialization.
        By default, returns the context as-is assuming it's JSON serializable.
        """
        return context

    @staticmethod
    def _serialize_deferred_requests(requests: DeferredToolRequests) -> dict[str, Any]:
        """Serialize DeferredToolRequests to a dictionary."""
        return {
            "approvals": [RunState._serialize_deferred_tool(t) for t in requests.approvals],
            "calls": [RunState._serialize_deferred_tool(t) for t in requests.calls],
        }

    @staticmethod
    def _serialize_deferred_tool(tool: DeferredToolCall) -> dict[str, Any]:
        """Serialize a DeferredToolCall to a dictionary."""
        return {
            "tool_call_id": tool.tool_call_id,
            "tool_name": tool.tool_name,
            "tool_arguments": tool.tool_arguments,
            "raw_arguments": tool.raw_arguments,
            "request_time": tool.request_time.isoformat(),
            "metadata": tool.metadata.to_dict() if tool.metadata is not None else None,
        }

    @staticmethod
    def _deserialize_deferred_requests(data: dict[str, Any]) -> DeferredToolRequests:
        """Deserialize DeferredToolRequests from a dictionary."""
        return DeferredToolRequests(
            approvals=[RunState._deserialize_deferred_tool(t) for t in data.get("approvals", [])],
            calls=[RunState._deserialize_deferred_tool(t) for t in data.get("calls", [])],
        )

    @staticmethod
    def _deserialize_deferred_tool(data: dict[str, Any]) -> DeferredToolCall:
        """Deserialize a DeferredToolCall from a dictionary."""
        raw_metadata = data.get("metadata")
        metadata = DeferredToolCallMetadata.from_dict(raw_metadata) if raw_metadata is not None else None
        return DeferredToolCall(
            tool_call_id=data["tool_call_id"],
            tool_name=data["tool_name"],
            tool_arguments=data.get("tool_arguments", {}),
            raw_arguments=data.get("raw_arguments", ""),
            request_time=datetime.fromisoformat(data["request_time"]),
            metadata=metadata,
        )

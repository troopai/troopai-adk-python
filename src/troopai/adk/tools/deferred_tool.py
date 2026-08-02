"""Deferred tool execution types for Human-in-the-Loop (HITL) support.

This module provides types for capturing tool calls that require human approval
or external execution. It enables the interruption/resumption pattern for
sensitive operations.

Example:
    # First run - may be interrupted for approval
    result = await Runner.arun(agent, "Delete user 123")

    # Handle interruptions
    while result.requires_action:
        for req in result.deferred_requests.approvals:
            logger.info(f"Approval requested: {req.tool_name}")
            confirmed = await ask_user_for_approval(req)

            if confirmed:
                result.state.approve(req)
            else:
                result.state.reject(req, message="User denied")

        # Resume execution
        result = await Runner.arun(agent, result.state)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DeferredToolCallMetadata:
    """Typed metadata for deferred tool calls, primarily for nested HITL.

    When a sub-agent (running via ``as_tool()``) defers a tool call for
    human approval, the parent runner wraps the deferral in a
    ``DeferredToolCall`` with this metadata. It carries everything needed
    to resume the sub-agent after the user approves or rejects.

    Frozen because metadata is set once at deferral time and must not be
    mutated afterward — any change would desynchronize the nested state.

    Attributes:
        nested_agent: Whether this deferral originates from a nested
            sub-agent invoked via ``as_tool()``.
        nested_agent_name: The ``name`` of the sub-agent that deferred.
            Used in rejection messages and user-facing display.
        nested_state: Serialized ``RunState`` of the sub-agent, produced
            by ``RunState.to_dict()``. Contains conversation history and
            deferred requests needed to resume the sub-agent.
        nested_deferred_requests: Typed summary of the sub-agent's
            deferred approval requests.

    Example:
        metadata = DeferredToolCallMetadata(
            nested_agent=True,
            nested_agent_name="Researcher",
            nested_state=sub_state.to_dict(),
            nested_deferred_requests=NestedDeferredToolRequests(
                approvals=[
                    DeferredToolCall(
                        tool_call_id="abc", tool_name="search",
                        tool_arguments={}, raw_arguments="{}",
                    )
                ]
            ),
        )
    """

    nested_agent: bool = False
    """bool: Whether this deferral comes from a nested sub-agent.

    When ``True``, the runner uses ``_resume_nested_agent_tool()``
    instead of directly executing the tool on resumption.
    """

    nested_agent_name: str | None = None
    """Optional[str]: Name of the sub-agent that deferred.

    Used in rejection messages to tell the parent LLM which
    sub-agent's operation was denied.
    """

    nested_state: dict[str, Any] | None = None
    """Optional[dict[str, Any]]: Serialized sub-agent RunState.

    Produced by ``RunState.to_dict()`` and consumed by
    ``RunState.from_dict()`` during resumption.
    """

    nested_deferred_requests: NestedDeferredToolRequests | None = None
    """Optional[NestedDeferredToolRequests]: Sub-agent's deferred approval requests.

    Each ``DeferredToolCall`` captures the identity of a tool call
    the sub-agent deferred (tool name, arguments, call ID).
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for persistence.

        Nested ``request_time`` datetimes are converted to ISO strings so
        the result is JSON-serializable; ``dataclasses.asdict()`` would
        leave them as raw ``datetime`` objects and break ``json.dumps``.

        Returns:
            A JSON-serializable dictionary representation.
        """
        nested_requests: dict[str, Any] | None = None
        if self.nested_deferred_requests is not None:
            nested_requests = {
                "approvals": [
                    {
                        "tool_call_id": a.tool_call_id,
                        "tool_name": a.tool_name,
                        "tool_arguments": a.tool_arguments,
                        "raw_arguments": a.raw_arguments,
                        "request_time": a.request_time.isoformat(),
                    }
                    for a in self.nested_deferred_requests.approvals
                ]
            }
        return {
            "nested_agent": self.nested_agent,
            "nested_agent_name": self.nested_agent_name,
            "nested_state": self.nested_state,
            "nested_deferred_requests": nested_requests,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeferredToolCallMetadata:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary produced by ``to_dict()``.

        Returns:
            A ``DeferredToolCallMetadata`` instance.
        """
        raw_requests = data.get("nested_deferred_requests")
        return cls(
            nested_agent=data.get("nested_agent", False),
            nested_agent_name=data.get("nested_agent_name"),
            nested_state=data.get("nested_state"),
            nested_deferred_requests=(
                NestedDeferredToolRequests.from_dict(raw_requests) if raw_requests is not None else None
            ),
        )


@dataclass
class DeferredToolCall:
    """A tool call captured for external approval or execution.

    This represents a tool invocation that was deferred instead of being
    executed immediately, typically because it requires human approval
    or external processing.

    Attributes:
        tool_call_id: Unique identifier for this tool call (from the LLM).
        tool_name: The name of the tool being invoked.
        tool_arguments: The parsed arguments for the tool call.
        raw_arguments: The raw JSON string of arguments from the LLM.
        request_time: When the tool call was captured.
        metadata: Typed metadata for nested HITL propagation. ``None``
            for regular (non-nested) deferrals.
    """

    tool_call_id: str
    """Unique identifier for this tool call (from the LLM)."""

    tool_name: str
    """The name of the tool being invoked."""

    tool_arguments: dict[str, Any]
    """The parsed arguments for the tool call."""

    raw_arguments: str
    """The raw JSON string of arguments from the LLM."""

    request_time: datetime = field(default_factory=datetime.now)
    """When the tool call was captured."""

    metadata: DeferredToolCallMetadata | None = None
    """Optional[DeferredToolCallMetadata]: Typed metadata for nested HITL.

    Set when this deferral originates from a sub-agent invoked via
    ``as_tool()``. ``None`` for regular (non-nested) deferrals.
    """


@dataclass(frozen=True)
class NestedDeferredToolRequests:
    """Typed container for a sub-agent's deferred approval requests.

    A typed container for the sub-agent's deferred approval requests;
    each approval is a standard ``DeferredToolCall`` (reused rather than
    duplicated).

    Attributes:
        approvals: Tool calls the sub-agent deferred for approval.
    """

    approvals: list[DeferredToolCall] = field(default_factory=list)
    """list[DeferredToolCall]: Tool calls the sub-agent deferred."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NestedDeferredToolRequests:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary with an ``"approvals"`` key.

        Returns:
            A ``NestedDeferredToolRequests`` instance.
        """
        approvals = []
        for a in data.get("approvals", []):
            request_time = None
            if "request_time" in a:
                rt = a["request_time"]
                request_time = datetime.fromisoformat(rt) if isinstance(rt, str) else rt
            # Default to datetime.now() when request_time is absent,
            # matching the field's default_factory behavior
            request_time_resolved = request_time if request_time is not None else datetime.now()
            approvals.append(
                DeferredToolCall(
                    tool_call_id=a["tool_call_id"],
                    tool_name=a["tool_name"],
                    tool_arguments=a.get("tool_arguments", {}),
                    raw_arguments=a.get("raw_arguments", ""),
                    request_time=request_time_resolved,
                )
            )
        return cls(approvals=approvals)


@dataclass
class DeferredToolRequests:
    """Container for all deferred tool calls in a run.

    When a run is interrupted because tools need approval or external
    execution, this object captures all the deferred requests.

    Per-call metadata (e.g., nested agent state) lives on each
    ``DeferredToolCall.metadata`` rather than at the container level.

    Attributes:
        approvals: Tool calls that need human approval before execution.
        calls: Tool calls for external execution (e.g., MCP tools).
    """

    approvals: list[DeferredToolCall] = field(default_factory=list)
    """Tool calls that need human approval before execution."""

    calls: list[DeferredToolCall] = field(default_factory=list)
    """Tool calls for external execution (e.g., MCP tools)."""

    def is_empty(self) -> bool:
        """Check if there are no deferred requests."""
        return len(self.approvals) == 0 and len(self.calls) == 0


@dataclass
class ToolApprovalResult:
    """Result of a human approval decision for a deferred tool call.

    Attributes:
        tool_call_id: The ID of the tool call being approved/rejected.
        approved: Whether the tool call was approved (True) or rejected (False).
        message: Optional rejection message if denied.
    """

    tool_call_id: str
    """The ID of the tool call being approved/rejected."""

    approved: bool
    """Whether the tool call was approved (True) or rejected (False)."""

    message: str | None = None
    """Optional rejection message if denied."""


@dataclass(frozen=True)
class ExternalToolCallResult:
    """Result from an externally executed tool call.

    When a tool is executed outside the agent runtime (e.g., an MCP tool
    run by an external service), this captures the result so it can be
    fed back into the agent on resumption.

    Attributes:
        call_id: The ID of the tool call this result corresponds to.
        output: The output produced by the external tool execution.
    """

    call_id: str
    """str: The ID of the tool call this result corresponds to.

    Must match the ``tool_call_id`` from the original ``DeferredToolCall``.
    """

    output: Any = None
    """Any: The output produced by the external tool execution.

    Can be any JSON-serializable value. Passed back to the LLM as
    the tool's response.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for persistence.

        Returns:
            A JSON-serializable dictionary representation.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalToolCallResult:
        """Deserialize from a dictionary.

        Args:
            data: Dictionary produced by ``to_dict()``.

        Returns:
            An ``ExternalToolCallResult`` instance.
        """
        return cls(
            call_id=data["call_id"],
            output=data.get("output"),
        )


@dataclass
class DeferredToolResults:
    """Results provided back for deferred tool execution.

    When resuming a run that was interrupted for approvals, this object
    contains the human decisions and any external tool results.

    Attributes:
        approvals: List of approval/rejection decisions.
        call_results: Results from externally executed tools.
    """

    approvals: list[ToolApprovalResult] = field(default_factory=list)
    """List of approval/rejection decisions."""

    call_results: list[ExternalToolCallResult] = field(default_factory=list)
    """list[ExternalToolCallResult]: Results from externally executed tools.

    Each entry maps a ``tool_call_id`` to its externally-produced result.
    """

    def get_approval_result(self, tool_call_id: str) -> ToolApprovalResult | None:
        """Get the approval result for a specific tool call.

        Args:
            tool_call_id: The ID of the tool call to look up.

        Returns:
            The ToolApprovalResult if found, None otherwise.
        """
        for result in self.approvals:
            if result.tool_call_id == tool_call_id:
                return result
        return None

    def is_approved(self, tool_call_id: str) -> bool:
        """Check if a specific tool call was approved.

        Args:
            tool_call_id: The ID of the tool call to check.

        Returns:
            True if the tool call was approved, False otherwise.
        """
        result = self.get_approval_result(tool_call_id)
        return result is not None and result.approved

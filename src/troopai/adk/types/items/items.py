"""Run item types — framework-level conversation items.

``RunItem`` is a Union of typed item classes that wrap raw types.
Every item has a required ``raw: T`` field holding the underlying
data (following the OpenAI agents SDK pattern).  Items provide:

- ``raw: T`` — the underlying data type (always required)
- ``type`` discriminator field for serialization / deserialization
- ``agent_name`` for multi-agent observability
- ``to_param()`` to produce a Layer 1 replay TypedDict for the converter

All data is accessed via ``raw``.  Properties exist only where
they transform data (e.g., concatenating text parts).

Factory functions on ``ItemHelpers`` convert between messages
(Layer 1 or Layer 2 dicts) and items at the handoff boundary.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, assert_never, override

from typing_extensions import TypeVar

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.tools.deferred_tool import DeferredToolCall
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.input.llm_input_easy_message import LLMInputEasyMessage
    from troopai.adk.types.output import FunctionToolCallResult
    from troopai.adk.types.responses.llm_response import (
        LLMResponse,
        LLMResponseFunctionToolCall,
        LLMResponseProviderItem,
        LLMResponseReasoning,
        LLMResponseRefusal,
        LLMResponseText,
    )
    from troopai.adk.types.tools.builtin_tool_types import (
        MCPApprovalRequest,
        MCPApprovalResponse,
        MCPListTools,
        ToolSearchToolCall,
        ToolSearchToolCallResult,
    )

T = TypeVar("T")
"""The raw type that this item wraps."""

logger = logging.getLogger(__name__)


# ==================================================================
# Base class
# ==================================================================


@dataclass(frozen=True)
class RunItemBase[T]:
    """Base class for all run items.

    Generic over ``T`` — the raw type this item wraps.  Every item
    has a required ``raw: T`` field holding the underlying data.
    Access data via ``raw`` directly.

    Provides ``to_param()`` for item → Layer 1 conversion.

    Attributes:
        type: Discriminator field for serialization. Overridden by each
            subclass with a ``Literal`` type.
        agent_name: Name of the agent that produced this item. ``None``
            for items not tied to a specific agent.
        raw: The underlying data for this item. Always required.
    """

    type: str = "run_item"
    """Discriminator field for serialization.  Overridden by each subclass
    with a ``Literal`` type."""

    agent_name: str | None = field(default=None, kw_only=True)
    """Name of the agent that produced this item.  Set by the runner
    for multi-agent observability.  ``None`` for items not tied to
    a specific agent (e.g., ``UserItem``, ``SystemItem``)."""

    raw: T = field(kw_only=True)
    """The underlying data for this item.  Always required."""

    def to_param(self) -> LLMInputContentItem:
        """Convert to Layer 1 replay param (TypedDict) for sending back to the LLM."""
        raise NotImplementedError


# ==================================================================
# Items wrapping input TypeDicts
# ==================================================================


@dataclass(frozen=True)
class SystemItem(RunItemBase["LLMInputEasyMessage"]):
    """A system or developer prompt message.

    Wraps ``LLMInputEasyMessage``.  Access via ``raw["content"]``,
    ``raw["role"]``.
    """

    type: Literal["system"] = "system"
    raw: LLMInputEasyMessage = field(kw_only=True)

    @override
    def to_param(self) -> LLMInputContentItem:
        return self.raw


@dataclass(frozen=True)
class UserItem(RunItemBase["LLMInputEasyMessage"]):
    """A user message.

    Wraps ``LLMInputEasyMessage``.  Access via ``raw["content"]``.
    """

    type: Literal["user"] = "user"
    raw: LLMInputEasyMessage = field(kw_only=True)

    @override
    def to_param(self) -> LLMInputContentItem:
        return self.raw


@dataclass(frozen=True)
class CompactionItem(RunItemBase["LLMInputEasyMessage"]):
    """Compacted/summarized conversation segment.

    Wraps ``LLMInputEasyMessage``.  Access via ``raw["content"]``.
    """

    type: Literal["compaction"] = "compaction"
    raw: LLMInputEasyMessage = field(kw_only=True)

    @override
    def to_param(self) -> LLMInputContentItem:
        return self.raw


# ==================================================================
# Items wrapping LLM output types
# ==================================================================


@dataclass(frozen=True)
class MessageOutputItem(RunItemBase["list[LLMResponseText | LLMResponseRefusal]"]):
    """An assistant message (text and/or refusal content).

    Wraps a list of ``LLMResponseText`` and ``LLMResponseRefusal`` parts
    from ``types/responses/``.  Use ``ItemHelpers`` for text extraction.
    """

    type: Literal["message_output"] = "message_output"
    raw: list[LLMResponseText | LLMResponseRefusal] = field(repr=False, kw_only=True)
    id: str | None = field(default=None, kw_only=True)
    """Unique message ID for session persistence."""
    status: str | None = field(default=None, kw_only=True)
    """Message processing status."""

    @override
    def to_param(self) -> LLMInputContentItem:
        # ``status`` on the TypedDict is ``Literal["in_progress",
        # "completed", "incomplete"]``; ``self.status`` is a plain ``str``
        # from the LLM. Narrow via equality so mypy can assign to the
        # literal slot. We construct whole-literal branches (rather than
        # mutating via subscript after construction) so pyright tracks
        # the TypedDict shape end-to-end.
        #
        # ``self.id`` is deliberately NOT emitted. It holds the provider
        # RESPONSE id (e.g. ``resp_...`` / ``chatcmpl-...`` / ``msg_...``),
        # captured for ``RunResult.last_response_id`` — it is NOT a
        # message-ITEM id. Replaying a response id in the message item's
        # ``id`` slot makes the OpenAI Responses API reject the turn. The
        # real per-message item id is not carried on the response parts,
        # so the message replays without one (the Responses converter then
        # replays it as an id-less assistant message, which is accepted).
        from troopai.adk.types.output import LLMResponseMessageParam

        content_parts = [part.to_param() for part in self.raw]
        status_lit: Literal["in_progress", "completed", "incomplete"] | None
        if self.status == "in_progress":
            status_lit = "in_progress"
        elif self.status == "completed":
            status_lit = "completed"
        elif self.status == "incomplete":
            status_lit = "incomplete"
        else:
            status_lit = None

        result: LLMResponseMessageParam
        if status_lit is not None:
            result = {
                "type": "message",
                "role": "assistant",
                "content": content_parts,
                "status": status_lit,
            }
        else:
            result = {
                "type": "message",
                "role": "assistant",
                "content": content_parts,
            }
        return result


@dataclass(frozen=True)
class ToolCallItem(RunItemBase["LLMResponseFunctionToolCall"]):
    """A function tool call from the assistant.

    Wraps ``LLMResponseFunctionToolCall`` from ``types/responses/``.
    """

    type: Literal["tool_call"] = "tool_call"
    raw: LLMResponseFunctionToolCall = field(repr=False, kw_only=True)
    description: str | None = field(default=None, kw_only=True)
    """Optional tool description for display/observability."""

    @override
    def to_param(self) -> LLMInputContentItem:
        return self.raw.to_param()


@dataclass(frozen=True)
class ToolCallOutputItem(RunItemBase["FunctionToolCallResult"]):
    """Result of executing a function tool call.

    Wraps ``FunctionToolCallResult``.  Use ``ItemHelpers.tool_call_output_str()``
    to coerce output to string.
    """

    type: Literal["tool_call_output"] = "tool_call_output"
    raw: FunctionToolCallResult = field(repr=False, kw_only=True)

    @override
    def to_param(self) -> LLMInputContentItem:
        # Build the typed TypedDict directly instead of round-tripping through
        # ``model_dump`` — ``model_dump`` erases to ``dict[str, Any]``. The
        # ``artifact`` field is intentionally omitted (app-side only, never
        # sent to the LLM). Construct both branches as whole literals so
        # pyright tracks the TypedDict shape end-to-end (post-construction
        # subscript mutation broadens to ``dict[str, Any]``).
        from troopai.adk.types.output import FunctionToolCallResultParam

        if self.raw.id is not None:
            result: FunctionToolCallResultParam = {
                "type": self.raw.type,
                "call_id": self.raw.call_id,
                "output": self.raw.output,
                "id": self.raw.id,
                "status": self.raw.status,
            }
        else:
            result = {
                "type": self.raw.type,
                "call_id": self.raw.call_id,
                "output": self.raw.output,
                "status": self.raw.status,
            }
        return result


@dataclass(frozen=True)
class ReasoningItem(RunItemBase["LLMResponseReasoning"]):
    """Reasoning / chain-of-thought item from the LLM.

    Wraps ``LLMResponseReasoning`` from ``types/responses/``.
    Use ``ItemHelpers.reasoning_summary_text()`` and
    ``ItemHelpers.reasoning_content_text()`` for text access.
    """

    type: Literal["reasoning"] = "reasoning"
    raw: LLMResponseReasoning = field(repr=False, kw_only=True)

    @override
    def to_param(self) -> LLMInputContentItem:
        return self.raw.to_param()


# ==================================================================
# Handoff items
# ==================================================================


@dataclass(frozen=True)
class HandoffCallItem(RunItemBase["LLMResponseFunctionToolCall"]):
    """The ``transfer_to_X`` tool call that triggered a handoff.

    Wraps ``LLMResponseFunctionToolCall`` from ``types/responses/``.
    """

    type: Literal["handoff_call"] = "handoff_call"
    raw: LLMResponseFunctionToolCall = field(repr=False, kw_only=True)
    target_agent: str | None = field(default=None, kw_only=True)
    """Name of the agent being handed off to."""

    @override
    def to_param(self) -> LLMInputContentItem:
        return self.raw.to_param()


@dataclass(frozen=True)
class HandoffOutputItem(RunItemBase["LLMInputContentItem"]):
    """Synthetic tool result for a handoff transfer.

    Wraps ``LLMInputContentItem`` — currently a ``FunctionToolCallResultParam``
    in Chat Completions, but broadened for future Responses API support.
    """

    type: Literal["handoff_output"] = "handoff_output"

    raw: LLMInputContentItem = field(kw_only=True)

    source: Agent | str | None = field(default=None, kw_only=True)
    """The agent that initiated the handoff (Agent object or name string)."""

    target: Agent | str | None = field(default=None, kw_only=True)
    """The agent being handed off to (Agent object or name string)."""

    @override
    def to_param(self) -> LLMInputContentItem:
        return self.raw


# ==================================================================
# MCP items
# ==================================================================


@dataclass(frozen=True)
class MCPListToolsItem(RunItemBase["MCPListTools"]):
    """MCP server tool listing event.

    Wraps ``MCPListTools``. Round-trips via the provider_item channel
    so the underlying wire payload (the verbatim ``mcp_list_tools``
    item the provider emitted) replays losslessly on subsequent
    turns.
    """

    type: Literal["mcp_list_tools"] = "mcp_list_tools"
    raw: MCPListTools = field(kw_only=True)

    @override
    def to_param(self) -> LLMInputContentItem:
        from troopai.adk.types.output import LLMResponseProviderItemParam
        from troopai.adk.types.tools.builtin_tool_types import MCPListToolsTool as _MCPListToolsTool

        def _tool_dict(t: _MCPListToolsTool) -> dict[str, Any]:
            d: dict[str, Any] = {"name": t.name}
            if t.input_schema is not None:
                d["input_schema"] = t.input_schema
            if t.description is not None:
                d["description"] = t.description
            if t.annotations is not None:
                d["annotations"] = t.annotations
            return d

        wire_raw: dict[str, Any] = {
            "type": "mcp_list_tools",
            "server_label": self.raw.server,
            "tools": [_tool_dict(t) for t in self.raw.tools],
        }
        if len(self.raw.id) > 0:
            wire_raw["id"] = self.raw.id
        if self.raw.error is not None:
            wire_raw["error"] = self.raw.error
        return LLMResponseProviderItemParam(
            type="provider_item",
            item_type="mcp_list_tools",
            raw=wire_raw,
        )


@dataclass(frozen=True)
class MCPApprovalRequestItem(RunItemBase["MCPApprovalRequest"]):
    """MCP tool call awaiting human approval.

    Wraps ``MCPApprovalRequest``. Round-trips via the provider_item
    channel so the developer can echo the request back to the
    provider on resume (alongside a ``MCPApprovalResponseItem`` with
    the approve/reject decision).
    """

    type: Literal["mcp_approval_request"] = "mcp_approval_request"
    raw: MCPApprovalRequest = field(kw_only=True)

    @override
    def to_param(self) -> LLMInputContentItem:
        from troopai.adk.types.output import LLMResponseProviderItemParam

        # Wire-format field names mirror OpenAI's
        # ``McpApprovalRequest`` TypedDict (``server_label``, not the
        # framework's ``server``). The framework name is kept on the
        # raw dataclass for ergonomics; the wire shape is reconstructed
        # here at the boundary.
        return LLMResponseProviderItemParam(
            type="provider_item",
            item_type="mcp_approval_request",
            raw={
                "type": "mcp_approval_request",
                "id": self.raw.id,
                "server_label": self.raw.server,
                "name": self.raw.name,
                "arguments": self.raw.arguments,
            },
        )


@dataclass(frozen=True)
class MCPApprovalResponseItem(RunItemBase["MCPApprovalResponse"]):
    """Human decision on an MCP approval request.

    Wraps ``MCPApprovalResponse``. Round-trips as the
    ``mcp_approval_response`` Responses-API input item so the
    provider sees the approve/reject decision on the next turn.

    The framework field is ``approved: bool``; the wire field is
    ``approve: bool``. This method maps between them.
    """

    type: Literal["mcp_approval_response"] = "mcp_approval_response"
    raw: MCPApprovalResponse = field(kw_only=True)

    @override
    def to_param(self) -> LLMInputContentItem:
        from troopai.adk.types.output import LLMResponseProviderItemParam

        wire_raw: dict[str, Any] = {
            "type": "mcp_approval_response",
            "approval_request_id": self.raw.approval_request_id,
            "approve": self.raw.approved,
        }
        if self.raw.id is not None:
            wire_raw["id"] = self.raw.id
        if self.raw.reason is not None:
            wire_raw["reason"] = self.raw.reason
        return LLMResponseProviderItemParam(
            type="provider_item",
            item_type="mcp_approval_response",
            raw=wire_raw,
        )


# ==================================================================
# Provider-hosted tool item (generic catch-all)
# ==================================================================


@dataclass(frozen=True)
class ProviderItem(RunItemBase["LLMResponseProviderItem"]):
    """A provider-hosted tool output item (generic catch-all).

    Wraps :class:`LLMResponseProviderItem`. Created when a provider
    (currently: OpenAI Responses API) emits a hosted-tool output that
    does not fit the function-call / text / reasoning / refusal
    taxonomy — e.g. ``file_search_call``, ``web_search_call``,
    ``image_generation_call``, ``code_interpreter_call``,
    ``computer_call``, ``mcp_call``.

    The raw provider payload is carried verbatim in ``raw.raw`` and
    replayed verbatim on subsequent turns. Framework code that needs
    to route or trace these items discriminates on ``raw.item_type``.
    """

    type: Literal["provider_item"] = "provider_item"
    raw: LLMResponseProviderItem = field(kw_only=True)

    @override
    def to_param(self) -> LLMInputContentItem:
        return self.raw.to_param()


# ==================================================================
# Tool search items
# ==================================================================


@dataclass(frozen=True)
class ToolSearchCallItem(RunItemBase["ToolSearchToolCall"]):
    """A tool search call from the LLM response.

    Wraps ``ToolSearchToolCall``.
    """

    type: Literal["tool_search_call"] = "tool_search_call"
    raw: ToolSearchToolCall = field(kw_only=True)

    @override
    def to_param(self) -> LLMInputContentItem:
        from troopai.adk.types.input import LLMInputEasyMessage

        return LLMInputEasyMessage(
            role="user",
            content=f"[tool_search] {self.raw.query}",
        )


@dataclass(frozen=True)
class ToolSearchOutputItem(RunItemBase["ToolSearchToolCallResult"]):
    """Result from a tool search call.

    Wraps ``ToolSearchToolCallResult``.
    """

    type: Literal["tool_search_output"] = "tool_search_output"
    raw: ToolSearchToolCallResult = field(kw_only=True)

    @override
    def to_param(self) -> LLMInputContentItem:
        from troopai.adk.types.input import LLMInputEasyMessage

        tool_names = ", ".join(t.name for t in self.raw.tools) if self.raw.tools else "(none)"
        return LLMInputEasyMessage(
            role="user",
            content=f"[tool_search_results] {tool_names}",
        )


# ==================================================================
# HITL item
# ==================================================================


@dataclass(frozen=True)
class ToolApprovalItem(RunItemBase["DeferredToolCall"]):
    """A tool call awaiting or having received human approval.

    Wraps ``DeferredToolCall``.  The ``approved`` and ``message``
    fields are framework-level state (not on ``DeferredToolCall``).
    """

    type: Literal["tool_approval"] = "tool_approval"
    raw: DeferredToolCall = field(kw_only=True)
    approved: bool | None = field(default=None, kw_only=True)
    """Approval decision — ``None`` while pending."""
    message: str | None = field(default=None, kw_only=True)
    """Optional rejection/approval message."""

    @override
    def to_param(self) -> LLMInputContentItem:
        from troopai.adk.types.output import FunctionToolCallResultParam as _Param

        if self.approved is None:
            output = f"Tool '{self.raw.tool_name}' awaiting approval"
        elif self.approved:
            output = f"Tool '{self.raw.tool_name}' approved"
        else:
            output = f"Tool '{self.raw.tool_name}' rejected"
            if self.message is not None:
                output = f"{output}: {self.message}"
        return _Param(
            type="function_call_output",
            call_id=self.raw.tool_call_id,
            output=output,
        )


# ==================================================================
# RunItem Union
# ==================================================================

type RunItem = (
    SystemItem
    | UserItem
    | MessageOutputItem
    | ToolCallItem
    | ToolCallOutputItem
    | ReasoningItem
    | HandoffCallItem
    | HandoffOutputItem
    | CompactionItem
    | MCPListToolsItem
    | MCPApprovalRequestItem
    | MCPApprovalResponseItem
    | ToolApprovalItem
    | ToolSearchCallItem
    | ToolSearchOutputItem
    | ProviderItem
)
"""A conversation item in the agent run."""


# ==================================================================
# Provider-item dispatch
# ==================================================================


def _json_arg_str(value: Any) -> str:
    """Coerce a provider ``arguments`` payload to a JSON string.

    Provider payloads carry tool-call ``arguments`` either as a JSON string
    (passed through) or as an already-parsed ``dict`` / ``list``. A plain
    ``str()`` on the latter yields a Python ``repr`` (single-quoted keys),
    which is not valid JSON and corrupts replay; JSON-encode those instead.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _typed_provider_item(
    part: LLMResponseProviderItem,
    agent_name: str | None,
) -> RunItem | None:
    """Build a typed RunItem from a provider_item when item_type maps
    to one of the framework's dedicated MCP item classes.

    Returns ``None`` when no typed mapping exists — the caller falls
    back to the generic ``ProviderItem`` so unknown provider payloads
    still round-trip.

    The raw payload's wire-shape field names (``server_label``,
    ``approve``) are remapped to the framework's dataclass field
    names (``server``, ``approved``). Missing fields default to
    empty / False so a partial provider payload never raises here
    — corruption would be surfaced at replay time, not at item-
    construction time.
    """
    from troopai.adk.types.tools.builtin_tool_types import (
        MCPApprovalRequest,
        MCPListTools,
        MCPListToolsTool,
    )

    raw = part.raw
    item_type = part.item_type

    if item_type == "mcp_approval_request":
        return MCPApprovalRequestItem(
            agent_name=agent_name,
            raw=MCPApprovalRequest(
                id=str(raw.get("id", "")),
                server=str(raw.get("server_label", "")),
                name=str(raw.get("name", "")),
                arguments=_json_arg_str(raw.get("arguments", "")),
            ),
        )

    if item_type == "mcp_list_tools":
        tools_raw = raw.get("tools")
        if tools_raw is not None and not isinstance(tools_raw, list):
            logger.warning(
                "mcp_list_tools payload has tools=%r (expected list); treating as empty.",
                type(tools_raw).__name__,
            )
            tools_raw = []
        elif tools_raw is None:
            tools_raw = []
        dropped = sum(1 for t in tools_raw if not isinstance(t, dict))
        if dropped > 0:
            logger.warning(
                "mcp_list_tools payload had %d non-dict tool entries; they were dropped.",
                dropped,
            )
        return MCPListToolsItem(
            agent_name=agent_name,
            raw=MCPListTools(
                id=str(raw.get("id", "")),
                server=str(raw.get("server_label", "")),
                tools=[
                    MCPListToolsTool(
                        name=str(t.get("name", "")),
                        input_schema=t.get("input_schema"),
                        description=t.get("description"),
                        annotations=t.get("annotations"),
                    )
                    for t in tools_raw
                    if isinstance(t, dict)
                ],
                error=raw.get("error"),
            ),
        )

    return None


# ==================================================================
# ItemHelpers
# ==================================================================


class ItemHelpers:
    """Static utility methods for working with RunItems.

    All data extraction logic lives here, not on item classes.
    """

    @staticmethod
    def extract_last_content(item: MessageOutputItem) -> str:
        """Extract the last text or refusal content from a MessageOutputItem.

        Checks the last content part: returns text for ``LLMResponseText``,
        refusal for ``LLMResponseRefusal``, or empty string if neither.

        Args:
            item: A MessageOutputItem to extract content from.

        Returns:
            The last text or refusal string, or empty string.
        """
        from troopai.adk.types.responses.llm_response import (
            LLMResponseRefusal as _Refusal,
            LLMResponseText as _Text,
        )

        if len(item.raw) == 0:
            return ""
        last = item.raw[-1]
        if isinstance(last, _Text):
            return last.text
        if isinstance(last, _Refusal):
            return last.refusal
        # Exhaustive on the ``LLMResponseText | LLMResponseRefusal`` union.
        assert_never(last)

    @staticmethod
    def input_to_new_input_list(
        user_input: str | list[LLMInputContentItem],
    ) -> list[LLMInputContentItem]:
        """Normalize a string or item list into a list of input items.

        Converts a plain string into a single user message item.
        Lists are returned as-is.

        Args:
            user_input: A string prompt or list of input items.

        Returns:
            A list of ``LLMInputContentItem`` items.
        """
        if isinstance(user_input, str):
            from troopai.adk.types.input import LLMInputEasyMessage

            return [LLMInputEasyMessage(role="user", content=user_input)]
        return list(user_input)

    @staticmethod
    def text_message_output(item: MessageOutputItem) -> str:
        """Concatenate all text parts from a MessageOutputItem.

        Args:
            item: A MessageOutputItem to extract text from.

        Returns:
            The concatenated text, or empty string if no text parts.
        """
        from troopai.adk.types.responses.llm_response import LLMResponseText as _Text

        parts: list[str] = []
        for part in item.raw:
            if isinstance(part, _Text):
                parts.append(part.text)
        return "".join(parts)

    @staticmethod
    def text_message_outputs(items: Sequence[RunItem]) -> str:
        """Concatenate all text from MessageOutputItems in a sequence.

        Args:
            items: A sequence of RunItems.

        Returns:
            Newline-joined text from all MessageOutputItems.
        """
        parts: list[str] = []
        for item in items:
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if len(text) != 0:
                    parts.append(text)
        return "\n".join(parts)

    @staticmethod
    def extract_last_text(items: Sequence[RunItem]) -> str | None:
        """Extract text from the last MessageOutputItem in a sequence.

        Args:
            items: A sequence of RunItems.

        Returns:
            The text from the last MessageOutputItem, or ``None``.
        """
        for item in reversed(items):
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                return text if len(text) > 0 else None
        return None

    @staticmethod
    def refusal_message_output(item: MessageOutputItem) -> str | None:
        """Extract refusal text from a MessageOutputItem.

        Args:
            item: A MessageOutputItem to check for refusal.

        Returns:
            The refusal text, or ``None`` if no refusal.
        """
        from troopai.adk.types.responses.llm_response import LLMResponseRefusal as _Refusal

        for part in item.raw:
            if isinstance(part, _Refusal):
                return part.refusal
        return None

    @staticmethod
    def tool_call_output_str(item: ToolCallOutputItem) -> str:
        """Coerce a ToolCallOutputItem's output to string.

        Args:
            item: A ToolCallOutputItem.

        Returns:
            The output as a string.
        """
        raw_output = item.raw.output
        return raw_output if isinstance(raw_output, str) else str(raw_output)

    @staticmethod
    def reasoning_summary_text(item: ReasoningItem) -> str | None:
        """Extract summary text from a ReasoningItem.

        Returns the explicit summary if present, otherwise the thinking text.

        Args:
            item: A ReasoningItem.

        Returns:
            Summary text, or ``None`` if empty.
        """
        if item.raw.summary is not None and len(item.raw.summary) > 0:
            return item.raw.summary
        return item.raw.thinking if len(item.raw.thinking) > 0 else None

    @staticmethod
    def reasoning_content_text(item: ReasoningItem) -> str | None:
        """Extract reasoning content text from a ReasoningItem.

        Args:
            item: A ReasoningItem.

        Returns:
            The thinking text, or ``None`` if empty.
        """
        return item.raw.thinking if len(item.raw.thinking) > 0 else None

    @staticmethod
    def tool_call_output_item(
        tool_call: ToolCallItem,
        output: Any,
    ) -> ToolCallOutputItem:
        """Create a ToolCallOutputItem from a ToolCallItem and output string.

        Args:
            tool_call: The tool call to create a result for.
            output: The tool's output string.

        Returns:
            A ToolCallOutputItem linked to the given tool call.
        """
        from troopai.adk.types.output import FunctionToolCallResult

        return ToolCallOutputItem(
            raw=FunctionToolCallResult(
                call_id=tool_call.raw.call_id,
                output=output,
            ),
            agent_name=tool_call.agent_name,
        )

    @staticmethod
    def message_to_run_items(
        msg: LLMInputContentItem,
        agent_name: str | None = None,
    ) -> list[RunItem]:
        """Convert a single message dict to one or more RunItems.

        Handles both Layer 1 (provider-agnostic) and Layer 2 (Chat
        Completions) message formats — they share the same dict keys.

        An assistant message with tool_calls produces:
        ``[ReasoningItem?, MessageOutputItem?, ToolCallItem, ...]``

        Args:
            msg: A message dict (Layer 1 or Layer 2 format).
            agent_name: If provided, sets ``agent_name`` on all
                returned items.

        Returns:
            One or more RunItem instances.
        """
        import dataclasses as dc

        from troopai.adk.types.output import (
            FunctionToolCallResult as _FTResult,
            FunctionToolCallResultParam as _ResultParam,
        )
        from troopai.adk.types.responses.llm_response import (
            LLMResponseAnnotation,
            LLMResponseFunctionToolCall as _FunctionToolCall,
            LLMResponseProviderItem,
            LLMResponseReasoning as _Thinking,
            LLMResponseRefusal as _Refusal,
            LLMResponseText as _Text,
        )

        # ``LLMInputContentItem`` is a union of TypedDicts — all dicts at
        # runtime. We dispatch dynamically on ``msg["type"]`` / ``msg["role"]``
        # and read heterogeneous optional keys (``id``, ``status``) that are
        # present on some union members and absent from others. A
        # ``dict[str, Any]`` alias keeps the reads typed-clean without one
        # ignore per variant-specific key access.
        data: dict[str, Any] = dict(msg)

        # Handle Layer 1 FunctionToolCallResultParam (type="function_call_output")
        if data.get("type") == "function_call_output":
            call_id = str(data.get("call_id", ""))
            raw_output = data.get("output", "")
            # ``output`` is ``str | list[LLMInputText | LLMInputImage]`` — preserve a
            # multimodal list verbatim; only coerce genuinely-unexpected shapes.
            output: str | list[Any] = raw_output if isinstance(raw_output, (str, list)) else str(raw_output)
            if isinstance(output, str) and output.startswith("Transferred to "):
                target_name = output.removeprefix("Transferred to ").rstrip(".")
                items: list[RunItem] = [
                    HandoffOutputItem(
                        raw=_ResultParam(type="function_call_output", call_id=call_id, output=output),
                        target=target_name,
                    )
                ]
            else:
                items = [ToolCallOutputItem(raw=_FTResult(call_id=call_id, output=output))]
            if agent_name is not None:
                items = [dc.replace(item, agent_name=agent_name) for item in items]
            return items

        # Handle Layer 1 LLMResponseFunctionToolCallParam (type="function_call")
        if data.get("type") == "function_call":
            # ``signature`` carries a thinking model's opaque per-tool-call
            # signature (base64 str, e.g. Gemini ``thought_signature``). Preserve
            # it across reload so the next replay can hand it back verbatim; a
            # persisted null / absent key leaves it ``None``.
            raw_sig = data.get("signature")
            raw_tc = _FunctionToolCall(
                call_id=str(data.get("call_id", "")),
                name=str(data.get("name", "")),
                arguments=str(data.get("arguments", "{}")),
                id=str(data["id"]) if "id" in data else None,
                status=data.get("status"),
                signature=str(raw_sig) if raw_sig is not None else None,
            )
            items = [ToolCallItem(raw=raw_tc)]
            if agent_name is not None:
                items = [dc.replace(item, agent_name=agent_name) for item in items]
            return items

        # Handle Layer 1 LLMResponseProviderItemParam (type="provider_item").
        # Mirror of response_to_run_items: rebuild the provider item, prefer a
        # typed MCP RunItem, else the generic ProviderItem — never let a
        # provider-hosted item fall through to the empty-UserItem path below.
        if data.get("type") == "provider_item":
            raw_payload = data.get("raw")
            provider_part = LLMResponseProviderItem(
                item_type=str(data.get("item_type", "")),
                raw=raw_payload if isinstance(raw_payload, dict) else {},
            )
            typed = _typed_provider_item(provider_part, agent_name)
            if typed is not None:
                return [typed]
            return [ProviderItem(raw=provider_part, agent_name=agent_name)]

        # Handle Layer 1 LLMResponseReasoningParam (type="reasoning")
        if data.get("type") == "reasoning":
            summary_data = data.get("summary", [])
            content_data = data.get("content")
            encrypted = data.get("encrypted_content")

            # Extract summary text from the structured summary list
            summary_text = ""
            if isinstance(summary_data, list):
                texts = []
                for s in summary_data:
                    if isinstance(s, dict):
                        texts.append(str(s.get("text", "")))
                summary_text = "".join(texts)

            # Extract thinking text from the structured content list and detect
            # Anthropic ``redacted_thinking`` blocks. A redacted block carries
            # its opaque payload under ``data`` (NOT ``text``); reading ``text``
            # would drop it and lose the ``is_redacted`` marker, so a later
            # ``to_param`` would re-emit a plain thinking block that Anthropic
            # rejects on multi-turn extended-thinking tool use.
            thinking_text = ""
            is_redacted = False
            redacted_payloads: list[str] = []
            if content_data is not None and isinstance(content_data, list):
                texts = []
                for c in content_data:
                    if isinstance(c, dict):
                        if c.get("type") == "redacted_thinking":
                            is_redacted = True
                            payload = c.get("data")
                            if payload is not None:
                                redacted_payloads.append(str(payload))
                        else:
                            texts.append(str(c.get("text", "")))
                thinking_text = "".join(texts)

            # Preserve the redacted payload as ``encrypted_content`` so the
            # round-trip re-emits the ``redacted_thinking`` block verbatim.
            if encrypted is not None:
                encrypted_content = str(encrypted)
            elif len(redacted_payloads) > 0:
                encrypted_content = "".join(redacted_payloads)
            else:
                encrypted_content = None

            # Omit a fabricated id: a synthesised ``reasoning_<uuid>`` (or the
            # string ``"None"`` from an explicit null) is not a valid provider
            # reasoning-item id and the OpenAI Responses API rejects it. Leave
            # it unset when the param carries none.
            raw_id = data.get("id")
            reasoning_id = str(raw_id) if raw_id is not None else None

            raw_thinking = _Thinking(
                thinking=thinking_text,
                id=reasoning_id,
                summary=summary_text if len(summary_text) > 0 else None,
                encrypted_content=encrypted_content,
                is_redacted=is_redacted,
                status=data.get("status"),
            )
            items = [ReasoningItem(raw=raw_thinking)]
            if agent_name is not None:
                items = [dc.replace(item, agent_name=agent_name) for item in items]
            return items

        # Handle Layer 1 LLMResponseMessageParam (type="message", role="assistant")
        # Content is a list of typed dicts, not a plain string.
        if data.get("type") == "message" and data.get("role") == "assistant":
            content_parts = data.get("content", [])
            msg_content: list[_Text | _Refusal] = []
            if isinstance(content_parts, list):
                for part in content_parts:
                    if isinstance(part, dict):
                        if part.get("type") == "output_text":
                            anns_raw = part.get("annotations")
                            annotations: list[LLMResponseAnnotation] | None = None
                            if isinstance(anns_raw, list) and len(anns_raw) > 0:
                                annotations = [LLMResponseAnnotation(**a) for a in anns_raw if isinstance(a, dict)]
                            msg_content.append(_Text(text=str(part.get("text", "")), annotations=annotations))
                        elif part.get("type") == "refusal":
                            msg_content.append(_Refusal(refusal=str(part.get("refusal", ""))))
            elif isinstance(content_parts, str):
                msg_content.append(_Text(text=content_parts))
            if len(msg_content) == 0:
                msg_content.append(_Text(text=""))

            raw_id = str(data["id"]) if "id" in data else None
            raw_status_val = data.get("status")
            raw_status: Literal["in_progress", "completed", "incomplete"] | None
            if raw_status_val == "in_progress":
                raw_status = "in_progress"
            elif raw_status_val == "completed":
                raw_status = "completed"
            elif raw_status_val == "incomplete":
                raw_status = "incomplete"
            else:
                raw_status = None
            items = [MessageOutputItem(raw=msg_content, id=raw_id, status=raw_status)]
            if agent_name is not None:
                items = [dc.replace(item, agent_name=agent_name) for item in items]
            return items

        role = data.get("role", "")

        if role in ("system", "developer"):
            from troopai.adk.types.input import LLMInputEasyMessage as _EasyMsg

            role_lit: Literal["user", "system", "developer", "assistant"] = (
                "developer" if role == "developer" else "system"
            )
            items = [
                SystemItem(
                    raw=_EasyMsg(
                        role=role_lit,
                        content=str(data.get("content", "")),
                    )
                )
            ]
            if agent_name is not None:
                items = [dc.replace(item, agent_name=agent_name) for item in items]
            return items

        if role == "user":
            from troopai.adk.types.input import LLMInputEasyMessage as _EasyMsg

            content = data.get("content", "")
            if isinstance(content, (str, list)):
                items = [UserItem(raw=_EasyMsg(role="user", content=content))]
            else:
                items = [UserItem(raw=_EasyMsg(role="user", content=str(content)))]
            if agent_name is not None:
                items = [dc.replace(item, agent_name=agent_name) for item in items]
            return items

        if role == "assistant":
            result_items: list[RunItem] = []

            # Thinking blocks → ReasoningItem
            thinking_blocks = data.get("thinking_blocks")
            if thinking_blocks is not None and isinstance(thinking_blocks, list):
                thinking_texts: list[str] = []
                signatures: list[str] = []

                for block in thinking_blocks:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type == "thinking":
                            thinking_texts.append(str(block.get("thinking", "")))
                            sig = block.get("signature")
                            if sig is not None:
                                signatures.append(str(sig))
                        elif block_type == "redacted_thinking":
                            # ``.get("data", "")`` only defaults on an absent
                            # key; an explicit ``None`` (e.g. ``"data": null``
                            # in a persisted/forwarded block) would reach
                            # ``len(None)`` and abort the whole history
                            # rebuild. Coerce ``None`` to "" before measuring.
                            redacted_data = block.get("data") or ""
                            if len(redacted_data) > 0:
                                signatures.append(str(redacted_data))

                if len(thinking_texts) > 0 or len(signatures) > 0:
                    raw_thinking = _Thinking(
                        thinking="\n".join(thinking_texts),
                        id=f"reasoning_{uuid.uuid4().hex[:12]}",
                        encrypted_content="\n".join(signatures) if len(signatures) > 0 else None,
                        status="completed",
                    )
                    result_items.append(ReasoningItem(raw=raw_thinking))

            # Content → MessageOutputItem
            content_val = data.get("content")
            refusal_val = data.get("refusal")
            if content_val is not None or refusal_val is not None:
                msg_content = []
                # Chat-Completions assistant content may be a list of typed
                # parts (``{"type": "text", ...}`` / ``{"type": "refusal", ...}``).
                # Extract each part rather than ``str()``-ing the whole list,
                # which would persist a Python ``repr`` instead of the text.
                if isinstance(content_val, list):
                    for part in content_val:
                        if isinstance(part, dict):
                            part_type = part.get("type")
                            if part_type == "text":
                                msg_content.append(_Text(text=str(part.get("text", ""))))
                            elif part_type == "refusal":
                                msg_content.append(_Refusal(refusal=str(part.get("refusal", ""))))
                elif content_val is not None:
                    msg_content.append(_Text(text=str(content_val)))
                if refusal_val is not None:
                    msg_content.append(_Refusal(refusal=str(refusal_val)))
                if len(msg_content) == 0:
                    msg_content.append(_Text(text=""))

                result_items.append(MessageOutputItem(raw=msg_content, status="completed"))

            # Tool calls → ToolCallItem
            tool_calls = data.get("tool_calls")
            if tool_calls is not None and isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        if isinstance(func, dict):
                            raw_tc = _FunctionToolCall(
                                call_id=str(tc.get("id", "")),
                                name=str(func.get("name", "")),
                                arguments=str(func.get("arguments", "{}")),
                                status="completed",
                            )
                            result_items.append(ToolCallItem(raw=raw_tc))

            # If nothing was produced, still represent the assistant turn.
            if len(result_items) == 0:
                result_items.append(
                    MessageOutputItem(
                        raw=[_Text(text="")],
                        status="completed",
                    )
                )

            if agent_name is not None:
                result_items = [dc.replace(item, agent_name=agent_name) for item in result_items]
            return result_items

        if role == "tool":
            content_str = str(data.get("content", ""))
            call_id = str(data.get("tool_call_id", ""))

            # Detect handoff synthetic result
            if content_str.startswith("Transferred to "):
                target_name = content_str.removeprefix("Transferred to ").rstrip(".")
                items = [
                    HandoffOutputItem(
                        raw=_ResultParam(
                            type="function_call_output",
                            call_id=call_id,
                            output=content_str,
                        ),
                        target=target_name,
                    )
                ]
            else:
                items = [ToolCallOutputItem(raw=_FTResult(call_id=call_id, output=content_str))]
            if agent_name is not None:
                items = [dc.replace(item, agent_name=agent_name) for item in items]
            return items

        # Unknown role → user item
        from troopai.adk.types.input import LLMInputEasyMessage as _EasyMsg

        logger.warning("Unknown message role %r — treating as user", role)
        items = [UserItem(raw=_EasyMsg(role="user", content=str(data.get("content", ""))))]
        if agent_name is not None:
            items = [dc.replace(item, agent_name=agent_name) for item in items]
        return items

    @staticmethod
    def messages_to_run_items(
        messages: Sequence[LLMInputContentItem],
    ) -> list[RunItem]:
        """Convert a sequence of messages/items to Layer 3 RunItems.

        Handles both Layer 1 (provider-agnostic) and Layer 2 (Chat
        Completions) message formats.

        Args:
            messages: A sequence of message dicts.

        Returns:
            A flat list of RunItem instances.
        """
        result: list[RunItem] = []
        for msg in messages:
            result.extend(ItemHelpers.message_to_run_items(msg))
        return result

    @staticmethod
    def response_to_run_items(
        response: LLMResponse,
        agent_name: str | None = None,
    ) -> list[RunItem]:
        """Convert an ``LLMResponse`` directly to RunItems.

        Eliminates the intermediate dict round-trip — maps response parts
        directly to their corresponding RunItem types.

        Args:
            response: An ``LLMResponse`` from ``LLM.acomplete()``.
            agent_name: If provided, sets ``agent_name`` on all items.

        Returns:
            A list of RunItem instances preserving part order.
        """
        from troopai.adk.types.responses.llm_response import (
            LLMResponseFunctionToolCall,
            LLMResponseProviderItem,
            LLMResponseReasoning,
            LLMResponseRefusal,
            LLMResponseText,
        )

        items: list[RunItem] = []

        # Collect thinking parts → ReasoningItem
        for part in response.response:
            if isinstance(part, LLMResponseReasoning):
                items.append(ReasoningItem(raw=part, agent_name=agent_name))

        # Collect text + refusal parts → MessageOutputItem
        text_refusal_parts: list[LLMResponseText | LLMResponseRefusal] = [
            p for p in response.response if isinstance(p, (LLMResponseText, LLMResponseRefusal))
        ]
        if len(text_refusal_parts) > 0:
            items.append(
                MessageOutputItem(
                    raw=text_refusal_parts,
                    id=response.response_id,
                    status="completed",
                    agent_name=agent_name,
                )
            )

        # Collect tool calls → ToolCallItem
        for part in response.response:
            if isinstance(part, LLMResponseFunctionToolCall):
                items.append(ToolCallItem(raw=part, agent_name=agent_name))

        # Collect provider-hosted tool items.
        #
        # Some item_types have a dedicated typed RunItem with a
        # round-trippable `raw` dataclass — currently
        # ``mcp_approval_request`` and ``mcp_list_tools``. For those we
        # build the typed item so callers can read structured fields
        # without parsing the raw provider dict. Every other item_type
        # (file_search_call, web_search_call, image_generation_call,
        # mcp_call, …) falls through to the generic ``ProviderItem``.
        for part in response.response:
            if not isinstance(part, LLMResponseProviderItem):
                continue
            typed = _typed_provider_item(part, agent_name)
            if typed is not None:
                items.append(typed)
            else:
                items.append(ProviderItem(raw=part, agent_name=agent_name))

        return items

    @staticmethod
    def run_items_to_params(
        items: Sequence[RunItem],
    ) -> list[LLMInputContentItem]:
        """Convert Layer 3 RunItems to Layer 1 params.

        Calls ``to_param()`` on each RunItemBase.  Dicts are passed
        through as-is.  Unknown types are warned and skipped.

        Args:
            items: A sequence of RunItem instances.

        Returns:
            A list of Layer 1 content items ready for the LLM layer.
        """
        params: list[LLMInputContentItem] = []
        for item in items:
            params.append(item.to_param())
        return params

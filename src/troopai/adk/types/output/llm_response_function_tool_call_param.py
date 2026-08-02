"""LLMResponseFunctionToolCallParam: TypedDict version of LLMResponseFunctionToolCall for replay."""

from __future__ import annotations

from typing import Literal, NotRequired, Required

from typing_extensions import TypedDict

__all__ = ["LLMResponseFunctionToolCallParam"]


class LLMResponseFunctionToolCallParam(TypedDict, total=False):
    """TypedDict version of ``LLMResponseFunctionToolCall`` for conversation replay.

    Used when a previous assistant tool call needs to be sent back to
    the LLM as part of conversation history.

    Attributes:
        type: Discriminator. Always ``"function_call"``.
        call_id: Correlation ID linking to the tool call result.
        name: The function name.
        arguments: JSON-encoded arguments string.
        id: Unique item ID.
        status: Item processing status.
        signature: Base64-encoded opaque provider signature (e.g. Gemini
            ``thought_signature``) preserved for multi-turn thinking context.
    """

    type: Required[Literal["function_call"]]
    """Discriminator. Always ``"function_call"``."""

    call_id: Required[str]
    """Correlation ID linking to the tool call result."""

    name: Required[str]
    """The function name."""

    arguments: Required[str]
    """JSON-encoded arguments string."""

    id: str
    """Unique item ID."""

    status: Literal["in_progress", "completed", "incomplete"]
    """Item processing status."""

    signature: NotRequired[str]
    """Base64-encoded opaque provider signature for context preservation.

    Present only when a thinking model attached a per-tool-call signature
    (e.g. Gemini's ``thought_signature``); omitted for providers that do
    not. Base64 keeps arbitrary, non-utf-8 bytes JSON-safe and lossless
    across serialize → reload → replay.
    """

"""LLMResponseReasoningParam: TypedDict version of LLMResponseReasoning for replay."""

from __future__ import annotations

from typing import Literal, Required

from typing_extensions import TypedDict

from troopai.adk.types.output.reasoning_content_text_param import ReasoningContentTextParam
from troopai.adk.types.output.reasoning_summary_text_param import ReasoningSummaryTextParam

__all__ = ["LLMResponseReasoningParam", "ReasoningRedactedThinkingParam"]


class ReasoningRedactedThinkingParam(TypedDict):
    """Replay entry for an Anthropic redacted-thinking block.

    Anthropic returns ``redacted_thinking`` blocks whose opaque ``data``
    must be replayed verbatim as a ``redacted_thinking`` block (NOT a
    plain thinking block) for multi-turn extended-thinking tool use.

    Attributes:
        type: Discriminator. Always ``"redacted_thinking"``.
        data: Opaque redacted-thinking payload, replayed verbatim.
    """

    type: Literal["redacted_thinking"]
    """Discriminator. Always ``"redacted_thinking"``."""

    data: str
    """Opaque redacted-thinking payload, replayed verbatim."""


class LLMResponseReasoningParam(TypedDict, total=False):
    """TypedDict version of ``LLMResponseReasoning`` for conversation replay.

    Used when reasoning / thinking blocks need to be replayed in
    subsequent turns (required for multi-turn tool use with
    extended thinking on Anthropic).

    Attributes:
        type: Discriminator. Always ``"reasoning"``.
        id: Unique item ID.
        summary: Summary text parts.
        content: Full reasoning text parts.
        encrypted_content: Opaque thinking block signature (Anthropic).
        status: Item processing status.
    """

    type: Required[Literal["reasoning"]]
    """Discriminator. Always ``"reasoning"``."""

    id: str
    """Unique item ID."""

    summary: Required[list[ReasoningSummaryTextParam]]
    """Summary text parts (list of ``{type: "summary_text", text: str}`` dicts)."""

    content: list[ReasoningContentTextParam | ReasoningRedactedThinkingParam]
    """Full reasoning parts: each is a ``reasoning_text`` entry, or an Anthropic
    ``redacted_thinking`` entry used to replay redacted blocks verbatim."""

    encrypted_content: str
    """Opaque thinking block signature (Anthropic)."""

    status: Literal["in_progress", "completed", "incomplete"]
    """Item processing status."""

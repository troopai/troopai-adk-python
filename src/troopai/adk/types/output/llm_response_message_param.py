"""LLMResponseMessageParam: TypedDict version of LLMResponseMessage for replay."""

from __future__ import annotations

from typing import Literal, Required

from typing_extensions import TypedDict

from troopai.adk.types.output.llm_response_refusal_param import LLMResponseRefusalParam
from troopai.adk.types.output.llm_response_text_param import LLMResponseTextParam

__all__ = ["LLMResponseMessageParam"]


class LLMResponseMessageParam(TypedDict, total=False):
    """TypedDict version of ``LLMResponseMessage`` for conversation replay.

    Used when an assistant message needs to be sent back to the LLM
    as part of conversation history (e.g., session replay, handoff).

    Attributes:
        type: Discriminator. Always ``"message"``.
        role: Always ``"assistant"``.
        content: List of output content dicts (text and/or refusal).
        id: Unique item ID.
        status: Item processing status.
    """

    type: Required[Literal["message"]]
    """Discriminator. Always ``"message"``."""

    role: Required[Literal["assistant"]]
    """Always ``"assistant"``."""

    content: Required[list[LLMResponseTextParam | LLMResponseRefusalParam]]
    """List of output content dicts (LLMResponseText / LLMResponseRefusal as dicts)."""

    id: str
    """Unique item ID."""

    status: Literal["in_progress", "completed", "incomplete"]
    """Item processing status."""

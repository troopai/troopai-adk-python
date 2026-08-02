"""LLMInputText: text content part for input messages."""

from __future__ import annotations

from typing import Literal, Required

from typing_extensions import TypedDict

__all__ = ["LLMInputText"]


class LLMInputText(TypedDict, total=False):
    """A text content part for input messages.

    Used inside ``LLMInputMessage.content`` for multimodal messages.
    Unlike ``LLMResponseText``, input text has no annotations.

    Attributes:
        type: Discriminator. Always ``"input_text"``.
        text: The text content.
    """

    type: Required[Literal["input_text"]]
    """Discriminator. Always ``"input_text"``."""

    text: Required[str]
    """The text content."""

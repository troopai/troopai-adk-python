"""LLMResponseTextParam: TypedDict version of LLMResponseText for replay."""

from __future__ import annotations

from typing import Literal, Required

from typing_extensions import TypedDict

__all__ = ["LLMResponseTextParam"]


class LLMResponseTextParam(TypedDict, total=False):
    """TypedDict version of ``LLMResponseText`` for conversation replay.

    Attributes:
        type: Discriminator. Always ``"output_text"``.
        text: The text content.
        annotations: Optional annotations (URL citations, etc.).
    """

    type: Required[Literal["output_text"]]
    """Discriminator. Always ``"output_text"``."""

    text: Required[str]
    """The text content."""

    annotations: list[object] | None
    """Optional annotations (URL citations, etc.)."""

"""ReasoningContentTextParam: TypedDict version of ReasoningContentText."""

from __future__ import annotations

from typing import Literal, Required

from typing_extensions import TypedDict

__all__ = ["ReasoningContentTextParam"]


class ReasoningContentTextParam(TypedDict, total=False):
    """TypedDict version of ``ReasoningContentText`` for conversation replay.

    Attributes:
        type: Always ``"reasoning_text"``.
        text: The reasoning text.
    """

    type: Required[Literal["reasoning_text"]]
    """Always ``"reasoning_text"``."""

    text: Required[str]
    """The reasoning text."""

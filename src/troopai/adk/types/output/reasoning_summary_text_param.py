"""ReasoningSummaryTextParam: TypedDict version of ReasoningSummaryText."""

from __future__ import annotations

from typing import Literal, Required

from typing_extensions import TypedDict

__all__ = ["ReasoningSummaryTextParam"]


class ReasoningSummaryTextParam(TypedDict, total=False):
    """TypedDict version of ``ReasoningSummaryText`` for conversation replay.

    Attributes:
        type: Always ``"summary_text"``.
        text: The summary text.
    """

    type: Required[Literal["summary_text"]]
    """Always ``"summary_text"``."""

    text: Required[str]
    """The summary text."""

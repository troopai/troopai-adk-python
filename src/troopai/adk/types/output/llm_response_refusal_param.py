"""LLMResponseRefusalParam: TypedDict version of LLMResponseRefusal for replay."""

from __future__ import annotations

from typing import Literal, Required

from typing_extensions import TypedDict

__all__ = ["LLMResponseRefusalParam"]


class LLMResponseRefusalParam(TypedDict, total=False):
    """TypedDict version of ``LLMResponseRefusal`` for conversation replay.

    Attributes:
        type: Discriminator. Always ``"refusal"``.
        refusal: The refusal explanation.
    """

    type: Required[Literal["refusal"]]
    """Discriminator. Always ``"refusal"``."""

    refusal: Required[str]
    """The refusal explanation."""

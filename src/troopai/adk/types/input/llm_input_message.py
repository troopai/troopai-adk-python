"""LLMInputMessage: strict message input with role and typed content."""

from __future__ import annotations

from typing import Literal, Required

from typing_extensions import TypedDict

from troopai.adk.types.input.llm_input_audio import LLMInputAudio
from troopai.adk.types.input.llm_input_image import LLMInputImage
from troopai.adk.types.input.llm_input_text import LLMInputText

__all__ = ["LLMInputContents", "LLMInputMessage"]

type LLMInputContents = list[LLMInputText | LLMInputImage | LLMInputAudio]
"""Content parts list for an input message."""


class LLMInputMessage(TypedDict, total=False):
    """A strict message input to the LLM with role and typed content.

    Requires ``content`` to be a list of typed content parts (not a plain string).
    For convenience with ``content: str | list``, use ``LLMInputEasyMessage``.

    Instructions given with the ``developer`` or ``system`` role take
    precedence over instructions given with the ``user`` role.

    Attributes:
        type: Discriminator. Always ``"message"``.
        role: The message role.
        content: A list of typed content parts.
        status: Item processing status.
    """

    type: Required[Literal["message"]]
    """Discriminator. Always ``"message"``."""

    role: Required[Literal["user", "system", "developer", "assistant"]]
    """The message role."""

    content: Required[LLMInputContents]
    """A list of typed content parts (text, image, audio)."""

    status: Literal["in_progress", "completed", "incomplete"]
    """Item processing status."""

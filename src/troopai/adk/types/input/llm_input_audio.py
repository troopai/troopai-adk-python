"""LLMInputAudio: audio content part for input messages."""

from __future__ import annotations

from typing import Literal, Required

from typing_extensions import TypedDict

__all__ = ["LLMInputAudio"]


class LLMInputAudio(TypedDict, total=False):
    """An audio content part for input messages.

    Used inside ``LLMInputMessage.content`` for multimodal messages.

    Supported by:
    - OpenAI (gpt-4o-audio-preview and later) — see
      https://platform.openai.com/docs/guides/audio

    Attributes:
        type: Discriminator. Always ``"input_audio"``.
        data: Base64-encoded audio bytes.
        format: Audio encoding format.
    """

    type: Required[Literal["input_audio"]]
    """Discriminator. Always ``"input_audio"``."""

    data: Required[str]
    """Base64-encoded audio bytes."""

    format: Required[Literal["wav", "mp3"]]
    """Audio encoding format. Either ``"wav"`` or ``"mp3"``."""

"""LLMInputEasyMessage: convenience message input accepting str or list content."""

from __future__ import annotations

from typing import Literal, Required

from typing_extensions import TypedDict

from troopai.adk.types.input.llm_input_message import LLMInputContents

__all__ = ["LLMInputEasyMessage"]


class LLMInputEasyMessage(TypedDict, total=False):
    """A convenience message input that accepts both plain strings and typed content.

    Unlike ``LLMInputMessage`` (which requires ``content: list``), this
    type accepts ``content: str | list`` for developer ergonomics.
    The converter normalizes string content to a single-element
    ``[{"type": "input_text", "text": content}]`` list before
    sending to the LLM.

    Attributes:
        role: The message role.
        content: Plain string or list of typed content parts.
    """

    role: Required[Literal["user", "system", "developer", "assistant"]]
    """The message role."""

    content: Required[str | LLMInputContents]
    """Plain string or list of typed content parts.

    - ``str``: Plain text message (most common).
    - ``list``: Multimodal content parts (text, image, audio).
    """

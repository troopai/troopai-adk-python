"""FunctionToolCallResultParam: TypedDict version of FunctionToolCallResult for replay."""

from __future__ import annotations

from typing import Literal, Required

from typing_extensions import TypedDict

from troopai.adk.types.input.llm_input_image import LLMInputImage
from troopai.adk.types.input.llm_input_text import LLMInputText

__all__ = ["FunctionToolCallResultParam"]


class FunctionToolCallResultParam(TypedDict, total=False):
    """TypedDict version of ``FunctionToolCallResult`` for conversation replay.

    Used when a tool execution result needs to be sent back to the LLM
    as part of conversation history (e.g., session replay).

    Attributes:
        type: Discriminator. Always ``"function_call_output"``.
        call_id: Correlation ID linking to the function tool call.
        output: The tool's output (string or multimodal content parts).
        id: Unique item ID.
        status: Processing status.
    """

    type: Required[Literal["function_call_output"]]
    """Discriminator. Always ``"function_call_output"``."""

    call_id: Required[str]
    """Correlation ID linking to ``LLMResponseFunctionToolCall.call_id``."""

    output: Required[str | list[LLMInputText | LLMInputImage]]
    """The tool's output (string or multimodal content parts)."""

    id: str
    """Unique item ID."""

    status: Literal["in_progress", "completed", "incomplete"]
    """Processing status."""

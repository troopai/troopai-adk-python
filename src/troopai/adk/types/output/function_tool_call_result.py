"""FunctionToolCallResult: result from executing a function tool call."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from troopai.adk.types.input.llm_input_image import LLMInputImage
from troopai.adk.types.input.llm_input_text import LLMInputText

__all__ = ["FunctionToolCallResult"]


class FunctionToolCallResult(BaseModel):
    """Result from executing a function tool call.

    A first-class conversation item produced by the framework's tool
    execution layer.  Carries a ``type`` discriminator and optional ``id``
    for union dispatch, session persistence, and tracing.

    Supports multimodal output — tools can return text, images, or a
    mix (e.g., a chart-generation tool returning an image alongside
    a description).

    The LLM implementation translates this to the appropriate
    provider-specific wire format before sending to the LLM.

    Attributes:
        type: Discriminator. Always ``"function_call_output"``.
        call_id: Correlation ID linking back to the
            ``LLMResponseFunctionToolCall`` that triggered this execution.
        output: The tool's output — a plain string (most common) or a
            list of multimodal content parts (text, images).
        id: Unique item ID for referencing and session persistence.
        status: Processing status. Defaults to ``"completed"``.
    """

    type: Literal["function_call_output"] = "function_call_output"
    """Discriminator. Always ``"function_call_output"``."""

    call_id: str
    """Correlation ID linking back to ``LLMResponseFunctionToolCall.call_id``."""

    output: str | list[LLMInputText | LLMInputImage]
    """The tool's output.

    - ``str``: Plain text result (most common). May be the tool's
      return value, an error message, a rejection message, or a
      disabled-tool notification.
    - ``list``: Multimodal content parts for tools that return
      structured content (images, mixed text + images).
    """

    id: str | None = None
    """Unique item ID for referencing and session persistence."""

    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    """Processing status. Defaults to ``"completed"``."""

    artifact: object | None = None
    """Application-side artifact not sent to the LLM.

    When a tool uses ``response_format="content_and_artifact"``, it returns
    ``(content_for_llm, artifact_for_app)``. The ``output`` field gets the
    content string; this field gets the artifact (documents, images, raw
    API responses, DataFrames, etc.).

    The LLM never sees this data — it's for the application to consume
    from ``RunResult``. This supports the cost optimization philosophy:
    the LLM gets a concise summary, the app gets the full data.

    ``None`` when no artifact is produced.
    """

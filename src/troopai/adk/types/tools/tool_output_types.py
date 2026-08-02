"""Structured tool output types for rich return values.

Tools can return these types (or plain strings) to provide structured
output that the framework handles appropriately:

- ``ToolOutputText`` — plain text (explicit wrapper)
- ``ToolOutputImage`` — image URL or base64-encoded data
- ``ToolOutputFileContent`` — file data with filename

When a tool returns a list containing these types, the framework
converts them to multimodal ``FunctionToolCallResult.output`` parts.

Example::

    @function_tool(name="chart")
    def generate_chart(data: str) -> list[ToolOutputText | ToolOutputImage]:
        chart = create_chart(data)
        return [
            ToolOutputText(text="Generated chart for the dataset"),
            ToolOutputImage(image_url=chart.to_data_url()),
        ]
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

__all__ = ["ToolOutputFileContent", "ToolOutputImage", "ToolOutputText"]


class ToolOutputText(BaseModel):
    """A text output from a tool.

    Attributes:
        type: Always ``"text"``.
        text: The text content.
    """

    type: Literal["text"] = "text"
    """Always ``"text"``."""

    text: str
    """The text content."""


class ToolOutputImage(BaseModel):
    """An image output from a tool.

    Attributes:
        type: Always ``"image"``.
        image_url: URL or base64-encoded data URI of the image.
        detail: Resolution hint for the LLM.
    """

    type: Literal["image"] = "image"
    """Always ``"image"``."""

    image_url: str
    """URL or base64-encoded data URI of the image."""

    detail: Literal["auto", "low", "high"] | None = None
    """Resolution hint for the LLM."""


class ToolOutputFileContent(BaseModel):
    """A file output from a tool.

    Attributes:
        type: Always ``"file"``.
        file_data: Base64-encoded file content, URL, or raw text.
        filename: Original filename.
    """

    type: Literal["file"] = "file"
    """Always ``"file"``."""

    file_data: str
    """Base64-encoded file content, URL, or raw text."""

    filename: str | None = None
    """Original filename."""

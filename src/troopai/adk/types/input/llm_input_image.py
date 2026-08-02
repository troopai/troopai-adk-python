"""LLMInputImage: image content part for input messages."""

from __future__ import annotations

from typing import Literal, Required

from typing_extensions import TypedDict

__all__ = ["LLMInputImage"]


class LLMInputImage(TypedDict, total=False):
    """An image content part for input messages.

    Used inside ``LLMInputMessage.content`` for multimodal messages.

    Learn about `image inputs <https://platform.openai.com/docs/guides/vision>`_.

    Attributes:
        type: Discriminator. Always ``"input_image"``.
        image_url: URL or base64-encoded data URI of the image.
        detail: Resolution level for image processing.
    """

    type: Required[Literal["input_image"]]
    """Discriminator. Always ``"input_image"``."""

    image_url: Required[str]
    """URL or base64-encoded data URI of the image.

    Supported formats: JPEG, PNG, GIF, WebP.
    Base64 example: ``data:image/jpeg;base64,{base64_data}``
    """

    detail: Literal["auto", "low", "high"]
    """Resolution level for image processing.

    - ``"low"``: Fewer tokens, faster processing.
    - ``"high"``: Tiled processing for detail.
    - ``"auto"`` (default): Model decides based on image size.
    """

    file_id: str
    """Provider-specific file ID (e.g., OpenAI file reference)."""

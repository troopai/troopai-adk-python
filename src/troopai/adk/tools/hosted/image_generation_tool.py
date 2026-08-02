"""Image generation hosted tool (OpenAI Responses only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from troopai.adk.tools.hosted.hosted_tool import HostedTool


@dataclass(kw_only=True)
class ImageGenerationTool(HostedTool):
    """Provider-hosted image generation.

    Provider matrix:
    - **OpenAI Responses**: native via the ``image_generation`` tool
      type. Honours :attr:`model`, :attr:`quality`, :attr:`size`,
      :attr:`output_format`.
    - All other providers raise :class:`UnsupportedHostedToolError`.

    Refs:
        - OpenAI Responses: https://platform.openai.com/docs/guides/tools-image-generation
    """

    SUPPORTED_PROVIDERS: ClassVar[tuple[str, ...]] = ("openai-responses",)

    model: str | None = None
    """Image model identifier (e.g. ``"gpt-image-1"``). ``None`` lets
    OpenAI choose. **OpenAI Responses only.**"""

    quality: Literal["low", "medium", "high", "auto"] | None = None
    """Quality tier for the generated image. **OpenAI Responses only.**"""

    size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"] | None = None
    """Output image size. **OpenAI Responses only.**"""

    output_format: Literal["png", "jpeg", "webp"] | None = None
    """Output file format. **OpenAI Responses only.**"""

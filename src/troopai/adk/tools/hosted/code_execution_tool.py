"""Code execution hosted tool (OpenAI Responses + Gemini)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from troopai.adk.tools.hosted.hosted_tool import HostedTool


@dataclass(kw_only=True)
class CodeExecutionTool(HostedTool):
    """Provider-hosted Python code interpreter.

    Provider matrix:
    - **OpenAI Responses**: native via the ``code_interpreter`` tool
      type. Optionally honours :attr:`container` to bind a specific
      ephemeral container.
    - **Gemini**: native via the ``code_execution`` tool type. No
      tunable knobs.
    - **Anthropic**: not supported via this class. Anthropic's code
      execution lives behind beta headers; pass via
      ``LLMConfig.extra_body`` if needed. The Anthropic converter
      raises :class:`UnsupportedHostedToolError`.
    - **OpenAI Chat Completions / LiteLLM**: not supported.

    Refs:
        - OpenAI Responses: https://platform.openai.com/docs/guides/tools-code-interpreter
        - Gemini: https://ai.google.dev/gemini-api/docs/code-execution
    """

    SUPPORTED_PROVIDERS: ClassVar[tuple[str, ...]] = ("openai-responses", "gemini")

    container: str | None = None
    """OpenAI container resource name (``cntr_...``) to bind the
    interpreter to. ``None`` lets OpenAI auto-provision a fresh
    container. **OpenAI Responses only.**"""

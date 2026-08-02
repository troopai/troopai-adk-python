"""File search hosted tool (OpenAI Responses only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from troopai.adk.tools.hosted.hosted_tool import HostedTool


@dataclass(kw_only=True)
class FileSearchTool(HostedTool):
    """Provider-hosted vector-store search over uploaded files.

    Provider matrix:
    - **OpenAI Responses**: native via the ``file_search`` tool type.
      Requires at least one vector store ID. Honours
      :attr:`max_num_results` and :attr:`ranking_options`.
    - All other providers raise :class:`UnsupportedHostedToolError`.
      The Anthropic / Gemini equivalents (Anthropic's
      ``file_search``, Gemini's RAG retrieval via Vertex) have
      different shapes and are not unified here. Use
      ``LLMConfig.extra_body`` to bypass this class for those paths.

    Refs:
        - OpenAI Responses: https://platform.openai.com/docs/guides/tools-file-search
    """

    SUPPORTED_PROVIDERS: ClassVar[tuple[str, ...]] = ("openai-responses",)

    vector_store_ids: list[str] = field(default_factory=list)
    """Vector store IDs to search. At least one is required.
    **OpenAI Responses only.**"""

    max_num_results: int | None = None
    """Maximum number of results to return. **OpenAI Responses only.**"""

    ranking_options: dict[str, Any] | None = None
    """Ranking parameters dict (``ranker``, ``score_threshold``).
    Typed against ``openai.types.responses.FileSearchToolParam.RankingOptions``.
    **OpenAI Responses only.**"""

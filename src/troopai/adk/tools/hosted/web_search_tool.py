"""Web search hosted tool (cross-provider)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

from troopai.adk.tools.hosted.hosted_tool import HostedTool


@dataclass(kw_only=True)
class WebSearchTool(HostedTool):
    """Provider-hosted web search.

    Provider matrix:
    - **Anthropic**: native via the ``web_search_20250305`` tool type.
      Honours :attr:`max_uses`, :attr:`allowed_domains`,
      :attr:`blocked_domains`, :attr:`user_location`.
    - **OpenAI Responses**: native via the ``web_search`` tool type
      (or ``web_search_preview`` for older Responses API deployments). Honours
      :attr:`search_context_size` and :attr:`user_location`.
    - **Gemini**: native via the ``google_search`` tool type. No tunable
      knobs at this time; all attribute fields are silently ignored on
      this provider.
    - **OpenAI Chat Completions / LiteLLM**: not supported via the
      tools array (Chat Completions exposes web search via the
      ``web_search_options`` config field instead). The
      ``OpenAIChatCompletionsLLM`` converter raises
      :class:`UnsupportedHostedToolError` — callers should pass via
      ``LLMConfig.extra_body`` or use a different provider.

    Attribute support is per provider. Setting an attribute that the
    active provider does not honour is silently dropped at the
    converter boundary (with a ``logger.debug`` line).

    Example::

        from troopai.adk.tools.hosted import WebSearchTool

        agent = Agent(
            llm=AnthropicLLM(model="claude-sonnet-4-5"),
            tools=[
                WebSearchTool(
                    max_uses=5,
                    allowed_domains=["arxiv.org", "nature.com"],
                ),
            ],
        )

    Refs:
        - Anthropic: https://docs.anthropic.com/en/docs/agents-and-tools/web-search-tool
        - OpenAI Responses: https://platform.openai.com/docs/guides/tools-web-search
        - Gemini: https://ai.google.dev/gemini-api/docs/grounding
    """

    SUPPORTED_PROVIDERS: ClassVar[tuple[str, ...]] = (
        "anthropic",
        "openai-responses",
        "gemini",
    )

    max_uses: int | None = None
    """Maximum number of search invocations the model may make per turn.
    **Anthropic only.**"""

    allowed_domains: list[str] | None = None
    """Whitelist of domains to include in results. Mutually exclusive
    with :attr:`blocked_domains`. **Anthropic only.**"""

    blocked_domains: list[str] | None = None
    """Blocklist of domains to exclude from results. Mutually exclusive
    with :attr:`allowed_domains`. **Anthropic only.**"""

    user_location: dict[str, str] | None = None
    """Approximate user-location dict for relevance ranking. Shape:
    ``{"type": "approximate", "city": "...", "country": "...", "region": "...", "timezone": "..."}``.
    **Anthropic + OpenAI Responses.**"""

    search_context_size: Literal["low", "medium", "high"] | None = None
    """Search context budget — controls how many results / how much text
    the model reads. **OpenAI Responses only.**"""

    def __post_init__(self) -> None:
        """Enforce the documented XOR constraint on allowed_domains / blocked_domains."""
        if self.allowed_domains is not None and self.blocked_domains is not None:
            raise ValueError("WebSearchTool requires at most one of allowed_domains or blocked_domains; both were set")

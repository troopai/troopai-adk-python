"""LiteLLM-backed token counting.

Provider-specific token estimation lives in this package so context
management can stay provider-agnostic at its public boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.types.input import LLMInputContentItem


class LiteLLMTokenCounter:
    """Token counting adapter for LiteLLM model identifiers."""

    @staticmethod
    def count_messages(messages: list[LLMInputContentItem], model: str) -> int:
        """Count tokens for Layer 1 messages after LiteLLM wire conversion."""
        import litellm

        from troopai.adk.llms.litellm.litellm_converter import ChatCompletionConverter

        wire_messages = ChatCompletionConverter.items_to_messages(messages, model=model)
        return litellm.token_counter(model=model, messages=wire_messages)

    @staticmethod
    def count_text(text: str, model: str) -> int:
        """Count tokens for plain text via LiteLLM."""
        import litellm

        return litellm.token_counter(model=model, text=text)

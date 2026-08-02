"""Provider-agnostic token counting facade.

The default implementation delegates to the LiteLLM adapter because
model names in this ADK are LiteLLM-compatible by default. Provider
imports stay under ``llms/litellm``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.types.input import LLMInputContentItem


class TokenCounter:
    """Estimates token counts for messages and text."""

    @staticmethod
    def count_messages(messages: list[LLMInputContentItem], model: str) -> int:
        """Count tokens in a list of chat messages.

        Args:
            messages: Layer 1 framework messages. Converted to Chat
                Completions wire format before counting so that assistant
                replay parts (``output_text``) and user content parts
                (``input_text``) — which litellm's counter does not
                recognise — are normalised to the ``text`` shape it
                expects.
            model: litellm model identifier (e.g. ``"gpt-4o"``).

        Returns:
            Estimated token count.
        """
        from troopai.adk.llms.litellm.litellm_token_counter import LiteLLMTokenCounter

        return LiteLLMTokenCounter.count_messages(messages, model)

    @staticmethod
    def count_text(text: str, model: str) -> int:
        """Count tokens in a plain text string.

        Args:
            text: The text to tokenize.
            model: litellm model identifier.

        Returns:
            Estimated token count.
        """
        from troopai.adk.llms.litellm.litellm_token_counter import LiteLLMTokenCounter

        return LiteLLMTokenCounter.count_text(text, model)

    @staticmethod
    def estimate_remaining(messages: list[LLMInputContentItem], model: str, max_context: int) -> int:
        """Estimate tokens remaining in the context window.

        Args:
            messages: Current conversation messages.
            model: litellm model identifier.
            max_context: Total context window size in tokens.

        Returns:
            Remaining token budget (can be negative if already over).
        """
        return max_context - TokenCounter.count_messages(messages, model)

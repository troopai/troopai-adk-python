from pydantic import BaseModel as PydanticBaseModel


class InputTokensDetails(PydanticBaseModel):
    """A detailed breakdown of the input tokens.

    Attributes:
        cached_tokens: Tokens read from cache (0.1x cost on Anthropic).
            Corresponds to ``cache_read_input_tokens`` in Anthropic's API
            and ``cached_tokens`` in OpenAI's ``prompt_tokens_details``.
        cache_creation_input_tokens: Tokens written to cache (1.25x cost
            on Anthropic).  Only populated for providers with explicit
            cache control (Anthropic, Bedrock).  Zero for providers with
            automatic caching (OpenAI, Azure).
    """

    cached_tokens: int = 0
    """The number of tokens that were retrieved from the cache.
    """

    cache_creation_input_tokens: int = 0
    """The number of tokens that were written to cache for the first time.

    Supported by:

    - Anthropic
    """


class OutputTokensDetails(PydanticBaseModel):
    """A detailed breakdown of the output tokens.

    Attributes:
        reasoning_tokens: Tokens used for reasoning (e.g., chain-of-thought).
    """

    reasoning_tokens: int
    """The number of reasoning tokens."""

from typing import TYPE_CHECKING

from .cost import CostEstimate
from .embedder import Embedder, Embedding, EmbeddingLRUCache
from .litellm import LiteLLM, LiteLLMEmbedder
from .llm import LLM
from .llm_config import LLMConfig

# Lazy import — anthropic SDK is optional.
# Only swallow "anthropic not installed" — any other ImportError
# (e.g. a typo inside anthropic_model.py) MUST surface instead of
# being masked as "extra missing".
if TYPE_CHECKING:
    # Type-check-time: the dep is assumed installed. Downstream code
    # that wants to narrow on missing-extra should compare
    # ``AnthropicLLM is None`` explicitly.
    from .anthropic import AnthropicConfig, AnthropicLLM
    from .gemini import GeminiConfig, GeminiLLM
    from .openai import OpenAIChatCompletionsLLM, OpenAIResponsesLLM
else:
    try:
        from .anthropic import AnthropicConfig, AnthropicLLM
    except ImportError as _exc:
        # Use ``ModuleNotFoundError.name`` (the exact top-level module the
        # interpreter failed to find) rather than a substring on the
        # message. Substring matching would also swallow genuine bugs like
        # ``No module named 'anthropic._legacy_response'`` raised from
        # deeper inside the anthropic package.
        if getattr(_exc, "name", None) != "anthropic":
            raise
        AnthropicLLM = None
        AnthropicConfig = None

    try:
        from .openai import OpenAIChatCompletionsLLM, OpenAIResponsesLLM
    except ImportError as _exc:
        # Same top-level-name guard as the anthropic case: only swallow
        # the "openai package itself not installed" case; anything
        # raised from deeper inside the openai SDK surfaces unmodified.
        if getattr(_exc, "name", None) != "openai":
            raise
        OpenAIChatCompletionsLLM = None
        OpenAIResponsesLLM = None

    try:
        from .gemini import GeminiConfig, GeminiLLM
    except ImportError as _exc:
        # Only swallow "google-genai not installed" — anything raised from
        # deeper inside ``google.genai`` surfaces unmodified. The failing
        # name is ``"google.genai"`` (not ``"google"``) whenever another
        # ``google.*`` distribution (google-auth, google-cloud-*) already
        # created the ``google`` namespace package but the genai wheel is
        # absent; guarding on ``"google"`` alone would let that common case
        # crash the whole ``llms`` import instead of degrading GeminiLLM.
        if getattr(_exc, "name", None) not in ("google", "google.genai"):
            raise
        GeminiLLM = None
        GeminiConfig = None

__all__ = [
    "LLM",
    "AnthropicConfig",
    "AnthropicLLM",
    "CostEstimate",
    "Embedder",
    "Embedding",
    "EmbeddingLRUCache",
    "GeminiConfig",
    "GeminiLLM",
    "LLMConfig",
    "LiteLLM",
    "LiteLLMEmbedder",
    "OpenAIChatCompletionsLLM",
    "OpenAIResponsesLLM",
]

"""LiteLLM-backed LLM implementation sub-package.

Contains all litellm-specific code: the ``LiteLLM`` class,
cache parameter resolution, and provider detection utilities.

The provider-agnostic ``LLM`` ABC, ``LLMConfig``, and ``LLMUsage``
live in the parent ``llms`` package.
"""

from .litellm_embedder import LiteLLMEmbedder
from .litellm_model import LiteLLM

__all__ = ["LiteLLM", "LiteLLMEmbedder"]

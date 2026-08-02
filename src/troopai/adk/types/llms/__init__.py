"""LLM-layer types (provider-agnostic).

Types here MUST NOT import from any provider SDK. They describe
framework-level configuration that every ``LLM`` implementation can
read and act on.
"""

from troopai.adk.types.llms.effort import EffortLevel
from troopai.adk.types.llms.retry_policy import (
    LLMRetryErrorKind,
    LLMRetryPolicy,
)

__all__ = [
    "EffortLevel",
    "LLMRetryErrorKind",
    "LLMRetryPolicy",
]

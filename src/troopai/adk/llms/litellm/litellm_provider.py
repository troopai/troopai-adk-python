"""Provider + capability lookups for a litellm model string.

Thin wrappers around litellm's local tables to identify the backing provider
(:func:`litellm.get_llm_provider`) and read model capabilities
(:func:`litellm.get_model_info`) for a model string.  Used by the Runner and
ContextManager to enable provider-specific optimisations (prompt caching,
server-side compaction, etc.), and to size request budgets to the model.
"""

from __future__ import annotations

import logging

# Import from the concrete modules: ``litellm.__init__`` re-exports these
# but does not list them in ``__all__``, which triggers
# ``reportPrivateImportUsage`` in pyright.
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.utils import get_model_info

logger = logging.getLogger(__name__)


def detect_provider(model: str) -> str:
    """Return the provider name for a litellm model identifier.

    Args:
        model: A litellm model string, e.g. ``"claude-sonnet-4-6"``,
            ``"gpt-4o"``, ``"gemini/gemini-2.5-flash"``.

    Returns:
        Lower-case provider name: ``"anthropic"``, ``"openai"``,
        ``"vertex_ai"``, ``"bedrock"``, etc.  Falls back to
        ``"unknown"`` if detection fails.
    """
    try:
        _model, provider, _api_base, _api_key = get_llm_provider(model)
        return str(provider).lower() if provider is not None else "unknown"
    except Exception:
        return "unknown"


def is_anthropic(model: str) -> bool:
    """Return True if *model* is served by Anthropic (or Bedrock Anthropic).

    Args:
        model: A litellm model string to classify.

    Returns:
        ``True`` when the detected provider is ``"anthropic"`` or
        ``"anthropic_text"``.
    """
    provider = detect_provider(model)
    return provider in ("anthropic", "anthropic_text")


def max_output_tokens(model: str) -> int | None:
    """Return the model's maximum output-token cap, or ``None`` if unknown.

    A local table lookup (no network) via litellm's model-info data. Useful for
    sizing a per-request output budget to the model instead of hard-coding one.

    Args:
        model: A litellm model string to look up.

    Returns:
        The model's maximum output tokens, or ``None`` when litellm has no
        capability data for the model (unmapped or a bare custom name), in which
        case the caller falls back to its own configured budget.
    """
    try:
        info = get_model_info(model)
    except Exception:
        # Unmapped model: litellm raises rather than returning empty.
        logger.debug("litellm model-info lookup: no data for model %s", model)
        return None
    cap = info.get("max_output_tokens")
    return cap if isinstance(cap, int) else None

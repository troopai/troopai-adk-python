"""Resolve thinking / reasoning configuration for Gemini.

Reads from two sources, in priority order:

1. :attr:`GeminiConfig.thinking_config` — typed
   ``google.genai.types.ThinkingConfig``. Highest priority.
2. ``LLMConfig.extra_args["thinking_config"]`` — passthrough escape
   hatch for callers using a plain ``LLMConfig`` instance.

If neither is set, returns ``None`` and Gemini uses the model's
default thinking budget (typically dynamic / "auto" on 2.5+ models).

Validates ``thinking_budget >= 1024`` for explicit-positive budgets
on Gemini 2.5 models — the API rejects below-floor values, so we
warn locally to save the round-trip.

Gemini thinking docs:
https://ai.google.dev/gemini-api/docs/thinking
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from troopai.adk.llms.gemini.gemini_config import GeminiConfig

if TYPE_CHECKING:
    from google.genai.types import ThinkingConfig

    from troopai.adk.llms.llm_config import LLMConfig

logger = logging.getLogger(__name__)

GEMINI_MIN_THINKING_BUDGET = 1024
"""Documented minimum for ``thinking_budget`` on Gemini 2.5 models.

``0`` disables thinking; ``-1`` lets Gemini choose dynamically;
otherwise the value must be ``>= 1024`` (Flash) or ``>= 4096``
(Pro). The 1024 floor is enforced as the conservative minimum;
Pro-specific validation is left to the API boundary.
"""


def resolve_thinking(config: LLMConfig) -> ThinkingConfig | None:
    """Resolve a thinking config from the LLMConfig.

    Args:
        config: The LLM configuration. Either a :class:`GeminiConfig`
            with a typed ``thinking_config`` field, or a plain
            :class:`LLMConfig` carrying the value under
            ``extra_args["thinking_config"]``.

    Returns:
        ``ThinkingConfig`` instance, or ``None`` when not configured.
    """
    candidate: ThinkingConfig | None = None

    if isinstance(config, GeminiConfig) and config.thinking_config is not None:
        candidate = config.thinking_config
    elif config.extra_args is not None:
        from_extra = config.extra_args.get("thinking_config")
        if from_extra is not None:
            # Trust the caller's instance. The SDK validates at the
            # network boundary; we only enforce the documented budget
            # floor below.
            candidate = from_extra

    if candidate is None:
        return None

    budget = getattr(candidate, "thinking_budget", None)
    if isinstance(budget, int) and 0 < budget < GEMINI_MIN_THINKING_BUDGET:
        logger.warning(
            "Gemini thinking_budget=%d is below the documented Flash-tier "
            "minimum of %d. Gemini will likely reject the request. Increase "
            "the budget, set 0 to disable thinking, or set -1 for dynamic.",
            budget,
            GEMINI_MIN_THINKING_BUDGET,
        )

    return candidate

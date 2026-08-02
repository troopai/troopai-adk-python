"""Resolve reasoning / extended-thinking configuration for Anthropic.

Reads from two sources, in priority order:

1. :attr:`AnthropicConfig.thinking` — typed
   ``anthropic.types.ThinkingConfigParam``. Highest priority.
2. ``LLMConfig.extra_args["thinking"]`` — passthrough escape hatch
   for callers using a plain ``LLMConfig`` instance.

If neither is set, returns ``None`` and Anthropic uses its default
(no extended thinking).

Validates ``budget_tokens >= 1024`` per the Anthropic contract and
emits a warning when the requested budget is below the floor —
Anthropic rejects the request server-side, but a clear local log
helps developers diagnose without a roundtrip.

When :attr:`AnthropicConfig.thinking_display` is set and the resolved
thinking config has type ``enabled`` or ``adaptive``, the ``display`` field is
merged into the returned dict.  This maps to the ``display`` key of
``anthropic.types.ThinkingConfigEnabledParam`` (``"omitted"`` or
``"summarized"``).  Callers supplying a pre-built dict via
``extra_args["thinking"]`` must include ``display`` themselves — the
resolver does not patch dicts that did not originate from a typed
:class:`~troopai.adk.llms.anthropic.anthropic_config.AnthropicConfig`.

Anthropic extended thinking docs:
https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from troopai.adk.llms.anthropic.anthropic_config import AnthropicConfig

if TYPE_CHECKING:
    from anthropic.types import ThinkingConfigParam

    from troopai.adk.llms.llm_config import LLMConfig

logger = logging.getLogger(__name__)

ANTHROPIC_MIN_THINKING_BUDGET = 1024
"""Anthropic's documented minimum value for ``thinking.budget_tokens``."""


def resolve_thinking(config: LLMConfig) -> ThinkingConfigParam | None:
    """Resolve a thinking config dict from the LLMConfig.

    Args:
        config: The LLM configuration. Either an :class:`AnthropicConfig`
            with a typed ``thinking`` field, or a plain :class:`LLMConfig`
            carrying the dict under ``extra_args["thinking"]``.

    Returns:
        Anthropic ``ThinkingConfigParam`` dict, or ``None`` when not
        configured.  When :attr:`AnthropicConfig.thinking_display` is set
        and the resolved config type is ``enabled`` or ``adaptive``, the returned
        dict is a shallow copy with the ``display`` field merged in — the
        original config dict is never mutated.
    """
    candidate: ThinkingConfigParam | None = None
    display_override: str | None = None

    if isinstance(config, AnthropicConfig) and config.thinking is not None:
        candidate = config.thinking
        display_override = config.thinking_display
    elif config.extra_args is not None:
        from_extra = config.extra_args.get("thinking")
        if isinstance(from_extra, dict):
            # ``ThinkingConfigParam`` is a TypedDict union; the
            # ``dict.get`` return is typed as ``dict[Unknown, Unknown]``
            # after the isinstance narrow and neither mypy nor pyright
            # can narrow further on the discriminator value at this
            # site. The Anthropic SDK validates the actual shape at
            # the network boundary; we only enforce the documented
            # budget floor below.
            candidate = from_extra  # type: ignore[assignment]
            # noinspection PyTypeChecker

    if candidate is None:
        return None

    if candidate.get("type") == "enabled":
        budget = candidate.get("budget_tokens")
        if isinstance(budget, int) and budget < ANTHROPIC_MIN_THINKING_BUDGET:
            logger.warning(
                "Anthropic thinking.budget_tokens=%d is below the documented "
                "minimum of %d — Anthropic will reject the request. Increase "
                "budget_tokens or disable thinking.",
                budget,
                ANTHROPIC_MIN_THINKING_BUDGET,
            )

    if display_override is not None and candidate.get("type") in ("enabled", "adaptive"):
        # Both the enabled and adaptive thinking params carry a display
        # field in the provider SDK. Return a shallow copy so the
        # caller's original dict is not mutated — the caller was told
        # "pass it through unchanged".
        # The merged dict stays structurally valid but the TypedDict union
        # prevents a narrow assignment; the SDK validates the actual shape
        # at the network boundary.
        candidate = {**candidate, "display": display_override}  # type: ignore[assignment]  # union-narrow merge, shape validated by the SDK

    return candidate

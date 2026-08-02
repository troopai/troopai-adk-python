"""Resolve ``auto_cache_control`` into litellm ``cache_control_injection_points``.

The litellm path caches by handing litellm a list of *injection points* — each
naming a message (by role or index) where litellm's ``AnthropicCacheControlHook``
inserts an ephemeral ``cache_control`` marker. Anthropic then caches the prefix
ending at that block and reads it back on the next request sharing that prefix.

Writing those points by hand is friction, so ``LiteLLMConfig.auto_cache_control``
is a one-bool opt-in (mirroring ``AnthropicConfig.auto_cache_control`` on the
native path). This module turns that bool into the canonical points: the system
message (caches the tools + system prefix) and the last input message (caches the
conversation up to the current turn). An explicit ``cache_control_injection_points``
value always wins, so power users keep full control.

litellm prompt-caching docs: https://docs.litellm.ai/docs/completion/prompt_caching
"""

from __future__ import annotations

from litellm.types.integrations.anthropic_cache_control_hook import (
    CacheControlInjectionPoint,
    CacheControlMessageInjectionPoint,
)
from litellm.types.llms.openai import ChatCompletionCachedContent


def resolve_cache_control_injection_points(
    auto_cache_control: bool | None,
    explicit: list[CacheControlInjectionPoint] | None,
) -> list[CacheControlInjectionPoint] | None:
    """Resolve where litellm should inject ``cache_control``.

    Args:
        auto_cache_control: The ``LiteLLMConfig.auto_cache_control`` opt-in. Only
            ``True`` generates points; ``None`` / ``False`` yield no auto points.
        explicit: A hand-written ``cache_control_injection_points`` list, or
            ``None``. When present it wins outright — auto points are not added.

    Returns:
        The injection points to hand litellm, or ``None`` when neither an
        explicit list nor the opt-in is set (caching left untouched).
    """
    if explicit is not None:
        return explicit
    if auto_cache_control is not True:
        return None
    control: ChatCompletionCachedContent = {"type": "ephemeral"}
    system_point: CacheControlMessageInjectionPoint = {
        "location": "message",
        "role": "system",
        "index": None,
        "control": control,
    }
    last_message_point: CacheControlMessageInjectionPoint = {
        "location": "message",
        "role": None,
        "index": -1,
        "control": control,
    }
    return [system_point, last_message_point]

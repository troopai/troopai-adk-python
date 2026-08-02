"""Anthropic Messages API configuration.

Extends the provider-agnostic :class:`LLMConfig` with fields that
``anthropic.AsyncAnthropic().messages.create()`` accepts natively. Every
non-primitive field is typed verbatim against ``anthropic.types.*`` —
the framework deliberately does NOT define parallel TypedDicts for
``thinking``, ``service_tier``, etc. Developers import the SDK types
directly when building a config.

Anthropic API reference:
https://docs.anthropic.com/en/api/messages
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.types.llms import EffortLevel

if TYPE_CHECKING:
    from anthropic.types import ThinkingConfigParam


@dataclass
class AnthropicConfig(LLMConfig):
    """Anthropic-Messages-specific config extending :class:`LLMConfig`.

    Only fields that have no provider-agnostic analogue on the base
    :class:`LLMConfig` live here. Generic fields (``temperature``,
    ``max_output_tokens``, ``tool_choice``, ``tool_execution_mode``,
    ``extra_body``, ``extra_args``, ``retry_policy`` …) are inherited
    unchanged. ``max_output_tokens`` maps to Anthropic's required
    ``max_tokens`` parameter at the call site.

    Attributes:
        temperature: Sampling temperature (0–1).
        top_k: Top-k filtering.
        top_p: Nucleus sampling.
        max_output_tokens: Maximum tokens in the response.  Maps to
            Anthropic's required ``max_tokens`` parameter at the call site.
        frequency_penalty: Penalise repeated tokens by frequency.
        presence_penalty: Penalise repeated tokens by presence.
        response_logprobs: Whether to return log probabilities for
            chosen tokens.
        top_logprobs: Number of top candidate tokens to return log
            probs for.
        stop_sequences: Stop generation on any of these strings.
        seed: Random seed for reproducibility.
        metadata: Arbitrary metadata passed to the API (only ``user_id``
            reaches Anthropic's ``MetadataParam``; other keys are dropped).
        extra_body: Extra fields merged into the API request body.
        extra_query: Extra query parameters for the API request.
        extra_headers: Extra HTTP headers for the API request.
        extra_args: Catch-all for provider-specific parameters.
        timeout: Request timeout in seconds or ``httpx.Timeout``.
        num_retries: SDK-level retries for transient API errors.
        retry_policy: Framework-level retry policy for transient LLM
            failures.
        fallbacks: Alternative model names to try on failure.
        include_usage: Include token usage in streaming responses.
        tool_choice: Tool selection strategy.
        tool_execution_mode: Sequential or parallel tool execution.
        reset_tool_choice: Reset ``tool_choice`` to ``"auto"`` after
            tools execute.
        thinking: Extended-thinking configuration. Typed against
            ``anthropic.types.ThinkingConfigParam`` directly. The
            ``budget_tokens`` field MUST be at least 1024 per the
            Anthropic contract — the reasoning resolver logs a
            warning when it is lower.
        service_tier: Latency / priority tier. ``"auto"`` lets
            Anthropic decide; ``"standard_only"`` forbids the
            priority tier even when available.
        auto_cache_control: When ``True``, inject ephemeral
            ``cache_control`` markers at the canonical Anthropic
            caching positions (last system block, last tool
            definition, last user-message text block). The marker
            tells Anthropic to cache the prefix ending at that block;
            subsequent calls with the same prefix read from the
            cache. ``None`` / ``False`` leaves messages untouched.
            The tool marker rides on the *last* tool definition, so a
            tool list that changes between turns shifts the marker and
            busts the cached prefix — keep the tool list stable across
            a conversation for cache hits.
        cache_control_ttl: TTL for injected cache_control markers.
            ``"5m"`` (default) is the short-lived tier; ``"1h"`` is
            the extended tier (priced higher).
        effort: Output/reasoning effort forwarded as
            ``output_config.effort``. ``None`` (default) omits the
            field so the model's own default applies. ``xhigh`` and
            ``max`` require a model that supports them.
        mid_system_messages: When ``True``, system/developer items that
            appear AFTER the first user/assistant turn stay in place as
            ``role:"system"`` entries in the messages array (preserving
            the cached prefix and operator authority) instead of being
            hoisted into the top-level ``system=`` parameter. Leading
            system items are always hoisted — the API rejects a system
            entry at ``messages[0]``. Sends the mid-conversation-system
            beta header; unsupported models return a request error.
            ``False`` (default) keeps today's hoist-everything behavior.
        thinking_display: Controls how thinking blocks are presented in
            the API response when thinking is configured (``enabled`` or
            ``adaptive`` types).
            ``"omitted"`` strips thinking blocks from the response body
            (they are still billed and influence the answer but are not
            returned).  ``"summarized"`` returns a condensed summary
            instead of the full reasoning trace.  ``None`` (default)
            omits this field entirely, preserving today's behavior (full
            thinking blocks returned).  Maps to the ``display`` field of
            ``anthropic.types.ThinkingConfigEnabledParam``; ignored when
            thinking is not configured.
    """

    thinking: ThinkingConfigParam | None = None
    """Extended-thinking configuration (``anthropic.types.ThinkingConfigParam``).

    Set to ``{"type": "enabled", "budget_tokens": N}`` to enable
    extended thinking. ``N`` MUST be at least 1024.
    """

    service_tier: Literal["auto", "standard_only"] | None = None
    """Anthropic service tier. ``None`` uses the account default."""

    auto_cache_control: bool | None = None
    """When ``True``, inject ephemeral cache_control markers automatically.

    Changing the tool list between turns shifts the last-tool marker and
    busts the cached prefix; keep tools stable across a conversation.
    """

    cache_control_ttl: Literal["5m", "1h"] | None = None
    """TTL for injected cache_control markers. ``None`` uses Anthropic's default (5m)."""

    effort: EffortLevel | None = None
    """Output/reasoning effort (``output_config.effort``). ``None`` omits the field."""

    mid_system_messages: bool = False
    """Keep non-leading system items in place as ``role:"system"`` messages (beta, model-gated)."""

    thinking_display: Literal["omitted", "summarized"] | None = field(default=None)
    """Controls thinking-block presentation when extended thinking is enabled.

    Maps to the ``display`` field shared by the enabled and adaptive thinking params.
    ``None`` (default) omits the field — full thinking blocks are returned.
    Ignored when thinking is not configured or is disabled.
    """

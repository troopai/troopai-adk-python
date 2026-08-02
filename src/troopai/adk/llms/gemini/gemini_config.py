"""Google Gemini configuration.

Extends the provider-agnostic :class:`LLMConfig` with fields that
``client.aio.models.generate_content`` accepts natively. Every
non-primitive field is typed verbatim against ``google.genai.types``
— the framework deliberately does NOT define parallel TypedDicts for
``ThinkingConfig`` / ``SafetySetting`` / etc. Developers import the
SDK types directly when building a config.

Gemini API reference:
https://ai.google.dev/api
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from troopai.adk.llms.llm_config import LLMConfig

if TYPE_CHECKING:
    from google.genai.types import (
        SafetySetting,
        ThinkingConfig,
    )


@dataclass
class GeminiConfig(LLMConfig):
    """Gemini-specific config extending :class:`LLMConfig`.

    Only fields that have no provider-agnostic analogue on the base
    :class:`LLMConfig` live here. Generic fields (``temperature``,
    ``max_output_tokens``, ``tool_choice``, ``tool_execution_mode``,
    ``extra_body``, ``extra_args``, ``retry_policy`` …) are inherited
    unchanged. ``max_output_tokens`` maps to Gemini's
    ``max_output_tokens`` field on ``GenerateContentConfig``.

    Attributes:
        temperature: Sampling temperature (0–1).
        top_k: Top-k filtering.
        top_p: Nucleus sampling.
        max_output_tokens: Maximum tokens in the response.  Maps to
            Gemini's ``max_output_tokens`` on ``GenerateContentConfig``.
        frequency_penalty: Penalise repeated tokens by frequency.
        presence_penalty: Penalise repeated tokens by presence.
        response_logprobs: Whether to return log probabilities for
            chosen tokens.
        top_logprobs: Number of top candidate tokens to return log
            probs for.
        stop_sequences: Stop generation on any of these strings.
        seed: Random seed for reproducibility.
        metadata: Arbitrary metadata passed to the LLM API.
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
        thinking_config: Extended-thinking configuration. Typed against
            ``google.genai.types.ThinkingConfig`` directly. Carries
            ``include_thoughts`` (bool), ``thinking_budget`` (int —
            ``0`` disables, ``-1`` lets Gemini decide, max 32_768),
            and ``thinking_level`` (Literal for Gemini 3+).
        safety_settings: Per-category safety thresholds. List of
            ``google.genai.types.SafetySetting`` instances. Each
            carries a ``HarmCategory`` and ``HarmBlockThreshold``.
        cached_content_name: Resource name of a pre-created cache
            (``"cachedContents/<id>"``). The cache must be created
            externally via ``client.aio.caches.create``; the framework
            does NOT manage cache lifecycle. When set, the model
            forwards it as ``config.cached_content`` so Gemini reads
            the cached prefix instead of recomputing.
        response_modalities: Output modalities the model may produce
            (e.g. ``["TEXT"]`` or ``["TEXT", "IMAGE"]``). Forwarded
            verbatim to ``GenerateContentConfig.response_modalities``.
    """

    thinking_config: ThinkingConfig | None = None
    """Extended-thinking configuration (``google.genai.types.ThinkingConfig``).

    Set to ``ThinkingConfig(thinking_budget=N, include_thoughts=True)``
    to enable reasoning. ``N >= 1024`` is the documented minimum on
    Gemini 2.5 models; ``-1`` lets the API choose.
    """

    safety_settings: list[SafetySetting] | None = None
    """Per-category harm thresholds.

    Each :class:`google.genai.types.SafetySetting` pairs a
    ``HarmCategory`` (e.g. ``HARM_CATEGORY_HATE_SPEECH``) with a
    ``HarmBlockThreshold`` (e.g. ``BLOCK_MEDIUM_AND_ABOVE``). When
    blocked, the response carries ``finish_reason="SAFETY"`` rather
    than raising — callers should inspect
    ``LLMResponse.finish_reason``.
    """

    cached_content_name: str | None = None
    """Resource name of a pre-created Gemini context cache.

    Format: ``"cachedContents/<id>"``. The cache must be created
    externally via ``client.aio.caches.create(model=..., contents=...,
    ttl=...)``. When set, the model forwards it as
    ``GenerateContentConfig.cached_content``; Gemini serves the
    cached prefix from cache (visible as ``cached_content_token_count``
    in the response usage). ``None`` disables caching for the call.
    """

    response_modalities: list[str] | None = None
    """Output modalities the model may produce.

    Defaults to text-only when unset. Pass ``["TEXT", "IMAGE"]`` for
    image-generating models (Gemini 2.0 Flash with image output).
    """

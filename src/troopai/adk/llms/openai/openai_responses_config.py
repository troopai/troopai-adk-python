"""OpenAI Responses API configuration.

Extends the provider-agnostic :class:`LLMConfig` with fields that
``client.responses.create()`` accepts natively. Every non-primitive
field is typed verbatim against ``openai.types.*`` — the framework
deliberately does NOT define parallel TypedDicts / dataclasses for
reasoning, metadata, or includable lists. Developers import those
SDK types directly when building a config.

OpenAI API reference:
https://platform.openai.com/docs/api-reference/responses/create
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

from troopai.adk.llms.llm_config import LLMConfig

if TYPE_CHECKING:
    from openai.types.responses.response_includable import ResponseIncludable
    from openai.types.shared_params import Reasoning


@dataclass
class OpenAIResponsesConfig(LLMConfig):
    """OpenAI-Responses-specific config extending :class:`LLMConfig`.

    Only fields that have no provider-agnostic analogue on the base
    :class:`LLMConfig` live here. Generic fields (``temperature``,
    ``max_output_tokens``, ``tool_choice``, ``tool_execution_mode``,
    ``extra_body``, ``extra_args``, ``retry_policy`` …) are inherited
    unchanged.

    Attributes:
        temperature: Sampling temperature (0–1).
        top_k: Top-k filtering.
        top_p: Nucleus sampling.
        max_output_tokens: Maximum tokens in the response.
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
        reasoning: Reasoning-effort configuration for o-series / gpt-5
            models. Typed against ``openai.types.shared_params.Reasoning``
            directly — the SDK is the source of truth for the shape.
        include: Response-side items to include in the output (e.g.
            ``"reasoning.encrypted_content"``,
            ``"web_search_call.results"``). Typed against
            ``openai.types.responses.response_includable.ResponseIncludable``.
        store: Whether OpenAI should persist the response server-side so
            that ``previous_response_id`` can chain onto it. Defaults to
            the API default (``True``) when unset.
        previous_response_id: Chain this request onto the output of a
            prior Responses-API response — lets the provider manage
            short-term conversational state without the client replaying
            history. Mutually useful with ``store=True``.
        truncation: How to handle inputs that exceed the model context
            window. ``"auto"`` lets the provider drop oldest items;
            ``"disabled"`` raises on overflow.
        service_tier: Latency / priority tier. ``"priority"`` requires a
            paid tier and expedites low-latency delivery.
        prompt_cache_key: Routing hint that improves prompt-cache hit
            rates for repeated-prefix calls.
        prompt_cache_retention: Cache duration policy for prompt caching.
        max_tool_calls: Upper bound on tool invocations the model may
            request per response. Distinct from framework-level retry /
            iteration budgets.
        background: Run the response asynchronously server-side. When
            ``True`` the call returns immediately with a response ID that
            can be polled; most framework users will leave this unset.
    """

    reasoning: Optional[Reasoning] = None
    """Reasoning config (``openai.types.shared_params.Reasoning``).

    Supported by: o-series and gpt-5 / gpt-5.1 models.
    """

    include: Optional[list[ResponseIncludable]] = None
    """Response-item inclusion list.

    Supported values are enumerated by
    ``openai.types.responses.response_includable.ResponseIncludable``.
    """

    store: Optional[bool] = None
    """Persist the response so that ``previous_response_id`` can chain onto it."""

    previous_response_id: Optional[str] = None
    """Chain this request onto a prior response ID (server-side state threading)."""

    truncation: Optional[Literal["auto", "disabled"]] = None
    """Oversize-input handling (``"auto"`` drops oldest, ``"disabled"`` raises)."""

    service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] = None
    """Latency / priority tier for the request."""

    prompt_cache_key: Optional[str] = None
    """Routing hint for improved prompt-cache hit rates (OpenAI)."""

    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    """Cache duration policy for prompt caching (OpenAI)."""

    max_tool_calls: Optional[int] = None
    """Upper bound on tool invocations the model may request per response."""

    background: Optional[bool] = None
    """Run the response asynchronously server-side (poll via response ID)."""

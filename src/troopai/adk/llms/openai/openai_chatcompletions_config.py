"""OpenAI Chat Completions configuration.

Extends the provider-agnostic :class:`LLMConfig` with fields that
``client.chat.completions.create()`` accepts natively. Every
non-primitive field is typed verbatim against ``openai.types.chat.*``
— the framework deliberately does NOT define parallel TypedDicts
for audio, prediction, or web-search-options. Developers import the
SDK types directly when building a config.

OpenAI API reference:
https://platform.openai.com/docs/api-reference/chat/create
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

from troopai.adk.llms.llm_config import LLMConfig

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionAudioParam, ChatCompletionPredictionContentParam
    from openai.types.chat.completion_create_params import WebSearchOptions


@dataclass
class OpenAIChatCompletionsConfig(LLMConfig):
    """OpenAI-Chat-Completions-specific config extending :class:`LLMConfig`.

    Only fields that have no provider-agnostic analogue on the base
    :class:`LLMConfig` live here. Generic fields (``temperature``,
    ``max_output_tokens``, ``tool_choice``, ``tool_execution_mode``,
    ``extra_body``, ``extra_args``, ``retry_policy`` …) are inherited
    unchanged. ``max_output_tokens`` is mapped to
    ``max_completion_tokens`` at the call site, which OpenAI's
    reasoning models require (they reject ``max_tokens``).

    Attributes:
        temperature: Sampling temperature (0–1).
        top_k: Top-k filtering.
        top_p: Nucleus sampling.
        max_output_tokens: Maximum tokens in the response.  Maps to
            ``max_completion_tokens`` at the call site (OpenAI's
            reasoning models reject ``max_tokens``).
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
        audio: Audio-output configuration. Typed against
            ``openai.types.chat.ChatCompletionAudioParam`` directly.
        web_search_options: Hosted web-search configuration (e.g.
            ``search_context_size``, ``user_location``). Typed against
            ``openai.types.chat.completion_create_params.WebSearchOptions``.
        prediction: Predicted-output content that lets the server skip
            regenerating large verbatim prefixes. Typed against
            ``openai.types.chat.ChatCompletionPredictionContentParam``.
        modalities: Output modalities the model may produce. ``"audio"``
            requires a model that supports audio output.
        store: Whether OpenAI should persist the completion for the
            eval / fine-tune / distillation pipelines.
        service_tier: Latency / priority tier (Chat Completions exposes
            a narrower set than the Responses API).
        prompt_cache_key: Routing hint for improved prompt-cache hit
            rates. Same semantics as the Responses API.
        prompt_cache_retention: Cache duration policy for prompt caching.
        verbosity: Response-length hint for gpt-5 / gpt-5.1-class models
            — ``"low"`` produces terser replies, ``"high"`` more
            expansive ones.
    """

    audio: Optional[ChatCompletionAudioParam] = None
    """Audio-output configuration (``openai.types.chat.ChatCompletionAudioParam``)."""

    web_search_options: Optional[WebSearchOptions] = None
    """Hosted web-search configuration.

    Typed against
    ``openai.types.chat.completion_create_params.WebSearchOptions``.
    """

    prediction: Optional[ChatCompletionPredictionContentParam] = None
    """Predicted-output content (``openai.types.chat.ChatCompletionPredictionContentParam``)."""

    modalities: Optional[list[Literal["text", "audio"]]] = None
    """Output modalities the model may produce."""

    store: Optional[bool] = None
    """Persist the completion for eval / fine-tune / distillation pipelines."""

    service_tier: Optional[Literal["auto", "default", "flex"]] = None
    """Latency / priority tier for the request."""

    prompt_cache_key: Optional[str] = None
    """Routing hint for improved prompt-cache hit rates (OpenAI)."""

    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    """Cache duration policy for prompt caching (OpenAI)."""

    verbosity: Optional[Literal["low", "medium", "high"]] = None
    """Response-length hint for gpt-5 / gpt-5.1-class models."""

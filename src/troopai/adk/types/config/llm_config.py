"""Provider-agnostic schema models for declarative LLM selection.

These Pydantic models describe the JSON ``llm`` / ``llm_config`` surface of
an agent config. Two forms are expressible: a bare model-name string on
``llm`` (handled in ``agent_config.py``), optionally paired with a
standalone agnostic ``llm_config`` block; or a typed provider block on
``llm`` that selects a provider-native LLM and carries its own
provider-specific ``config``.

This module is provider-agnostic by construction: it imports no provider
SDK and no runtime provider config. JSON-shaped fields are modeled with
their real types (reused from the runtime types where they are
JSON-shaped, to avoid drift); structured fields whose only typed form lives
in a provider SDK (``thinking``, ``reasoning``, ``audio`` …) are accepted
as free JSON maps and forwarded verbatim. The provider factories in
``config/providers.py`` are the only place that turns these blocks into a
concrete ``LLM`` and runtime config.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from troopai.adk.types.common import Body, Headers, Metadata, Query
from troopai.adk.types.tools import ToolChoice, ToolExecutionMode


class LLMRetryPolicyBlock(BaseModel):
    """JSON mirror of ``LLMRetryPolicy`` for declarative configs.

    Every field is optional so that an unset key falls back to the runtime
    policy's own default rather than a duplicated default that could drift.
    ``retry_on`` is a list in JSON; the provider factory rebuilds the
    runtime ``frozenset``.

    Attributes:
        max_retries: Maximum retry attempts before giving up.
        initial_delay: Delay in seconds before the first retry.
        max_delay: Upper bound on the delay between retries.
        multiplier: Exponential growth factor applied each retry.
        jitter: Randomize the delay to avoid synchronized retry storms.
        retry_on: Error kinds to retry; absent falls back to the runtime
            policy's default (rate-limit only).
    """

    model_config = ConfigDict(extra="forbid")

    max_retries: int | None = None
    """Maximum number of retry attempts before giving up."""

    initial_delay: float | None = None
    """Delay in seconds before the first retry."""

    max_delay: float | None = None
    """Upper bound on the delay between retries."""

    multiplier: float | None = None
    """Exponential growth factor applied each retry."""

    jitter: bool | None = None
    """Randomize the delay to avoid synchronized retry storms."""

    retry_on: list[Literal["rate_limit", "server_error", "timeout"]] | None = None
    """Error kinds to retry; absent falls back to the runtime default (rate-limit only)."""


class LLMConfigBlock(BaseModel):
    """JSON mirror of the provider-agnostic ``LLMConfig`` surface.

    This is the type of a standalone top-level ``llm_config`` block and the
    base for every provider config sub-model. Only JSON-shaped fields are
    surfaced; ``timeout`` is a float (an ``httpx.Timeout`` object has no
    JSON form) and ``retry_policy`` is the nested
    :class:`LLMRetryPolicyBlock`.

    Attributes:
        temperature: Sampling temperature (0–1).
        top_k: Top-k filtering.
        top_p: Nucleus sampling.
        max_output_tokens: Maximum tokens in the response.
        frequency_penalty: Penalise repeated tokens by frequency.
        presence_penalty: Penalise repeated tokens by presence.
        response_logprobs: Whether to return log probabilities.
        top_logprobs: Number of top candidate tokens to return log probs for.
        stop_sequences: Stop generation on any of these strings.
        seed: Random seed for reproducibility.
        metadata: Arbitrary metadata passed to the LLM API.
        extra_body: Extra fields merged into the API request body.
        extra_query: Extra query parameters for the API request.
        extra_headers: Extra HTTP headers for the API request.
        extra_args: Catch-all for provider-specific parameters.
        timeout: Request timeout in seconds (float only).
        num_retries: SDK-level retries for transient API errors.
        retry_policy: Framework-level retry policy.
        fallbacks: Alternative model names to try on failure.
        include_usage: Include token usage in streaming responses.
        tool_choice: Tool selection strategy.
        tool_execution_mode: Sequential or parallel tool execution.
        reset_tool_choice: Reset ``tool_choice`` to ``"auto"`` after tools.
    """

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    """Sampling temperature (0–1)."""

    top_k: float | None = None
    """Top-k filtering."""

    top_p: float | None = None
    """Nucleus sampling."""

    max_output_tokens: int | None = None
    """Maximum number of tokens to generate."""

    frequency_penalty: float | None = None
    """Penalise repeated tokens based on their frequency."""

    presence_penalty: float | None = None
    """Penalise repeated tokens based on their presence."""

    response_logprobs: bool | None = None
    """Whether to return log probabilities for the chosen tokens."""

    top_logprobs: int | None = None
    """Number of top candidate tokens to return log probs for."""

    stop_sequences: list[str] | None = None
    """Stop generation when any of these strings is encountered."""

    seed: int | None = None
    """Random seed for reproducibility."""

    metadata: Metadata | None = None
    """Arbitrary metadata passed to the LLM API."""

    extra_body: Body | None = None
    """Extra fields merged into the API request body."""

    extra_query: Query | None = None
    """Extra query parameters for the API request."""

    extra_headers: Headers | None = None
    """Extra HTTP headers for the API request."""

    extra_args: dict[str, Any] | None = None
    """Catch-all for provider-specific parameters spread into the API call."""

    timeout: float | None = None
    """Request timeout in seconds (float only)."""

    num_retries: int | None = None
    """Number of SDK-level retries for transient API errors."""

    retry_policy: LLMRetryPolicyBlock | None = None
    """Framework-level retry policy for transient LLM failures."""

    fallbacks: list[str] | None = None
    """Alternative model names to try if the primary model fails."""

    include_usage: bool | None = None
    """Include token usage in streaming responses."""

    tool_choice: ToolChoice | None = None
    """Tool selection strategy (``"auto"`` / ``"required"`` / ``"none"`` / tool name)."""

    tool_execution_mode: ToolExecutionMode | None = None
    """Whether the LLM may invoke multiple tools in a single turn."""

    reset_tool_choice: bool | None = None
    """Reset ``tool_choice`` to ``"auto"`` after tools execute."""


class AnthropicConfigBlock(LLMConfigBlock):
    """Anthropic-specific config extending the agnostic block.

    Mirrors the JSON-shaped fields of the runtime ``AnthropicConfig``. The
    structured ``thinking`` field is accepted as a free JSON map (its only
    typed form lives in ``anthropic.types``); the factory forwards it
    verbatim to ``AnthropicConfig.thinking``.

    Anthropic API reference: https://docs.anthropic.com/en/api/messages

    Attributes:
        thinking: Extended-thinking config, e.g.
            ``{"type": "enabled", "budget_tokens": N}``.
        service_tier: Latency / priority tier.
        auto_cache_control: Inject ephemeral ``cache_control`` markers.
        cache_control_ttl: TTL for injected cache_control markers.
    """

    thinking: dict[str, Any] | None = None
    """Extended-thinking config (free JSON map forwarded to the SDK)."""

    service_tier: Literal["auto", "standard_only"] | None = None
    """Anthropic service tier."""

    auto_cache_control: bool | None = None
    """When ``True``, inject ephemeral cache_control markers automatically."""

    cache_control_ttl: Literal["5m", "1h"] | None = None
    """TTL for injected cache_control markers."""


class OpenAIResponsesConfigBlock(LLMConfigBlock):
    """OpenAI Responses-specific config extending the agnostic block.

    Mirrors the JSON-shaped fields of the runtime ``OpenAIResponsesConfig``.
    ``reasoning`` is a free JSON map (typed form in
    ``openai.types.shared_params``).

    OpenAI API reference:
    https://platform.openai.com/docs/api-reference/responses/create

    Attributes:
        reasoning: Reasoning config (free JSON map).
        include: Response-item inclusion list.
        store: Persist the response server-side.
        previous_response_id: Chain onto a prior response ID.
        truncation: Oversize-input handling.
        service_tier: Latency / priority tier.
        prompt_cache_key: Prompt-cache routing hint.
        prompt_cache_retention: Cache duration policy.
        max_tool_calls: Upper bound on tool invocations per response.
        background: Run the response asynchronously server-side.
    """

    reasoning: dict[str, Any] | None = None
    """Reasoning config (free JSON map forwarded to the SDK)."""

    include: list[str] | None = None
    """Response-item inclusion list."""

    store: bool | None = None
    """Persist the response so ``previous_response_id`` can chain onto it."""

    previous_response_id: str | None = None
    """Chain this request onto a prior response ID."""

    truncation: Literal["auto", "disabled"] | None = None
    """Oversize-input handling."""

    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    """Latency / priority tier for the request."""

    prompt_cache_key: str | None = None
    """Routing hint for improved prompt-cache hit rates."""

    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    """Cache duration policy for prompt caching."""

    max_tool_calls: int | None = None
    """Upper bound on tool invocations the model may request per response."""

    background: bool | None = None
    """Run the response asynchronously server-side."""


class OpenAIChatConfigBlock(LLMConfigBlock):
    """OpenAI Chat-Completions-specific config extending the agnostic block.

    Mirrors the JSON-shaped fields of the runtime
    ``OpenAIChatCompletionsConfig``. ``audio`` / ``web_search_options`` /
    ``prediction`` are free JSON maps (typed forms in ``openai.types.chat``).

    OpenAI API reference:
    https://platform.openai.com/docs/api-reference/chat/create

    Attributes:
        audio: Audio-output configuration (free JSON map).
        web_search_options: Hosted web-search configuration (free JSON map).
        prediction: Predicted-output content (free JSON map).
        modalities: Output modalities the model may produce.
        store: Persist the completion for eval / fine-tune pipelines.
        service_tier: Latency / priority tier.
        prompt_cache_key: Prompt-cache routing hint.
        prompt_cache_retention: Cache duration policy.
        verbosity: Response-length hint for gpt-5-class models.
    """

    audio: dict[str, Any] | None = None
    """Audio-output configuration (free JSON map)."""

    web_search_options: dict[str, Any] | None = None
    """Hosted web-search configuration (free JSON map)."""

    prediction: dict[str, Any] | None = None
    """Predicted-output content (free JSON map)."""

    modalities: list[Literal["text", "audio"]] | None = None
    """Output modalities the model may produce."""

    store: bool | None = None
    """Persist the completion for eval / fine-tune / distillation pipelines."""

    service_tier: Literal["auto", "default", "flex"] | None = None
    """Latency / priority tier for the request."""

    prompt_cache_key: str | None = None
    """Routing hint for improved prompt-cache hit rates."""

    prompt_cache_retention: Literal["in_memory", "24h"] | None = None
    """Cache duration policy for prompt caching."""

    verbosity: Literal["low", "medium", "high"] | None = None
    """Response-length hint for gpt-5 / gpt-5.1-class models."""


class GeminiConfigBlock(LLMConfigBlock):
    """Gemini-specific config extending the agnostic block.

    Mirrors the JSON-shaped fields of the runtime ``GeminiConfig``.
    ``thinking_config`` and ``safety_settings`` are free JSON maps (typed
    forms in ``google.genai.types``).

    Gemini API reference: https://ai.google.dev/api

    Attributes:
        thinking_config: Extended-thinking config (free JSON map).
        safety_settings: Per-category harm thresholds (free JSON maps).
        cached_content_name: Resource name of a pre-created context cache.
        response_modalities: Output modalities the model may produce.
    """

    thinking_config: dict[str, Any] | None = None
    """Extended-thinking config (free JSON map forwarded to the SDK)."""

    safety_settings: list[dict[str, Any]] | None = None
    """Per-category harm thresholds (free JSON maps forwarded to the SDK)."""

    cached_content_name: str | None = None
    """Resource name of a pre-created Gemini context cache."""

    response_modalities: list[str] | None = None
    """Output modalities the model may produce."""


class LiteLLMConfigBlock(LLMConfigBlock):
    """LiteLLM-specific config extending the agnostic block.

    Mirrors the JSON-shaped fields of the runtime ``LiteLLMConfig``.
    ``thinking`` and ``cache_control_injection_points`` are free JSON maps.

    Refs:
        - https://docs.litellm.ai/docs/reasoning_content
        - https://docs.litellm.ai/docs/completion/prompt_caching

    Attributes:
        reasoning_effort: Reasoning effort level.
        thinking: Anthropic thinking budget (free JSON map).
        cache_control_injection_points: Where litellm injects cache_control.
        auto_cache_control: One-bool opt-in that injects ``cache_control`` at
            the canonical positions (system message + last input message)
            instead of a hand-written ``cache_control_injection_points``.
        cached_content: Gemini pre-created CachedContent ID.
        prompt_cache_key: OpenAI/Deepseek routing hint for cache hits.
        prompt_cache_retention: OpenAI/Deepseek cache duration.
    """

    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    """Reasoning effort level (litellm maps to the provider parameter)."""

    thinking: dict[str, Any] | None = None
    """Anthropic thinking budget (free JSON map)."""

    cache_control_injection_points: list[dict[str, Any]] | None = None
    """Where litellm should inject ``cache_control`` blocks (free JSON maps)."""

    auto_cache_control: bool | None = None
    """Inject ``cache_control`` at the canonical positions (system message +
    last input message). ``None`` / ``False`` leaves messages untouched — off by
    default so the caller opts into the cache-write premium. An explicit
    ``cache_control_injection_points`` wins when both are set.

    Honoured by providers that read explicit markers — Anthropic and Google
    Gemini/Vertex AI. OpenAI/Deepseek cache prefixes automatically and ignore
    the markers (use ``prompt_cache_key`` / ``prompt_cache_retention`` there)."""

    cached_content: str | None = None
    """Gemini pre-created CachedContent ID."""

    prompt_cache_key: str | None = None
    """OpenAI/Deepseek routing hint for cache hits."""

    prompt_cache_retention: str | None = None
    """OpenAI/Deepseek cache duration, forwarded verbatim (e.g. ``"in_memory"`` or ``"24h"``)."""


class AnthropicProviderBlock(BaseModel):
    """Typed ``llm`` block selecting the native Anthropic provider.

    Attributes:
        provider: Discriminator literal ``"anthropic"``.
        model: Model identifier (e.g. ``"claude-sonnet-4-5"``).
        api_key: API key; falls back to ``ANTHROPIC_API_KEY`` when unset.
        base_url: Override the Anthropic base URL.
        max_retries: SDK-level client retries.
        config: Optional provider-specific configuration.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["anthropic"]
    """Provider discriminator."""

    model: str = Field(min_length=1)
    """Model identifier passed to the Anthropic Messages API."""

    api_key: str | None = None
    """API key; falls back to ``ANTHROPIC_API_KEY`` when unset."""

    base_url: str | None = None
    """Override the Anthropic base URL."""

    max_retries: int | None = None
    """SDK-level client retries (the client default applies when unset)."""

    config: AnthropicConfigBlock | None = None
    """Optional provider-specific configuration."""


class OpenAIResponsesProviderBlock(BaseModel):
    """Typed ``llm`` block selecting the native OpenAI Responses provider.

    Attributes:
        provider: Discriminator literal ``"openai-responses"``.
        model: Model identifier (e.g. ``"gpt-4o"``).
        api_key: API key; falls back to ``OPENAI_API_KEY`` when unset.
        base_url: Override the OpenAI base URL.
        organization: OpenAI organization id.
        project: OpenAI project id.
        max_retries: SDK-level client retries.
        config: Optional provider-specific configuration.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai-responses"]
    """Provider discriminator."""

    model: str = Field(min_length=1)
    """Model identifier passed to ``client.responses.create()``."""

    api_key: str | None = None
    """API key; falls back to ``OPENAI_API_KEY`` when unset."""

    base_url: str | None = None
    """Override the OpenAI base URL."""

    organization: str | None = None
    """OpenAI organization id."""

    project: str | None = None
    """OpenAI project id."""

    max_retries: int | None = None
    """SDK-level client retries (the client default applies when unset)."""

    config: OpenAIResponsesConfigBlock | None = None
    """Optional provider-specific configuration."""


class OpenAIChatProviderBlock(BaseModel):
    """Typed ``llm`` block selecting the native OpenAI Chat-Completions provider.

    Attributes:
        provider: Discriminator literal ``"openai-chat"``.
        model: Model identifier (e.g. ``"gpt-4o"``).
        api_key: API key; falls back to ``OPENAI_API_KEY`` when unset.
        base_url: Override the OpenAI base URL.
        organization: OpenAI organization id.
        project: OpenAI project id.
        max_retries: SDK-level client retries.
        config: Optional provider-specific configuration.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai-chat"]
    """Provider discriminator."""

    model: str = Field(min_length=1)
    """Model identifier passed to ``client.chat.completions.create()``."""

    api_key: str | None = None
    """API key; falls back to ``OPENAI_API_KEY`` when unset."""

    base_url: str | None = None
    """Override the OpenAI base URL."""

    organization: str | None = None
    """OpenAI organization id."""

    project: str | None = None
    """OpenAI project id."""

    max_retries: int | None = None
    """SDK-level client retries (the client default applies when unset)."""

    config: OpenAIChatConfigBlock | None = None
    """Optional provider-specific configuration."""


class GeminiProviderBlock(BaseModel):
    """Typed ``llm`` block selecting the native Google Gemini provider.

    The Vertex ``credentials`` object has no JSON form and is not
    expressible here — construct the agent in Python for custom Vertex
    credentials. ``api_key`` / ``vertexai`` / ``project`` / ``location`` are
    covered.

    Attributes:
        provider: Discriminator literal ``"gemini"``.
        model: Model identifier (e.g. ``"gemini-2.5-pro"``).
        api_key: API key; falls back to ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``.
        vertexai: Use Vertex AI instead of the Gemini Developer API.
        project: Google Cloud project (Vertex).
        location: Google Cloud location (Vertex).
        base_url: Override the Gemini base URL.
        config: Optional provider-specific configuration.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["gemini"]
    """Provider discriminator."""

    model: str = Field(min_length=1)
    """Model identifier passed to ``client.aio.models.generate_content``."""

    api_key: str | None = None
    """API key; falls back to ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` when unset."""

    vertexai: bool | None = None
    """Use Vertex AI instead of the Gemini Developer API."""

    project: str | None = None
    """Google Cloud project (Vertex)."""

    location: str | None = None
    """Google Cloud location (Vertex)."""

    base_url: str | None = None
    """Override the Gemini base URL."""

    config: GeminiConfigBlock | None = None
    """Optional provider-specific configuration."""


class LiteLLMProviderBlock(BaseModel):
    """Typed ``llm`` block selecting the LiteLLM multi-provider backend.

    Attributes:
        provider: Discriminator literal ``"litellm"``.
        model: litellm model identifier (e.g. ``"gpt-4o"``,
            ``"anthropic/claude-sonnet-4-5"``).
        api_key: API key for the underlying provider.
        base_url: Override the provider base URL.
        extra_params: Default parameters merged into every litellm call.
        config: Optional provider-specific configuration.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["litellm"]
    """Provider discriminator."""

    model: str = Field(min_length=1)
    """litellm model identifier."""

    api_key: str | None = None
    """API key for the underlying provider."""

    base_url: str | None = None
    """Override the provider base URL."""

    extra_params: dict[str, Any] | None = None
    """Default parameters merged into every litellm call."""

    config: LiteLLMConfigBlock | None = None
    """Optional provider-specific configuration."""


LLMProviderConfig = Annotated[
    AnthropicProviderBlock
    | OpenAIResponsesProviderBlock
    | OpenAIChatProviderBlock
    | GeminiProviderBlock
    | LiteLLMProviderBlock,
    Field(discriminator="provider"),
]
"""Discriminated union of typed ``llm`` provider blocks, keyed on ``provider``."""

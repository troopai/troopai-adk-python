"""LiteLLM-backed LLM implementation.

Provides ``LiteLLM`` — the default ``LLM`` implementation that uses
`litellm <https://github.com/BerriAI/litellm>`_ to communicate with
100+ language model providers through a unified API.

Key responsibilities:

- **Parameter mapping**: Maps ``LLMConfig`` fields to litellm parameter
  names via explicit named arguments (e.g., ``max_output_tokens`` →
  ``max_tokens``).
- **Structured output**: Uses ``response_format`` (JSON schema mode)
  instead of instructor, then validates via ``AgentOutputSchemaBase``.
- **Response parsing**: Converts litellm's ``ModelResponse`` into our
  ``LLMResponse`` type with proper usage tracking.
- **Streaming**: Yields ``LLMStreamEvent`` objects for real-time token
  delivery, with a final ``"done"`` event containing the complete response.
- **Prompt caching**: Applies cache control parameters when configured.

Example::

    from troopai.adk.llms import LiteLLM

    llm = LiteLLM(model="gpt-4o")

    # Non-streaming
    response = await llm.acomplete(
        messages=[{"role": "user", "content": "Hello!"}],
        llm_config=LLMConfig(temperature=0.7),
    )
    logger.info(response.content)

    # Streaming
    async for event in await llm.acomplete(
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True,
    ):
        if event.type == "part_delta" and event.delta is not None:
            logger.info(event.delta)
        elif event.type == "done" and event.response is not None:
            logger.info(f"\\nTokens: {event.response.usage.total_tokens}")

    # Structured output (no instructor needed)
    from troopai.adk.schemas import AgentOutputSchema

    schema = AgentOutputSchema(MyModel)
    response = await llm.acomplete(
        messages=[{"role": "user", "content": "Analyze this."}],
        output_schema=schema,
    )
    logger.info(response.output)  # MyModel instance
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast, overload, override

from litellm.types.llms.openai import (
    REASONING_EFFORT,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)
from typing_extensions import TypedDict


class ThinkingParam(TypedDict, total=False):
    """Framework-owned thinking budget config, shape-compatible with Anthropic.

    Mirrors ``litellm.types.llms.anthropic.AnthropicThinkingParam`` but is
    defined in this codebase so ``LiteLLMConfig`` does not re-export a
    provider-SDK type to developers. litellm accepts the same dict shape
    at the wire level — no conversion needed.

    Supported by: Anthropic (via litellm).

    Ref: https://docs.litellm.ai/docs/reasoning_content
    """

    type: Literal["enabled"]
    budget_tokens: int


from datetime import UTC

from troopai.adk.llms.litellm.litellm_cache_applicator import resolve_cache_control_injection_points
from troopai.adk.llms.litellm.litellm_retry import call_with_retry
from troopai.adk.llms.llm import LLM
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.types.input import LLMInputContentItem

if TYPE_CHECKING:
    from litellm import ModelResponse
    from litellm.types.integrations.anthropic_cache_control_hook import CacheControlInjectionPoint
    from litellm.utils import CustomStreamWrapper  # pyright: ignore[reportPrivateImportUsage]

    from troopai.adk.llms.llm_usage import LLMUsage
    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.tools import Tool
    from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseReasoning, LLMStreamEvent


# ---------------------------------------------------------------------------
# LiteLLM-specific LLM config
#
# Extends the provider-agnostic LLMConfig with litellm-specific fields
# for reasoning, thinking, and prompt caching.  The base LLMConfig has
# no reasoning or caching fields — those are provider-specific.
#
# Following Pydantic-AI's pattern: base settings are flat and agnostic,
# provider-specific settings extend the base with prefixed fields.
#
# Refs:
#   - litellm reasoning: https://docs.litellm.ai/docs/reasoning_content
#   - litellm prompt caching: https://docs.litellm.ai/docs/completion/prompt_caching
# ---------------------------------------------------------------------------


@dataclass
class LiteLLMConfig(LLMConfig):
    """LiteLLM-specific config extending the base :class:`LLMConfig`.

    Adds fields for reasoning and prompt caching that litellm supports.
    Provider-specific fields use litellm types directly.

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
        extra_args: Catch-all for provider-specific parameters spread
            into the API call.
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
        reasoning_effort: Reasoning effort level.  litellm maps this
            to the appropriate provider parameter.

            Supported by: OpenAI, Anthropic, Gemini, DeepSeek, Bedrock.

            Ref: https://docs.litellm.ai/docs/reasoning_content
        thinking: Anthropic-specific thinking budget control.
            Shape: ``{"type": "enabled", "budget_tokens": N}``.

            Supported by: Anthropic (via litellm).

            Ref: https://docs.litellm.ai/docs/reasoning_content
        cache_control_injection_points: Where litellm should inject
            ``cache_control`` blocks.  Processed by litellm's
            ``AnthropicCacheControlHook``.

            Supported by: Anthropic, Google Gemini/Vertex AI.

            Ref: https://docs.litellm.ai/docs/completion/prompt_caching
        cached_content: Pre-created CachedContent ID for Gemini.

            Supported by: Google Gemini.

            Ref: https://docs.litellm.ai/docs/completion/prompt_caching#google-gemini
        prompt_cache_key: Routing hint for improved cache hit rates.

            Supported by: OpenAI, Deepseek.

            Ref: https://docs.litellm.ai/docs/completion/prompt_caching#openai
        prompt_cache_retention: Cache duration policy.

            Supported by: OpenAI, Deepseek.

            Ref: https://docs.litellm.ai/docs/completion/prompt_caching#openai
    """

    reasoning_effort: REASONING_EFFORT | None = None
    """Reasoning effort level (``"none"``/``"minimal"``/``"low"``/``"medium"``/``"high"``/``"xhigh"``).

    Supported by: OpenAI, Anthropic, Gemini, DeepSeek, Bedrock.
    """

    thinking: ThinkingParam | None = None
    """Anthropic thinking budget: ``{"type": "enabled", "budget_tokens": N}``.

    Supported by: Anthropic (via litellm).
    """

    cache_control_injection_points: list[CacheControlInjectionPoint] | None = None
    """Where to inject ``cache_control`` blocks (Anthropic/Gemini).

    Supported by: Anthropic, Google Gemini/Vertex AI.
    """

    cached_content: str | None = None
    """Gemini pre-created CachedContent ID.

    Supported by: Google Gemini.
    """

    prompt_cache_key: str | None = None
    """OpenAI/Deepseek routing hint for cache hits.

    Supported by: OpenAI, Deepseek.
    """

    prompt_cache_retention: str | None = None
    """OpenAI/Deepseek cache duration (``"in_memory"`` or ``"24h"``).

    Supported by: OpenAI, Deepseek.
    """

    auto_cache_control: bool | None = None
    """When ``True``, inject ``cache_control`` at the canonical caching positions
    (the system message, then the last input message) instead of hand-writing
    ``cache_control_injection_points``. The one-bool opt-in mirrors
    ``AnthropicConfig.auto_cache_control`` on the native path; an explicit
    ``cache_control_injection_points`` value still wins when both are set.
    ``None`` / ``False`` leaves messages untouched — off by default so the caller
    opts into the cache-write premium rather than out of it.

    Markers use the ephemeral (default ~5m) tier; the extended tier and other
    fine control remain available through explicit
    ``cache_control_injection_points`` (litellm's typed injection ``control`` does
    not carry a TTL, unlike the native ``AnthropicConfig.cache_control_ttl``).

    Supported by: Anthropic, Google Gemini/Vertex AI (providers that honour
    ``cache_control`` markers). OpenAI/Deepseek cache automatically and ignore
    the markers.
    """


# ---------------------------------------------------------------------------
# Streaming accumulation buffers
#
# These are NOT the same as litellm's streaming chunk types
# (ChatCompletionDeltaToolCallChunk, ChatCompletionToolCallFunctionChunk).
# litellm's chunk types have optional fields and extra required fields
# (index, type, provider_specific_fields).  These are minimal mutable
# buffers that grow across multiple stream chunks — all fields are required
# and directly accessible without None checks.
# ---------------------------------------------------------------------------


class _StreamedFunctionData(TypedDict):
    """Accumulated function name + arguments during streaming."""

    name: str
    arguments: str


class _StreamedToolCall(TypedDict):
    """Accumulated tool call during streaming."""

    id: str
    function: _StreamedFunctionData


logger = logging.getLogger(__name__)


# Parameters that ``LiteLLM.acomplete()`` passes to ``litellm.acompletion()`` by
# explicit keyword. Any ``extra_body`` / ``extra_args`` key matching one of these
# would raise "multiple values for keyword argument" when spread as
# ``**extra_kwargs``, so colliding keys are stripped (the explicit LLMConfig
# value wins). Keep in sync with the named arguments in the acompletion call.
_ACOMPLETION_NAMED_PARAMS: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "tools",
        "temperature",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "max_tokens",
        "stop",
        "seed",
        "logprobs",
        "top_logprobs",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "stream",
        "stream_options",
        "timeout",
        "num_retries",
        "fallbacks",
        "reasoning_effort",
        "thinking",
        "extra_headers",
        "extra_query",
        "api_key",
        "base_url",
        "cache_control_injection_points",
        "cached_content",
        "prompt_cache_key",
        "prompt_cache_retention",
    }
)


class LiteLLM(LLM):
    """LLM implementation backed by litellm.

    Provides access to 100+ language models (OpenAI, Anthropic, Google,
    Mistral, Cohere, Groq, Bedrock, Azure, etc.) through litellm's
    unified API.

    The class handles:

    - Explicit mapping of ``LLMConfig`` fields to litellm parameters
    - Structured output via ``response_format`` (JSON schema mode)
    - Response parsing into ``LLMResponse``
    - Streaming via ``LLMStreamEvent`` async iteration
    - Prompt caching parameter injection

    Args:
        model: Model identifier (e.g., ``"gpt-4o"``, ``"claude-sonnet-4-20250514"``).
        api_key: API key for the LLM provider.
        base_url: Base URL for the LLM provider API.
        extra_params: Additional provider-specific parameters applied
            to every call (e.g., ``organization``, ``project``).
            Overridden by ``LLMConfig.extra_args`` on a per-call basis.

    Example::

        llm = LiteLLM(model="gpt-4o")

        # Or with provider credentials
        llm = LiteLLM(model="gpt-4o", api_key="sk-...", base_url="https://...")

        # Or with extra provider-specific defaults
        llm = LiteLLM(model="gpt-4o", extra_params={"organization": "org-..."})

    Attributes:
        model: The model identifier for this LLM instance.
        api_key: API key passed to litellm on every call.
        base_url: Base URL passed to litellm on every call.
        extra_params: Extra default parameters for litellm calls.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        self._model: str = model

        self.api_key: str | None = api_key
        """API key for the LLM provider."""

        self.base_url: str | None = base_url
        """Base URL for the LLM provider API."""

        self.extra_params: dict[str, Any] = extra_params or {}
        """Additional default parameters for litellm calls.

        These are merged into every ``litellm.acompletion()`` call.
        ``LLMConfig.extra_args`` takes precedence over these defaults.
        """

    @property
    def model(self) -> str:
        """The model identifier for this LLM instance."""
        return self._model

    @overload
    async def acomplete(
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        stream: Literal[False] = False,
    ) -> LLMResponse: ...

    @overload
    async def acomplete(
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        *,
        stream: Literal[True],
    ) -> AsyncIterator[LLMStreamEvent]: ...

    @override
    async def acomplete(
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
        """Call the LLM via litellm with explicit named parameters.

        Accepts provider-agnostic input (``UserPrompt``) and converts
        to Chat Completions wire format via ``ChatCompletionConverter``
        before calling ``litellm.acompletion()``.

        Tool choice and parallel tool calls are read from ``llm_config``:

        - ``llm_config.tool_choice`` → ``tool_choice`` parameter
        - ``llm_config.tool_execution_mode`` → ``parallel_tool_calls`` bool

        Args:
            messages: Conversation input — a plain string or list of
                provider-agnostic ``LLMInputContentItem`` items.
            llm_config: Optional LLM parameters (includes tool choice
                and execution mode).
            tools: Optional tool definitions.
            output_schema: Optional structured output schema.
            stream: Whether to stream the response.

        Returns:
            ``LLMResponse`` (non-streaming) or
            ``AsyncIterator[LLMStreamEvent]`` (streaming).
        """
        import litellm as litellm_lib

        from troopai.adk.llms.litellm.litellm_converter import ChatCompletionConverter

        config = llm_config if llm_config is not None else LLMConfig()

        # Determine if thinking blocks should be preserved in message history
        preserve_thinking = isinstance(config, LiteLLMConfig) and (
            config.reasoning_effort is not None or config.thinking is not None
        )

        # Convert provider-agnostic items to Chat Completions wire format
        model = self._model
        converted_messages = ChatCompletionConverter.items_to_messages(
            messages,
            model=model,
            preserve_thinking_blocks=preserve_thinking,
            preserve_tool_output_all_content=True,
        )

        # Reorder tool messages for providers that require adjacency
        model_lower = model.lower() if len(model) > 0 else ""
        is_gemini = "gemini" in model_lower
        is_anthropic = "claude" in model_lower or "anthropic" in model_lower
        if is_anthropic or is_gemini:
            converted_messages = ChatCompletionConverter.fix_tool_message_ordering(converted_messages)

        # Convert Gemini thought signatures to litellm's format
        if is_gemini:
            converted_messages = self._convert_gemini_thought_signatures(converted_messages)

        logger.debug(
            "LLM call: model=%s, stream=%s, tools=%d, output_schema=%s",
            model,
            stream,
            len(tools) if tools is not None else 0,
            output_schema.name() if output_schema is not None else None,
        )

        # Early check: warn when the likely API key is missing.
        # litellm would eventually raise, but the error is often cryptic.
        if self.api_key is None:
            import os

            provider_key_map: dict[str, str] = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",
                "gemini": "GEMINI_API_KEY",
            }
            try:
                # Import from the concrete module rather than the
                # package re-export: ``litellm.__init__`` re-exports
                # ``get_llm_provider`` but does not list it in
                # ``__all__``, which triggers
                # ``reportPrivateImportUsage`` in pyright.
                from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

                _model_name, _provider, _api_base, _resolved_key = get_llm_provider(model)
                provider_name = str(_provider).lower() if _provider is not None else "unknown"
            except Exception as e:
                logger.debug("Provider detection failed for model=%s: %s; skipping the early API-key check", model, e)
                provider_name = "unknown"
            env_var = provider_key_map.get(provider_name)
            if env_var is not None and os.environ.get(env_var) is None:
                logger.error(
                    "API key not found: %s is not set. "
                    "Set it in your environment or in a .env file (see .env.example).",
                    env_var,
                )

        # Build response_format for structured output.
        # Uses a three-tier strategy based on model capabilities:
        #   Tier 1: json_schema — model natively constrains output to schema
        #   Tier 2: json_object — model outputs valid JSON, we validate schema
        #   Tier 3: none — no JSON support, rely on prompt + our validation
        # Validation happens downstream in
        # ``run/turn_resolution.resolve_structured_output_step`` — the LLM
        # layer no longer parses the JSON itself.
        response_format = None
        if output_schema is not None and not output_schema.is_plain_text():
            response_format = self._resolve_response_format(
                output_schema=output_schema,
                model=model,
                litellm_lib=litellm_lib,
            )

        # Build stream_options for streaming usage tracking.
        # Without this, most providers omit usage from streaming responses,
        # so _parse_usage() would return None and we'd lose all token/cost
        # tracking.
        stream_options = None
        if stream:
            include_usage = config.include_usage if config.include_usage is not None else True
            stream_options = {"include_usage": include_usage}

        # Read reasoning and caching from LiteLLMConfig if provided.
        # These fields are provider-specific and only exist on
        # LiteLLMConfig, not on the base LLMConfig.
        reasoning_effort: Any = None
        thinking = None
        cache_control_injection_points = None
        cached_content = None
        prompt_cache_key = None
        prompt_cache_retention = None
        if isinstance(config, LiteLLMConfig):
            reasoning_effort = config.reasoning_effort
            thinking = config.thinking
            cache_control_injection_points = resolve_cache_control_injection_points(
                config.auto_cache_control, config.cache_control_injection_points
            )
            cached_content = config.cached_content
            prompt_cache_key = config.prompt_cache_key
            prompt_cache_retention = config.prompt_cache_retention

        # Assemble extra kwargs for genuinely dynamic provider-specific
        # parameters (extra_params, extra_body, extra_args).
        extra_kwargs: dict[str, Any] = {**self.extra_params}
        if config.extra_body is not None and isinstance(config.extra_body, dict):
            extra_kwargs.update(config.extra_body)
        if config.extra_args is not None:
            extra_kwargs.update(config.extra_args)

        # Prevent duplicate reasoning_effort from extra_body/extra_args
        # (user might pass it via extra_body AND via Reasoning config)
        if reasoning_effort is not None:
            extra_kwargs.pop("reasoning_effort", None)
        elif "reasoning_effort" in extra_kwargs:
            # Fallback: use reasoning_effort from extra_body/extra_args
            reasoning_effort = extra_kwargs.pop("reasoning_effort")

        # Generalized dedup: any remaining extra_body/extra_args key that
        # collides with a parameter passed explicitly by name below would raise
        # "multiple values for keyword argument". The explicit LLMConfig-mapped
        # value is authoritative — drop the colliding copy and warn so the misuse
        # is visible. (reasoning_effort, handled just above, supports an extra_*
        # fallback and is already removed by this point.)
        for key in _ACOMPLETION_NAMED_PARAMS & extra_kwargs.keys():
            logger.warning(
                "extra_body/extra_args key %r collides with a mapped LLMConfig parameter; "
                "the explicit value takes precedence and the extra copy is ignored.",
                key,
            )
            del extra_kwargs[key]

        # Only pass metadata when explicitly set — litellm forwards it
        # to the provider, and some providers reject None metadata.
        if config.metadata is not None:
            extra_kwargs["metadata"] = config.metadata

        # Convert ``LLMConfig.timeout`` (``float | httpx.Timeout | None``)
        # down to litellm's typed surface (``float | int | None``).
        # Use the read-phase value of an httpx.Timeout since that is the
        # one litellm/httpx ultimately applies to the response read —
        # fall back to the connect timeout, then to a 60s default.
        timeout_value: float | int | None = None
        if config.timeout is not None:
            if isinstance(config.timeout, (int, float)):
                timeout_value = config.timeout
            else:
                timeout_value = config.timeout.read or config.timeout.connect or 60.0

        # Convert ``LLMConfig.extra_headers`` (``Mapping[str, object]``)
        # to the concrete ``dict`` that litellm's typed signature wants.
        extra_headers_dict: dict[str, object] | None = (
            dict(config.extra_headers) if config.extra_headers is not None else None
        )

        # Convert Tool instances to litellm wire-format dicts
        # (schema → parameters, etc.)
        wire_tools = self._convert_tools(model, tools) if tools is not None and len(tools) > 0 else None

        # Convert tool_choice through the converter so a bare tool-name string
        # ("my_tool") becomes the wire-format named-tool shape. litellm passes
        # bare strings through unchanged, which providers reject; "auto"/
        # "required"/"none" round-trip identically.
        wire_tool_choice = ChatCompletionConverter.convert_tool_choice(config.tool_choice)

        # Map tool_execution_mode to parallel_tool_calls: True for PARALLEL,
        # False for SEQUENTIAL, None (omit) when unset. Sending None for
        # SEQUENTIAL would let providers default to parallel, ignoring the
        # explicit one-tool-per-turn request.
        parallel_tool_calls = (
            (config.tool_execution_mode == "parallel") if config.tool_execution_mode is not None else None
        )

        # Anthropic requires `tools=` when the conversation contains tool
        # messages (e.g. after a handoff from an agent that had tools); it
        # returns a 400 otherwise. Add a no-op dummy tool so the provider does
        # not reject the request. Restrict this to Anthropic-family models:
        # OpenAI/Gemini accept historical tool messages without re-declaring
        # tools, and injecting a callable tool the developer never defined would
        # add an un-opted-in token and could be invoked spuriously.
        if wire_tools is None and is_anthropic:
            has_tool_content = any(
                msg.get("role") == "tool" or (msg.get("role") == "assistant" and "tool_calls" in msg)
                for msg in converted_messages
            )
            if has_tool_content:
                wire_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "_placeholder",
                            "description": "Placeholder tool (do not call).",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]
                logger.debug("Added placeholder tool — conversation contains tool messages but agent has no tools")

        # Single litellm call with explicit named parameters.
        # LLMConfig field → litellm parameter name mapping:
        #   max_output_tokens → max_tokens
        #   stop_sequences    → stop
        #   response_logprobs → logprobs (bool)
        #   top_logprobs      → top_logprobs (int)
        async def _call_litellm() -> ModelResponse | CustomStreamWrapper:
            # litellm.acompletion is typed to return Any; its runtime contract
            # is ModelResponse (stream=False) | CustomStreamWrapper (stream=True).
            result = await litellm_lib.acompletion(
                model=model,
                messages=converted_messages,
                tools=wire_tools,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                max_tokens=config.max_output_tokens,
                stop=config.stop_sequences,
                seed=config.seed,
                logprobs=config.response_logprobs,
                top_logprobs=config.top_logprobs,
                # litellm's stub types tool_choice as str | dict; convert_tool_choice
                # returns the precise wire TypedDict, which is a dict at runtime.
                tool_choice=wire_tool_choice,  # pyright: ignore[reportArgumentType]
                parallel_tool_calls=parallel_tool_calls,
                response_format=response_format,
                stream=stream,
                stream_options=stream_options,
                timeout=timeout_value,
                num_retries=config.num_retries,
                fallbacks=config.fallbacks,
                reasoning_effort=reasoning_effort,
                # litellm's stub widened AnthropicThinkingParam.type to
                # "enabled" | "adaptive"; our ThinkingParam.type is "enabled", a
                # valid subset litellm accepts at runtime.
                thinking=thinking,  # pyright: ignore[reportArgumentType]
                extra_headers=extra_headers_dict,
                extra_query=config.extra_query,
                api_key=self.api_key,
                base_url=self.base_url,
                cache_control_injection_points=cache_control_injection_points,
                cached_content=cached_content,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
                **extra_kwargs,
            )
            return cast("ModelResponse | CustomStreamWrapper", result)

        # Streaming calls MUST NOT be retried: reconnecting mid-stream
        # would double-emit or silently drop tokens. Only non-streaming
        # responses flow through the framework-level retry policy.
        if not stream and config.retry_policy is not None:
            response = await call_with_retry(_call_litellm, config.retry_policy, model=model)
        else:
            response = await _call_litellm()

        if stream:
            logger.debug("Streaming response started for model=%s", model)
            # stream=True guarantees the streaming arm (CustomStreamWrapper, an
            # AsyncIterator); narrow at the call site per the SDK-return contract.
            return self._stream(cast("CustomStreamWrapper", response))
        else:
            parsed = self._parse_response(response)
            if parsed.usage is not None:
                logger.info(
                    "LLM response: model=%s, input_tokens=%d, output_tokens=%d",
                    parsed.model,
                    parsed.usage.input_tokens,
                    parsed.usage.output_tokens,
                )
            return parsed

    # ------------------------------------------------------------------
    # Cost lookup
    # ------------------------------------------------------------------

    @override
    def cost(self, model: str, usage: LLMUsage) -> float | None:
        """USD cost via litellm's per-model pricing table.

        Docs: https://docs.litellm.ai/docs/completion/token_usage

        Args:
            model: The model identifier to look up in litellm's pricing table.
            usage: Token usage carrying input and output counts.

        Returns:
            Estimated USD cost, or ``None`` when litellm has no price
            data for the model.
        """
        import litellm

        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
            )
        except litellm.NotFoundError:  # pyright: ignore[reportPrivateImportUsage]
            # Normal case: litellm has no pricing data for this model.
            logger.debug("litellm cost lookup: no pricing data for model %s", model)
            return None
        except Exception:
            # Unexpected infrastructure error (network, type error, etc.) —
            # surface at WARNING so it is visible, then return None gracefully.
            logger.warning("litellm cost lookup failed for model %s", model, exc_info=True)
            return None
        return prompt_cost + completion_cost

    # ------------------------------------------------------------------
    # Tool conversion (provider-agnostic → wire format)
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(
        model: str,  # noqa: ARG004
        tools: list[Tool],
    ) -> list[ChatCompletionToolParam]:
        """Convert ``Tool`` instances to litellm wire-format dicts.

        Returns a list of ``ChatCompletionToolParam`` dicts. Both
        ``FunctionTool`` and ``ExecutableBuiltinTool`` are mapped to the
        same function-call format — the LLM layer treats executable
        builtins like function tools on the wire, and the framework
        runs ``on_invoke`` locally when the LLM calls them.

        The litellm path does not natively translate typed
        provider-hosted tool classes (e.g. ``WebSearchTool``,
        ``FileSearchTool``). Pass raw provider JSON via
        ``LLMConfig.extra_body`` / ``LLMConfig.extra_args`` instead;
        native provider modules (``llms/openai/``) handle the typed
        subclasses directly.

        Args:
            model: Model identifier (unused — kept for interface stability).
            tools: List of ``Tool`` instances from the Runner.

        Returns:
            List of typed ``ChatCompletionToolParam`` wire-format dicts.
        """
        from troopai.adk.schemas.utils import SchemaEnforcement
        from troopai.adk.tools.builtin.builtin_tool import ExecutableBuiltinTool
        from troopai.adk.tools.function_tool import FunctionTool

        wire_tools: list[ChatCompletionToolParam] = []
        for tool in tools:
            if isinstance(tool, FunctionTool):
                function_dict = ChatCompletionToolParamFunctionChunk(
                    name=tool.name,
                    description=tool.description or "",
                )
                json_schema = tool.get_json_schema()
                if json_schema is not None:
                    function_dict["parameters"] = json_schema
                    function_dict["strict"] = tool.schema_enforcement in (
                        SchemaEnforcement.STRICT,
                        SchemaEnforcement.COMPACT,
                    )
                wire_tools.append(
                    ChatCompletionToolParam(
                        type="function",
                        function=function_dict,
                    )
                )
            elif isinstance(tool, ExecutableBuiltinTool):
                # Executable builtins have schema + on_invoke — convert to function-call format
                from pydantic import BaseModel as _BaseModel

                from troopai.adk.schemas.utils import normalize_schema

                raw_schema = (
                    tool.schema.model_json_schema()
                    if isinstance(tool.schema, type) and issubclass(tool.schema, _BaseModel)
                    else tool.schema
                )
                schema = normalize_schema(raw_schema)
                function_dict = ChatCompletionToolParamFunctionChunk(
                    name=tool.name,
                    description=tool.description or "",
                )
                if schema is not None:
                    function_dict["parameters"] = schema
                wire_tools.append(
                    ChatCompletionToolParam(
                        type="function",
                        function=function_dict,
                    )
                )
            else:
                logger.warning("Unknown tool type: %s — skipping", type(tool))
        return wire_tools

    # ------------------------------------------------------------------
    # Gemini thought signature handling
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_gemini_thought_signatures(
        messages: list[Any],
    ) -> list[Any]:
        """Convert Gemini thought signatures to litellm's provider_specific_fields.

        Transforms tool calls from the converter's format to litellm's format:
        - Reads ``provider_data.google.thought_signature`` from tool call dicts
        - Sets ``provider_specific_fields.thought_signature`` for litellm
        - For tool calls without a valid thought signature (after the last user
          message), inserts ``"skip_thought_signature_validator"`` to prevent
          Gemini API validation errors

        Only processes assistant messages after the last user message (current turn).

        See: https://ai.google.dev/gemini-api/docs/thought-signatures

        Args:
            messages: Converted Chat Completions messages. Not mutated —
                messages that need rewriting are shallow-copied first.

        Returns:
            A new list; messages that carry current-turn tool calls are copies
            with ``provider_specific_fields`` set, the rest pass through by
            reference.
        """
        # Find the last user message index
        last_user_index = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], dict) and messages[i].get("role") == "user":
                last_user_index = i
                break

        converted: list[Any] = []
        for i, message in enumerate(messages):
            # Only current-turn (after the last user message) assistant messages
            # with tool calls carry thought signatures; pass everything else
            # through unchanged.
            is_target = (
                isinstance(message, dict)
                and (last_user_index == -1 or i > last_user_index)
                and message.get("role") == "assistant"
                and bool(message.get("tool_calls"))
            )
            if not is_target:
                converted.append(message)
                continue

            # Copy the message and each tool call before rewriting so the
            # caller's history is never mutated in place: popping ``provider_data``
            # off a shared dict would corrupt a later retry or re-run.
            new_message = dict(message)
            new_tool_calls: list[Any] = []
            for tool_call in message.get("tool_calls", []):
                if not isinstance(tool_call, dict):
                    new_tool_calls.append(tool_call)
                    continue
                new_call = dict(tool_call)
                provider_data = new_call.pop("provider_data", None)
                thought_sig = None
                if isinstance(provider_data, dict):
                    google_fields = provider_data.get("google")
                    if isinstance(google_fields, dict):
                        thought_sig = google_fields.get("thought_signature")
                # A real signature validates; otherwise skip validation so Gemini
                # does not reject a tool call that never carried one.
                new_call["provider_specific_fields"] = {
                    "thought_signature": thought_sig if thought_sig is not None else "skip_thought_signature_validator"
                }
                new_tool_calls.append(new_call)
            new_message["tool_calls"] = new_tool_calls
            converted.append(new_message)

        return converted

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    @staticmethod
    def _build_response_format(
        output_schema: AgentOutputSchemaBase,
    ) -> dict[str, Any]:
        """Build ``response_format`` for structured output.

        Uses JSON schema mode (``type: "json_schema"``) to constrain
        the model's output to a specific schema.  This replaces the
        ``instructor`` library for structured output.

        The model returns JSON in ``response.choices[0].message.content``,
        which is then validated by ``output_schema.validate_json()``.

        Returns a plain dict because litellm's ``response_format`` parameter
        accepts ``dict | BaseModel | None`` and litellm does not define
        typed response format parameters in its type system.

        Shape follows the Chat Completions API convention::

            {"type": "json_schema", "json_schema": {"name": str, "schema": dict, "strict": bool}}

        Args:
            output_schema: The output schema to enforce.

        Returns:
            A response_format dict for ``litellm.acompletion()``.

        References:
            - litellm JSON mode: https://docs.litellm.ai/docs/completion/json_mode
            - Chat Completions response_format:
              https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": output_schema.name(),
                "schema": output_schema.json_schema(),
                "strict": output_schema.is_strict_json_schema(),
            },
        }

    def _resolve_response_format(
        self,
        output_schema: AgentOutputSchemaBase,
        model: str,
        litellm_lib: Any,
    ) -> dict[str, Any] | None:
        """Choose the best ``response_format`` based on model capabilities.

        Implements a three-tier fallback strategy:

        1. **JSON Schema mode** — if the model supports
           ``supports_response_schema()``, use full schema constraints.
           The model is natively constrained to produce valid JSON
           matching the schema.
        2. **JSON Object mode** — if the model supports ``response_format``
           but not schema mode, use ``{"type": "json_object"}``.  The
           model outputs valid JSON; validation happens downstream in
           ``run/turn_resolution.resolve_structured_output_step``.
        3. **No JSON support** — if the model has no native JSON mode,
           return ``None`` and rely on prompt instructions plus
           downstream client-side validation in
           ``run/turn_resolution.resolve_structured_output_step``.

        Args:
            output_schema: The structured output schema.
            model: The model identifier.
            litellm_lib: The litellm module (passed to avoid re-importing).

        Returns:
            A response_format dict, or ``None`` if the model has no
            JSON support.
        """
        # Tier 1: Full JSON schema mode
        try:
            if litellm_lib.supports_response_schema(model=model):
                logger.debug(
                    "Model %s supports json_schema mode for structured output",
                    model,
                )
                return self._build_response_format(output_schema)
        except Exception as e:
            # litellm may raise for unrecognized models; fall through
            logger.debug(
                "litellm supports_response_schema raised for model=%s: %s; trying the json_object tier", model, e
            )

        # Tier 2: JSON object mode (model outputs JSON, we validate schema)
        try:
            supported_params = litellm_lib.get_supported_openai_params(
                model=model,
            )
            if supported_params is not None and "response_format" in supported_params:
                logger.debug(
                    "Model %s supports json_object but not json_schema; "
                    "using json_object with client-side schema validation",
                    model,
                )
                # Tier 2 shape: {"type": "json_object"}
                return {"type": "json_object"}
        except Exception as e:
            logger.debug(
                "litellm get_supported_openai_params raised for model=%s: %s; falling to the prompt tier", model, e
            )

        # Tier 3: No JSON support — rely on prompt + client-side validation
        logger.warning(
            "Model %s does not support response_format. "
            "Structured output will rely on prompt instructions "
            "and client-side validation.",
            model,
        )
        return None

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        response: Any,
    ) -> LLMResponse:
        """Convert a litellm ``ModelResponse`` into our ``LLMResponse``.

        Extracts content, tool calls, and usage from the provider's
        response. Structured-output validation happens downstream in
        ``run/turn_resolution.resolve_structured_output_step``.

        Args:
            response: The raw response from ``litellm.acompletion()``.

        Returns:
            A fully populated ``LLMResponse``.
        """
        from datetime import datetime

        from troopai.adk.types.responses.llm_response import (
            LLMResponse,
            LLMResponseAnnotation,
            LLMResponseFunctionToolCall,
            LLMResponsePart,
            LLMResponseReasoning,
            LLMResponseRefusal,
            LLMResponseText,
        )

        # Extract message from first choice
        message = response.choices[0].message if response.choices else None
        resp_model = getattr(response, "model", "") or ""
        finish_reason = getattr(response.choices[0], "finish_reason", None) if response.choices else None

        # Build parts list in order
        parts: list[LLMResponsePart] = []

        # --- Thinking blocks (appear before text in conversation order) ---
        raw_thinking_blocks = getattr(message, "thinking_blocks", None) if message is not None else None
        if raw_thinking_blocks is not None:
            for block in raw_thinking_blocks:
                block_dict = dict(block) if not isinstance(block, dict) else block
                block_type = block_dict.get("type", "")
                if block_type == "thinking":
                    parts.append(
                        LLMResponseReasoning(
                            thinking=str(block_dict.get("thinking", "")),
                            signature=str(block_dict.get("signature", ""))
                            if block_dict.get("signature") is not None
                            else None,
                        )
                    )
                elif block_type == "redacted_thinking":
                    # Redacted: preserve signature data, empty thinking
                    parts.append(
                        LLMResponseReasoning(
                            thinking="",
                            signature=str(block_dict.get("data", "")),
                        )
                    )

        # --- Reasoning content (unified string — fallback if no structured blocks) ---
        reasoning_content = getattr(message, "reasoning_content", None) if message is not None else None
        if reasoning_content is not None and len(parts) == 0:
            # Only add as a part if no structured thinking blocks were found
            parts.append(LLMResponseReasoning(thinking=str(reasoning_content)))

        # --- Text content ---
        content = getattr(message, "content", None) if message is not None else None
        annotations: list[LLMResponseAnnotation] = []
        raw_annotations = getattr(message, "annotations", None) if message is not None else None
        if raw_annotations is not None and isinstance(raw_annotations, list):
            for ann in raw_annotations:
                ann_dict = dict(ann) if not isinstance(ann, dict) else ann
                url_citation = ann_dict.get("url_citation", {})
                if isinstance(url_citation, dict):
                    annotations.append(
                        LLMResponseAnnotation(
                            url=str(url_citation.get("url", "")),
                            title=str(url_citation.get("title", "")),
                            start_index=int(url_citation.get("start_index", 0)),
                            end_index=int(url_citation.get("end_index", 0)),
                        )
                    )

        if content is not None:
            parts.append(
                LLMResponseText(
                    text=content,
                    annotations=annotations if len(annotations) > 0 else None,
                )
            )

        # --- Refusal ---
        refusal = getattr(message, "refusal", None) if message is not None else None
        if refusal is None and message is not None:
            provider_fields = getattr(message, "provider_specific_fields", None)
            if isinstance(provider_fields, dict):
                refusal = provider_fields.get("refusal")
        if refusal is not None:
            parts.append(LLMResponseRefusal(refusal=str(refusal)))

        # --- Tool calls ---
        raw_tool_calls = getattr(message, "tool_calls", None) if message is not None else None
        if raw_tool_calls is not None:
            for tc in raw_tool_calls:
                tc_id = tc.id or ""
                if "gemini" in resp_model.lower() and "__thought__" in tc_id:
                    tc_id = tc_id.split("__thought__")[0]
                parts.append(
                    LLMResponseFunctionToolCall(
                        call_id=tc_id,
                        name=tc.function.name or "",
                        arguments=tc.function.arguments or "",
                    )
                )

        # Parse usage
        usage = self._parse_usage(response)

        # Timestamp from response.created (unix epoch)
        created = getattr(response, "created", None)
        timestamp = datetime.fromtimestamp(created, tz=UTC) if created is not None else None

        return LLMResponse(
            response_id=getattr(response, "id", "") or "",
            model=resp_model,
            response=parts,
            usage=usage,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            timestamp=timestamp,
        )

    @staticmethod
    def _parse_usage(response: Any) -> LLMUsage | None:
        """Extract token usage from a litellm response.

        Handles differences between providers:

        - OpenAI: ``prompt_tokens_details.cached_tokens``
        - Anthropic: ``cache_creation_input_tokens``, ``cache_read_input_tokens``
        - Others: Basic ``prompt_tokens`` / ``completion_tokens``

        Args:
            response: The raw response from litellm.

        Returns:
            An ``LLMUsage`` instance, or ``None`` if usage data is absent.
        """
        from troopai.adk.llms.llm_usage import LLMUsage
        from troopai.adk.types.tokens import InputTokensDetails, OutputTokensDetails

        raw_usage = getattr(response, "usage", None)
        if raw_usage is None:
            return None

        input_tokens = getattr(raw_usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(raw_usage, "completion_tokens", 0) or 0
        # litellm leaves total_tokens at 0 when a provider omits it (it does
        # not synthesize prompt+completion). Synthesize here so the per-request
        # usage entry is recorded — LLMUsage.__add__ drops requests with
        # total_tokens == 0 from its breakdown.
        total_tokens = getattr(raw_usage, "total_tokens", 0) or 0
        if total_tokens == 0 and (input_tokens > 0 or output_tokens > 0):
            total_tokens = input_tokens + output_tokens

        # Extract cached token details
        cached_tokens = 0
        cache_creation_tokens = 0

        # OpenAI-style: prompt_tokens_details.cached_tokens
        prompt_details = getattr(raw_usage, "prompt_tokens_details", None)
        if prompt_details is not None:
            cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0

        # Anthropic-style: cache_read_input_tokens.  litellm normalises this
        # value into prompt_tokens_details.cached_tokens as well, so prefer
        # the explicit field when present (takes priority over the OpenAI-style
        # cached_tokens read above).
        cache_read = getattr(raw_usage, "cache_read_input_tokens", None)
        if cache_read is not None:
            cached_tokens = cache_read

        cache_creation = getattr(raw_usage, "cache_creation_input_tokens", None)
        if cache_creation is not None:
            cache_creation_tokens = cache_creation

        # Extract reasoning token details
        reasoning_tokens = 0
        completion_details = getattr(raw_usage, "completion_tokens_details", None)
        if completion_details is not None:
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0

        return LLMUsage(
            requests=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_tokens_details=InputTokensDetails(
                cached_tokens=cached_tokens,
                cache_creation_input_tokens=cache_creation_tokens,
            ),
            output_tokens_details=OutputTokensDetails(
                reasoning_tokens=reasoning_tokens,
            ),
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def _stream(
        self,
        response_chunks: AsyncIterator,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Process streaming chunks into ``LLMStreamEvent`` objects.

        Iterates over the async chunk iterator returned by
        ``litellm.acompletion(stream=True)`` and emits:

        - ``LLMStreamEvent(type="part_start")`` when a new response part begins
        - ``LLMStreamEvent(type="part_delta")`` for incremental text/reasoning fragments
        - ``LLMStreamEvent(type="part_end")`` when a response part is finalized
        - ``LLMStreamEvent(type="done")`` with the complete ``LLMResponse``

        Structured-output validation (when the agent has ``output_schema``
        set) happens downstream in
        ``run/turn_resolution.resolve_structured_output_step``.

        Args:
            response_chunks: Async iterator of streaming chunks from
                ``litellm.acompletion()``.

        Yields:
            ``LLMStreamEvent`` instances.
        """
        from troopai.adk.types.responses.llm_response import (
            LLMResponseFunctionToolCall,
            LLMResponseReasoning,
            LLMResponseText,
            LLMStreamEvent,
        )

        accumulated_content = ""
        accumulated_reasoning = ""
        # Structured thinking blocks (carry the signatures Anthropic requires on
        # replay) arrive fragmented across per-chunk deltas. They MUST be
        # accumulated here: the final usage-sentinel chunk has ``choices=[]`` and
        # never carries them, so reading them off the last chunk always lost them.
        accumulated_thinking_blocks: list[dict[str, Any]] = []
        tool_call_data: dict[int, _StreamedToolCall] = {}
        last_chunk: Any = None
        # Tracks the most recent chunk that actually carried choices.  When
        # ``include_usage=True`` (the default), litellm/OpenAI sends a final
        # usage-sentinel chunk with ``choices=[]``; that sentinel becomes
        # ``last_chunk`` but has no finish_reason.  We read finish_reason from
        # ``last_content_chunk`` so it is always the chunk that carried the
        # model's actual stop signal.
        last_content_chunk: Any = None

        # Monotonically incrementing counter for part indices.  Indices are
        # assigned lazily as each part type first appears, so when reasoning is
        # absent the text part gets index 0 (no gap) and tool calls start
        # immediately after text.
        next_part_index = 0
        reasoning_part_index: int | None = None
        text_part_index: int | None = None
        # Maps litellm tc.index → assigned stream part index for tool calls.
        tool_part_index: dict[int, int] = {}

        try:
            async for chunk in response_chunks:
                last_chunk = chunk

                # Hoist to a local so the two uses of `choices` below share a
                # single attribute lookup instead of two.
                chunk_choices = getattr(chunk, "choices", None)
                if not chunk_choices:
                    continue

                last_content_chunk = chunk

                delta = chunk_choices[0].delta

                # Content delta → part_delta for text. Guard against the empty
                # priming chunk (litellm/OpenAI send delta.content="" on the
                # role chunk), which would otherwise emit a phantom empty text
                # part and consume a part index.
                delta_content = getattr(delta, "content", None)
                if delta_content is not None and len(delta_content) > 0:
                    if text_part_index is None:
                        text_part_index = next_part_index
                        next_part_index += 1
                        yield LLMStreamEvent(
                            type="part_start",
                            index=text_part_index,
                            part=LLMResponseText(text=""),
                        )
                    accumulated_content += delta_content
                    yield LLMStreamEvent(type="part_delta", index=text_part_index, delta=delta_content)

                # Reasoning delta → part_delta for thinking. Same empty-priming
                # guard as the content branch: an empty reasoning_content chunk
                # would emit a phantom reasoning part and shift the text part's
                # index.
                delta_reasoning = getattr(delta, "reasoning_content", None)
                if delta_reasoning is not None and len(delta_reasoning) > 0:
                    if reasoning_part_index is None:
                        reasoning_part_index = next_part_index
                        next_part_index += 1
                        yield LLMStreamEvent(
                            type="part_start",
                            index=reasoning_part_index,
                            part=LLMResponseReasoning(thinking=""),
                        )
                    accumulated_reasoning += delta_reasoning
                    yield LLMStreamEvent(type="part_delta", index=reasoning_part_index, delta=delta_reasoning)

                # Structured thinking-block fragments → accumulate for the final
                # response. These carry the per-block signature (emitted in a
                # later delta than the thinking text) that the plain
                # reasoning_content stream above does not preserve.
                delta_thinking = getattr(delta, "thinking_blocks", None)
                if isinstance(delta_thinking, list):
                    for block in delta_thinking:
                        accumulated_thinking_blocks.append(block if isinstance(block, dict) else dict(block))

                # Tool call deltas → accumulate (emit part_start on first chunk)
                delta_tool_calls = getattr(delta, "tool_calls", None)
                if delta_tool_calls is not None:
                    for tc in delta_tool_calls:
                        idx = tc.index
                        if idx not in tool_call_data:
                            tool_call_data[idx] = _StreamedToolCall(
                                id="",
                                function=_StreamedFunctionData(
                                    name="",
                                    arguments="",
                                ),
                            )
                            # Assign a fresh part index for this tool call.
                            assigned = next_part_index
                            next_part_index += 1
                            tool_part_index[idx] = assigned
                            yield LLMStreamEvent(
                                type="part_start",
                                index=assigned,
                                part=LLMResponseFunctionToolCall(),
                            )

                        tc_id = getattr(tc, "id", None)
                        if tc_id is not None:
                            tool_call_data[idx]["id"] = tc_id

                        tc_func = getattr(tc, "function", None)
                        if tc_func is not None:
                            func_name = getattr(tc_func, "name", None)
                            if func_name is not None:
                                tool_call_data[idx]["function"]["name"] = func_name

                            func_args = getattr(tc_func, "arguments", None)
                            if func_args is not None:
                                tool_call_data[idx]["function"]["arguments"] += func_args
                                yield LLMStreamEvent(
                                    type="part_delta",
                                    index=tool_part_index[idx],
                                    delta=func_args,
                                )
        except Exception as exc:
            # Mid-stream provider error: emit a terminal done(finish_reason="error")
            # carrying the partial response (and any usage on last_chunk) so
            # consumers awaiting `done` — usage flush, transcript finalize —
            # finalize cleanly, THEN re-raise so the error still propagates.
            # `Exception` excludes CancelledError/KeyboardInterrupt/SystemExit
            # (all BaseException), so cancellation is never masked. This is the
            # cross-provider streaming-error contract.
            logger.error("LLM stream failed mid-stream: model=%s: %s", self.model, exc)
            partial = replace(
                self._build_stream_response(
                    accumulated_content=accumulated_content,
                    accumulated_reasoning=accumulated_reasoning,
                    accumulated_thinking_blocks=accumulated_thinking_blocks,
                    tool_call_data=tool_call_data,
                    last_chunk=last_chunk,
                    last_content_chunk=last_content_chunk,
                ),
                finish_reason="error",
            )
            yield LLMStreamEvent(type="done", response=partial)
            raise

        # Emit part_end for every started part so the API contract matches the
        # other provider implementations (anthropic_model, openai_chatcompletions,
        # openai_responses, gemini_model all emit part_end).  Indices are emitted
        # in ascending order — reasoning first (lowest), then text, then tool
        # calls in their litellm-assigned order.
        parts_in_order: list[int] = []
        if reasoning_part_index is not None:
            parts_in_order.append(reasoning_part_index)
        if text_part_index is not None:
            parts_in_order.append(text_part_index)
        for tc_idx in sorted(tool_part_index.keys()):
            parts_in_order.append(tool_part_index[tc_idx])
        for part_idx in sorted(parts_in_order):
            yield LLMStreamEvent(type="part_end", index=part_idx)

        # Build final response
        response = self._build_stream_response(
            accumulated_content=accumulated_content,
            accumulated_reasoning=accumulated_reasoning,
            accumulated_thinking_blocks=accumulated_thinking_blocks,
            tool_call_data=tool_call_data,
            last_chunk=last_chunk,
            last_content_chunk=last_content_chunk,
        )

        if response.usage is not None:
            logger.info(
                "LLM stream complete: model=%s, input_tokens=%d, output_tokens=%d",
                response.model,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )

        yield LLMStreamEvent(type="done", response=response)

    @staticmethod
    def _combine_streamed_thinking(
        raw_blocks: list[dict[str, Any]],
    ) -> list[LLMResponseReasoning]:
        """Combine per-chunk thinking-block fragments into complete signed blocks.

        Mirrors litellm's own streaming reconstruction: a thinking block's text
        arrives in fragments and the block is finalized when its ``signature``
        delta appears; ``redacted_thinking`` blocks carry opaque ``data`` and
        stand alone. A trailing block with text but no signature is still
        emitted so nothing is dropped.

        Args:
            raw_blocks: Thinking-block dicts gathered in order from the deltas.

        Returns:
            One ``LLMResponseReasoning`` per reconstructed block, in order.
        """
        from troopai.adk.types.responses.llm_response import LLMResponseReasoning

        combined: list[LLMResponseReasoning] = []
        text_parts: list[str] = []
        signature: str | None = None

        def flush() -> None:
            nonlocal text_parts, signature
            if len(text_parts) > 0 or signature is not None:
                combined.append(LLMResponseReasoning(thinking="".join(text_parts), signature=signature))
            text_parts = []
            signature = None

        for block in raw_blocks:
            if block.get("type") == "redacted_thinking":
                flush()
                data = block.get("data")
                if data is not None:
                    combined.append(LLMResponseReasoning(thinking="", signature=str(data)))
                continue
            text = block.get("thinking")
            if text is not None and len(str(text)) > 0:
                text_parts.append(str(text))
            sig = block.get("signature")
            if sig is not None and len(str(sig)) > 0:
                signature = str(sig)
                flush()
        flush()
        return combined

    def _build_stream_response(
        self,
        accumulated_content: str,
        accumulated_reasoning: str,
        accumulated_thinking_blocks: list[dict[str, Any]],
        tool_call_data: dict[int, _StreamedToolCall],
        last_chunk: Any,
        last_content_chunk: Any = None,
    ) -> LLMResponse:
        """Reconstruct a complete ``LLMResponse`` from streaming chunks.

        Called after all chunks have been received.  Assembles tool calls
        from accumulated fragments, extracts usage from the last chunk,
        and optionally validates structured output.

        Args:
            accumulated_content: Full text content from all chunks.
            accumulated_reasoning: Full reasoning content from all chunks.
            accumulated_thinking_blocks: Structured thinking-block fragments
                gathered from every per-chunk delta. Combined into signed
                reasoning parts; the plain ``accumulated_reasoning`` text is a
                fallback only when no structured blocks arrived.
            tool_call_data: Accumulated tool call fragments by index.
            last_chunk: The last streaming chunk (contains usage).
            last_content_chunk: The most recent chunk with non-empty ``choices``.
                Used for ``finish_reason`` to avoid the usage-sentinel chunk
                (``choices=[]``) overwriting the real stop signal when
                ``include_usage=True`` is set.

        Returns:
            A complete ``LLMResponse``.
        """
        from troopai.adk.types.responses.llm_response import (
            LLMResponse,
            LLMResponseFunctionToolCall,
            LLMResponsePart,
            LLMResponseReasoning,
            LLMResponseText,
        )

        parts: list[LLMResponsePart] = []

        # --- Thinking ---
        # Prefer structured blocks combined from the per-chunk deltas: they
        # carry the per-block signature Anthropic requires on replay. Fall back
        # to the plain reasoning text only when no structured blocks streamed.
        combined_thinking = self._combine_streamed_thinking(accumulated_thinking_blocks)
        if len(combined_thinking) > 0:
            parts.extend(combined_thinking)
        elif len(accumulated_reasoning) > 0:
            parts.append(LLMResponseReasoning(thinking=accumulated_reasoning))

        # --- Text content ---
        content = accumulated_content if len(accumulated_content) > 0 else None
        if content is not None:
            parts.append(LLMResponseText(text=content))

        # --- Tool calls ---
        # Derive model name from last_content_chunk (or last_chunk) so the
        # Gemini __thought__ suffix stripping matches the non-streaming path.
        _model_lower = ""
        _fr_chunk_for_model = last_content_chunk if last_content_chunk is not None else last_chunk
        if _fr_chunk_for_model is not None:
            _model_lower = (getattr(_fr_chunk_for_model, "model", "") or "").lower()
        for idx in sorted(tool_call_data.keys()):
            data = tool_call_data[idx]
            tc_id = data["id"]
            # Gemini appends a ``__thought__`` suffix to tool-call IDs in some
            # streaming responses; strip it to match the non-streaming path so
            # tool-result IDs round-trip correctly on the next turn.
            if "gemini" in _model_lower and "__thought__" in tc_id:
                tc_id = tc_id.split("__thought__")[0]
            parts.append(
                LLMResponseFunctionToolCall(
                    call_id=tc_id,
                    name=data["function"]["name"],
                    arguments=data["function"]["arguments"],
                )
            )

        # Extract usage from the final sentinel chunk (which may have empty
        # choices when ``include_usage=True``).  Extract finish_reason from
        # the last chunk that actually carried choices — the usage sentinel
        # has ``choices=[]`` and would leave finish_reason=None.
        usage = self._parse_usage(last_chunk) if last_chunk is not None else None
        finish_reason: str | None = None
        fr_chunk = last_content_chunk if last_content_chunk is not None else last_chunk
        if fr_chunk is not None:
            fr_choices = getattr(fr_chunk, "choices", None)
            if fr_choices:
                raw_fr = getattr(fr_choices[0], "finish_reason", None)
                finish_reason = str(raw_fr) if raw_fr is not None else None

        return LLMResponse(
            response_id=getattr(last_chunk, "id", "") or "" if last_chunk is not None else "",
            model=(getattr(last_chunk, "model", "") or "" if last_chunk is not None else ""),
            response=parts,
            usage=usage,
            finish_reason=finish_reason,
        )


# Type alias for the streaming return type (string annotation avoids
# importing LLMStreamEvent at runtime, which lives under TYPE_CHECKING).
LLMStreamResponse = "AsyncIterator[LLMStreamEvent]"
"""Type alias for the async iterator returned by streaming LLM calls."""

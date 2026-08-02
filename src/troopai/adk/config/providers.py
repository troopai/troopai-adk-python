"""LLM provider registry and factories for declarative configs.

A provider factory turns a validated typed ``llm`` block into a concrete
``LLM`` instance plus an optional runtime ``LLMConfig``. This module is the
ONLY config-layer module that imports the concrete LLM implementations and
their runtime configs — lazily, inside each factory, so the import (and any
missing-optional-dependency error) happens only when that provider block is
first built, never at application import time. A provider whose package is
not installed raises ``ModuleNotFoundError`` from the provider module; the
assembler's ``build_llm`` wraps that into a ``ConfigResolutionError`` naming
the provider. Everything above this module (the schema models, the assembler
dispatch) stays provider-agnostic.

``register_llm_provider`` exposes the registry for extension, mirroring the
eval loader's ``register_grader`` pattern. The built-in factories are
registered at import time.

Security: building a provider block constructs an ``LLM`` from the named
provider. Load only config files you trust.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from troopai.adk.exceptions import ConfigResolutionError
from troopai.adk.llms.llm import LLM
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.types.config.llm_config import (
    AnthropicProviderBlock,
    GeminiProviderBlock,
    LiteLLMProviderBlock,
    LLMConfigBlock,
    LLMProviderConfig,
    OpenAIChatProviderBlock,
    OpenAIResponsesProviderBlock,
)
from troopai.adk.types.llms.retry_policy import LLMRetryPolicy

logger = logging.getLogger(__name__)

ProviderFactory = Callable[[LLMProviderConfig], "tuple[LLM, LLMConfig | None]"]
"""Factory taking a validated provider block, returning ``(LLM, LLMConfig | None)``."""

PROVIDER_REGISTRY: dict[str, ProviderFactory] = {}
"""Registry mapping a ``provider`` discriminator to its factory."""


def register_llm_provider(name: str, factory: ProviderFactory) -> None:
    """Register (or override) the factory for a provider name.

    Mirrors the eval loader's ``register_grader`` extension point. Registering
    a name the schema's ``LLMProviderConfig`` union does not accept leaves an
    unreachable entry — Pydantic rejects an unknown ``provider`` block before
    dispatch — so in practice this overrides how a built-in provider is built
    rather than introducing a new schema variant.

    Args:
        name: The ``provider`` discriminator value (e.g. ``"anthropic"``).
        factory: Callable taking the validated provider block and returning
            ``(LLM, LLMConfig | None)``.
    """
    PROVIDER_REGISTRY[name] = factory
    logger.debug("Registered LLM provider factory: %r", name)


def _split_fields(block: LLMConfigBlock) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a config block's set fields into agnostic vs provider-specific.

    Returns ``(agnostic, provider_specific)`` where ``agnostic`` maps to
    base :class:`LLMConfig` constructor kwargs and ``provider_specific`` to
    the extra kwargs of a provider config subclass (callers building an
    agnostic config discard the latter). ``mode="python"`` keeps enums
    (``ToolExecutionMode``) intact. The agnostic ``retry_policy`` (dumped to
    a dict) is reconstructed into a runtime :class:`LLMRetryPolicy` with
    ``retry_on`` back as a frozenset; an omitted ``retry_on`` is left off the
    constructor call so the dataclass default (rate-limit only) applies
    rather than ``None``, which would broaden the scope to every transient
    error kind the developer never opted into. ``timeout`` is already a float and
    needs no reconstruction — owning this here keeps the config layer
    decoupled from the temporal serializer.
    """
    dumped = block.model_dump(exclude_none=True, mode="python")
    agnostic = {key: value for key, value in dumped.items() if key in LLMConfigBlock.model_fields}
    provider_specific = {key: value for key, value in dumped.items() if key not in LLMConfigBlock.model_fields}
    retry = agnostic.get("retry_policy")
    if isinstance(retry, dict):
        kwargs = {key: value for key, value in retry.items() if key != "retry_on"}
        if "retry_on" in retry:
            kwargs["retry_on"] = frozenset(retry["retry_on"])
        agnostic["retry_policy"] = LLMRetryPolicy(**kwargs)
    return agnostic, provider_specific


def build_agnostic_config(block: LLMConfigBlock | None) -> LLMConfig | None:
    """Build a base ``LLMConfig`` from an agnostic config block.

    Used by the string-``llm`` path.

    Args:
        block: The agnostic block, or ``None``.

    Returns:
        An ``LLMConfig``, or ``None`` when ``block`` is ``None``.
    """
    if block is None:
        return None
    agnostic, _ = _split_fields(block)
    return LLMConfig(**agnostic)


def _provider_config(block: LLMConfigBlock, config_cls: type[LLMConfig]) -> LLMConfig:
    """Build a runtime provider config subclass from a provider config block.

    The agnostic fields are reconstructed (``retry_policy`` frozenset) and
    the provider-specific fields layered on as constructor kwargs.
    ``config_cls`` is the runtime dataclass (e.g. ``AnthropicConfig``).

    Raises:
        ConfigResolutionError: If a config-block field has no matching field
            on ``config_cls`` (a schema-vs-runtime drift), turning the bare
            ``TypeError`` into an actionable message.
    """
    agnostic, provider_specific = _split_fields(block)
    try:
        return config_cls(**agnostic, **provider_specific)
    except TypeError as exc:
        raise ConfigResolutionError(
            f"Could not build {config_cls.__name__} from the config block ({exc}). "
            f"Provider-specific fields supplied: {sorted(provider_specific)}."
        ) from exc


# Each factory re-checks its block type with ``isinstance``. Normal dispatch
# (``PROVIDER_REGISTRY[block.provider]``) cannot mismatch, so the guard is a
# defense against registry misuse — a factory registered under the wrong
# ``register_llm_provider`` name — and the narrowing the type checker needs.
def _build_anthropic(block: LLMProviderConfig) -> tuple[LLM, LLMConfig | None]:
    """Factory for the native Anthropic provider."""
    if not isinstance(block, AnthropicProviderBlock):
        raise ConfigResolutionError(
            f"anthropic factory received {type(block).__name__}, expected AnthropicProviderBlock."
        )
    from troopai.adk.llms.anthropic.anthropic_config import AnthropicConfig
    from troopai.adk.llms.anthropic.anthropic_model import AnthropicLLM

    extra: dict[str, int] = {} if block.max_retries is None else {"max_retries": block.max_retries}
    llm = AnthropicLLM(
        model=block.model,
        api_key=block.api_key,
        base_url=block.base_url,
        **extra,
    )
    config = _provider_config(block.config, AnthropicConfig) if block.config is not None else None
    logger.debug("Built anthropic LLM for model %r", block.model)
    return llm, config


def _build_openai_responses(block: LLMProviderConfig) -> tuple[LLM, LLMConfig | None]:
    """Factory for the native OpenAI Responses provider."""
    if not isinstance(block, OpenAIResponsesProviderBlock):
        raise ConfigResolutionError(
            f"openai-responses factory received {type(block).__name__}, expected OpenAIResponsesProviderBlock."
        )
    from troopai.adk.llms.openai.openai_responses_config import OpenAIResponsesConfig
    from troopai.adk.llms.openai.openai_responses_model import OpenAIResponsesLLM

    extra = {} if block.max_retries is None else {"max_retries": block.max_retries}
    llm = OpenAIResponsesLLM(
        model=block.model,
        api_key=block.api_key,
        base_url=block.base_url,
        organization=block.organization,
        project=block.project,
        **extra,
    )
    config = _provider_config(block.config, OpenAIResponsesConfig) if block.config is not None else None
    logger.debug("Built openai-responses LLM for model %r", block.model)
    return llm, config


def _build_openai_chat(block: LLMProviderConfig) -> tuple[LLM, LLMConfig | None]:
    """Factory for the native OpenAI Chat-Completions provider."""
    if not isinstance(block, OpenAIChatProviderBlock):
        raise ConfigResolutionError(
            f"openai-chat factory received {type(block).__name__}, expected OpenAIChatProviderBlock."
        )
    from troopai.adk.llms.openai.openai_chatcompletions_config import OpenAIChatCompletionsConfig
    from troopai.adk.llms.openai.openai_chatcompletions_model import OpenAIChatCompletionsLLM

    extra = {} if block.max_retries is None else {"max_retries": block.max_retries}
    llm = OpenAIChatCompletionsLLM(
        model=block.model,
        api_key=block.api_key,
        base_url=block.base_url,
        organization=block.organization,
        project=block.project,
        **extra,
    )
    config = _provider_config(block.config, OpenAIChatCompletionsConfig) if block.config is not None else None
    logger.debug("Built openai-chat LLM for model %r", block.model)
    return llm, config


def _build_gemini(block: LLMProviderConfig) -> tuple[LLM, LLMConfig | None]:
    """Factory for the native Google Gemini provider."""
    if not isinstance(block, GeminiProviderBlock):
        raise ConfigResolutionError(f"gemini factory received {type(block).__name__}, expected GeminiProviderBlock.")
    from troopai.adk.llms.gemini.gemini_config import GeminiConfig
    from troopai.adk.llms.gemini.gemini_model import GeminiLLM

    llm = GeminiLLM(
        model=block.model,
        api_key=block.api_key,
        vertexai=block.vertexai if block.vertexai is not None else False,
        project=block.project,
        location=block.location,
        base_url=block.base_url,
    )
    config = _provider_config(block.config, GeminiConfig) if block.config is not None else None
    logger.debug("Built gemini LLM for model %r", block.model)
    return llm, config


def _build_litellm(block: LLMProviderConfig) -> tuple[LLM, LLMConfig | None]:
    """Factory for the LiteLLM multi-provider backend."""
    if not isinstance(block, LiteLLMProviderBlock):
        raise ConfigResolutionError(f"litellm factory received {type(block).__name__}, expected LiteLLMProviderBlock.")
    from troopai.adk.llms.litellm.litellm_model import LiteLLM, LiteLLMConfig

    llm = LiteLLM(
        model=block.model,
        api_key=block.api_key,
        base_url=block.base_url,
        extra_params=block.extra_params,
    )
    config = _provider_config(block.config, LiteLLMConfig) if block.config is not None else None
    logger.debug("Built litellm LLM for model %r", block.model)
    return llm, config


register_llm_provider("anthropic", _build_anthropic)
register_llm_provider("openai-responses", _build_openai_responses)
register_llm_provider("openai-chat", _build_openai_chat)
register_llm_provider("gemini", _build_gemini)
register_llm_provider("litellm", _build_litellm)

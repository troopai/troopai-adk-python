"""Tests for the LLM provider registry and factories.

Factories construct a concrete ``LLM`` plus an optional runtime
``LLMConfig`` from a validated provider block. Provider SDKs are NOT
required for construction: the LLM constructors set attributes only
(clients lazy-init), and the configs are SDK-free dataclasses, so these
tests build configs without making network calls.
"""

from __future__ import annotations

import pytest

from troopai.adk.config.providers import (
    PROVIDER_REGISTRY,
    build_agnostic_config,
    register_llm_provider,
)
from troopai.adk.llms.anthropic.anthropic_config import AnthropicConfig
from troopai.adk.llms.anthropic.anthropic_model import AnthropicLLM
from troopai.adk.llms.gemini.gemini_config import GeminiConfig
from troopai.adk.llms.gemini.gemini_model import GeminiLLM
from troopai.adk.llms.litellm.litellm_model import LiteLLM, LiteLLMConfig
from troopai.adk.llms.llm import LLM
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.llms.openai.openai_chatcompletions_config import OpenAIChatCompletionsConfig
from troopai.adk.llms.openai.openai_chatcompletions_model import OpenAIChatCompletionsLLM
from troopai.adk.llms.openai.openai_responses_config import OpenAIResponsesConfig
from troopai.adk.llms.openai.openai_responses_model import OpenAIResponsesLLM
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


class TestBuiltinProviders:
    def test_anthropic_builds_llm_and_config(self) -> None:
        block = AnthropicProviderBlock.model_validate(
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-5",
                "api_key": "k",
                "config": {
                    "temperature": 0.5,
                    "auto_cache_control": True,
                    "thinking": {"type": "enabled", "budget_tokens": 2048},
                },
            }
        )
        llm, config = PROVIDER_REGISTRY["anthropic"](block)
        assert isinstance(llm, AnthropicLLM)
        assert llm.model == "claude-sonnet-4-5"
        assert isinstance(config, AnthropicConfig)
        assert config.temperature == 0.5
        assert config.auto_cache_control is True
        assert config.thinking == {"type": "enabled", "budget_tokens": 2048}

    def test_openai_responses_builds(self) -> None:
        block = OpenAIResponsesProviderBlock.model_validate(
            {"provider": "openai-responses", "model": "gpt-4o", "config": {"truncation": "auto", "max_tool_calls": 2}}
        )
        llm, config = PROVIDER_REGISTRY["openai-responses"](block)
        assert isinstance(llm, OpenAIResponsesLLM)
        assert isinstance(config, OpenAIResponsesConfig)
        assert config.truncation == "auto"
        assert config.max_tool_calls == 2

    def test_openai_chat_builds(self) -> None:
        block = OpenAIChatProviderBlock.model_validate(
            {"provider": "openai-chat", "model": "gpt-4o", "config": {"verbosity": "low"}}
        )
        llm, config = PROVIDER_REGISTRY["openai-chat"](block)
        assert isinstance(llm, OpenAIChatCompletionsLLM)
        assert isinstance(config, OpenAIChatCompletionsConfig)
        assert config.verbosity == "low"

    def test_gemini_builds(self) -> None:
        block = GeminiProviderBlock.model_validate(
            {
                "provider": "gemini",
                "model": "gemini-2.5-pro",
                "vertexai": True,
                "config": {"response_modalities": ["TEXT"]},
            }
        )
        llm, config = PROVIDER_REGISTRY["gemini"](block)
        assert isinstance(llm, GeminiLLM)
        assert isinstance(config, GeminiConfig)
        assert config.response_modalities == ["TEXT"]

    def test_litellm_builds(self) -> None:
        block = LiteLLMProviderBlock.model_validate(
            {"provider": "litellm", "model": "gpt-4o", "config": {"reasoning_effort": "high"}}
        )
        llm, config = PROVIDER_REGISTRY["litellm"](block)
        assert isinstance(llm, LiteLLM)
        assert isinstance(config, LiteLLMConfig)
        assert config.reasoning_effort == "high"

    def test_config_absent_returns_none(self) -> None:
        block = AnthropicProviderBlock.model_validate({"provider": "anthropic", "model": "claude-sonnet-4-5"})
        llm, config = PROVIDER_REGISTRY["anthropic"](block)
        assert isinstance(llm, AnthropicLLM)
        assert config is None


class TestAgnosticRoundTrip:
    def test_retry_policy_list_to_frozenset(self) -> None:
        block = AnthropicProviderBlock.model_validate(
            {
                "provider": "anthropic",
                "model": "m",
                "config": {"retry_policy": {"max_retries": 5, "retry_on": ["rate_limit", "timeout"]}},
            }
        )
        _, config = PROVIDER_REGISTRY["anthropic"](block)
        assert isinstance(config, AnthropicConfig)
        assert isinstance(config.retry_policy, LLMRetryPolicy)
        assert config.retry_policy.max_retries == 5
        assert config.retry_policy.retry_on == frozenset({"rate_limit", "timeout"})

    def test_retry_policy_omitted_retry_on_keeps_rate_limit_default(self) -> None:
        """Omitting ``retry_on`` in the config must NOT broaden the retry
        scope to every transient kind — the runtime dataclass default
        (rate-limit only) must apply, matching ``LLMRetryPolicy(max_retries=5)``
        built in Python. Passing ``retry_on=None`` would silently retry
        server_error and timeout too, a cost the developer never opted into.
        """
        block = AnthropicProviderBlock.model_validate(
            {
                "provider": "anthropic",
                "model": "m",
                "config": {"retry_policy": {"max_retries": 5}},
            }
        )
        _, config = PROVIDER_REGISTRY["anthropic"](block)
        assert isinstance(config, AnthropicConfig)
        assert isinstance(config.retry_policy, LLMRetryPolicy)
        assert config.retry_policy.max_retries == 5
        assert config.retry_policy.retry_on == frozenset({"rate_limit"})
        assert config.retry_policy.should_retry("server_error", attempt=0) is False
        assert config.retry_policy.should_retry("timeout", attempt=0) is False
        assert config.retry_policy.should_retry("rate_limit", attempt=0) is True

    def test_timeout_float_preserved(self) -> None:
        block = AnthropicProviderBlock.model_validate(
            {"provider": "anthropic", "model": "m", "config": {"timeout": 12.5}}
        )
        _, config = PROVIDER_REGISTRY["anthropic"](block)
        assert isinstance(config, AnthropicConfig)
        assert config.timeout == 12.5

    def test_build_agnostic_config_none(self) -> None:
        assert build_agnostic_config(None) is None

    def test_build_agnostic_config_base_llmconfig(self) -> None:
        config = build_agnostic_config(LLMConfigBlock.model_validate({"temperature": 0.3, "max_output_tokens": 500}))
        assert isinstance(config, LLMConfig)
        assert config.temperature == 0.3
        assert config.max_output_tokens == 500


class TestRegistry:
    def test_register_overrides_dispatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = LiteLLM(model="sentinel")

        def _fake(block: LLMProviderConfig) -> tuple[LLM, LLMConfig | None]:
            return sentinel, None

        monkeypatch.setitem(PROVIDER_REGISTRY, "litellm", _fake)
        block = LiteLLMProviderBlock.model_validate({"provider": "litellm", "model": "ignored"})
        llm, config = PROVIDER_REGISTRY["litellm"](block)
        assert llm is sentinel
        assert config is None

    def test_register_llm_provider_sets_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(PROVIDER_REGISTRY, "litellm", PROVIDER_REGISTRY["litellm"])

        def _fake(block: LLMProviderConfig) -> tuple[LLM, LLMConfig | None]:
            return LiteLLM(model="x"), None

        register_llm_provider("litellm", _fake)
        assert PROVIDER_REGISTRY["litellm"] is _fake


class TestExports:
    def test_register_llm_provider_exported_from_config(self) -> None:
        from troopai.adk.config import register_llm_provider as exported

        assert exported is register_llm_provider

    def test_provider_models_exported_from_types_config(self) -> None:
        from troopai.adk.types import config as types_config

        assert types_config.LLMConfigBlock is LLMConfigBlock
        assert types_config.LLMProviderConfig is LLMProviderConfig


class TestProviderSpecificFields:
    def test_gemini_free_map_fields_thread_through(self) -> None:
        block = GeminiProviderBlock.model_validate(
            {
                "provider": "gemini",
                "model": "gemini-2.5-pro",
                "config": {
                    "thinking_config": {"thinking_budget": 2048},
                    "safety_settings": [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}],
                },
            }
        )
        _, config = PROVIDER_REGISTRY["gemini"](block)
        assert isinstance(config, GeminiConfig)
        assert config.thinking_config == {"thinking_budget": 2048}
        assert config.safety_settings == [{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}]

    def test_openai_responses_reasoning_free_map(self) -> None:
        block = OpenAIResponsesProviderBlock.model_validate(
            {"provider": "openai-responses", "model": "gpt-4o", "config": {"reasoning": {"effort": "high"}}}
        )
        _, config = PROVIDER_REGISTRY["openai-responses"](block)
        assert isinstance(config, OpenAIResponsesConfig)
        assert config.reasoning == {"effort": "high"}

    def test_openai_chat_audio_free_map(self) -> None:
        block = OpenAIChatProviderBlock.model_validate(
            {"provider": "openai-chat", "model": "gpt-4o", "config": {"audio": {"voice": "alloy", "format": "wav"}}}
        )
        _, config = PROVIDER_REGISTRY["openai-chat"](block)
        assert isinstance(config, OpenAIChatCompletionsConfig)
        assert config.audio == {"voice": "alloy", "format": "wav"}

    def test_litellm_connection_params_thread_through(self) -> None:
        block = LiteLLMProviderBlock.model_validate(
            {
                "provider": "litellm",
                "model": "gpt-4o",
                "api_key": "k",
                "base_url": "https://x",
                "extra_params": {"top_k": 50},
            }
        )
        llm, _ = PROVIDER_REGISTRY["litellm"](block)
        assert isinstance(llm, LiteLLM)
        assert llm.api_key == "k"
        assert llm.base_url == "https://x"
        assert llm.extra_params == {"top_k": 50}

    def test_litellm_auto_cache_control_threads_through(self) -> None:
        # The declarative opt-in reaches the runtime config that the LLM path reads.
        block = LiteLLMProviderBlock.model_validate(
            {"provider": "litellm", "model": "claude-sonnet-4-5", "config": {"auto_cache_control": True}}
        )
        _, config = PROVIDER_REGISTRY["litellm"](block)
        assert isinstance(config, LiteLLMConfig)
        assert config.auto_cache_control is True

    def test_litellm_auto_cache_control_absent_stays_off(self) -> None:
        # No opt-in in the block → no cache-write premium at runtime.
        block = LiteLLMProviderBlock.model_validate(
            {"provider": "litellm", "model": "claude-sonnet-4-5", "config": {"reasoning_effort": "low"}}
        )
        _, config = PROVIDER_REGISTRY["litellm"](block)
        assert isinstance(config, LiteLLMConfig)
        assert config.auto_cache_control is None


class TestNoImplicitMaxRetries:
    """Config assembler must NOT inject max_retries=2 when the developer
    omitted it from the provider block.

    The developer's intent when omitting ``max_retries`` is to let the
    underlying SDK decide. The assembler must not silently override that
    with a hard-coded value, which would impose retry costs the developer
    did not opt into.
    """

    def test_anthropic_omitted_max_retries_uses_constructor_default(self) -> None:
        """No max_retries in block → LLM uses its own constructor default."""
        block = AnthropicProviderBlock.model_validate({"provider": "anthropic", "model": "claude-sonnet-4-5"})
        assert block.max_retries is None, "block.max_retries must be None when omitted"
        llm, _ = PROVIDER_REGISTRY["anthropic"](block)
        assert isinstance(llm, AnthropicLLM)
        # The constructor default is 0 (no hidden SDK-level retries) —
        # that default is owned by AnthropicLLM, not the assembler.
        assert llm._max_retries == 0

    def test_anthropic_explicit_max_retries_forwarded(self) -> None:
        """Explicit max_retries is forwarded as-is."""
        block = AnthropicProviderBlock.model_validate({"provider": "anthropic", "model": "m", "max_retries": 5})
        llm, _ = PROVIDER_REGISTRY["anthropic"](block)
        assert isinstance(llm, AnthropicLLM)
        assert llm._max_retries == 5

    def test_anthropic_zero_max_retries_forwarded(self) -> None:
        """max_retries=0 (opt-out of retries) is forwarded without coercion."""
        block = AnthropicProviderBlock.model_validate({"provider": "anthropic", "model": "m", "max_retries": 0})
        llm, _ = PROVIDER_REGISTRY["anthropic"](block)
        assert isinstance(llm, AnthropicLLM)
        assert llm._max_retries == 0

    def test_openai_responses_omitted_max_retries_uses_constructor_default(self) -> None:
        block = OpenAIResponsesProviderBlock.model_validate({"provider": "openai-responses", "model": "gpt-4o"})
        assert block.max_retries is None
        llm, _ = PROVIDER_REGISTRY["openai-responses"](block)
        assert isinstance(llm, OpenAIResponsesLLM)
        assert llm._max_retries == 0  # constructor default, not assembler-injected

    def test_openai_chat_explicit_max_retries_forwarded(self) -> None:
        block = OpenAIChatProviderBlock.model_validate({"provider": "openai-chat", "model": "gpt-4o", "max_retries": 3})
        llm, _ = PROVIDER_REGISTRY["openai-chat"](block)
        assert isinstance(llm, OpenAIChatCompletionsLLM)
        assert llm._max_retries == 3

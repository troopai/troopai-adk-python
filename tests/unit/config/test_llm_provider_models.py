"""Tests for the declarative LLM provider-block schema models.

Pins the agnostic config block, the per-provider config sub-models, the
provider-discriminated union, and the one-source validator. These exercise
the Pydantic models directly (the loader wraps ValidationError into
ConfigParseError, covered elsewhere).
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from troopai.adk.types.config import AgentConfig
from troopai.adk.types.config.llm_config import (
    AnthropicConfigBlock,
    AnthropicProviderBlock,
    GeminiConfigBlock,
    GeminiProviderBlock,
    LiteLLMConfigBlock,
    LLMConfigBlock,
    LLMProviderConfig,
    LLMRetryPolicyBlock,
    OpenAIChatConfigBlock,
    OpenAIResponsesConfigBlock,
    OpenAIResponsesProviderBlock,
)
from troopai.adk.types.tools import ToolExecutionMode

_PROVIDER_ADAPTER: TypeAdapter[object] = TypeAdapter(LLMProviderConfig)


class TestLLMConfigBlock:
    def test_minimal_scalars(self) -> None:
        block = LLMConfigBlock.model_validate({"temperature": 0.7, "max_output_tokens": 2000})
        assert block.temperature == 0.7
        assert block.max_output_tokens == 2000

    def test_all_unset_is_valid(self) -> None:
        block = LLMConfigBlock.model_validate({})
        assert block.temperature is None
        assert block.retry_policy is None

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfigBlock.model_validate({"temprature": 0.7})

    def test_tool_execution_mode_enum(self) -> None:
        block = LLMConfigBlock.model_validate({"tool_execution_mode": "parallel"})
        assert block.tool_execution_mode is ToolExecutionMode.PARALLEL

    def test_nested_retry_policy(self) -> None:
        block = LLMConfigBlock.model_validate(
            {"retry_policy": {"max_retries": 5, "retry_on": ["rate_limit", "timeout"]}}
        )
        assert isinstance(block.retry_policy, LLMRetryPolicyBlock)
        assert block.retry_policy.max_retries == 5
        assert block.retry_policy.retry_on == ["rate_limit", "timeout"]

    def test_retry_policy_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfigBlock.model_validate({"retry_policy": {"retry_on": ["nonsense"]}})

    def test_timeout_is_float_only(self) -> None:
        block = LLMConfigBlock.model_validate({"timeout": 30.0})
        assert block.timeout == 30.0


class TestProviderConfigBlocks:
    def test_anthropic_typed_scalars_and_free_map(self) -> None:
        block = AnthropicConfigBlock.model_validate(
            {
                "temperature": 0.5,
                "auto_cache_control": True,
                "cache_control_ttl": "1h",
                "service_tier": "auto",
                "thinking": {"type": "enabled", "budget_tokens": 2048},
            }
        )
        assert block.temperature == 0.5
        assert block.auto_cache_control is True
        assert block.cache_control_ttl == "1h"
        assert block.thinking == {"type": "enabled", "budget_tokens": 2048}

    def test_anthropic_bad_cache_ttl_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnthropicConfigBlock.model_validate({"cache_control_ttl": "10m"})

    def test_openai_responses_fields(self) -> None:
        block = OpenAIResponsesConfigBlock.model_validate(
            {
                "truncation": "auto",
                "max_tool_calls": 3,
                "reasoning": {"effort": "high"},
                "include": ["reasoning.encrypted_content"],
            }
        )
        assert block.truncation == "auto"
        assert block.max_tool_calls == 3
        assert block.reasoning == {"effort": "high"}
        assert block.include == ["reasoning.encrypted_content"]

    def test_openai_chat_fields(self) -> None:
        block = OpenAIChatConfigBlock.model_validate(
            {"verbosity": "low", "modalities": ["text"], "web_search_options": {"search_context_size": "high"}}
        )
        assert block.verbosity == "low"
        assert block.modalities == ["text"]
        assert block.web_search_options == {"search_context_size": "high"}

    def test_gemini_fields(self) -> None:
        block = GeminiConfigBlock.model_validate(
            {
                "cached_content_name": "cachedContents/x",
                "response_modalities": ["TEXT"],
                "safety_settings": [{"category": "X", "threshold": "Y"}],
            }
        )
        assert block.cached_content_name == "cachedContents/x"
        assert block.response_modalities == ["TEXT"]
        assert block.safety_settings == [{"category": "X", "threshold": "Y"}]

    def test_litellm_fields(self) -> None:
        block = LiteLLMConfigBlock.model_validate(
            {
                "reasoning_effort": "high",
                "cached_content": "abc",
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "auto_cache_control": True,
            }
        )
        assert block.reasoning_effort == "high"
        assert block.cached_content == "abc"
        assert block.auto_cache_control is True

    def test_litellm_auto_cache_control_defaults_off(self) -> None:
        # Cost-conservative default: the caller opts INTO the cache-write premium.
        block = LiteLLMConfigBlock.model_validate({"reasoning_effort": "low"})
        assert block.auto_cache_control is None

    def test_inherits_agnostic_fields(self) -> None:
        block = AnthropicConfigBlock.model_validate({"max_output_tokens": 1000})
        assert block.max_output_tokens == 1000


class TestProviderBlockUnion:
    def test_anthropic_discriminated(self) -> None:
        block = _PROVIDER_ADAPTER.validate_python(
            {"provider": "anthropic", "model": "claude-sonnet-4-5", "api_key": "k", "config": {"temperature": 0.5}}
        )
        assert isinstance(block, AnthropicProviderBlock)
        assert block.model == "claude-sonnet-4-5"
        assert block.api_key == "k"
        assert block.config is not None
        assert block.config.temperature == 0.5

    def test_openai_responses_connection_params(self) -> None:
        block = _PROVIDER_ADAPTER.validate_python(
            {"provider": "openai-responses", "model": "gpt-4o", "organization": "org", "project": "proj"}
        )
        assert isinstance(block, OpenAIResponsesProviderBlock)
        assert block.organization == "org"
        assert block.project == "proj"

    def test_gemini_vertex_params(self) -> None:
        block = _PROVIDER_ADAPTER.validate_python(
            {"provider": "gemini", "model": "gemini-2.5-pro", "vertexai": True, "project": "p", "location": "us"}
        )
        assert isinstance(block, GeminiProviderBlock)
        assert block.vertexai is True
        assert block.location == "us"

    def test_unknown_provider_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _PROVIDER_ADAPTER.validate_python({"provider": "mystery", "model": "x"})

    def test_missing_provider_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _PROVIDER_ADAPTER.validate_python({"model": "x"})

    def test_unknown_connection_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _PROVIDER_ADAPTER.validate_python({"provider": "anthropic", "model": "x", "organization": "nope"})


class TestAgentConfigLLM:
    def test_string_llm_still_valid(self) -> None:
        config = AgentConfig.model_validate({"name": "a", "system_prompt": "p", "llm": "gpt-4o"})
        assert config.llm == "gpt-4o"

    def test_string_llm_with_llm_config(self) -> None:
        config = AgentConfig.model_validate(
            {"name": "a", "system_prompt": "p", "llm": "gpt-4o", "llm_config": {"temperature": 0.5}}
        )
        assert config.llm == "gpt-4o"
        assert config.llm_config is not None
        assert config.llm_config.temperature == 0.5

    def test_provider_block_llm(self) -> None:
        config = AgentConfig.model_validate(
            {"name": "a", "system_prompt": "p", "llm": {"provider": "anthropic", "model": "claude-sonnet-4-5"}}
        )
        assert isinstance(config.llm, AnthropicProviderBlock)

    def test_both_sources_rejected(self) -> None:
        with pytest.raises(ValidationError, match="one source"):
            AgentConfig.model_validate(
                {
                    "name": "a",
                    "system_prompt": "p",
                    "llm": {"provider": "anthropic", "model": "m", "config": {"temperature": 0.5}},
                    "llm_config": {"temperature": 0.7},
                }
            )

    def test_llm_config_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig.model_validate({"name": "a", "system_prompt": "p", "llm_config": {"temprature": 0.5}})

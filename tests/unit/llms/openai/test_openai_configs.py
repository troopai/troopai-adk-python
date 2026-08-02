"""Tests for OpenAI config dataclasses.

Verifies that :class:`OpenAIResponsesConfig` and
:class:`OpenAIChatCompletionsConfig`:

- Default every field to ``None`` (unset).
- Inherit all provider-agnostic :class:`LLMConfig` fields unchanged.
- Round-trip ``openai.types.*`` field values verbatim (no translation).
- Merge correctly via :meth:`LLMConfig.resolve` — override wins per field,
  ``extra_args`` deep-merges.
"""

from __future__ import annotations

from openai.types.chat import ChatCompletionAudioParam, ChatCompletionPredictionContentParam
from openai.types.chat.completion_create_params import WebSearchOptions
from openai.types.shared_params import Reasoning

from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.llms.openai import OpenAIChatCompletionsConfig, OpenAIResponsesConfig


class TestOpenAIResponsesConfigDefaults:
    def test_all_added_fields_default_to_none(self) -> None:
        cfg = OpenAIResponsesConfig()
        assert cfg.reasoning is None
        assert cfg.include is None
        assert cfg.store is None
        assert cfg.previous_response_id is None
        assert cfg.truncation is None
        assert cfg.service_tier is None
        assert cfg.prompt_cache_key is None
        assert cfg.prompt_cache_retention is None
        assert cfg.max_tool_calls is None
        assert cfg.background is None

    def test_inherits_llm_config_fields(self) -> None:
        """Generic fields from the base MUST still default to None."""
        cfg = OpenAIResponsesConfig(temperature=0.7, max_output_tokens=2048)
        assert cfg.temperature == 0.7
        assert cfg.max_output_tokens == 2048
        # Fields not explicitly set stay None.
        assert cfg.top_p is None
        assert cfg.extra_body is None

    def test_is_llm_config_subclass(self) -> None:
        assert issubclass(OpenAIResponsesConfig, LLMConfig)


class TestOpenAIResponsesConfigRoundTrip:
    def test_reasoning_field_accepts_sdk_typeddict(self) -> None:
        """The field MUST hold a verbatim ``openai.types.shared_params.Reasoning``."""
        reasoning: Reasoning = {"effort": "high", "summary": "auto"}
        cfg = OpenAIResponsesConfig(reasoning=reasoning)
        assert cfg.reasoning is reasoning
        # The TypedDict is literally a dict at runtime — no conversion.
        assert cfg.reasoning == {"effort": "high", "summary": "auto"}

    def test_include_field_accepts_literal_list(self) -> None:
        """Values come from ``ResponseIncludable`` — a ``Literal[...]`` union."""
        cfg = OpenAIResponsesConfig(
            include=["reasoning.encrypted_content", "web_search_call.results"],
        )
        assert cfg.include == ["reasoning.encrypted_content", "web_search_call.results"]

    def test_previous_response_id_threading(self) -> None:
        cfg = OpenAIResponsesConfig(previous_response_id="resp_abc", store=True)
        assert cfg.previous_response_id == "resp_abc"
        assert cfg.store is True


class TestOpenAIResponsesConfigResolve:
    def test_override_wins_per_field(self) -> None:
        base = OpenAIResponsesConfig(temperature=0.7, max_tool_calls=10)
        override = OpenAIResponsesConfig(temperature=0.2, background=True)
        merged = base.resolve(override)
        assert isinstance(merged, OpenAIResponsesConfig)
        assert merged.temperature == 0.2
        assert merged.max_tool_calls == 10
        assert merged.background is True

    def test_resolve_keeps_subclass_type(self) -> None:
        """``resolve()`` on a subclass MUST NOT collapse to the base class."""
        base = OpenAIResponsesConfig(service_tier="priority")
        merged = base.resolve(None)
        assert isinstance(merged, OpenAIResponsesConfig)
        assert merged.service_tier == "priority"

    def test_extra_args_deep_merges(self) -> None:
        """Base-class ``resolve`` deep-merges ``extra_args``; subclass inherits that."""
        base = OpenAIResponsesConfig(extra_args={"key_a": 1})
        override = OpenAIResponsesConfig(extra_args={"key_b": 2})
        merged = base.resolve(override)
        assert merged.extra_args == {"key_a": 1, "key_b": 2}


class TestOpenAIChatCompletionsConfigDefaults:
    def test_all_added_fields_default_to_none(self) -> None:
        cfg = OpenAIChatCompletionsConfig()
        assert cfg.audio is None
        assert cfg.web_search_options is None
        assert cfg.prediction is None
        assert cfg.modalities is None
        assert cfg.store is None
        assert cfg.service_tier is None
        assert cfg.prompt_cache_key is None
        assert cfg.prompt_cache_retention is None
        assert cfg.verbosity is None

    def test_inherits_llm_config_fields(self) -> None:
        cfg = OpenAIChatCompletionsConfig(temperature=0.5, max_output_tokens=1024)
        assert cfg.temperature == 0.5
        assert cfg.max_output_tokens == 1024
        assert cfg.stop_sequences is None

    def test_is_llm_config_subclass(self) -> None:
        assert issubclass(OpenAIChatCompletionsConfig, LLMConfig)


class TestOpenAIChatCompletionsConfigRoundTrip:
    def test_audio_field_accepts_sdk_typeddict(self) -> None:
        audio: ChatCompletionAudioParam = {"format": "mp3", "voice": "alloy"}
        cfg = OpenAIChatCompletionsConfig(audio=audio)
        assert cfg.audio is audio

    def test_web_search_options_accepts_sdk_typeddict(self) -> None:
        opts: WebSearchOptions = {"search_context_size": "medium"}
        cfg = OpenAIChatCompletionsConfig(web_search_options=opts)
        assert cfg.web_search_options == {"search_context_size": "medium"}

    def test_prediction_accepts_sdk_typeddict(self) -> None:
        pred: ChatCompletionPredictionContentParam = {
            "type": "content",
            "content": "def foo(): return 42",
        }
        cfg = OpenAIChatCompletionsConfig(prediction=pred)
        assert cfg.prediction is pred

    def test_modalities_text_and_audio(self) -> None:
        cfg = OpenAIChatCompletionsConfig(modalities=["text", "audio"])
        assert cfg.modalities == ["text", "audio"]


class TestOpenAIChatCompletionsConfigResolve:
    def test_override_wins_per_field(self) -> None:
        base = OpenAIChatCompletionsConfig(temperature=0.7, verbosity="low")
        override = OpenAIChatCompletionsConfig(verbosity="high", store=True)
        merged = base.resolve(override)
        assert isinstance(merged, OpenAIChatCompletionsConfig)
        assert merged.temperature == 0.7
        assert merged.verbosity == "high"
        assert merged.store is True

    def test_resolve_keeps_subclass_type(self) -> None:
        base = OpenAIChatCompletionsConfig(service_tier="flex")
        merged = base.resolve(None)
        assert isinstance(merged, OpenAIChatCompletionsConfig)
        assert merged.service_tier == "flex"


class TestConfigsAreDistinct:
    """Responses and Chat Completions expose disjoint fields beyond the base."""

    def test_responses_specific_fields_absent_from_chat(self) -> None:
        chat_cfg = OpenAIChatCompletionsConfig()
        # ``reasoning`` / ``include`` / ``previous_response_id`` are Responses-API only.
        assert not hasattr(chat_cfg, "reasoning")
        assert not hasattr(chat_cfg, "include")
        assert not hasattr(chat_cfg, "previous_response_id")
        assert not hasattr(chat_cfg, "max_tool_calls")
        assert not hasattr(chat_cfg, "background")

    def test_chat_specific_fields_absent_from_responses(self) -> None:
        resp_cfg = OpenAIResponsesConfig()
        # ``audio`` / ``web_search_options`` / ``prediction`` / ``verbosity`` /
        # ``modalities`` are Chat-Completions-specific in our split.
        assert not hasattr(resp_cfg, "audio")
        assert not hasattr(resp_cfg, "web_search_options")
        assert not hasattr(resp_cfg, "prediction")
        assert not hasattr(resp_cfg, "modalities")
        assert not hasattr(resp_cfg, "verbosity")

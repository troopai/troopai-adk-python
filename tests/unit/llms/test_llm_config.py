"""Tests for LLMConfig, focusing on num_retries and fallbacks fields."""

from troopai.adk.llms.litellm.litellm_model import LiteLLMConfig
from troopai.adk.llms.llm_config import LLMConfig


class TestLLMConfigRetryFields:
    """Test num_retries and fallbacks fields on LLMConfig."""

    def test_defaults_are_none(self):
        """Both num_retries and fallbacks default to None."""
        config = LLMConfig()
        assert config.num_retries is None
        assert config.fallbacks is None

    def test_num_retries_set(self):
        """num_retries can be set to a positive integer."""
        config = LLMConfig(num_retries=3)
        assert config.num_retries == 3

    def test_num_retries_zero(self):
        """num_retries=0 means no retries."""
        config = LLMConfig(num_retries=0)
        assert config.num_retries == 0

    def test_fallbacks_set(self):
        """fallbacks accepts a list of model name strings."""
        config = LLMConfig(fallbacks=["gpt-4o-mini", "claude-sonnet-4-20250514"])
        assert config.fallbacks == ["gpt-4o-mini", "claude-sonnet-4-20250514"]

    def test_resolve_override_num_retries(self):
        """resolve() merges num_retries from override."""
        base = LLMConfig(num_retries=2, temperature=0.7)
        override = LLMConfig(num_retries=5)
        merged = base.resolve(override)
        assert merged.num_retries == 5
        assert merged.temperature == 0.7

    def test_resolve_override_fallbacks(self):
        """resolve() merges fallbacks from override."""
        base = LLMConfig(fallbacks=["gpt-4o"])
        override = LLMConfig(fallbacks=["claude-sonnet-4-20250514"])
        merged = base.resolve(override)
        assert merged.fallbacks == ["claude-sonnet-4-20250514"]

    def test_resolve_keeps_base_when_override_is_none(self):
        """resolve() keeps base values when override fields are None."""
        base = LLMConfig(num_retries=3, fallbacks=["gpt-4o"])
        override = LLMConfig(temperature=0.5)
        merged = base.resolve(override)
        assert merged.num_retries == 3
        assert merged.fallbacks == ["gpt-4o"]
        assert merged.temperature == 0.5

    def test_to_json_dict_includes_retry_fields(self):
        """to_json_dict() includes num_retries and fallbacks when set."""
        config = LLMConfig(num_retries=2, fallbacks=["gpt-4o-mini"])
        d = config.to_json_dict()
        assert d["num_retries"] == 2
        assert d["fallbacks"] == ["gpt-4o-mini"]

    def test_to_json_dict_excludes_none_retry_fields(self):
        """to_json_dict() excludes num_retries and fallbacks when None."""
        config = LLMConfig(temperature=0.5)
        d = config.to_json_dict()
        assert "num_retries" not in d
        assert "fallbacks" not in d

    def test_combined_with_other_fields(self):
        """num_retries and fallbacks work alongside all other fields."""
        config = LLMConfig(
            temperature=0.7,
            max_output_tokens=2000,
            num_retries=3,
            fallbacks=["gpt-4o-mini"],
            timeout=30.0,
        )
        assert config.temperature == 0.7
        assert config.max_output_tokens == 2000
        assert config.num_retries == 3
        assert config.fallbacks == ["gpt-4o-mini"]
        assert config.timeout == 30.0


class TestResolveAcrossSubclassAndBase:
    """resolve() must not crash or drop fields when subclass/base configs mix."""

    def test_subclass_self_base_override_does_not_crash(self):
        """A subclass ``self`` resolving a base override keeps subclass fields.

        Regression: ``getattr(override, subclass_only_field)`` raised
        AttributeError because the base override lacks the subclass's fields.
        """
        sub = LiteLLMConfig(reasoning_effort="high", temperature=0.1)
        base_override = LLMConfig(temperature=0.5)

        merged = sub.resolve(base_override)

        assert isinstance(merged, LiteLLMConfig)
        assert merged.temperature == 0.5  # override wins
        assert merged.reasoning_effort == "high"  # subclass-only field preserved

    def test_base_self_subclass_override_keeps_subclass_fields(self):
        """A base ``self`` resolving a subclass override keeps the extra fields.

        Regression: iterating only the base's fields silently dropped the
        subclass override's provider-specific fields and returned a base config.
        """
        base = LLMConfig(temperature=0.7)
        sub_override = LiteLLMConfig(reasoning_effort="low", top_p=0.9)

        merged = base.resolve(sub_override)

        assert isinstance(merged, LiteLLMConfig)
        assert merged.temperature == 0.7  # base value preserved
        assert merged.top_p == 0.9  # override value applied
        assert merged.reasoning_effort == "low"  # subclass-only override survives

    def test_both_subclass_merges_provider_fields(self):
        """Both subclass: override wins per field, base fills the rest."""
        base = LiteLLMConfig(reasoning_effort="high", cached_content="cc1")
        override = LiteLLMConfig(reasoning_effort="low")

        merged = base.resolve(override)

        assert isinstance(merged, LiteLLMConfig)
        assert merged.reasoning_effort == "low"  # override wins
        assert merged.cached_content == "cc1"  # base retained

    def test_subclass_base_extra_args_still_deep_merged(self):
        """extra_args deep-merge still works across a subclass/base mix."""
        sub = LiteLLMConfig(extra_args={"a": 1})
        base_override = LLMConfig(extra_args={"b": 2})

        merged = sub.resolve(base_override)

        assert merged.extra_args == {"a": 1, "b": 2}

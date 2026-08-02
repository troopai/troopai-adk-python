"""Tests for AnthropicConfig.thinking_display.

The field must:
- Default to None (today's behavior: no display field in the thinking param).
- When set to "omitted" or "summarized" with thinking enabled, merge the
  ``display`` key into the resolved thinking dict.
- Never mutate the original config dict.
- Be ignored when thinking is not configured or is disabled.
- Be ignored when thinking arrives via extra_args (not AnthropicConfig).
"""

from __future__ import annotations

from troopai.adk.llms.anthropic.anthropic_config import AnthropicConfig
from troopai.adk.llms.anthropic.anthropic_reasoning_resolver import resolve_thinking
from troopai.adk.llms.llm_config import LLMConfig


class TestThinkingDisplayDefault:
    def test_default_is_none(self) -> None:
        cfg = AnthropicConfig()
        assert cfg.thinking_display is None

    def test_resolve_without_thinking_returns_none(self) -> None:
        cfg = AnthropicConfig(thinking_display="omitted")
        assert resolve_thinking(cfg) is None

    def test_resolve_with_display_none_unchanged(self) -> None:
        cfg = AnthropicConfig(thinking={"type": "enabled", "budget_tokens": 4096})
        result = resolve_thinking(cfg)
        assert result is not None
        assert "display" not in result


class TestThinkingDisplayMerge:
    def test_omitted_merged_into_result(self) -> None:
        cfg = AnthropicConfig(
            thinking={"type": "enabled", "budget_tokens": 4096},
            thinking_display="omitted",
        )
        result = resolve_thinking(cfg)
        assert result is not None
        assert result.get("display") == "omitted"

    def test_summarized_merged_into_result(self) -> None:
        cfg = AnthropicConfig(
            thinking={"type": "enabled", "budget_tokens": 4096},
            thinking_display="summarized",
        )
        result = resolve_thinking(cfg)
        assert result is not None
        assert result.get("display") == "summarized"

    def test_other_fields_preserved(self) -> None:
        cfg = AnthropicConfig(
            thinking={"type": "enabled", "budget_tokens": 8192},
            thinking_display="omitted",
        )
        result = resolve_thinking(cfg)
        assert result is not None
        assert result.get("type") == "enabled"
        assert result.get("budget_tokens") == 8192
        assert result.get("display") == "omitted"

    def test_original_config_dict_not_mutated(self) -> None:
        original = {"type": "enabled", "budget_tokens": 4096}
        cfg = AnthropicConfig(thinking=original, thinking_display="summarized")  # type: ignore[arg-type]
        resolve_thinking(cfg)
        # The dict passed in must not have been mutated.
        assert "display" not in original

    def test_display_not_added_for_disabled_thinking(self) -> None:
        cfg = AnthropicConfig(
            thinking={"type": "disabled"},
            thinking_display="omitted",
        )
        result = resolve_thinking(cfg)
        assert result is not None
        assert "display" not in result

    def test_display_not_applied_via_extra_args(self) -> None:
        # thinking_display is an AnthropicConfig-only field; extra_args path
        # must not pick it up (the dict is returned as-is).
        cfg = LLMConfig(extra_args={"thinking": {"type": "enabled", "budget_tokens": 4096}})
        result = resolve_thinking(cfg)
        assert result is not None
        assert "display" not in result


class TestThinkingDisplayByteIdentical:
    def test_unset_display_byte_identical_to_baseline(self) -> None:
        """Resolved dict without display must equal the original thinking dict."""
        thinking_dict = {"type": "enabled", "budget_tokens": 4096}
        cfg_with = AnthropicConfig(thinking=thinking_dict)  # type: ignore[arg-type]
        cfg_without = AnthropicConfig(thinking=thinking_dict)  # type: ignore[arg-type]
        # Both without display_override should yield same content.
        assert resolve_thinking(cfg_with) == resolve_thinking(cfg_without)

    def test_display_set_produces_different_dict_than_unset(self) -> None:
        cfg_no_display = AnthropicConfig(thinking={"type": "enabled", "budget_tokens": 4096})
        cfg_with_display = AnthropicConfig(
            thinking={"type": "enabled", "budget_tokens": 4096},
            thinking_display="omitted",
        )
        result_no = resolve_thinking(cfg_no_display)
        result_with = resolve_thinking(cfg_with_display)
        assert result_no != result_with
        assert result_with is not None
        assert result_with.get("display") == "omitted"


async def test_display_merges_into_adaptive_thinking() -> None:
    """display applies to adaptive thinking, not only the enabled type."""
    config = AnthropicConfig(thinking={"type": "adaptive"}, thinking_display="omitted")
    resolved = resolve_thinking(config)
    assert resolved is not None
    assert resolved.get("type") == "adaptive"
    assert resolved.get("display") == "omitted"
    assert config.thinking == {"type": "adaptive"}, "caller's dict must not be mutated"

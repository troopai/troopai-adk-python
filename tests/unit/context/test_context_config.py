"""Tests for context config validation (Field constraints).

Regression: CompactionConfig.trigger_tokens, preserve_recent_items, and
ContextManagementConfig.token_budget_warning_threshold previously accepted
out-of-range values with no validation error.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from troopai.adk.context.context_config import (
    CompactionConfig,
    ContextEditingConfig,
    ContextManagementConfig,
    TokenUsage,
)

# ── CompactionConfig field constraints ────────────────────────────────


class TestCompactionConfigValidation:
    def test_trigger_tokens_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            CompactionConfig(trigger_tokens=0)

    def test_trigger_tokens_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CompactionConfig(trigger_tokens=-1)

    def test_trigger_tokens_one_is_valid(self) -> None:
        cfg = CompactionConfig(trigger_tokens=1)
        assert cfg.trigger_tokens == 1

    def test_preserve_recent_items_zero_is_valid(self) -> None:
        cfg = CompactionConfig(preserve_recent_items=0)
        assert cfg.preserve_recent_items == 0

    def test_preserve_recent_items_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CompactionConfig(preserve_recent_items=-1)


# ── ContextEditingConfig field constraints ────────────────────────────


class TestContextEditingConfigValidation:
    def test_tool_result_trigger_tokens_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextEditingConfig(tool_result_trigger_tokens=0)

    def test_tool_result_trigger_tokens_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextEditingConfig(tool_result_trigger_tokens=-5)

    def test_thinking_turns_to_keep_zero_valid(self) -> None:
        cfg = ContextEditingConfig(thinking_turns_to_keep=0)
        assert cfg.thinking_turns_to_keep == 0

    def test_thinking_turns_to_keep_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextEditingConfig(thinking_turns_to_keep=-1)


# ── ContextManagementConfig.token_budget_warning_threshold ───────────


class TestContextManagementConfigValidation:
    def test_threshold_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextManagementConfig(token_budget_warning_threshold=-0.1)

    def test_threshold_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextManagementConfig(token_budget_warning_threshold=1.01)

    def test_threshold_zero_valid(self) -> None:
        cfg = ContextManagementConfig(token_budget_warning_threshold=0.0)
        assert cfg.token_budget_warning_threshold == 0.0

    def test_threshold_one_valid(self) -> None:
        cfg = ContextManagementConfig(token_budget_warning_threshold=1.0)
        assert cfg.token_budget_warning_threshold == 1.0

    def test_threshold_default_valid(self) -> None:
        cfg = ContextManagementConfig()
        assert cfg.token_budget_warning_threshold == 0.8


# ── TokenUsage TypedDict ──────────────────────────────────────────────


class TestTokenUsage:
    def test_token_usage_has_required_keys(self) -> None:
        usage: TokenUsage = {
            "used": 1000,
            "max": 5000,
            "remaining": 4000,
            "utilisation": 0.2,
            "compaction_count": 0,
        }
        assert usage["used"] == 1000
        assert usage["remaining"] == 4000

    def test_token_usage_annotations_match_expected(self) -> None:
        expected = {"used", "max", "remaining", "utilisation", "compaction_count"}
        assert set(TokenUsage.__annotations__.keys()) == expected

"""Tests for the Anthropic thinking / reasoning resolver."""

from __future__ import annotations

import logging

import pytest

from troopai.adk.llms.anthropic.anthropic_config import AnthropicConfig
from troopai.adk.llms.anthropic.anthropic_reasoning_resolver import (
    ANTHROPIC_MIN_THINKING_BUDGET,
    resolve_thinking,
)
from troopai.adk.llms.llm_config import LLMConfig


class TestResolveThinking:
    def test_returns_none_when_unset(self) -> None:
        assert resolve_thinking(LLMConfig()) is None

    def test_reads_anthropic_config_thinking_field(self) -> None:
        cfg = AnthropicConfig(thinking={"type": "enabled", "budget_tokens": 4096})
        result = resolve_thinking(cfg)
        assert result == {"type": "enabled", "budget_tokens": 4096}

    def test_reads_extra_args_when_no_typed_field(self) -> None:
        cfg = LLMConfig(extra_args={"thinking": {"type": "enabled", "budget_tokens": 2048}})
        result = resolve_thinking(cfg)
        assert result == {"type": "enabled", "budget_tokens": 2048}

    def test_anthropic_config_thinking_beats_extra_args(self) -> None:
        cfg = AnthropicConfig(
            thinking={"type": "enabled", "budget_tokens": 4096},
            extra_args={"thinking": {"type": "enabled", "budget_tokens": 1024}},
        )
        result = resolve_thinking(cfg)
        assert result == {"type": "enabled", "budget_tokens": 4096}

    def test_warns_below_minimum_budget(self, caplog: pytest.LogCaptureFixture) -> None:
        cfg = AnthropicConfig(thinking={"type": "enabled", "budget_tokens": 100})
        with caplog.at_level(logging.WARNING):
            resolve_thinking(cfg)
        assert any("budget_tokens" in rec.getMessage() for rec in caplog.records)
        assert any(str(ANTHROPIC_MIN_THINKING_BUDGET) in rec.getMessage() for rec in caplog.records)

    def test_disabled_thinking_passes_through(self) -> None:
        cfg = AnthropicConfig(thinking={"type": "disabled"})
        assert resolve_thinking(cfg) == {"type": "disabled"}

    def test_non_dict_extra_args_thinking_ignored(self) -> None:
        cfg = LLMConfig(extra_args={"thinking": "enabled"})
        assert resolve_thinking(cfg) is None

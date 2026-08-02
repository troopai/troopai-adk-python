"""Tests for the Gemini thinking / reasoning resolver."""

from __future__ import annotations

import logging

import pytest
from google.genai.types import ThinkingConfig

from troopai.adk.llms.gemini.gemini_config import GeminiConfig
from troopai.adk.llms.gemini.gemini_reasoning_resolver import (
    GEMINI_MIN_THINKING_BUDGET,
    resolve_thinking,
)
from troopai.adk.llms.llm_config import LLMConfig


class TestResolveThinking:
    def test_returns_none_when_unset(self) -> None:
        assert resolve_thinking(LLMConfig()) is None

    def test_reads_gemini_config_thinking(self) -> None:
        cfg = GeminiConfig(
            thinking_config=ThinkingConfig(thinking_budget=4096, include_thoughts=True),
        )
        result = resolve_thinking(cfg)
        assert result is not None
        assert result.thinking_budget == 4096
        assert result.include_thoughts is True

    def test_reads_extra_args_when_no_typed_field(self) -> None:
        cfg = LLMConfig(
            extra_args={
                "thinking_config": ThinkingConfig(thinking_budget=2048, include_thoughts=False),
            }
        )
        result = resolve_thinking(cfg)
        assert result is not None
        assert result.thinking_budget == 2048

    def test_typed_field_beats_extra_args(self) -> None:
        cfg = GeminiConfig(
            thinking_config=ThinkingConfig(thinking_budget=8192),
            extra_args={"thinking_config": ThinkingConfig(thinking_budget=1024)},
        )
        result = resolve_thinking(cfg)
        assert result is not None
        assert result.thinking_budget == 8192

    def test_warns_below_floor(self, caplog: pytest.LogCaptureFixture) -> None:
        cfg = GeminiConfig(thinking_config=ThinkingConfig(thinking_budget=512))
        with caplog.at_level(logging.WARNING):
            resolve_thinking(cfg)
        assert any("thinking_budget" in r.getMessage() for r in caplog.records)
        assert any(str(GEMINI_MIN_THINKING_BUDGET) in r.getMessage() for r in caplog.records)

    def test_zero_budget_disables_thinking_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        # 0 disables thinking; not below the floor.
        cfg = GeminiConfig(thinking_config=ThinkingConfig(thinking_budget=0))
        with caplog.at_level(logging.WARNING):
            resolve_thinking(cfg)
        assert not any("thinking_budget" in r.getMessage() for r in caplog.records)

    def test_dynamic_budget_no_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        # -1 enables dynamic; not below the floor.
        cfg = GeminiConfig(thinking_config=ThinkingConfig(thinking_budget=-1))
        with caplog.at_level(logging.WARNING):
            resolve_thinking(cfg)
        assert not any("thinking_budget" in r.getMessage() for r in caplog.records)

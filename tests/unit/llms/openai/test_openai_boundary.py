"""Tests for :mod:`troopai.adk.llms.openai.openai_boundary`.

Focus: ``sanitize_for_log`` must neutralize every Unicode line terminator so a
model name echoed back by the API cannot forge log lines (CWE-117). Parity with
the anthropic/gemini boundaries — stripping only ``\\n``/``\\r`` would still let
NEL / LS / PS split a single-line log record.
"""

from __future__ import annotations

import pytest

from troopai.adk.llms.openai.openai_boundary import sanitize_for_log


class TestSanitizeForLog:
    @pytest.mark.parametrize(
        ("name", "terminator"),
        [
            ("LF", "\n"),
            ("CR", "\r"),
            ("NEL", "\x85"),
            ("LS", " "),
            ("PS", " "),
        ],
    )
    def test_each_line_terminator_is_replaced(self, name: str, terminator: str) -> None:
        forged = f"gpt-4o{terminator}ERROR fake-injected-line"
        cleaned = sanitize_for_log(forged)
        assert terminator not in cleaned, f"{name} ({terminator!r}) survived sanitization"
        # The break is replaced by a space, never dropped (no token concatenation).
        assert cleaned == "gpt-4o ERROR fake-injected-line"

    def test_plain_model_name_unchanged(self) -> None:
        # Routed model names legitimately contain "/" and ":" — those must pass through.
        assert sanitize_for_log("openrouter/anthropic/claude-3:beta") == ("openrouter/anthropic/claude-3:beta")

    def test_multiple_terminators_all_replaced(self) -> None:
        forged = "a\nb\rc\x85d e f"
        assert sanitize_for_log(forged) == "a b c d e f"

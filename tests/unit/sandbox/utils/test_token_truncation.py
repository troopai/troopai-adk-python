"""Tests for ``troopai.adk.sandbox.utils.token_truncation``."""

from __future__ import annotations

import pytest

from troopai.adk.sandbox.utils.token_truncation import (
    APPROX_BYTES_PER_TOKEN,
    TruncationPolicy,
    approx_bytes_for_tokens,
    approx_token_count,
    approx_tokens_from_byte_count,
    format_truncation_marker,
    formatted_truncate_text,
    formatted_truncate_text_with_token_count,
    split_budget,
    split_string,
    truncate_text,
    truncate_with_byte_estimate,
    truncate_with_token_budget,
)


class TestTruncationPolicy:
    def test_bytes_factory_clamps_negative(self) -> None:
        p = TruncationPolicy.bytes(-50)
        assert p.limit == 0
        assert p.mode == "bytes"

    def test_tokens_factory_clamps_negative(self) -> None:
        p = TruncationPolicy.tokens(-1)
        assert p.limit == 0
        assert p.mode == "tokens"

    def test_token_budget_conversion(self) -> None:
        assert TruncationPolicy.bytes(40).token_budget() == 10
        assert TruncationPolicy.tokens(7).token_budget() == 7

    def test_byte_budget_conversion(self) -> None:
        assert TruncationPolicy.tokens(10).byte_budget() == 40
        assert TruncationPolicy.bytes(100).byte_budget() == 100


class TestApproximations:
    def test_approx_token_count_round_up(self) -> None:
        assert approx_token_count("") == 0
        assert approx_token_count("a") == 1  # 1 byte → 1 token
        assert approx_token_count("a" * 4) == 1
        assert approx_token_count("a" * 5) == 2

    def test_approx_bytes_for_tokens(self) -> None:
        assert approx_bytes_for_tokens(0) == 0
        assert approx_bytes_for_tokens(3) == 12
        assert approx_bytes_for_tokens(-5) == 0

    def test_approx_tokens_from_byte_count(self) -> None:
        assert approx_tokens_from_byte_count(0) == 0
        assert approx_tokens_from_byte_count(-1) == 0
        assert approx_tokens_from_byte_count(7) == 2

    def test_approx_bytes_per_token_constant(self) -> None:
        assert APPROX_BYTES_PER_TOKEN == 4


class TestTruncation:
    def test_no_truncation_when_within_budget(self) -> None:
        assert truncate_text("hello", TruncationPolicy.bytes(100)) == "hello"
        assert truncate_text("hello", TruncationPolicy.tokens(100)) == "hello"

    def test_byte_mode_truncates(self) -> None:
        out = truncate_with_byte_estimate("a" * 50, TruncationPolicy.bytes(10))
        assert "truncated" in out
        assert len(out.encode("utf-8")) < 50

    def test_zero_byte_budget_returns_marker(self) -> None:
        out = truncate_with_byte_estimate("hello", TruncationPolicy.bytes(0))
        assert "truncated" in out
        assert "hello" not in out

    def test_token_mode_truncates(self) -> None:
        out, original = truncate_with_token_budget("a" * 100, TruncationPolicy.tokens(5))
        assert "truncated" in out
        assert original is not None
        assert original >= 25

    def test_token_mode_empty_input(self) -> None:
        out, original = truncate_with_token_budget("", TruncationPolicy.tokens(10))
        assert out == ""
        assert original is None

    def test_formatted_prepends_total_lines(self) -> None:
        content = "line1\nline2\nline3\n" + "x" * 1000
        out = formatted_truncate_text(content, TruncationPolicy.bytes(20))
        assert out.startswith("Total output lines: 4")

    def test_formatted_with_token_count_none_passthrough(self) -> None:
        out, original = formatted_truncate_text_with_token_count("hello", None)
        assert out == "hello"
        assert original is None

    def test_formatted_with_token_count_truncates(self) -> None:
        out, original = formatted_truncate_text_with_token_count("a" * 200, 5)
        assert "truncated" in out
        assert original is not None


class TestHelpers:
    def test_split_budget_even(self) -> None:
        assert split_budget(10) == (5, 5)

    def test_split_budget_odd(self) -> None:
        assert split_budget(11) == (5, 6)

    def test_split_string_basic(self) -> None:
        removed, left, right = split_string("abcdefghij", 3, 3)
        assert left == "abc"
        assert right == "hij"
        assert removed == 4

    def test_split_string_empty(self) -> None:
        removed, left, right = split_string("", 5, 5)
        assert removed == 0
        assert left == ""
        assert right == ""

    @pytest.mark.parametrize(
        "mode, count, expected",
        [
            ("tokens", 5, "…5 tokens truncated…"),
            ("bytes", 7, "…7 chars truncated…"),
        ],
    )
    def test_format_truncation_marker(self, mode: str, count: int, expected: str) -> None:
        policy = TruncationPolicy.tokens(10) if mode == "tokens" else TruncationPolicy.bytes(10)
        assert format_truncation_marker(policy, count) == expected

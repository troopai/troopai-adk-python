"""Tests for ``troopai.adk.sandbox.apply_diff``."""

from __future__ import annotations

import pytest

from troopai.adk.sandbox.apply_diff import apply_diff


class TestCreateMode:
    def test_basic_create(self) -> None:
        diff = "+line one\n+line two"
        out = apply_diff("", diff, mode="create")
        assert out == "line one\nline two"

    def test_empty_create(self) -> None:
        out = apply_diff("", "", mode="create")
        assert out == ""

    def test_create_rejects_non_plus_lines(self) -> None:
        with pytest.raises(ValueError, match="Invalid Add File Line"):
            apply_diff("", " line one", mode="create")

    def test_create_preserves_crlf(self) -> None:
        diff = "+a\r\n+b"
        out = apply_diff("", diff, mode="create")
        # CRLF in the diff → CRLF in output (no input to override).
        assert out == "a\r\nb"


class TestUpdateMode:
    def test_simple_replace(self) -> None:
        input_text = "alpha\nbeta\ngamma"
        diff = "@@\n alpha\n-beta\n+BETA\n gamma"
        out = apply_diff(input_text, diff)
        assert out == "alpha\nBETA\ngamma"

    def test_pure_insert(self) -> None:
        input_text = "a\nc"
        diff = "@@\n a\n+b\n c"
        out = apply_diff(input_text, diff)
        assert out == "a\nb\nc"

    def test_pure_delete(self) -> None:
        input_text = "a\nb\nc"
        diff = "@@\n a\n-b\n c"
        out = apply_diff(input_text, diff)
        assert out == "a\nc"

    def test_invalid_context_raises(self) -> None:
        input_text = "alpha\nbeta"
        diff = "@@\n NOPE\n-x\n+y"
        with pytest.raises(ValueError, match="Invalid Context"):
            apply_diff(input_text, diff)

    def test_overlapping_chunk_raises(self) -> None:
        # Two anchored hunks at the same line position → overlapping cursor.
        input_text = "a\nb\nc"
        diff = "@@\n a\n-b\n+B\n c\n@@\n a\n-b\n+B\n c"
        with pytest.raises(ValueError, match="Invalid Context|overlapping"):
            apply_diff(input_text, diff)


class TestFuzzyMatching:
    def test_rstrip_fuzz(self) -> None:
        # Trailing space in input shouldn't break the match.
        input_text = "alpha  \nbeta\ngamma"
        diff = "@@\n alpha\n-beta\n+BETA\n gamma"
        out = apply_diff(input_text, diff)
        assert "BETA" in out

    def test_strip_fuzz(self) -> None:
        # Leading whitespace difference: input has indented content; the
        # diff's context lines use the engine's required space-prefix
        # encoding but the inner text omits the indent — the strip-fuzz
        # tier (penalty 100) handles this.
        input_text = "    alpha\n    beta\n    gamma"
        diff = "@@\n alpha\n- beta\n+ BETA\n gamma"
        out = apply_diff(input_text, diff)
        assert "BETA" in out


class TestNewlineHandling:
    def test_lf_input_lf_output(self) -> None:
        input_text = "a\nb\nc"
        diff = "@@\n a\n-b\n+B\n c"
        out = apply_diff(input_text, diff)
        assert "\r\n" not in out

    def test_crlf_input_crlf_output(self) -> None:
        input_text = "a\r\nb\r\nc"
        diff = "@@\n a\n-b\n+B\n c"
        out = apply_diff(input_text, diff)
        assert "\r\n" in out

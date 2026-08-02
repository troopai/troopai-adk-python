"""Unit tests for the litellm capability lookups (no network — local tables)."""

from __future__ import annotations

from troopai.adk.llms.litellm.litellm_provider import max_output_tokens


class TestMaxOutputTokens:
    def test_known_model_returns_positive_cap(self) -> None:
        # gpt-4o is in litellm's model-info table; its output cap is a positive int.
        cap = max_output_tokens("gpt-4o")
        assert isinstance(cap, int)
        assert cap > 0

    def test_unmapped_model_returns_none(self) -> None:
        # An unmapped model makes litellm raise; the helper degrades to None so the
        # caller falls back to its configured budget instead of crashing.
        assert max_output_tokens("this-model-does-not-exist-xyz") is None

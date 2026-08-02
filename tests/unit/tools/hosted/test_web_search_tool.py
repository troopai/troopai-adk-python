"""Regression tests for WebSearchTool construction-time validation."""

from __future__ import annotations

import pytest

from troopai.adk.tools.hosted import WebSearchTool


class TestWebSearchToolDomainXor:
    """The documented XOR on allowed_domains / blocked_domains is enforced."""

    def test_allowed_domains_only_is_valid(self) -> None:
        tool = WebSearchTool(allowed_domains=["arxiv.org"])
        assert tool.allowed_domains == ["arxiv.org"]
        assert tool.blocked_domains is None

    def test_blocked_domains_only_is_valid(self) -> None:
        tool = WebSearchTool(blocked_domains=["spam.example"])
        assert tool.blocked_domains == ["spam.example"]
        assert tool.allowed_domains is None

    def test_neither_is_valid(self) -> None:
        tool = WebSearchTool()
        assert tool.allowed_domains is None
        assert tool.blocked_domains is None

    def test_both_raises(self) -> None:
        with pytest.raises(ValueError, match="both were set"):
            WebSearchTool(allowed_domains=["a.com"], blocked_domains=["b.com"])

    def test_both_empty_lists_still_raises(self) -> None:
        # Empty lists are still "set" (not None) — the XOR is on presence.
        with pytest.raises(ValueError, match="both were set"):
            WebSearchTool(allowed_domains=[], blocked_domains=[])

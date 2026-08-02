"""Anthropic-converter hosted-tool dispatch tests."""

from __future__ import annotations

from typing import Any, cast

import pytest

from troopai.adk.llms.anthropic.anthropic_converter import AnthropicConverter
from troopai.adk.tools.hosted import (
    CodeExecutionTool,
    FileSearchTool,
    ImageGenerationTool,
    UnsupportedHostedToolError,
    URLContextTool,
    WebSearchTool,
)


class TestAnthropicWebSearch:
    def test_minimal(self) -> None:
        tools = AnthropicConverter.convert_tools([WebSearchTool()])
        assert len(tools) == 1
        param = cast("dict[str, Any]", tools[0])
        assert param["type"] == "web_search_20250305"
        assert param["name"] == "web_search"

    def test_max_uses_propagated(self) -> None:
        tools = AnthropicConverter.convert_tools([WebSearchTool(max_uses=7)])
        param = cast("dict[str, Any]", tools[0])
        assert param["max_uses"] == 7

    def test_allowed_domains_propagated(self) -> None:
        tools = AnthropicConverter.convert_tools([WebSearchTool(allowed_domains=["a.com", "b.org"])])
        param = cast("dict[str, Any]", tools[0])
        assert param["allowed_domains"] == ["a.com", "b.org"]

    def test_blocked_domains_propagated(self) -> None:
        tools = AnthropicConverter.convert_tools([WebSearchTool(blocked_domains=["spam.example"])])
        param = cast("dict[str, Any]", tools[0])
        assert param["blocked_domains"] == ["spam.example"]

    def test_user_location_propagated(self) -> None:
        tools = AnthropicConverter.convert_tools(
            [WebSearchTool(user_location={"type": "approximate", "city": "Paris", "country": "FR"})]
        )
        param = cast("dict[str, Any]", tools[0])
        assert param["user_location"]["city"] == "Paris"
        assert param["user_location"]["country"] == "FR"
        assert param["user_location"]["type"] == "approximate"

    def test_openai_only_attrs_silently_dropped(self) -> None:
        # search_context_size is OpenAI-only — Anthropic converter ignores
        # it without error (just a debug log).
        tools = AnthropicConverter.convert_tools([WebSearchTool(search_context_size="high")])
        param = cast("dict[str, Any]", tools[0])
        assert "search_context_size" not in param


class TestAnthropicUnsupportedHostedTools:
    @pytest.mark.parametrize(
        "tool",
        [
            CodeExecutionTool(),
            FileSearchTool(vector_store_ids=["vs_1"]),
            ImageGenerationTool(),
            URLContextTool(),
        ],
    )
    def test_unsupported_raises(self, tool: object) -> None:
        with pytest.raises(UnsupportedHostedToolError) as info:
            AnthropicConverter.convert_tools([tool])  # type: ignore[list-item]
        # Error message includes the tool class name and provider.
        assert type(tool).__name__ in str(info.value)
        assert "anthropic" in str(info.value)

    def test_error_lists_supported_providers(self) -> None:
        with pytest.raises(UnsupportedHostedToolError) as info:
            AnthropicConverter.convert_tools([URLContextTool()])
        # URLContextTool is gemini-only.
        assert "gemini" in str(info.value)

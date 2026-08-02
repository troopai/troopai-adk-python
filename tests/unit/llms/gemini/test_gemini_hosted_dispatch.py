"""Gemini-converter hosted-tool dispatch tests."""

from __future__ import annotations

import pytest

from troopai.adk.llms.gemini.gemini_converter import GeminiConverter
from troopai.adk.tools.hosted import (
    CodeExecutionTool,
    FileSearchTool,
    ImageGenerationTool,
    UnsupportedHostedToolError,
    URLContextTool,
    WebSearchTool,
)


class TestGeminiWebSearch:
    def test_emits_google_search(self) -> None:
        tools = GeminiConverter.convert_tools([WebSearchTool()])
        assert tools is not None
        assert tools[0].google_search is not None

    def test_silently_drops_provider_specific_attrs(self) -> None:
        # max_uses (Anthropic), search_context_size (OpenAI) — Gemini
        # ignores these but doesn't raise.
        tools = GeminiConverter.convert_tools(
            [
                WebSearchTool(
                    max_uses=5,
                    search_context_size="high",
                    allowed_domains=["a.com"],
                )
            ]
        )
        assert tools is not None
        assert tools[0].google_search is not None


class TestGeminiCodeExecution:
    def test_emits_code_execution(self) -> None:
        tools = GeminiConverter.convert_tools([CodeExecutionTool()])
        assert tools is not None
        assert tools[0].code_execution is not None

    def test_silently_drops_container(self) -> None:
        # container is OpenAI-only.
        tools = GeminiConverter.convert_tools([CodeExecutionTool(container="cntr_x")])
        assert tools is not None
        assert tools[0].code_execution is not None


class TestGeminiURLContext:
    def test_emits_url_context(self) -> None:
        tools = GeminiConverter.convert_tools([URLContextTool()])
        assert tools is not None
        assert tools[0].url_context is not None


class TestGeminiUnsupported:
    @pytest.mark.parametrize(
        "tool",
        [FileSearchTool(vector_store_ids=["vs_1"]), ImageGenerationTool()],
    )
    def test_unsupported_raises(self, tool: object) -> None:
        with pytest.raises(UnsupportedHostedToolError) as info:
            GeminiConverter.convert_tools([tool])  # type: ignore[list-item]
        assert "gemini" in str(info.value)
        assert "openai-responses" in str(info.value)


class TestGeminiCombinedTools:
    def test_function_and_hosted_in_same_request(self) -> None:
        from troopai.adk.tools import function_tool

        @function_tool
        def lookup(x: int) -> int:
            """Square."""
            return x * x

        tools = GeminiConverter.convert_tools([lookup, WebSearchTool(), URLContextTool()])
        assert tools is not None
        # All collapsed into a single Tool instance.
        assert len(tools) == 1
        t = tools[0]
        assert t.function_declarations is not None
        assert len(t.function_declarations) == 1
        assert t.google_search is not None
        assert t.url_context is not None

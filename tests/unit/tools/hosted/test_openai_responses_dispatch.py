"""OpenAI Responses converter hosted-tool dispatch tests."""

from __future__ import annotations

from typing import Any, cast

import pytest

from troopai.adk.llms.openai.openai_responses_converter import OpenAIResponsesConverter
from troopai.adk.tools.hosted import (
    CodeExecutionTool,
    FileSearchTool,
    ImageGenerationTool,
    UnsupportedHostedToolError,
    URLContextTool,
    WebSearchTool,
)


class TestOpenAIResponsesWebSearch:
    def test_minimal(self) -> None:
        tools = OpenAIResponsesConverter.convert_tools([WebSearchTool()])
        param = cast("dict[str, Any]", tools[0])
        assert param["type"] == "web_search"

    def test_search_context_size_propagated(self) -> None:
        tools = OpenAIResponsesConverter.convert_tools([WebSearchTool(search_context_size="medium")])
        param = cast("dict[str, Any]", tools[0])
        assert param["search_context_size"] == "medium"

    def test_user_location_propagated(self) -> None:
        tools = OpenAIResponsesConverter.convert_tools(
            [WebSearchTool(user_location={"type": "approximate", "city": "Paris"})]
        )
        param = cast("dict[str, Any]", tools[0])
        assert param["user_location"]["city"] == "Paris"

    def test_anthropic_only_attrs_silently_dropped(self) -> None:
        # max_uses, allowed_domains, blocked_domains are Anthropic-only.
        tools = OpenAIResponsesConverter.convert_tools([WebSearchTool(max_uses=5, allowed_domains=["a.com"])])
        param = cast("dict[str, Any]", tools[0])
        assert "max_uses" not in param
        assert "allowed_domains" not in param


class TestOpenAIResponsesCodeExecution:
    def test_minimal(self) -> None:
        tools = OpenAIResponsesConverter.convert_tools([CodeExecutionTool()])
        param = cast("dict[str, Any]", tools[0])
        assert param["type"] == "code_interpreter"
        assert param["container"] == "auto"

    def test_container_propagated(self) -> None:
        tools = OpenAIResponsesConverter.convert_tools([CodeExecutionTool(container="cntr_abc")])
        param = cast("dict[str, Any]", tools[0])
        assert param["container"] == "cntr_abc"


class TestOpenAIResponsesFileSearch:
    def test_minimal(self) -> None:
        tools = OpenAIResponsesConverter.convert_tools([FileSearchTool(vector_store_ids=["vs_1"])])
        param = cast("dict[str, Any]", tools[0])
        assert param["type"] == "file_search"
        assert param["vector_store_ids"] == ["vs_1"]

    def test_max_results(self) -> None:
        tools = OpenAIResponsesConverter.convert_tools([FileSearchTool(vector_store_ids=["vs_1"], max_num_results=20)])
        param = cast("dict[str, Any]", tools[0])
        assert param["max_num_results"] == 20


class TestOpenAIResponsesImageGeneration:
    def test_minimal(self) -> None:
        tools = OpenAIResponsesConverter.convert_tools([ImageGenerationTool()])
        param = cast("dict[str, Any]", tools[0])
        assert param["type"] == "image_generation"

    def test_full(self) -> None:
        tools = OpenAIResponsesConverter.convert_tools(
            [
                ImageGenerationTool(
                    quality="high",
                    size="1024x1024",
                    output_format="png",
                )
            ]
        )
        param = cast("dict[str, Any]", tools[0])
        assert param["quality"] == "high"
        assert param["size"] == "1024x1024"
        assert param["output_format"] == "png"


class TestOpenAIResponsesUnsupported:
    def test_url_context_raises(self) -> None:
        # URLContextTool is Gemini-only; OpenAI Responses raises.
        with pytest.raises(UnsupportedHostedToolError) as info:
            OpenAIResponsesConverter.convert_tools([URLContextTool()])
        assert "openai-responses" in str(info.value)
        assert "gemini" in str(info.value)

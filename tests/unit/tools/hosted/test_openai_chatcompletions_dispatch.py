"""OpenAI Chat Completions hosted-tool dispatch tests.

Chat Completions does not support hosted tools via the ``tools=``
array — it uses dedicated config fields like ``web_search_options``
instead. So every hosted tool variant should raise.
"""

from __future__ import annotations

import pytest

from troopai.adk.llms.openai.openai_chatcompletions_converter import (
    OpenAIChatCompletionsConverter,
)
from troopai.adk.tools.hosted import (
    CodeExecutionTool,
    FileSearchTool,
    ImageGenerationTool,
    UnsupportedHostedToolError,
    URLContextTool,
    WebSearchTool,
)


class TestChatCompletionsHostedToolsAllRaise:
    @pytest.mark.parametrize(
        "tool",
        [
            WebSearchTool(),
            CodeExecutionTool(),
            FileSearchTool(vector_store_ids=["vs_1"]),
            ImageGenerationTool(),
            URLContextTool(),
        ],
    )
    def test_each_hosted_tool_raises(self, tool: object) -> None:
        with pytest.raises(UnsupportedHostedToolError) as info:
            OpenAIChatCompletionsConverter.convert_tools([tool])  # type: ignore[list-item]
        assert "openai-chatcompletions" in str(info.value)

    def test_error_message_lists_alternative_providers(self) -> None:
        with pytest.raises(UnsupportedHostedToolError) as info:
            OpenAIChatCompletionsConverter.convert_tools([WebSearchTool()])
        message = str(info.value)
        # WebSearchTool supports anthropic + openai-responses + gemini.
        assert "anthropic" in message
        assert "openai-responses" in message
        assert "gemini" in message

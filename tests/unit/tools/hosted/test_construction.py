"""Construction tests for hosted-tool dataclasses."""

from __future__ import annotations

from troopai.adk.tools.hosted import (
    CodeExecutionTool,
    FileSearchTool,
    ImageGenerationTool,
    URLContextTool,
    WebSearchTool,
)


class TestWebSearchToolConstruction:
    def test_default(self) -> None:
        tool = WebSearchTool()
        assert tool.max_uses is None
        assert tool.allowed_domains is None
        assert tool.blocked_domains is None
        assert tool.user_location is None
        assert tool.search_context_size is None

    def test_anthropic_attrs(self) -> None:
        tool = WebSearchTool(
            max_uses=5,
            allowed_domains=["arxiv.org"],
        )
        assert tool.max_uses == 5
        assert tool.allowed_domains == ["arxiv.org"]
        assert tool.blocked_domains is None

    def test_blocked_domains_attr(self) -> None:
        tool = WebSearchTool(blocked_domains=["spam.example"])
        assert tool.blocked_domains == ["spam.example"]
        assert tool.allowed_domains is None

    def test_openai_attrs(self) -> None:
        tool = WebSearchTool(
            search_context_size="high",
            user_location={"type": "approximate", "city": "Paris"},
        )
        assert tool.search_context_size == "high"
        assert tool.user_location == {"type": "approximate", "city": "Paris"}

    def test_supported_providers(self) -> None:
        assert WebSearchTool.SUPPORTED_PROVIDERS == (
            "anthropic",
            "openai-responses",
            "gemini",
        )


class TestCodeExecutionToolConstruction:
    def test_default(self) -> None:
        tool = CodeExecutionTool()
        assert tool.container is None

    def test_with_container(self) -> None:
        tool = CodeExecutionTool(container="cntr_abc123")
        assert tool.container == "cntr_abc123"

    def test_supported_providers(self) -> None:
        assert CodeExecutionTool.SUPPORTED_PROVIDERS == ("openai-responses", "gemini")


class TestFileSearchToolConstruction:
    def test_default(self) -> None:
        tool = FileSearchTool()
        assert tool.vector_store_ids == []
        assert tool.max_num_results is None
        assert tool.ranking_options is None

    def test_with_stores(self) -> None:
        tool = FileSearchTool(
            vector_store_ids=["vs_1", "vs_2"],
            max_num_results=10,
        )
        assert tool.vector_store_ids == ["vs_1", "vs_2"]
        assert tool.max_num_results == 10

    def test_supported_providers(self) -> None:
        assert FileSearchTool.SUPPORTED_PROVIDERS == ("openai-responses",)


class TestImageGenerationToolConstruction:
    def test_default(self) -> None:
        tool = ImageGenerationTool()
        assert tool.model is None
        assert tool.quality is None
        assert tool.size is None
        assert tool.output_format is None

    def test_full(self) -> None:
        tool = ImageGenerationTool(
            model="gpt-image-1",
            quality="high",
            size="1024x1024",
            output_format="png",
        )
        assert tool.model == "gpt-image-1"
        assert tool.quality == "high"
        assert tool.size == "1024x1024"
        assert tool.output_format == "png"

    def test_supported_providers(self) -> None:
        assert ImageGenerationTool.SUPPORTED_PROVIDERS == ("openai-responses",)


class TestURLContextToolConstruction:
    def test_default(self) -> None:
        tool = URLContextTool()
        # No knobs to assert; just confirm construction succeeds.
        assert isinstance(tool, URLContextTool)

    def test_supported_providers(self) -> None:
        assert URLContextTool.SUPPORTED_PROVIDERS == ("gemini",)


class TestHostedMCPToolConstruction:
    """Finding 7: HostedMCPTool must enforce XOR on server_url / connector_id."""

    def test_server_url_only_is_valid(self) -> None:
        from troopai.adk.tools.hosted.mcp_tool import HostedMCPTool

        tool = HostedMCPTool(server_label="my_server", server_url="https://mcp.example.com")
        assert tool.server_url == "https://mcp.example.com"
        assert tool.connector_id is None

    def test_connector_id_only_is_valid(self) -> None:
        from troopai.adk.tools.hosted.mcp_tool import HostedMCPTool

        tool = HostedMCPTool(server_label="gmail", connector_id="connector_gmail")
        assert tool.connector_id == "connector_gmail"
        assert tool.server_url is None

    def test_neither_raises(self) -> None:
        import pytest

        from troopai.adk.tools.hosted.mcp_tool import HostedMCPTool

        with pytest.raises(ValueError, match="got neither"):
            HostedMCPTool(server_label="bad")

    def test_both_raises(self) -> None:
        import pytest

        from troopai.adk.tools.hosted.mcp_tool import HostedMCPTool

        with pytest.raises(ValueError, match="both were set"):
            HostedMCPTool(
                server_label="bad",
                server_url="https://mcp.example.com",
                connector_id="connector_gmail",
            )

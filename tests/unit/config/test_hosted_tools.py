"""Tests for the hosted-tool config model, registry, and factories."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from troopai.adk.config.hosted_tools import (
    HOSTED_TOOL_REGISTRY,
    build_hosted_tool,
    register_hosted_tool,
)
from troopai.adk.exceptions import ConfigResolutionError
from troopai.adk.tools.hosted.code_execution_tool import CodeExecutionTool
from troopai.adk.tools.hosted.file_search_tool import FileSearchTool
from troopai.adk.tools.hosted.image_generation_tool import ImageGenerationTool
from troopai.adk.tools.hosted.mcp_tool import HostedMCPTool
from troopai.adk.tools.hosted.url_context_tool import URLContextTool
from troopai.adk.tools.hosted.web_search_tool import WebSearchTool
from troopai.adk.types.config.tool_config import HostedToolRef


class TestHostedToolRef:
    def test_minimal(self) -> None:
        ref = HostedToolRef.model_validate({"type": "web_search"})
        assert ref.type == "web_search"
        assert ref.args == {}

    def test_with_args(self) -> None:
        ref = HostedToolRef.model_validate({"type": "web_search", "args": {"max_uses": 5}})
        assert ref.args == {"max_uses": 5}

    def test_unknown_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HostedToolRef.model_validate({"type": "telepathy"})

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HostedToolRef.model_validate({"type": "web_search", "extra": 1})


class TestHostedToolFactories:
    def test_web_search_with_args(self) -> None:
        tool = build_hosted_tool(HostedToolRef.model_validate({"type": "web_search", "args": {"max_uses": 5}}))
        assert isinstance(tool, WebSearchTool)
        assert tool.max_uses == 5

    def test_code_execution(self) -> None:
        tool = build_hosted_tool(HostedToolRef.model_validate({"type": "code_execution"}))
        assert isinstance(tool, CodeExecutionTool)

    def test_file_search(self) -> None:
        tool = build_hosted_tool(
            HostedToolRef.model_validate({"type": "file_search", "args": {"vector_store_ids": ["vs_1"]}})
        )
        assert isinstance(tool, FileSearchTool)
        assert tool.vector_store_ids == ["vs_1"]

    def test_image_generation(self) -> None:
        tool = build_hosted_tool(
            HostedToolRef.model_validate({"type": "image_generation", "args": {"quality": "high"}})
        )
        assert isinstance(tool, ImageGenerationTool)
        assert tool.quality == "high"

    def test_url_context(self) -> None:
        tool = build_hosted_tool(HostedToolRef.model_validate({"type": "url_context"}))
        assert isinstance(tool, URLContextTool)

    def test_hosted_mcp(self) -> None:
        tool = build_hosted_tool(
            HostedToolRef.model_validate(
                {"type": "hosted_mcp", "args": {"server_label": "docs", "server_url": "https://mcp.example.com"}}
            )
        )
        assert isinstance(tool, HostedMCPTool)
        assert tool.server_label == "docs"

    def test_bad_arg_raises_config_error(self) -> None:
        with pytest.raises(ConfigResolutionError, match="web_search"):
            build_hosted_tool(HostedToolRef.model_validate({"type": "web_search", "args": {"nonsense": 1}}))

    def test_hosted_mcp_xor_violation_raises(self) -> None:
        with pytest.raises(ConfigResolutionError):
            build_hosted_tool(HostedToolRef.model_validate({"type": "hosted_mcp", "args": {"server_label": "x"}}))

    def test_register_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = URLContextTool()

        def _fake(args: dict[str, object]) -> URLContextTool:
            return sentinel

        monkeypatch.setitem(HOSTED_TOOL_REGISTRY, "url_context", HOSTED_TOOL_REGISTRY["url_context"])
        register_hosted_tool("url_context", _fake)
        tool = build_hosted_tool(HostedToolRef.model_validate({"type": "url_context"}))
        assert tool is sentinel


class TestExports:
    def test_register_hosted_tool_exported(self) -> None:
        from troopai.adk.config import register_hosted_tool as exported
        from troopai.adk.config.hosted_tools import register_hosted_tool as direct

        assert exported is direct

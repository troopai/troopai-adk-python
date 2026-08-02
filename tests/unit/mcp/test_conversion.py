"""Tests for ``troopai.adk.mcp.conversion``.

Covers:
- ``mcp_tool_to_function_tool`` produces a ``FunctionTool`` with the
  raw ``inputSchema`` (with ``"properties"`` filled when missing) and
  ``SchemaEnforcement.NONE``.
- ``call_tool_result_to_str`` concatenates text parts and raises
  ``MCPToolCallError`` when ``isError=True``.
- non-text MCP content is retained in the artifact channel.
- Argument parsing rejects non-object JSON and surfaces decode errors
  as ``MCPToolCallError``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

from troopai.adk.mcp.conversion import (
    call_tool_result_to_artifact,
    call_tool_result_to_str,
    mcp_tool_to_function_tool,
)
from troopai.adk.mcp.exceptions import MCPToolCallError
from troopai.adk.schemas import SchemaEnforcement


def _make_mcp_tool(name: str = "lookup", schema: dict[str, Any] | None = None) -> Any:
    # Mock(name=...) sets the mock's repr-name, not an attribute named "name".
    # Build the bare mock then assign attrs explicitly so ``mcp_tool.name``
    # returns the string we want, not the auto-generated repr.
    tool = MagicMock()
    tool.name = name
    tool.description = "Look up a record"
    tool.inputSchema = schema if schema is not None else {"type": "object"}
    return tool


def _make_server(name: str = "test") -> Any:
    server = MagicMock()
    server.name = name
    server.call_tool = AsyncMock()
    return server


# ---------------------------------------------------------------- conversion


def test_conversion_uses_strict_none_and_keeps_raw_schema() -> None:
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    mcp_tool = _make_mcp_tool(name="search", schema=schema)
    server = _make_server("svc")

    ft = mcp_tool_to_function_tool(mcp_tool, server)

    assert ft.name == "search"
    assert ft.schema_enforcement == SchemaEnforcement.NONE
    assert ft.schema["type"] == "object"
    assert ft.schema["properties"] == {"q": {"type": "string"}}


def test_conversion_pads_missing_properties_for_parameterless_tool() -> None:
    mcp_tool = _make_mcp_tool(name="ping", schema={"type": "object"})
    server = _make_server("svc")

    ft = mcp_tool_to_function_tool(mcp_tool, server)

    assert ft.schema["properties"] == {}


async def test_on_invoke_passes_parsed_args_to_server_call_tool() -> None:
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    mcp_tool = _make_mcp_tool(name="search", schema=schema)
    server = _make_server("svc")
    server.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text="ok")])

    ft = mcp_tool_to_function_tool(mcp_tool, server)
    out = await ft.on_invoke(MagicMock(), json.dumps({"q": "hi"}))  # type: ignore[misc]

    server.call_tool.assert_awaited_once_with("search", {"q": "hi"})
    assert out == ("ok", None)
    assert ft.response_format == "content_and_artifact"


async def test_on_invoke_handles_empty_args() -> None:
    mcp_tool = _make_mcp_tool(name="ping")
    server = _make_server("svc")
    server.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text="pong")])

    ft = mcp_tool_to_function_tool(mcp_tool, server)
    out = await ft.on_invoke(MagicMock(), "")  # type: ignore[misc]

    server.call_tool.assert_awaited_once_with("ping", None)
    assert out == ("pong", None)


async def test_on_invoke_raises_on_invalid_json_args() -> None:
    mcp_tool = _make_mcp_tool(name="search")
    server = _make_server("svc")

    ft = mcp_tool_to_function_tool(mcp_tool, server)

    with pytest.raises(MCPToolCallError) as exc_info:
        await ft.on_invoke(MagicMock(), "not-json")  # type: ignore[misc]

    assert "invalid JSON arguments" in str(exc_info.value)


async def test_on_invoke_raises_when_args_not_a_json_object() -> None:
    mcp_tool = _make_mcp_tool(name="search")
    server = _make_server("svc")

    ft = mcp_tool_to_function_tool(mcp_tool, server)

    with pytest.raises(MCPToolCallError) as exc_info:
        await ft.on_invoke(MagicMock(), json.dumps([1, 2, 3]))  # type: ignore[misc]

    assert "must be a JSON object" in str(exc_info.value)


# ---------------------------------------------------- call_tool_result_to_str


def test_result_to_str_concatenates_text_parts() -> None:
    result = CallToolResult(
        content=[
            TextContent(type="text", text="hello"),
            TextContent(type="text", text="world"),
        ]
    )
    out = call_tool_result_to_str(result, tool_name="t", server_name="s")
    assert out == "hello\nworld"


def test_result_to_str_raises_on_iserror_true() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="boom")],
        isError=True,
    )
    with pytest.raises(MCPToolCallError) as exc_info:
        call_tool_result_to_str(result, tool_name="t", server_name="s")
    assert exc_info.value.tool_name == "t"
    assert exc_info.value.server == "s"
    assert "boom" in str(exc_info.value)


def test_result_to_artifact_preserves_non_text_parts() -> None:
    result = CallToolResult(
        content=[
            TextContent(type="text", text="text-part"),
            ImageContent(type="image", data="base64data", mimeType="image/png"),
        ]
    )
    out = call_tool_result_to_str(result, tool_name="t", server_name="s")
    artifact = call_tool_result_to_artifact(result)

    assert out == "text-part"
    assert artifact == {"content": [{"type": "image", "data": "base64data", "mimeType": "image/png"}]}


def test_result_to_str_ignores_structured_content_in_text_path() -> None:
    """``call_tool_result_to_str`` returns the text body unchanged when
    ``structuredContent`` is present; the artifact helper includes it
    only when requested by the converted tool.
    """
    result = CallToolResult(
        content=[TextContent(type="text", text="ok")],
        structuredContent={"value": 42},
    )
    out = call_tool_result_to_str(result, tool_name="t", server_name="s")
    assert out == "ok"
    assert call_tool_result_to_artifact(result) is None


def test_result_to_str_uses_isinstance_not_string_comparison() -> None:
    """Regression: call_tool_result_to_str used string comparison
    ('type == text') instead of isinstance for TextContent narrowing.
    A non-TextContent part with .type == 'text' must NOT be treated
    as text content.
    """

    class _FakePart:
        type = "text"
        text = "should be ignored"

    result = CallToolResult.model_construct(content=[_FakePart()])
    # The fake part is not a TextContent instance; it must be skipped
    out = call_tool_result_to_str(result, tool_name="t", server_name="s")
    assert out == ""


async def test_structured_content_artifact_path_returns_tuple() -> None:
    """When ``use_structured_content=True``, the converted tool's
    on_invoke returns ``(text, artifact)`` so the runner stores
    structured content on ``FunctionToolCallResult.artifact``.
    """
    schema = {"type": "object"}
    mcp_tool = _make_mcp_tool(name="lookup", schema=schema)
    server = _make_server("svc")
    server.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="ok")],
        structuredContent={"value": 42},
    )

    ft = mcp_tool_to_function_tool(mcp_tool, server, use_structured_content=True)
    out = await ft.on_invoke(MagicMock(), "{}")  # type: ignore[misc]
    assert out == ("ok", {"structuredContent": {"value": 42}})
    assert ft.response_format == "content_and_artifact"


async def test_non_text_content_artifact_path_returns_tuple_by_default() -> None:
    """Converted MCP tools retain non-text content even when structuredContent is disabled."""
    schema = {"type": "object"}
    mcp_tool = _make_mcp_tool(name="lookup", schema=schema)
    server = _make_server("svc")
    server.call_tool.return_value = CallToolResult(
        content=[
            TextContent(type="text", text="preview"),
            ImageContent(type="image", data="base64data", mimeType="image/png"),
        ],
        structuredContent={"ignored": True},
    )

    ft = mcp_tool_to_function_tool(mcp_tool, server)
    out = await ft.on_invoke(MagicMock(), "{}")  # type: ignore[misc]

    assert out == (
        "preview",
        {"content": [{"type": "image", "data": "base64data", "mimeType": "image/png"}]},
    )

"""Tests for the faithful tool-use sampling turn."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp import types as mcp_types

from troopai.adk.mcp.sampling import make_sampling_callback
from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseFunctionToolCall, LLMResponseText


async def test_tool_call_response_returns_with_tools_result() -> None:
    """When the LLM responds with tool calls, the result must be CreateMessageResultWithTools
    with stopReason='toolUse' and ToolUseContent blocks."""
    fake_llm = MagicMock()
    fake_response = LLMResponse(
        response_id="r1",
        model="fake-model",
        response=[
            LLMResponseFunctionToolCall(
                call_id="call_123",
                name="search",
                arguments='{"query": "test"}',
            )
        ],
        finish_reason="tool_calls",
    )
    fake_llm.acomplete = AsyncMock(return_value=fake_response)
    cb = make_sampling_callback(fake_llm)

    tool = mcp_types.Tool(
        name="search",
        description="Search the web",
        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    params = mcp_types.CreateMessageRequestParams(
        messages=[
            mcp_types.SamplingMessage(
                role="user",
                content=mcp_types.TextContent(type="text", text="find something"),
            )
        ],
        maxTokens=100,
        tools=[tool],
    )
    result = await cb(None, params)
    assert isinstance(result, mcp_types.CreateMessageResultWithTools)
    assert result.stopReason == "toolUse"
    # content_as_list property should return the list
    content_list = result.content_as_list
    assert len(content_list) == 1
    assert isinstance(content_list[0], mcp_types.ToolUseContent)
    assert content_list[0].name == "search"
    assert content_list[0].id == "call_123"
    assert content_list[0].input == {"query": "test"}


async def test_tool_result_in_messages_forwarded_to_llm() -> None:
    """ToolResultContent in incoming messages must be converted to Layer 1 tool-result items."""
    captured_messages: list[Any] = []

    async def fake_acomplete(messages: Any, **kwargs: Any) -> LLMResponse:
        captured_messages.extend(messages)
        return LLMResponse(
            response_id="r2",
            model="fake-model",
            response=[LLMResponseText(text="done")],
            finish_reason="stop",
        )

    fake_llm = MagicMock()
    fake_llm.acomplete = AsyncMock(side_effect=fake_acomplete)
    cb = make_sampling_callback(fake_llm)

    params = mcp_types.CreateMessageRequestParams(
        messages=[
            mcp_types.SamplingMessage(
                role="user",
                content=mcp_types.ToolResultContent(
                    type="tool_result",
                    toolUseId="call_abc",
                    content=[mcp_types.TextContent(type="text", text='{"result": "42"}')],
                ),
            )
        ],
        maxTokens=100,
    )
    result = await cb(None, params)
    assert isinstance(result, mcp_types.CreateMessageResult)
    # Check that the tool result was converted to a FunctionToolCallResultParam
    assert len(captured_messages) >= 1
    tool_result_items = [
        m for m in captured_messages if isinstance(m, dict) and m.get("type") == "function_call_output"
    ]
    assert len(tool_result_items) == 1
    assert tool_result_items[0]["call_id"] == "call_abc"


async def test_tool_use_in_messages_forwarded_to_llm() -> None:
    """ToolUseContent in incoming messages must be converted to Layer 1 tool-call items."""
    captured_messages: list[Any] = []

    async def fake_acomplete(messages: Any, **kwargs: Any) -> LLMResponse:
        captured_messages.extend(messages)
        return LLMResponse(
            response_id="r3",
            model="fake-model",
            response=[LLMResponseText(text="answer")],
            finish_reason="stop",
        )

    fake_llm = MagicMock()
    fake_llm.acomplete = AsyncMock(side_effect=fake_acomplete)
    cb = make_sampling_callback(fake_llm)

    params = mcp_types.CreateMessageRequestParams(
        messages=[
            mcp_types.SamplingMessage(
                role="assistant",
                content=mcp_types.ToolUseContent(
                    type="tool_use",
                    id="call_xyz",
                    name="calculator",
                    input={"expression": "2+2"},
                ),
            )
        ],
        maxTokens=100,
    )
    result = await cb(None, params)
    assert isinstance(result, mcp_types.CreateMessageResult)
    tool_call_items = [m for m in captured_messages if isinstance(m, dict) and m.get("type") == "function_call"]
    assert len(tool_call_items) == 1
    assert tool_call_items[0]["call_id"] == "call_xyz"
    assert tool_call_items[0]["name"] == "calculator"
    assert tool_call_items[0]["arguments"] == json.dumps({"expression": "2+2"})


async def test_finish_reason_tool_use_maps_to_stop_reason_tooluse() -> None:
    """finish_reason='tool_use' (Anthropic style) must map to stopReason='toolUse'."""
    fake_llm = MagicMock()
    fake_response = LLMResponse(
        response_id="r4",
        model="fake-model",
        response=[
            LLMResponseFunctionToolCall(
                call_id="call_1",
                name="tool_a",
                arguments="{}",
            )
        ],
        finish_reason="tool_use",
    )
    fake_llm.acomplete = AsyncMock(return_value=fake_response)
    cb = make_sampling_callback(fake_llm)

    tool = mcp_types.Tool(
        name="tool_a",
        description="A tool",
        inputSchema={"type": "object", "properties": {}},
    )
    params = mcp_types.CreateMessageRequestParams(
        messages=[
            mcp_types.SamplingMessage(
                role="user",
                content=mcp_types.TextContent(type="text", text="go"),
            )
        ],
        maxTokens=50,
        tools=[tool],
    )
    result = await cb(None, params)
    assert isinstance(result, mcp_types.CreateMessageResultWithTools)
    assert result.stopReason == "toolUse"


async def test_text_only_path_unchanged_when_no_tools() -> None:
    """Text-only servers (no tools in params) must get CreateMessageResult, as before."""
    fake_llm = MagicMock()
    fake_response = LLMResponse(
        response_id="r5",
        model="fake-model",
        response=[LLMResponseText(text="hello")],
        finish_reason="stop",
    )
    fake_llm.acomplete = AsyncMock(return_value=fake_response)
    cb = make_sampling_callback(fake_llm)

    params = mcp_types.CreateMessageRequestParams(
        messages=[
            mcp_types.SamplingMessage(
                role="user",
                content=mcp_types.TextContent(type="text", text="hi"),
            )
        ],
        maxTokens=100,
    )
    result = await cb(None, params)
    assert isinstance(result, mcp_types.CreateMessageResult)
    assert result.content.text == "hello"  # type: ignore[union-attr]


async def test_no_warning_logged_when_tools_forwarded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When tools are properly forwarded, no warning about tools being unsupported should appear."""
    fake_llm = MagicMock()
    fake_response = LLMResponse(
        response_id="r6",
        model="fake-model",
        response=[LLMResponseText(text="answer")],
        finish_reason="stop",
    )
    fake_llm.acomplete = AsyncMock(return_value=fake_response)
    cb = make_sampling_callback(fake_llm)

    tool = mcp_types.Tool(
        name="search",
        description="Search",
        inputSchema={"type": "object", "properties": {}},
    )
    params = mcp_types.CreateMessageRequestParams(
        messages=[
            mcp_types.SamplingMessage(
                role="user",
                content=mcp_types.TextContent(type="text", text="find X"),
            )
        ],
        maxTokens=100,
        tools=[tool],
    )
    with caplog.at_level("WARNING"):
        await cb(None, params)
    # No warning about tools being unsupported / not forwarded
    assert not any("cannot be forwarded" in r.message for r in caplog.records)


async def test_tool_intent_without_calls_answers_as_end_turn() -> None:
    """finish_reason=tool_calls with an empty call list must not emit toolUse.

    A text body paired with stopReason="toolUse" contradicts the protocol;
    the degenerate response is answered as a plain endTurn completion.
    """
    fake_llm = MagicMock()
    fake_response = LLMResponse(
        response_id="r-degenerate",
        model="fake-model",
        response=[LLMResponseText(text="partial thought")],
        finish_reason="tool_calls",
    )
    fake_llm.acomplete = AsyncMock(return_value=fake_response)
    cb = make_sampling_callback(fake_llm)

    params = mcp_types.CreateMessageRequestParams(
        messages=[
            mcp_types.SamplingMessage(
                role="user",
                content=mcp_types.TextContent(type="text", text="hi"),
            )
        ],
        maxTokens=100,
    )
    result = await cb(None, params)
    assert isinstance(result, mcp_types.CreateMessageResult)
    assert result.stopReason == "endTurn"
    assert isinstance(result.content, mcp_types.TextContent)
    assert result.content.text == "partial thought"


async def test_tool_calls_detected_by_presence_not_finish_reason() -> None:
    """Gemini reports finish_reason="STOP" even with function calls present.

    Tool calls must be detected by their presence on the response, not by
    gating on finish_reason — otherwise Gemini tool calls are silently dropped
    and answered as plain text.
    """
    fake_llm = MagicMock()
    fake_response = LLMResponse(
        response_id="r-gemini",
        model="gemini-2.5-flash",
        response=[
            LLMResponseFunctionToolCall(
                call_id="call_g",
                name="search",
                arguments='{"q": "x"}',
            )
        ],
        # Gemini-style terminal reason despite emitting a function call.
        finish_reason="STOP",
    )
    fake_llm.acomplete = AsyncMock(return_value=fake_response)
    cb = make_sampling_callback(fake_llm)

    tool = mcp_types.Tool(
        name="search",
        description="Search",
        inputSchema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    params = mcp_types.CreateMessageRequestParams(
        messages=[
            mcp_types.SamplingMessage(
                role="user",
                content=mcp_types.TextContent(type="text", text="find x"),
            )
        ],
        maxTokens=100,
        tools=[tool],
    )
    result = await cb(None, params)
    assert isinstance(result, mcp_types.CreateMessageResultWithTools)
    assert result.stopReason == "toolUse"
    content_list = result.content_as_list
    assert len(content_list) == 1
    assert isinstance(content_list[0], mcp_types.ToolUseContent)
    assert content_list[0].name == "search"


async def test_generation_controls_mapped_to_llm_config() -> None:
    """maxTokens / temperature / stopSequences on the request must reach the
    host LLM as an ``LLMConfig`` — the MCP spec requires them to be honoured."""
    captured: dict[str, Any] = {}

    async def fake_acomplete(messages: Any, **kwargs: Any) -> LLMResponse:
        del messages
        captured.update(kwargs)
        return LLMResponse(
            response_id="r-cfg",
            model="fake-model",
            response=[LLMResponseText(text="ok")],
            finish_reason="stop",
        )

    fake_llm = MagicMock()
    fake_llm.acomplete = AsyncMock(side_effect=fake_acomplete)
    cb = make_sampling_callback(fake_llm)

    params = mcp_types.CreateMessageRequestParams(
        messages=[
            mcp_types.SamplingMessage(
                role="user",
                content=mcp_types.TextContent(type="text", text="hi"),
            )
        ],
        maxTokens=256,
        temperature=0.3,
        stopSequences=["STOP", "END"],
    )
    await cb(None, params)

    config = captured.get("llm_config")
    assert config is not None
    assert config.max_output_tokens == 256
    assert config.temperature == 0.3
    assert config.stop_sequences == ["STOP", "END"]

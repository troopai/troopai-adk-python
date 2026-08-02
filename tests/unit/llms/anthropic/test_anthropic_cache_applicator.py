"""Tests for the Anthropic cache_control injector."""

from __future__ import annotations

from typing import Any, cast

from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolUnionParam,
)

from troopai.adk.llms.anthropic.anthropic_cache_applicator import apply_cache_control
from troopai.adk.llms.anthropic.anthropic_config import AnthropicConfig
from troopai.adk.llms.anthropic.anthropic_converter import AnthropicConverter
from troopai.adk.llms.llm_config import LLMConfig


def _user(content: str | list[TextBlockParam]) -> MessageParam:
    return MessageParam(role="user", content=content)


def _tool(name: str) -> ToolParam:
    return ToolParam(name=name, input_schema={"type": "object"})


class TestApplyCacheControlOff:
    def test_no_anthropic_config_passes_through(self) -> None:
        system = "You are a helpful assistant."
        messages: list[MessageParam] = [_user("hello")]
        tools: list[ToolUnionParam] = [_tool("foo")]
        out_system, out_msgs, out_tools = apply_cache_control(system, messages, tools, LLMConfig())
        assert out_system is system
        assert out_msgs is messages
        assert out_tools is tools

    def test_anthropic_config_without_flag_passes_through(self) -> None:
        cfg = AnthropicConfig()  # auto_cache_control is None
        system = "be helpful"
        messages: list[MessageParam] = [_user("hi")]
        out_system, out_msgs, out_tools = apply_cache_control(system, messages, None, cfg)
        assert out_system is system
        assert out_msgs is messages
        assert out_tools is None

    def test_flag_false_passes_through(self) -> None:
        cfg = AnthropicConfig(auto_cache_control=False)
        out_system, _out_msgs, _out_tools = apply_cache_control("be helpful", [_user("hi")], None, cfg)
        assert out_system == "be helpful"


class TestApplyCacheControlOn:
    def test_string_system_becomes_marked_textblock_list(self) -> None:
        cfg = AnthropicConfig(auto_cache_control=True)
        out_system, _msgs, _tools = apply_cache_control("long system prompt", [_user("q")], None, cfg)
        assert isinstance(out_system, list)
        assert len(out_system) == 1
        block = cast("dict[str, Any]", out_system[0])
        assert block["type"] == "text"
        assert block["text"] == "long system prompt"
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_ttl_propagates_to_marker(self) -> None:
        cfg = AnthropicConfig(auto_cache_control=True, cache_control_ttl="1h")
        out_system, _msgs, _tools = apply_cache_control("sys", [_user("q")], None, cfg)
        assert isinstance(out_system, list)
        block = cast("dict[str, Any]", out_system[0])
        assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_empty_string_system_unchanged(self) -> None:
        cfg = AnthropicConfig(auto_cache_control=True)
        out_system, _msgs, _tools = apply_cache_control("", [_user("q")], None, cfg)
        assert out_system == ""

    def test_last_user_text_block_marked(self) -> None:
        cfg = AnthropicConfig(auto_cache_control=True)
        messages: list[MessageParam] = [
            _user([TextBlockParam(type="text", text="A"), TextBlockParam(type="text", text="B")]),
        ]
        _system, out_msgs, _tools = apply_cache_control(None, messages, None, cfg)
        content = out_msgs[0]["content"]
        assert isinstance(content, list)
        # Last text block is marked
        last = cast("dict[str, Any]", content[-1])
        assert last["cache_control"] == {"type": "ephemeral"}
        # Earlier block is not
        first = cast("dict[str, Any]", content[0])
        assert "cache_control" not in first

    def test_string_user_content_becomes_marked_block_list(self) -> None:
        cfg = AnthropicConfig(auto_cache_control=True)
        _system, out_msgs, _tools = apply_cache_control(None, [_user("hello")], None, cfg)
        content = out_msgs[0]["content"]
        assert isinstance(content, list)
        block = cast("dict[str, Any]", content[0])
        assert block["text"] == "hello"
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_only_last_user_message_marked_when_multiple_users(self) -> None:
        cfg = AnthropicConfig(auto_cache_control=True)
        messages: list[MessageParam] = [
            _user([TextBlockParam(type="text", text="first")]),
            MessageParam(role="assistant", content="ok"),
            _user([TextBlockParam(type="text", text="second")]),
        ]
        _system, out_msgs, _tools = apply_cache_control(None, messages, None, cfg)
        # First user untouched
        first_content = out_msgs[0]["content"]
        assert isinstance(first_content, list)
        first_block = cast("dict[str, Any]", first_content[0])
        assert "cache_control" not in first_block
        # Second user marked
        second_content = out_msgs[2]["content"]
        assert isinstance(second_content, list)
        second_block = cast("dict[str, Any]", second_content[0])
        assert second_block["cache_control"] == {"type": "ephemeral"}

    def test_last_tool_marked(self) -> None:
        cfg = AnthropicConfig(auto_cache_control=True)
        tools: list[ToolUnionParam] = [_tool("first"), _tool("last")]
        _system, _msgs, out_tools = apply_cache_control(None, [_user("q")], tools, cfg)
        assert out_tools is not None
        first = cast("dict[str, Any]", out_tools[0])
        last = cast("dict[str, Any]", out_tools[1])
        assert "cache_control" not in first
        assert last["cache_control"] == {"type": "ephemeral"}

    def test_inputs_not_mutated(self) -> None:
        cfg = AnthropicConfig(auto_cache_control=True)
        original_messages: list[MessageParam] = [
            _user([TextBlockParam(type="text", text="A")]),
        ]
        original_tools: list[ToolUnionParam] = [_tool("foo")]
        apply_cache_control("sys", original_messages, original_tools, cfg)
        # Original objects untouched
        original_content = original_messages[0]["content"]
        assert isinstance(original_content, list)
        original_block = cast("dict[str, Any]", original_content[0])
        assert "cache_control" not in original_block
        original_tool = cast("dict[str, Any]", original_tools[0])
        assert "cache_control" not in original_tool


class TestCacheControlWithMidSystemMessages:
    """auto_cache_control must skip in-place role:"system" entries.

    The applicator's reverse scan for the last user message has to land on
    a ``role:"user"`` entry even when mid-conversation system messages are
    preserved in place — a regression here would silently mark the wrong
    message (or a system entry the API would reject).
    """

    def test_marker_lands_on_last_user_not_in_place_system(self) -> None:
        items: list[Any] = [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "message", "role": "assistant", "content": "hello"},
            {"type": "message", "role": "system", "content": "Terse mode."},
            {"type": "message", "role": "user", "content": "go"},
        ]
        _system, msgs = AnthropicConverter.items_to_messages(items, preserve_mid_system=True)
        config = AnthropicConfig(auto_cache_control=True, mid_system_messages=True)
        _wire_system, out_msgs, _tools = apply_cache_control("sys", msgs, None, config)

        roles = [m["role"] for m in out_msgs]
        assert roles == ["user", "assistant", "system", "user"]
        system_blocks = out_msgs[2]["content"]
        assert isinstance(system_blocks, list)
        assert all("cache_control" not in block for block in system_blocks), (
            "in-place system entries must stay unmarked"
        )
        last_user_blocks = out_msgs[3]["content"]
        assert isinstance(last_user_blocks, list)
        assert last_user_blocks[-1].get("cache_control") == {"type": "ephemeral"}

    def test_marker_skips_trailing_system_entry(self) -> None:
        items: list[Any] = [
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "message", "role": "system", "content": "Switch to French."},
        ]
        _system, msgs = AnthropicConverter.items_to_messages(items, preserve_mid_system=True)
        config = AnthropicConfig(auto_cache_control=True, mid_system_messages=True)
        _wire_system, out_msgs, _tools = apply_cache_control(None, msgs, None, config)

        assert [m["role"] for m in out_msgs] == ["user", "system"]
        first_user_blocks = out_msgs[0]["content"]
        assert isinstance(first_user_blocks, list)
        assert first_user_blocks[-1].get("cache_control") == {"type": "ephemeral"}
        trailing_system_blocks = out_msgs[1]["content"]
        assert isinstance(trailing_system_blocks, list)
        assert all("cache_control" not in block for block in trailing_system_blocks)

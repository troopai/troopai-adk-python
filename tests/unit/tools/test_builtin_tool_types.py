"""Unit tests for built-in tool types.

Tests for WebSearch, FileSearch, and Computer use type definitions
in ``troopai.adk.types.tools.builtin_tool_types``.
"""

import pytest

from troopai.adk.types.tools.builtin_tool_types import (
    ComputerAction,
    ComputerToolCall,
    ComputerToolCallResult,
    FileSearchResult,
    FileSearchToolCall,
    FileSearchToolCallResult,
    WebSearchResult,
    WebSearchToolCall,
    WebSearchToolCallResult,
)

# ---------------------------------------------------------------------------
# Web search types
# ---------------------------------------------------------------------------


class TestWebSearchTypes:
    """Tests for WebSearchResult, WebSearchToolCall, WebSearchToolCallResult."""

    def test_web_search_result_construction(self) -> None:
        result = WebSearchResult(
            url="https://example.com",
            title="Example",
            snippet="An example page.",
        )
        assert result.url == "https://example.com"
        assert result.title == "Example"
        assert result.snippet == "An example page."

    def test_web_search_result_snippet_defaults_to_none(self) -> None:
        result = WebSearchResult(url="https://example.com", title="Example")
        assert result.snippet is None

    def test_web_search_tool_call_construction(self) -> None:
        call = WebSearchToolCall(
            id="ws_001",
            query="python dataclasses",
            status="completed",
        )
        assert call.id == "ws_001"
        assert call.query == "python dataclasses"
        assert call.status == "completed"

    def test_web_search_tool_call_status_defaults_to_none(self) -> None:
        call = WebSearchToolCall(id="ws_002", query="test")
        assert call.status is None

    def test_web_search_tool_call_result_construction(self) -> None:
        entries = [
            WebSearchResult(url="https://a.com", title="A"),
            WebSearchResult(url="https://b.com", title="B", snippet="B snippet"),
        ]
        result = WebSearchToolCallResult(call_id="ws_001", results=entries)
        assert result.call_id == "ws_001"
        assert len(result.results) == 2
        assert result.results[0].url == "https://a.com"
        assert result.results[1].snippet == "B snippet"

    def test_web_search_tool_call_result_defaults_to_empty_list(self) -> None:
        result = WebSearchToolCallResult(call_id="ws_003")
        assert result.results == []

    def test_web_search_result_immutable(self) -> None:
        result = WebSearchResult(url="https://example.com", title="Example")
        with pytest.raises(AttributeError):
            result.url = "https://other.com"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            result.title = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# File search types
# ---------------------------------------------------------------------------


class TestFileSearchTypes:
    """Tests for FileSearchResult, FileSearchToolCall, FileSearchToolCallResult."""

    def test_file_search_result_defaults(self) -> None:
        result = FileSearchResult()
        assert result.file_id is None
        assert result.filename is None
        assert result.score is None
        assert result.text is None

    def test_file_search_result_construction(self) -> None:
        result = FileSearchResult(
            file_id="f_123",
            filename="report.pdf",
            score=0.95,
            text="Quarterly results...",
        )
        assert result.file_id == "f_123"
        assert result.filename == "report.pdf"
        assert result.score == 0.95
        assert result.text == "Quarterly results..."

    def test_file_search_tool_call_construction(self) -> None:
        call = FileSearchToolCall(
            id="fs_001",
            queries=["revenue 2025", "quarterly report"],
            status="completed",
        )
        assert call.id == "fs_001"
        assert call.queries == ["revenue 2025", "quarterly report"]
        assert call.status == "completed"

    def test_file_search_tool_call_queries_defaults_to_empty_list(self) -> None:
        call = FileSearchToolCall(id="fs_002")
        assert call.queries == []

    def test_file_search_tool_call_result_construction(self) -> None:
        entries = [
            FileSearchResult(file_id="f_1", filename="a.txt", score=0.9),
            FileSearchResult(file_id="f_2", filename="b.txt", score=0.7, text="match"),
        ]
        result = FileSearchToolCallResult(call_id="fs_001", results=entries)
        assert result.call_id == "fs_001"
        assert len(result.results) == 2
        assert result.results[0].score == 0.9
        assert result.results[1].text == "match"

    def test_file_search_tool_call_result_defaults_to_empty_list(self) -> None:
        result = FileSearchToolCallResult(call_id="fs_003")
        assert result.results == []


# ---------------------------------------------------------------------------
# Computer use types
# ---------------------------------------------------------------------------


class TestComputerTypes:
    """Tests for ComputerAction, ComputerToolCall, ComputerToolCallResult."""

    def test_computer_action_click(self) -> None:
        action = ComputerAction(type="click", x=100, y=200, button="left")
        assert action.type == "click"
        assert action.x == 100
        assert action.y == 200
        assert action.button == "left"
        assert action.text is None
        assert action.keys is None
        assert action.scroll_x is None
        assert action.scroll_y is None

    def test_computer_action_type_text(self) -> None:
        action = ComputerAction(type="type", text="Hello, world!")
        assert action.type == "type"
        assert action.text == "Hello, world!"
        assert action.x is None
        assert action.y is None

    def test_computer_action_keypress(self) -> None:
        action = ComputerAction(type="keypress", keys=["ctrl", "c"])
        assert action.type == "keypress"
        assert action.keys == ["ctrl", "c"]

    def test_computer_action_scroll(self) -> None:
        action = ComputerAction(type="scroll", scroll_x=0, scroll_y=-120)
        assert action.type == "scroll"
        assert action.scroll_x == 0
        assert action.scroll_y == -120

    def test_computer_tool_call_construction(self) -> None:
        action = ComputerAction(type="click", x=50, y=75, button="left")
        call = ComputerToolCall(
            id="cu_001",
            action=action,
            call_id="call_001",
            status="completed",
        )
        assert call.id == "cu_001"
        assert call.action is action
        assert call.action.type == "click"
        assert call.call_id == "call_001"
        assert call.status == "completed"

    def test_computer_tool_call_status_defaults_to_none(self) -> None:
        action = ComputerAction(type="screenshot")
        call = ComputerToolCall(id="cu_002", action=action, call_id="call_002")
        assert call.status is None

    def test_computer_tool_call_result_construction(self) -> None:
        result = ComputerToolCallResult(
            call_id="call_001",
            output="Button clicked.",
            screenshot="iVBORw0KGgo=",
        )
        assert result.call_id == "call_001"
        assert result.output == "Button clicked."
        assert result.screenshot == "iVBORw0KGgo="

    def test_computer_tool_call_result_defaults(self) -> None:
        result = ComputerToolCallResult(call_id="call_002")
        assert result.output is None
        assert result.screenshot is None

    def test_computer_action_immutable(self) -> None:
        action = ComputerAction(type="click", x=10, y=20)
        with pytest.raises(AttributeError):
            action.type = "type"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            action.x = 999  # type: ignore[misc]

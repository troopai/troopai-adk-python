"""Tests for the Live streaming surface on :class:`PanelRenderer`.

Validates ``open_stream_panel`` / ``update_stream_panel`` /
``close_stream_panel`` behaviour, the ``_just_streamed_final_answer``
flag transitions, and the LIFO single-Live-per-Console invariant.

Tests use ``Console(force_terminal=True, record=True, file=StringIO)``
so the Live widget actually starts (Rich auto-detects non-TTY and
refuses to refresh otherwise).
"""

from __future__ import annotations

import io
from unittest import mock

import pytest
from rich.console import Console

from troopai.adk.verbose.config import VerboseConfig
from troopai.adk.verbose.panel_renderer import (
    PanelRenderer,
    _stream_title_and_border,
    _truncate_stream_text,
)


@pytest.fixture(autouse=True)
def clear_tool_counter() -> None:
    """Reset class-level state between tests."""
    PanelRenderer._tool_usage_counts.clear()


def _renderer_with_live_console() -> tuple[PanelRenderer, Console]:
    cfg = VerboseConfig(mode="panel", use_color=True)
    renderer = PanelRenderer(cfg)
    console = Console(
        file=io.StringIO(),
        record=True,
        force_terminal=True,
        width=100,
        no_color=False,
    )
    renderer._console = console
    return renderer, console


# ---------------------------------------------------------------------------
# _stream_title_and_border helper
# ---------------------------------------------------------------------------


class TestStreamTitleAndBorder:
    def test_text_call_type(self) -> None:
        title, border = _stream_title_and_border("text")
        assert title == "✅ Agent Final Answer"
        assert border == "green"

    def test_tool_call_call_type(self) -> None:
        title, border = _stream_title_and_border("tool_call")
        assert title == "🔧 Tool Arguments"
        assert border == "yellow"


# ---------------------------------------------------------------------------
# _truncate_stream_text helper
# ---------------------------------------------------------------------------


class TestTruncateStreamText:
    def test_short_text_unchanged(self) -> None:
        text = "line 1\nline 2\nline 3"
        assert _truncate_stream_text(text, max_lines=20) == text

    def test_long_text_keeps_tail_with_prefix(self) -> None:
        text = "\n".join(f"line {i}" for i in range(30))
        result = _truncate_stream_text(text, max_lines=20)
        assert result.startswith("...\n")
        # Last line is preserved.
        assert result.endswith("line 29")
        # Should contain exactly 20 trailing lines after the "...\n".
        assert result.count("\n") == 20

    def test_max_lines_zero_disables_truncation(self) -> None:
        text = "a\nb\nc\nd"
        assert _truncate_stream_text(text, max_lines=0) == text


# ---------------------------------------------------------------------------
# PanelRenderer Live lifecycle
# ---------------------------------------------------------------------------


class TestOpenStreamPanel:
    def test_opens_live_widget(self) -> None:
        renderer, _ = _renderer_with_live_console()
        renderer.open_stream_panel("coordinator", "text")
        try:
            assert renderer._streaming_live is not None
            assert renderer._live_call_type == "text"
            assert renderer._just_streamed_final_answer is False
        finally:
            renderer.close_stream_panel()

    def test_subsequent_open_stops_prior_live(self) -> None:
        renderer, _ = _renderer_with_live_console()
        renderer.open_stream_panel("coordinator", "text")
        first_live = renderer._streaming_live
        try:
            renderer.open_stream_panel("coordinator", "tool_call")
            assert renderer._streaming_live is not first_live
            assert renderer._live_call_type == "tool_call"
        finally:
            renderer.close_stream_panel()

    def test_no_live_when_console_unavailable(self) -> None:
        cfg = VerboseConfig(mode="panel")
        renderer = PanelRenderer(cfg)
        with mock.patch.object(renderer, "_get_console", return_value=None):
            renderer.open_stream_panel("coordinator", "text")
        assert renderer._streaming_live is None


class TestUpdateStreamPanel:
    def test_noop_without_open_live(self) -> None:
        renderer, _ = _renderer_with_live_console()
        # Without open_stream_panel first, update is silent.
        renderer.update_stream_panel("hello", "text")
        assert renderer._streaming_live is None

    def test_update_propagates_call_type(self) -> None:
        renderer, _ = _renderer_with_live_console()
        renderer.open_stream_panel("coordinator", "text")
        try:
            renderer.update_stream_panel("partial answer", "text")
            assert renderer._live_call_type == "text"
            # Mid-stream switch to tool_call is allowed.
            renderer.update_stream_panel('{"foo":', "tool_call")
            assert renderer._live_call_type == "tool_call"
        finally:
            renderer.close_stream_panel()


class TestCloseStreamPanel:
    def test_clears_live_state(self) -> None:
        renderer, _ = _renderer_with_live_console()
        renderer.open_stream_panel("coordinator", "text")
        renderer.update_stream_panel("partial", "text")
        renderer.close_stream_panel()
        assert renderer._streaming_live is None

    def test_text_call_type_sets_just_streamed_flag(self) -> None:
        renderer, _ = _renderer_with_live_console()
        renderer.open_stream_panel("coordinator", "text")
        renderer.update_stream_panel("done", "text")
        renderer.close_stream_panel()
        # Subsequent agent_finish close should be suppressed.
        assert renderer._just_streamed_final_answer is True

    def test_tool_call_call_type_clears_just_streamed_flag(self) -> None:
        renderer, _ = _renderer_with_live_console()
        renderer.open_stream_panel("coordinator", "tool_call")
        renderer.update_stream_panel('{"foo":1}', "tool_call")
        renderer.close_stream_panel()
        # Tool-call streams must not suppress the agent_finish panel.
        assert renderer._just_streamed_final_answer is False

    def test_close_without_open_is_noop(self) -> None:
        renderer, _ = _renderer_with_live_console()
        # Must not raise.
        renderer.close_stream_panel()
        assert renderer._streaming_live is None


class TestPauseResumeForHITL:
    def test_pause_stops_live(self) -> None:
        renderer, _ = _renderer_with_live_console()
        renderer.open_stream_panel("coordinator", "text")
        renderer.pause_live_updates()
        assert renderer._streaming_live is None
        # Resume is a flag-clear no-op (next chunk reopens via emit path).
        renderer.resume_live_updates()
        assert renderer._streaming_live is None

    def test_pause_without_live_is_noop(self) -> None:
        renderer, _ = _renderer_with_live_console()
        renderer.pause_live_updates()
        renderer.resume_live_updates()
        assert renderer._streaming_live is None

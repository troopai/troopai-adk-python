"""Tests for :class:`VerboseRenderer`.

Focuses on the ANSI fallback path to keep tests deterministic
regardless of whether Rich is installed. Rich is exercised indirectly
only through a soft-import probe guard.
"""

from __future__ import annotations

import io
from typing import override

import pytest

from troopai.adk.verbose.config import (
    EVENT_AGENT_START,
    EVENT_TOOL_START,
    EventStyle,
    VerboseConfig,
)
from troopai.adk.verbose.renderer import (
    VerboseRenderer,
    _strip_ansi,
    format_payload,
    redact_secrets,
)


def _make_renderer(**kwargs) -> tuple[VerboseRenderer, io.StringIO]:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False, **kwargs)
    return VerboseRenderer(cfg), stream


def test_disabled_config_emits_nothing() -> None:
    renderer, stream = _make_renderer(enabled=False)
    renderer.render_line(EVENT_AGENT_START, "Alice started")
    assert stream.getvalue() == ""


def test_renders_headline() -> None:
    renderer, stream = _make_renderer()
    renderer.render_line(EVENT_AGENT_START, "Alice started")
    output = stream.getvalue()
    assert "Alice started" in output
    # prefix tag
    assert "[agent]" in output
    # icon (CrewAI-aligned emoji)
    assert "🤖" in output


def test_ansi_colour_present_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    renderer, stream = _make_renderer()
    renderer.render_line(EVENT_TOOL_START, "run_tool")
    # ANSI escape for "yellow" = \033[33m
    assert "\033[" in stream.getvalue()


def test_no_ansi_when_use_color_false() -> None:
    renderer, stream = _make_renderer(use_color=False)
    renderer.render_line(EVENT_TOOL_START, "run_tool")
    assert "\033[" not in stream.getvalue()


def test_payload_shown_when_enabled() -> None:
    renderer, stream = _make_renderer()
    renderer.render_line(EVENT_TOOL_START, "run_tool", "arg=42")
    assert "arg=42" in stream.getvalue()


def test_payload_suppressed_when_show_payload_false() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=False)
    cfg.styles[EVENT_TOOL_START] = EventStyle(
        color="yellow",
        icon="→",
        prefix="tool",
        show_payload=False,
    )
    renderer = VerboseRenderer(cfg)
    renderer.render_line(EVENT_TOOL_START, "run_tool", "should_not_appear")
    assert "should_not_appear" not in stream.getvalue()


def test_unknown_event_renders_plain() -> None:
    renderer, stream = _make_renderer()
    renderer.render_line("some.future.event", "hello")
    assert "hello" in stream.getvalue()


def test_render_line_swallows_no_exception_path() -> None:
    # Exercise the Rich probe path - should not blow up even if Rich is absent
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream, use_rich=True)
    renderer = VerboseRenderer(cfg)
    renderer.render_line(EVENT_AGENT_START, "ok")
    # Either Rich or ANSI wrote something - never nothing when enabled.
    assert len(stream.getvalue()) > 0


def test_timestamp_prefix_when_enabled() -> None:
    renderer, stream = _make_renderer(show_timestamps=True)
    renderer.render_line(EVENT_AGENT_START, "Alice")
    output = stream.getvalue()
    # HH:MM:SS pattern
    import re

    assert re.search(r"\d{2}:\d{2}:\d{2}", output) is not None


def test_format_payload_none() -> None:
    assert format_payload(None) == ""


def test_format_payload_truncates() -> None:
    long = "x" * 5000
    out = format_payload(long, max_chars=100)
    assert len(out) < len(long)
    assert "more chars" in out


def test_format_payload_passthrough_short() -> None:
    assert format_payload("hi") == "hi"


def test_format_payload_repr_for_non_string() -> None:
    out = format_payload({"a": 1})
    assert "a" in out and "1" in out


def test_format_payload_guards_against_raising_repr() -> None:
    class _Hostile:
        @override
        def __repr__(self) -> str:
            raise RuntimeError("boom")

    out = format_payload(_Hostile())
    # Must not raise; must flag the failure inline.
    assert "repr failed" in out
    assert "_Hostile" in out


def test_format_payload_guards_against_oversize_object() -> None:
    import sys

    class _Bloated:
        @override
        def __sizeof__(self) -> int:
            # Report a size that exceeds the 4 MB guard.
            return 10 * 1024 * 1024

        @override
        def __repr__(self) -> str:  # pragma: no cover — must not be called
            raise AssertionError("repr() reached despite oversize guard")

    obj = _Bloated()
    # Sanity: sys.getsizeof honours __sizeof__ (plus a small overhead).
    assert sys.getsizeof(obj) > 4 * 1024 * 1024
    out = format_payload(obj)
    assert "too large" in out
    assert "_Bloated" in out


def test_strip_ansi_removes_escape_sequences() -> None:
    # Cursor-up + clear-line + OSC can forge log lines.
    nasty = "\x1b[1A\x1b[2Kattacker-replaced\x1b]0;title\x07"
    safe = _strip_ansi(nasty)
    assert "\x1b" not in safe
    assert "attacker-replaced" in safe


def test_strip_ansi_preserves_plain_text() -> None:
    assert _strip_ansi("hello world") == "hello world"


def test_ansi_render_strips_payload_escapes() -> None:
    renderer, stream = _make_renderer()
    # Payload carrying forgery-capable escape sequences.
    renderer.render_line(
        EVENT_TOOL_START,
        "calling evil",
        "\x1b[1A\x1b[2Kinjected",
    )
    output = stream.getvalue()
    # Body line is sanitised; no raw escape bytes survive.
    assert "\x1b[1A" not in output
    assert "\x1b[2K" not in output
    assert "injected" in output


def test_redact_secrets_masks_common_keys() -> None:
    payload = {
        "query": "anthropic",
        "api_key": "sk-abc123",
        "Authorization": "Bearer xyz",
        "nested": {"password": "hunter2", "keep": "ok"},
    }
    redacted = redact_secrets(payload)
    assert redacted["query"] == "anthropic"
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["Authorization"] == "[REDACTED]"
    assert redacted["nested"]["password"] == "[REDACTED]"
    assert redacted["nested"]["keep"] == "ok"


def test_redact_secrets_passes_through_non_dict() -> None:
    assert redact_secrets("plain string") == "plain string"
    assert redact_secrets(42) == 42
    assert redact_secrets([{"token": "x"}, {"ok": 1}]) == [
        {"token": "[REDACTED]"},
        {"ok": 1},
    ]


def test_redact_secrets_handles_self_referential_dict() -> None:
    # A tool returning a Python cycle must not raise RecursionError out
    # of the verbose hooks (which are required never to raise).
    cyclic: dict[str, object] = {"api_key": "sk-secret"}
    cyclic["self"] = cyclic
    result = redact_secrets(cyclic)
    # Top-level secret is still masked; descent stops at the depth bound
    # instead of recursing forever on the cycle.
    assert result["api_key"] == "[REDACTED]"


def test_redact_secrets_handles_self_referential_list() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    # Must terminate rather than raise RecursionError.
    result = redact_secrets(cyclic)
    assert isinstance(result, list)


def test_redact_secrets_handles_deeply_nested_structure() -> None:
    # Acyclic but nested far deeper than the redaction depth bound.
    # Must not blow the recursion limit.
    node: dict[str, object] = {"leaf": True}
    for _ in range(2000):
        node = {"child": node}
    result = redact_secrets(node)
    assert isinstance(result, dict)

"""Tests for :class:`VerboseConfig`.

Covers style resolution, registration, ``NO_COLOR`` compliance, and the
stream default.
"""

from __future__ import annotations

import io

import pytest

from troopai.adk.verbose.config import (
    EVENT_AGENT_START,
    EVENT_TOOL_START,
    EventStyle,
    VerboseConfig,
)


def test_defaults_populated() -> None:
    cfg = VerboseConfig()
    assert cfg.enabled is True
    assert cfg.use_color is True
    assert cfg.use_rich is True
    assert cfg.show_timestamps is False
    assert cfg.output is None
    assert EVENT_AGENT_START in cfg.styles
    assert EVENT_TOOL_START in cfg.styles


def test_get_style_returns_registered() -> None:
    cfg = VerboseConfig()
    style = cfg.get_style(EVENT_AGENT_START)
    assert len(style.icon) > 0
    assert len(style.prefix) > 0


def test_get_style_unknown_returns_neutral() -> None:
    cfg = VerboseConfig()
    style = cfg.get_style("some.future.event")
    assert style == EventStyle()
    assert len(style.color) == 0
    assert len(style.icon) == 0


def test_register_event_overrides() -> None:
    cfg = VerboseConfig()
    cfg.register_event(
        "memory.read",
        EventStyle(color="blue", icon="⇲", prefix="memory"),
    )
    assert cfg.get_style("memory.read").color == "blue"
    assert cfg.get_style("memory.read").icon == "⇲"


def test_register_event_replaces_existing() -> None:
    cfg = VerboseConfig()
    original = cfg.get_style(EVENT_TOOL_START)
    cfg.register_event(EVENT_TOOL_START, EventStyle(color="red"))
    assert cfg.get_style(EVENT_TOOL_START).color == "red"
    assert cfg.get_style(EVENT_TOOL_START) != original


def test_resolve_output_defaults_to_stderr() -> None:
    import sys

    cfg = VerboseConfig()
    assert cfg.resolve_output() is sys.stderr


def test_resolve_output_honours_explicit() -> None:
    stream = io.StringIO()
    cfg = VerboseConfig(output=stream)
    assert cfg.resolve_output() is stream


def test_resolve_use_color_true_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    cfg = VerboseConfig()
    assert cfg.resolve_use_color() is True


def test_resolve_use_color_respects_field() -> None:
    cfg = VerboseConfig(use_color=False)
    assert cfg.resolve_use_color() is False


def test_resolve_use_color_respects_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    cfg = VerboseConfig()
    assert cfg.resolve_use_color() is False


def test_resolve_use_color_ignores_empty_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "")
    cfg = VerboseConfig()
    assert cfg.resolve_use_color() is True


def test_event_style_is_frozen() -> None:
    style = EventStyle(color="red")
    with pytest.raises(Exception):
        style.color = "blue"  # type: ignore[misc]


def test_disabled_config_round_trip() -> None:
    cfg = VerboseConfig(enabled=False)
    assert cfg.enabled is False
    assert cfg.get_style(EVENT_TOOL_START) is not None

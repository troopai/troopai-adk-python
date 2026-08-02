"""Tests for :mod:`troopai.adk.verbose.mode`.

Covers the full precedence ladder of :func:`resolve_mode` — explicit
developer override, ``NO_COLOR`` / ``FORCE_COLOR`` envs, CI detection,
TTY detection, and Rich-availability fallback — plus the individual
environment probes.

The tests patch ``os.environ`` via ``monkeypatch`` and inject fake
streams (``io.StringIO`` for non-TTY, a stub object with
``isatty=lambda: True`` for TTY) rather than manipulating real stdout,
so they are safe to run in any CI environment.
"""

from __future__ import annotations

import io
from typing import override
from unittest import mock

import pytest

from troopai.adk.verbose.config import VerboseConfig
from troopai.adk.verbose.mode import (
    is_ci,
    is_force_color,
    is_no_color,
    is_rich_available,
    is_tty,
    resolve_mode,
)

# ---------------------------------------------------------------------------
# Env fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrub colour / CI env vars so each test starts from a known state."""
    for var in ("NO_COLOR", "FORCE_COLOR", "CI", "TERM"):
        monkeypatch.delenv(var, raising=False)


class _TTYStream(io.StringIO):
    """Minimal TTY-looking stream for mode-resolution tests.

    Inherits :class:`io.StringIO` (itself a ``TextIO``) and overrides
    ``isatty`` so the stream looks like an interactive terminal. This
    avoids a ``cast(TextIO, ...)`` at call sites — the type is correct
    by inheritance.
    """

    @override
    def isatty(self) -> bool:
        return True


def _tty_stream() -> _TTYStream:
    return _TTYStream()


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------


class TestIsNoColor:
    def test_unset_returns_false(self) -> None:
        assert is_no_color() is False

    def test_non_empty_returns_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        assert is_no_color() is True

    def test_empty_returns_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NO_COLOR", "")
        assert is_no_color() is False


class TestIsForceColor:
    def test_unset_returns_false(self) -> None:
        assert is_force_color() is False

    def test_non_empty_returns_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FORCE_COLOR", "1")
        assert is_force_color() is True

    def test_zero_returns_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``FORCE_COLOR=0`` is the Node.js convention for "off"."""
        monkeypatch.setenv("FORCE_COLOR", "0")
        assert is_force_color() is False

    def test_empty_returns_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FORCE_COLOR", "")
        assert is_force_color() is False


class TestIsCi:
    def test_unset_returns_false(self) -> None:
        assert is_ci() is False

    def test_ci_true_returns_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("CI", "true")
        assert is_ci() is True

    def test_ci_false_returns_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """GitHub Actions set ``CI=true`` for jobs running in CI and
        some tools set ``CI=false`` for local emulation; we honour that."""
        monkeypatch.setenv("CI", "false")
        assert is_ci() is False

    def test_term_dumb_returns_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TERM", "dumb")
        assert is_ci() is True

    def test_term_xterm_returns_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TERM", "xterm-256color")
        assert is_ci() is False


class TestIsTty:
    def test_none_stream_returns_false(self) -> None:
        assert is_tty(None) is False

    def test_stringio_returns_false(self) -> None:
        """A StringIO has no isatty; treat as non-TTY."""
        stream = io.StringIO()
        assert is_tty(stream) is False

    def test_fake_tty_stream_returns_true(self) -> None:
        assert is_tty(_tty_stream()) is True

    def test_isatty_raises_returns_false(self) -> None:
        """A closed stream's isatty() raises ValueError; don't propagate."""

        class _Closed:
            def isatty(self) -> bool:
                raise ValueError("closed stream")

        assert is_tty(_Closed()) is False  # type: ignore[arg-type]


class TestIsRichAvailable:
    def test_rich_available_when_installed(self) -> None:
        """Rich is a direct dependency in this repo, so this is True."""
        assert is_rich_available() is True

    def test_downgrade_when_find_spec_returns_none(self) -> None:
        """When find_spec returns None, the function reports unavailable."""
        with mock.patch(
            "troopai.adk.verbose.mode.importlib.util.find_spec",
            return_value=None,
        ):
            assert is_rich_available() is False


# ---------------------------------------------------------------------------
# resolve_mode — precedence ladder
# ---------------------------------------------------------------------------


class TestResolveMode:
    def test_disabled_config_returns_off(self) -> None:
        """``enabled=False`` short-circuits regardless of mode."""
        cfg = VerboseConfig(enabled=False, mode="panel")
        assert resolve_mode(cfg) == "off"

    def test_mode_off_returns_off(self) -> None:
        cfg = VerboseConfig(mode="off")
        assert resolve_mode(cfg) == "off"

    def test_mode_line_returns_line(self) -> None:
        """Explicit line override wins even with a TTY."""
        cfg = VerboseConfig(mode="line", output=_tty_stream())
        assert resolve_mode(cfg) == "line"

    def test_mode_panel_returns_panel_when_rich_available(self) -> None:
        cfg = VerboseConfig(mode="panel")
        assert resolve_mode(cfg) == "panel"

    def test_mode_panel_downgrades_when_rich_missing(self) -> None:
        cfg = VerboseConfig(mode="panel")
        with mock.patch(
            "troopai.adk.verbose.mode.is_rich_available",
            return_value=False,
        ):
            assert resolve_mode(cfg) == "line"

    def test_auto_with_no_color_returns_line(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        cfg = VerboseConfig(mode="auto", output=_tty_stream())
        assert resolve_mode(cfg) == "line"

    def test_auto_with_force_color_returns_panel_when_rich_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """FORCE_COLOR overrides non-TTY detection, per Node/Rich convention."""
        monkeypatch.setenv("FORCE_COLOR", "1")
        cfg = VerboseConfig(mode="auto", output=io.StringIO())
        assert resolve_mode(cfg) == "panel"

    def test_auto_with_force_color_downgrades_when_rich_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FORCE_COLOR", "1")
        cfg = VerboseConfig(mode="auto", output=_tty_stream())
        with mock.patch(
            "troopai.adk.verbose.mode.is_rich_available",
            return_value=False,
        ):
            assert resolve_mode(cfg) == "line"

    def test_auto_with_non_tty_stream_returns_line(self) -> None:
        cfg = VerboseConfig(mode="auto", output=io.StringIO())
        assert resolve_mode(cfg) == "line"

    def test_auto_in_ci_returns_line(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("CI", "true")
        cfg = VerboseConfig(mode="auto", output=_tty_stream())
        assert resolve_mode(cfg) == "line"

    def test_auto_with_term_dumb_returns_line(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TERM", "dumb")
        cfg = VerboseConfig(mode="auto", output=_tty_stream())
        assert resolve_mode(cfg) == "line"

    def test_auto_without_rich_returns_line(self) -> None:
        cfg = VerboseConfig(mode="auto", output=_tty_stream())
        with mock.patch(
            "troopai.adk.verbose.mode.is_rich_available",
            return_value=False,
        ):
            assert resolve_mode(cfg) == "line"

    def test_auto_happy_path_returns_panel(self) -> None:
        """TTY + Rich + no env overrides → panel."""
        cfg = VerboseConfig(mode="auto", output=_tty_stream())
        assert resolve_mode(cfg) == "panel"

    def test_no_color_beats_force_color(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When both are set, NO_COLOR wins (higher precedence)."""
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("FORCE_COLOR", "1")
        cfg = VerboseConfig(mode="auto", output=_tty_stream())
        assert resolve_mode(cfg) == "line"

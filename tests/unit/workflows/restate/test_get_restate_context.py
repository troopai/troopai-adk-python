"""Tests for :func:`~troopai.adk.workflows.restate.llm.get_restate_context`.

The accessor reads ``restate.extensions.current_context()``, which raises
``LookupError`` when its ``ContextVar`` is unset (called outside a
handler).  These tests use ``sys.modules`` injection so they run with or
without the live ``restate`` SDK installed.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from troopai.adk.workflows.restate.llm import get_restate_context


def _install_fake_extensions(monkeypatch: pytest.MonkeyPatch, current_context: Any) -> None:
    """Inject a fake ``restate.extensions`` module exposing ``current_context``."""
    fake_pkg = types.ModuleType("restate")
    fake_ext = types.ModuleType("restate.extensions")
    fake_ext.current_context = current_context  # type: ignore[attr-defined]
    fake_pkg.extensions = fake_ext  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "restate", fake_pkg)
    monkeypatch.setitem(sys.modules, "restate.extensions", fake_ext)


class TestGetRestateContext:
    """get_restate_context must only swallow ImportError and LookupError.

    Regression: an earlier implementation swallowed broad error classes,
    silently dropping unrelated SDK errors and falling back to
    un-journaled direct LLM calls that cause replay divergence.
    """

    def test_returns_none_outside_handler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LookupError (ContextVar unset — outside a handler) returns None."""
        _install_fake_extensions(monkeypatch, MagicMock(side_effect=LookupError("restate_context")))
        assert get_restate_context() is None

    def test_returns_context_inside_handler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The active context object is returned verbatim when set."""
        sentinel = object()
        _install_fake_extensions(monkeypatch, MagicMock(return_value=sentinel))
        assert get_restate_context() is sentinel

    def test_unrelated_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Errors other than ImportError/LookupError must NOT be swallowed."""
        _install_fake_extensions(monkeypatch, MagicMock(side_effect=RuntimeError("event loop is closed")))
        with pytest.raises(RuntimeError, match="event loop is closed"):
            get_restate_context()

    def test_returns_none_on_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns None when the restate SDK is not installed."""
        monkeypatch.setitem(sys.modules, "restate", None)  # type: ignore[arg-type]
        monkeypatch.setitem(sys.modules, "restate.extensions", None)  # type: ignore[arg-type]
        assert get_restate_context() is None

    def test_live_sdk_outside_handler_returns_none(self) -> None:
        """With the real SDK installed, outside any handler the accessor is None."""
        pytest.importorskip("restate")
        assert get_restate_context() is None

"""Enforce the soft-import contract for ``rich``.

The :mod:`troopai.adk.verbose` module declares ``rich`` as **optional**
(installed via ``pip install 'troopai-adk-python[verbose]'``). If any future
change accidentally promotes ``rich`` to a hard dependency — by adding
a top-level ``from rich import ...`` anywhere on the import path of
``VerboseHooks`` / ``VerboseRenderer`` / ``PanelRenderer`` — these
tests break loudly instead of the failure surfacing only at install
time on a minimal-install consumer.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any, Optional

import pytest


class _RichBlocker:
    """Meta-path finder that raises ``ImportError`` for any ``rich`` module.

    Installed ahead of the real finders so an attempt to import ``rich``
    fails even if the wheel is present in ``site-packages``. Simulates
    the consumer who ran ``pip install troopai-adk-python`` without the
    ``[verbose]`` extra.
    """

    def find_spec(self, name: str, path: Optional[Any] = None, target: Optional[Any] = None) -> None:
        del path, target
        if name == "rich" or name.startswith("rich."):
            raise ImportError(f"simulated missing: {name}")
        return None


@pytest.fixture
def block_rich(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent ``rich`` from being imported for the duration of the test.

    Also purges any already-cached ``rich`` / ``troopai.adk.verbose.*``
    modules from :data:`sys.modules` so the next import executes module
    top-level against the blocked finder.
    """
    # Evict any cached verbose + rich modules — guarantees re-execution.
    to_drop = [name for name in list(sys.modules) if name.startswith(("rich", "troopai.adk.verbose"))]
    for name in to_drop:
        monkeypatch.delitem(sys.modules, name, raising=False)

    # Also evict stub finder for importlib so find_spec sees the blocker.
    monkeypatch.setattr("importlib.util.find_spec", _shim_find_spec(importlib.util.find_spec))

    blocker = _RichBlocker()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])


def _shim_find_spec(real_find_spec: Any) -> Any:
    """Wrap :func:`importlib.util.find_spec` so the blocker is honored.

    ``is_rich_available`` calls ``importlib.util.find_spec("rich")`` —
    that bypasses the ``sys.meta_path`` blocker (it goes through the
    actual finder chain). Intercept it here.
    """

    def _wrapped(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "rich" or name.startswith("rich."):
            return None
        return real_find_spec(name, *args, **kwargs)

    return _wrapped


class TestRichSoftImport:
    """Contract: ``rich`` must stay off the baseline import path."""

    def test_core_package_imports_without_rich(self, block_rich: None) -> None:
        del block_rich
        # The flagship surface — ``Agent`` and ``Runner`` — must load.
        mod = importlib.import_module("troopai.adk")
        assert hasattr(mod, "Agent")
        assert hasattr(mod, "Runner")
        assert "rich" not in sys.modules, "importing troopai.adk pulled in rich"

    def test_verbose_hooks_imports_without_rich(self, block_rich: None) -> None:
        del block_rich
        mod = importlib.import_module("troopai.adk.verbose")
        # VerboseHooks must be constructible — exercises hooks.py top-level.
        # ``run_config_verbose=None`` keeps the hook in a pass-through mode
        # where no renderer is ever instantiated, exactly what a consumer
        # without rich expects when they don't opt into verbose output.
        hooks_cls = mod.VerboseHooks
        hooks = hooks_cls(run_config_verbose=None)
        assert hooks is not None
        assert "rich" not in sys.modules, "importing troopai.adk.verbose pulled in rich"

    def test_is_rich_available_returns_false(self, block_rich: None) -> None:
        del block_rich
        mode = importlib.import_module("troopai.adk.verbose.mode")
        assert mode.is_rich_available() is False

    def test_resolve_mode_falls_back_to_line(self, block_rich: None) -> None:
        del block_rich
        mode = importlib.import_module("troopai.adk.verbose.mode")
        cfg_mod = importlib.import_module("troopai.adk.verbose.config")

        cfg = cfg_mod.VerboseConfig(enabled=True, mode="panel")
        resolved = mode.resolve_mode(cfg)
        # With rich unavailable, ``panel`` must degrade to ``line``.
        assert resolved == "line", f"expected 'line' fallback, got {resolved!r}"


class TestRichAvailableHappyPath:
    """Sanity: when rich IS available, the panel backend is selected."""

    def test_resolve_mode_selects_panel_when_rich_present(self) -> None:
        from troopai.adk.verbose.config import VerboseConfig
        from troopai.adk.verbose.mode import is_rich_available, resolve_mode

        # Only meaningful when the test environment has rich installed
        # (which it does via the ``[dev]`` extra).
        if is_rich_available() is False:
            pytest.skip("rich not installed in this environment")
        cfg = VerboseConfig(enabled=True, mode="panel")
        assert resolve_mode(cfg) == "panel"

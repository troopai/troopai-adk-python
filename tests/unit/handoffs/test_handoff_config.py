"""Tests for HandoffConfig construction-time validation."""

from __future__ import annotations

import pytest

from troopai.adk.handoffs.handoff_config import HandoffConfig
from troopai.adk.handoffs.handoff_strategy import HandoffStrategy


def test_last_n_strategy_without_window_is_rejected() -> None:
    """strategy=LAST_N without window must be rejected at construction.

    Otherwise the executor silently falls back to a hardcoded default
    window (hidden token cost) — a no-implicit-behaviour / cost-conservative
    violation. The misconfiguration should surface immediately.
    """
    with pytest.raises(ValueError, match="window"):
        HandoffConfig(strategy=HandoffStrategy.LAST_N)


def test_last_n_strategy_with_window_is_accepted() -> None:
    cfg = HandoffConfig(strategy=HandoffStrategy.LAST_N, window=5)
    assert cfg.window == 5


def test_full_strategy_without_window_is_fine() -> None:
    cfg = HandoffConfig(strategy=HandoffStrategy.FULL)
    assert cfg.strategy == HandoffStrategy.FULL

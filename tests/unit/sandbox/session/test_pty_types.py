"""Tests for ``troopai.adk.sandbox.session.pty_types``."""

from __future__ import annotations

import time

import pytest

from troopai.adk.sandbox.session.pty_types import (
    PTY_EMPTY_YIELD_TIME_MS_MIN,
    PTY_YIELD_TIME_MS_MAX,
    PTY_YIELD_TIME_MS_MIN,
    allocate_pty_process_id,
    clamp_pty_yield_time_ms,
    process_id_to_prune_from_meta,
    resolve_pty_write_yield_time_ms,
    truncate_text_by_tokens,
)


class TestClamp:
    def test_clamps_below_floor(self) -> None:
        assert clamp_pty_yield_time_ms(10) == PTY_YIELD_TIME_MS_MIN

    def test_clamps_above_ceiling(self) -> None:
        assert clamp_pty_yield_time_ms(60_000) == PTY_YIELD_TIME_MS_MAX

    def test_passes_through_in_range(self) -> None:
        assert clamp_pty_yield_time_ms(1000) == 1000


class TestResolveWriteYieldTime:
    def test_input_empty_uses_drain_floor(self) -> None:
        out = resolve_pty_write_yield_time_ms(yield_time_ms=500, input_empty=True)
        assert out == PTY_EMPTY_YIELD_TIME_MS_MIN

    def test_input_present_uses_normal_clamp(self) -> None:
        out = resolve_pty_write_yield_time_ms(yield_time_ms=500, input_empty=False)
        assert out == 500


class TestAllocatePtyProcessId:
    def test_returns_unused_id(self) -> None:
        used = {1500, 2000, 3000}
        new_id = allocate_pty_process_id(used)
        assert new_id not in used
        assert 1_000 <= new_id < 100_000

    def test_raises_when_range_exhausted(self) -> None:
        # Construct an impossibly-large used set to trip the early guard.
        used = set(range(1_000, 100_000))
        with pytest.raises(RuntimeError, match="range exhausted"):
            allocate_pty_process_id(used)


class TestProcessIdToPrune:
    def test_empty_meta_returns_none(self) -> None:
        assert process_id_to_prune_from_meta([]) is None

    def test_prefers_exited_lru(self) -> None:
        now = time.time()
        # 10 entries — 8 recent + 2 older; the older of the two is "exited".
        meta = [(i, now - 1000 + i, False) for i in range(8)]  # recent (protected)
        meta.append((100, now - 5000, True))  # older + exited → prune target
        meta.append((101, now - 4000, False))  # older + running
        out = process_id_to_prune_from_meta(meta)
        assert out == 100

    def test_falls_back_to_lru_when_none_exited(self) -> None:
        now = time.time()
        meta = [(i, now - 1000 + i, False) for i in range(8)]
        meta.append((100, now - 5000, False))  # oldest unprotected → prune
        out = process_id_to_prune_from_meta(meta)
        assert out == 100

    def test_no_unprotected_returns_none(self) -> None:
        now = time.time()
        meta = [(i, now - 1000 + i, False) for i in range(8)]  # all 8 protected
        assert process_id_to_prune_from_meta(meta) is None


class TestTruncateTextByTokens:
    def test_no_truncation_returns_none(self) -> None:
        out, original = truncate_text_by_tokens("hello", None)
        assert out == "hello"
        assert original is None

    def test_truncates_when_budget_exceeded(self) -> None:
        out, original = truncate_text_by_tokens("x" * 1000, 10)
        assert "truncated" in out
        assert original is not None

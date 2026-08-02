"""Tests for the litellm ``auto_cache_control`` → injection-points resolver."""

from __future__ import annotations

from troopai.adk.llms.litellm.litellm_cache_applicator import resolve_cache_control_injection_points
from troopai.adk.llms.litellm.litellm_model import LiteLLMConfig


class TestResolveOff:
    def test_none_flag_no_explicit_returns_none(self) -> None:
        assert resolve_cache_control_injection_points(None, None) is None

    def test_false_flag_no_explicit_returns_none(self) -> None:
        # False is not True — no auto points; caching left untouched.
        assert resolve_cache_control_injection_points(False, None) is None

    def test_config_default_is_off(self) -> None:
        # Cost-conservative default: the caller opts INTO the cache-write premium.
        assert LiteLLMConfig().auto_cache_control is None


class TestResolveExplicitWins:
    def test_explicit_points_returned_verbatim_even_with_auto_on(self) -> None:
        explicit = [{"location": "message", "role": "user", "index": None, "control": {"type": "ephemeral"}}]
        out = resolve_cache_control_injection_points(True, explicit)  # type: ignore[arg-type]
        assert out is explicit

    def test_explicit_wins_over_disabled_flag(self) -> None:
        explicit = [{"location": "message", "role": "system", "index": None, "control": {"type": "ephemeral"}}]
        out = resolve_cache_control_injection_points(None, explicit)  # type: ignore[arg-type]
        assert out is explicit


class TestResolveAuto:
    def test_auto_returns_system_then_last_message_points(self) -> None:
        out = resolve_cache_control_injection_points(True, None)
        assert out is not None
        assert len(out) == 2
        system_point, last_point = out
        # System message is cached by role.
        assert system_point == {
            "location": "message",
            "role": "system",
            "index": None,
            "control": {"type": "ephemeral"},
        }
        # The last input message is cached by negative index (role left None).
        assert last_point == {
            "location": "message",
            "role": None,
            "index": -1,
            "control": {"type": "ephemeral"},
        }

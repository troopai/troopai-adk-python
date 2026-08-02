"""Tests for ``troopai.adk.sandbox.capabilities.Capabilities`` (P11)."""

from __future__ import annotations

from troopai.adk.sandbox.capabilities import Capabilities, CompactionCapability


class TestDefaultsAreCostConservative:
    def test_default_is_compaction_only(self) -> None:
        result = Capabilities.default()
        assert len(result) == 1
        assert isinstance(result[0], CompactionCapability)

    def test_default_returns_fresh_list(self) -> None:
        a = Capabilities.default()
        b = Capabilities.default()
        assert a is not b
        # Mutating one does not affect the other.
        a.append(CompactionCapability())
        assert len(b) == 1


class TestListConcatErgonomic:
    def test_concat_with_extras(self) -> None:
        extras = [CompactionCapability()]  # placeholder until P13/P14
        combined = Capabilities.default() + extras
        assert len(combined) == 2

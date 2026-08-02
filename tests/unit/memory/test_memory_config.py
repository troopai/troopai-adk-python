"""Tests for MemoryConfig validation."""

import logging

from troopai.adk.memory.in_memory import TemporaryMemory
from troopai.adk.memory.memory_config import MemoryConfig


class TestMemoryConfigValidation:
    def test_auto_extract_without_extractor_raises(self):
        """MemoryConfig must raise ValueError when auto_extract=True but extractor is None."""
        import pytest

        with pytest.raises(ValueError, match="auto_extract=True requires an extractor"):
            MemoryConfig(
                memory=TemporaryMemory(),
                namespace="ns",
                auto_extract=True,
            )

    def test_auto_extract_with_extractor_no_warning(self, caplog):
        """No warning when auto_extract=True and extractor is provided."""

        class FakeExtractor:
            async def extract(self, _messages, *, namespace):
                return []

        with caplog.at_level(logging.WARNING):
            MemoryConfig(
                memory=TemporaryMemory(),
                namespace="ns",
                auto_extract=True,
                extractor=FakeExtractor(),
            )
        assert "auto_extract" not in caplog.text

    def test_no_warning_when_auto_extract_false(self, caplog):
        """No warning when auto_extract is False (default)."""
        with caplog.at_level(logging.WARNING):
            MemoryConfig(
                memory=TemporaryMemory(),
                namespace="ns",
            )
        assert "auto_extract" not in caplog.text

"""Tests for memory type definitions."""

import pytest

from troopai.adk.memory.memory_types import (
    MemoryEntry,
    MemoryKind,
    MemoryMetadata,
    MemorySearchFilter,
    MemorySearchResult,
    MemorySource,
)


class TestMemorySource:
    def test_values(self):
        assert MemorySource.EXTRACTION == "extraction"
        assert MemorySource.MANUAL == "manual"
        assert MemorySource.TOOL == "tool"

    def test_str_serialization(self):
        assert str(MemorySource.MANUAL) == "manual"


class TestMemoryMetadata:
    def test_defaults(self):
        meta = MemoryMetadata(source=MemorySource.MANUAL)
        assert meta.importance == 3
        assert meta.categories == ()
        assert meta.session_id is None
        assert meta.agent_name is None
        assert meta.kind is MemoryKind.EPISODIC
        assert meta.custom == {}

    def test_custom_values(self):
        meta = MemoryMetadata(
            source=MemorySource.EXTRACTION,
            importance=5,
            categories=("fact", "preference"),
            session_id="sess-1",
            agent_name="support",
            custom={"key": "value"},
        )
        assert meta.importance == 5
        assert meta.categories == ("fact", "preference")
        assert meta.session_id == "sess-1"
        assert meta.agent_name == "support"

    def test_frozen(self):
        meta = MemoryMetadata(source=MemorySource.MANUAL)
        with pytest.raises(AttributeError):
            meta.importance = 5  # type: ignore[misc]


class TestMemoryEntry:
    def test_creation(self):
        meta = MemoryMetadata(source=MemorySource.MANUAL)
        entry = MemoryEntry(
            id="abc-123",
            content="User prefers dark mode",
            namespace="user:1",
            metadata=meta,
            created_at=1000.0,
            updated_at=1000.0,
        )
        assert entry.id == "abc-123"
        assert entry.content == "User prefers dark mode"
        assert entry.namespace == "user:1"

    def test_frozen(self):
        meta = MemoryMetadata(source=MemorySource.MANUAL)
        entry = MemoryEntry(
            id="abc-123",
            content="test",
            namespace="ns",
            metadata=meta,
            created_at=1000.0,
            updated_at=1000.0,
        )
        with pytest.raises(AttributeError):
            entry.content = "modified"  # type: ignore[misc]


class TestMemorySearchResult:
    def test_creation(self):
        meta = MemoryMetadata(source=MemorySource.MANUAL)
        entry = MemoryEntry(
            id="abc-123",
            content="test",
            namespace="ns",
            metadata=meta,
            created_at=1000.0,
            updated_at=1000.0,
        )
        result = MemorySearchResult(entry=entry, score=0.85)
        assert result.score == 0.85
        assert result.entry.id == "abc-123"


class TestMemoryMetadataImportanceBounds:
    def test_valid_importance_values(self):
        for imp in (1, 2, 3, 4, 5):
            m = MemoryMetadata(source=MemorySource.MANUAL, importance=imp)
            assert m.importance == imp

    def test_importance_below_1_raises(self):
        with pytest.raises(ValueError, match="importance"):
            MemoryMetadata(source=MemorySource.MANUAL, importance=0)

    def test_importance_above_5_raises(self):
        with pytest.raises(ValueError, match="importance"):
            MemoryMetadata(source=MemorySource.MANUAL, importance=6)

    def test_negative_importance_raises(self):
        with pytest.raises(ValueError, match="importance"):
            MemoryMetadata(source=MemorySource.MANUAL, importance=-1)


class TestMemorySearchFilter:
    def test_defaults(self):
        f = MemorySearchFilter()
        assert f.namespace is None
        assert f.categories is None
        assert f.importance is None
        assert f.agent_name is None
        assert f.kind is None
        assert f.after is None
        assert f.before is None

    def test_frozen(self):
        """MemorySearchFilter must be immutable (frozen=True)."""
        f = MemorySearchFilter(importance=3)
        with pytest.raises(AttributeError):
            f.importance = 5  # type: ignore[misc]

    def test_categories_accepts_tuple(self):
        """categories field must be tuple[str, ...] | None when frozen."""
        f = MemorySearchFilter(categories=("a", "b"))
        assert f.categories == ("a", "b")

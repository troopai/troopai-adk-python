"""Tests for TemporaryMemory backend."""

import pytest

from troopai.adk.memory.in_memory import TemporaryMemory
from troopai.adk.memory.memory_types import (
    MemoryMetadata,
    MemorySearchFilter,
    MemorySource,
)


@pytest.fixture
def memory():
    return TemporaryMemory()


class TestAdd:
    @pytest.mark.asyncio
    async def test_add_returns_entry(self, memory):
        entry = await memory.add("User likes Python", namespace="user:1")
        assert entry.content == "User likes Python"
        assert entry.namespace == "user:1"
        assert entry.id  # non-empty UUID
        assert entry.metadata.source == MemorySource.MANUAL

    @pytest.mark.asyncio
    async def test_add_with_metadata(self, memory):
        meta = MemoryMetadata(
            source=MemorySource.TOOL,
            importance=5,
            categories=("preference",),
        )
        entry = await memory.add("Dark mode preferred", namespace="user:1", metadata=meta)
        assert entry.metadata.source == MemorySource.TOOL
        assert entry.metadata.importance == 5
        assert entry.metadata.categories == ("preference",)


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_finds_matching(self, memory):
        await memory.add("User prefers dark mode", namespace="user:1")
        await memory.add("User lives in Paris", namespace="user:1")

        results = await memory.search("dark mode", namespace="user:1")
        assert len(results) >= 1
        assert any("dark mode" in r.entry.content for r in results)

    @pytest.mark.asyncio
    async def test_search_respects_namespace(self, memory):
        await memory.add("User likes Python", namespace="user:1")
        await memory.add("User likes Python", namespace="user:2")

        results = await memory.search("Python", namespace="user:1")
        assert len(results) == 1
        assert results[0].entry.namespace == "user:1"

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, memory):
        for i in range(10):
            await memory.add(f"Fact number {i} about Python", namespace="ns")

        results = await memory.search("Python", namespace="ns", limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_empty_query(self, memory):
        await memory.add("Something", namespace="ns")
        results = await memory.search("", namespace="ns")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_no_match(self, memory):
        await memory.add("User likes Python", namespace="ns")
        results = await memory.search("JavaScript", namespace="ns")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_importance_filter(self, memory):
        meta_low = MemoryMetadata(source=MemorySource.MANUAL, importance=1)
        meta_high = MemoryMetadata(source=MemorySource.MANUAL, importance=5)

        await memory.add("Low importance fact about Python", namespace="ns", metadata=meta_low)
        await memory.add("High importance fact about Python", namespace="ns", metadata=meta_high)

        filter = MemorySearchFilter(importance=3)
        results = await memory.search("Python", namespace="ns", filter=filter)
        assert len(results) == 1
        assert results[0].entry.metadata.importance == 5

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self, memory):
        meta1 = MemoryMetadata(source=MemorySource.MANUAL, categories=("preference",))
        meta2 = MemoryMetadata(source=MemorySource.MANUAL, categories=("fact",))

        await memory.add("Prefers Python programming", namespace="ns", metadata=meta1)
        await memory.add("Python is a language", namespace="ns", metadata=meta2)

        filter = MemorySearchFilter(categories=("preference",))
        results = await memory.search("Python", namespace="ns", filter=filter)
        assert len(results) == 1
        assert "Prefers" in results[0].entry.content

    @pytest.mark.asyncio
    async def test_search_scores_ordered(self, memory):
        await memory.add("dark mode is the best display theme", namespace="ns")
        await memory.add("dark", namespace="ns")

        results = await memory.search("dark mode", namespace="ns")
        assert len(results) == 2
        # More relevant result should have higher score
        assert results[0].score >= results[1].score


class TestGet:
    @pytest.mark.asyncio
    async def test_get_existing(self, memory):
        entry = await memory.add("Test content", namespace="ns")
        retrieved = await memory.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == "Test content"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, memory):
        result = await memory.get("nonexistent-id")
        assert result is None


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_existing(self, memory):
        entry = await memory.add("Test", namespace="ns")
        deleted = await memory.delete(entry.id)
        assert deleted is True
        assert await memory.get(entry.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory):
        deleted = await memory.delete("nonexistent-id")
        assert deleted is False


class TestClear:
    @pytest.mark.asyncio
    async def test_clear_namespace(self, memory):
        await memory.add("A", namespace="ns1")
        await memory.add("B", namespace="ns1")
        await memory.add("C", namespace="ns2")

        count = await memory.clear(namespace="ns1")
        assert count == 2

        # ns2 untouched
        results = await memory.search("C", namespace="ns2")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_clear_empty_namespace(self, memory):
        count = await memory.clear(namespace="empty")
        assert count == 0


class TestAddFromSession:
    @pytest.mark.asyncio
    async def test_add_from_session(self, memory):
        class FakeExtractor:
            async def extract(self, _messages, *, namespace):
                from troopai.adk.memory.extractor import ExtractionResult

                return [
                    ExtractionResult(content="User likes Python", importance=4, categories=("preference",)),
                    ExtractionResult(content="User is a developer", importance=3),
                ]

        messages = [{"role": "user", "content": "I'm a Python developer"}]
        entries = await memory.add_from_session(
            messages,
            namespace="user:1",
            extractor=FakeExtractor(),
            session_id="sess-1",
            agent_name="assistant",
        )
        assert len(entries) == 2
        assert entries[0].content == "User likes Python"
        assert entries[0].metadata.source == MemorySource.EXTRACTION
        assert entries[0].metadata.session_id == "sess-1"
        assert entries[0].metadata.agent_name == "assistant"


class TestAddEvents:
    @pytest.mark.asyncio
    async def test_add_events(self, memory):
        from troopai.adk.memory.extractor import ExtractionResult
        from troopai.adk.session.session_event import create_session_event

        class FakeExtractor:
            async def extract(self, _messages, *, namespace):
                return [ExtractionResult(content="User prefers Python", importance=4)]

        events = [
            create_session_event(author="user", content={"role": "user", "content": "I love Python"}),
            create_session_event(author="assistant", content={"role": "assistant", "content": "Great!"}),
        ]
        entries = await memory.add_events(
            events,
            namespace="user:1",
            extractor=FakeExtractor(),
            session_id="sess-1",
            agent_name="assistant",
        )
        assert len(entries) == 1
        assert entries[0].content == "User prefers Python"
        assert entries[0].metadata.session_id == "sess-1"


class TestAddSession:
    @pytest.mark.asyncio
    async def test_add_session(self, memory):
        from troopai.adk.memory.extractor import ExtractionResult
        from troopai.adk.session import SQLiteMultiSessions

        class FakeExtractor:
            async def extract(self, _messages, *, namespace):
                return [ExtractionResult(content="User is a developer", importance=3)]

        store = SQLiteMultiSessions()
        session = await store.create("test-session")

        from troopai.adk.session.session_event import create_session_event

        await session.add(
            [
                create_session_event(author="user", content={"role": "user", "content": "I code daily"}),
            ]
        )

        entries = await memory.add_session(
            session,
            namespace="user:1",
            extractor=FakeExtractor(),
            agent_name="assistant",
        )
        assert len(entries) == 1
        assert entries[0].content == "User is a developer"
        assert entries[0].metadata.session_id == "test-session"
        await store.close()

"""Tests for SQLiteMemory backend."""

import pytest

from troopai.adk.memory.memory_types import (
    MemoryMetadata,
    MemorySearchFilter,
    MemorySource,
)
from troopai.adk.memory.sqlite_memory import SQLiteMemory


@pytest.fixture
def memory():
    return SQLiteMemory(path=":memory:")


class TestTableNameValidation:
    """Constructor-time allowlist for ``table`` — closes SQL-injection surface.

    The table name is interpolated into f-string SQL (SQLite cannot
    parameterize identifiers), so a malicious value would execute
    arbitrary SQL on every memory operation. These tests pin the
    fail-fast contract.
    """

    @pytest.mark.parametrize(
        "bad_name",
        [
            "",
            "memories; DROP TABLE users;--",
            "memories users",
            "memories'",
            '"memories"',
            "memories-foo",
            "1memories",
            "m" * 65,
        ],
    )
    def test_rejects_unsafe_table_name(self, bad_name: str) -> None:
        with pytest.raises(ValueError):
            SQLiteMemory(path=":memory:", table=bad_name)

    @pytest.mark.parametrize(
        "good_name",
        ["memories", "my_memories", "_private", "t1", "A_1", "m" * 64],
    )
    def test_accepts_safe_table_name(self, good_name: str) -> None:
        mem = SQLiteMemory(path=":memory:", table=good_name)
        assert mem._table == good_name
        assert mem._fts_table == f"{good_name}_fts"


class TestAdd:
    @pytest.mark.asyncio
    async def test_add_returns_entry(self, memory):
        entry = await memory.add("User likes Python", namespace="user:1")
        assert entry.content == "User likes Python"
        assert entry.namespace == "user:1"
        assert entry.id
        assert entry.metadata.source == MemorySource.MANUAL

    @pytest.mark.asyncio
    async def test_add_with_metadata(self, memory):
        meta = MemoryMetadata(
            source=MemorySource.TOOL,
            importance=5,
            categories=("preference",),
            agent_name="support",
        )
        entry = await memory.add("Prefers dark mode", namespace="user:1", metadata=meta)
        assert entry.metadata.source == MemorySource.TOOL
        assert entry.metadata.importance == 5
        assert entry.metadata.categories == ("preference",)
        assert entry.metadata.agent_name == "support"


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_finds_matching(self, memory):
        await memory.add("User prefers dark mode for displays", namespace="user:1")
        await memory.add("User lives in Paris France", namespace="user:1")

        results = await memory.search("dark mode", namespace="user:1")
        assert len(results) >= 1
        assert any("dark mode" in r.entry.content for r in results)

    @pytest.mark.asyncio
    async def test_search_respects_namespace(self, memory):
        await memory.add("User likes Python programming", namespace="user:1")
        await memory.add("User likes Python programming", namespace="user:2")

        results = await memory.search("Python", namespace="user:1")
        assert len(results) == 1
        assert results[0].entry.namespace == "user:1"

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, memory):
        for i in range(10):
            await memory.add(f"Fact number {i} about Python programming", namespace="ns")

        results = await memory.search("Python", namespace="ns", limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_empty_query(self, memory):
        await memory.add("Something interesting here", namespace="ns")
        results = await memory.search("", namespace="ns")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_no_match_falls_back_to_recent(self, memory):
        """No FTS match falls back to most recent memories."""
        await memory.add("User likes Python programming", namespace="ns")
        results = await memory.search("JavaScript", namespace="ns")
        assert len(results) == 1
        assert "Python" in results[0].entry.content

    @pytest.mark.asyncio
    async def test_search_scores_normalized(self, memory):
        await memory.add("dark mode theme preference setting", namespace="ns")
        await memory.add("dark mode is enabled on the display", namespace="ns")

        results = await memory.search("dark mode", namespace="ns")
        assert len(results) == 2
        for r in results:
            assert 0.0 <= r.score <= 1.0

    @pytest.mark.asyncio
    async def test_search_with_importance_filter(self, memory):
        meta_low = MemoryMetadata(source=MemorySource.MANUAL, importance=1)
        meta_high = MemoryMetadata(source=MemorySource.MANUAL, importance=5)

        await memory.add("Low importance Python fact", namespace="ns", metadata=meta_low)
        await memory.add("High importance Python fact", namespace="ns", metadata=meta_high)

        filter = MemorySearchFilter(importance=3)
        results = await memory.search("Python", namespace="ns", filter=filter)
        assert len(results) == 1
        assert results[0].entry.metadata.importance == 5

    @pytest.mark.asyncio
    async def test_search_or_semantics(self, memory):
        """FTS5 search uses OR — partial word overlap matches documents."""
        await memory.add("User prefers using PyTorch over TensorFlow", namespace="ns")
        await memory.add("User is a Python developer working on ML", namespace="ns")

        results = await memory.search("Tell me about PyTorch", namespace="ns")
        assert len(results) >= 1
        assert any("PyTorch" in r.entry.content for r in results)

    @pytest.mark.asyncio
    async def test_search_mixed_overlap_query(self, memory):
        """Queries with partial content overlap find matches via OR."""
        await memory.add("User likes Python programming", namespace="ns")

        results = await memory.search("What is the user's favorite Python language?", namespace="ns")
        assert len(results) >= 1
        assert any("Python" in r.entry.content for r in results)

    @pytest.mark.asyncio
    async def test_search_punctuation_in_query(self, memory):
        """Punctuation in query does not break FTS5 matching."""
        await memory.add("User prefers dark mode", namespace="ns")
        results = await memory.search("dark mode?", namespace="ns")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_zero_overlap_returns_recent(self, memory):
        """Query with zero token overlap falls back to recent memories."""
        await memory.add("User prefers PyTorch over TensorFlow", namespace="ns")
        await memory.add("User is a Python developer", namespace="ns")

        # No token overlap with stored content — recency fallback
        results = await memory.search("What do you remember about me?", namespace="ns")
        assert len(results) == 2
        # Most recent first
        assert "Python developer" in results[0].entry.content

    @pytest.mark.asyncio
    async def test_search_non_english_query(self, memory):
        """Non-English queries work without language-specific stopwords."""
        await memory.add("L'utilisateur préfère PyTorch", namespace="ns")

        results = await memory.search("Parlez-moi de PyTorch", namespace="ns")
        assert len(results) >= 1
        assert any("PyTorch" in r.entry.content for r in results)

    @pytest.mark.asyncio
    async def test_search_with_agent_name_filter(self, memory):
        meta1 = MemoryMetadata(source=MemorySource.MANUAL, agent_name="support")
        meta2 = MemoryMetadata(source=MemorySource.MANUAL, agent_name="sales")

        await memory.add("Support Python interaction note", namespace="ns", metadata=meta1)
        await memory.add("Sales Python interaction note", namespace="ns", metadata=meta2)

        filter = MemorySearchFilter(agent_name="support")
        results = await memory.search("Python", namespace="ns", filter=filter)
        assert len(results) == 1
        assert results[0].entry.metadata.agent_name == "support"


class TestSearchEmbeddedDoubleQuote:
    """A query token with an interior double-quote must not crash FTS5.

    Search text is arbitrary developer/end-user input (inch marks, code
    snippets, quoted speech). An interior ``"`` survives token cleanup
    and, when wrapped as an FTS5 phrase, produced a malformed MATCH
    expression that raised ``OperationalError('unterminated string')``.
    """

    @pytest.mark.asyncio
    async def test_search_query_with_interior_double_quote(self, memory):
        await memory.add('The 5"display screen is bright', namespace="ns")

        # Token ``5"display`` carries an interior double-quote.
        results = await memory.search('5"display', namespace="ns")
        assert len(results) >= 1
        assert any("display" in r.entry.content for r in results)

    @pytest.mark.asyncio
    async def test_search_query_with_multiple_interior_quotes(self, memory):
        await memory.add("She said hello to everyone", namespace="ns")

        # Must not raise even with stray quotes that match nothing; the
        # query should fall back to recency rather than crash.
        results = await memory.search('a"b"c hello', namespace="ns")
        assert len(results) >= 1


class TestSearchCategoriesFilter:
    """``MemorySearchFilter.categories`` must be honored by SQLiteMemory.

    The filter restricts to entries carrying any of the given categories,
    matching the contract every sibling backend implements. Exercised on
    both the BM25 path and the recency-fallback path.
    """

    @pytest.mark.asyncio
    async def test_categories_filter_on_bm25_path(self, memory):
        billing = MemoryMetadata(source=MemorySource.MANUAL, categories=("billing",))
        support = MemoryMetadata(source=MemorySource.MANUAL, categories=("support",))
        await memory.add("Python invoice billing question", namespace="ns", metadata=billing)
        await memory.add("Python password reset support", namespace="ns", metadata=support)

        results = await memory.search("Python", namespace="ns", filter=MemorySearchFilter(categories=("billing",)))
        assert len(results) == 1
        assert results[0].entry.metadata.categories == ("billing",)

    @pytest.mark.asyncio
    async def test_categories_filter_matches_any(self, memory):
        billing = MemoryMetadata(source=MemorySource.MANUAL, categories=("billing",))
        support = MemoryMetadata(source=MemorySource.MANUAL, categories=("support",))
        sales = MemoryMetadata(source=MemorySource.MANUAL, categories=("sales",))
        await memory.add("Python billing", namespace="ns", metadata=billing)
        await memory.add("Python support", namespace="ns", metadata=support)
        await memory.add("Python sales", namespace="ns", metadata=sales)

        results = await memory.search(
            "Python", namespace="ns", filter=MemorySearchFilter(categories=("billing", "support"))
        )
        assert len(results) == 2
        returned = {r.entry.metadata.categories[0] for r in results}
        assert returned == {"billing", "support"}

    @pytest.mark.asyncio
    async def test_categories_filter_on_recency_fallback(self, memory):
        billing = MemoryMetadata(source=MemorySource.MANUAL, categories=("billing",))
        support = MemoryMetadata(source=MemorySource.MANUAL, categories=("support",))
        await memory.add("User likes Python", namespace="ns", metadata=billing)
        await memory.add("User likes Rust", namespace="ns", metadata=support)

        # Zero token overlap forces the recency-fallback path.
        results = await memory.search(
            "completelydifferenttokens",
            namespace="ns",
            filter=MemorySearchFilter(categories=("billing",)),
        )
        assert len(results) == 1
        assert results[0].entry.metadata.categories == ("billing",)


class TestGet:
    @pytest.mark.asyncio
    async def test_get_existing(self, memory):
        entry = await memory.add("Test content here", namespace="ns")
        retrieved = await memory.get(entry.id)
        assert retrieved is not None
        assert retrieved.content == "Test content here"
        assert retrieved.metadata.source == MemorySource.MANUAL

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, memory):
        result = await memory.get("nonexistent-id")
        assert result is None


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_existing(self, memory):
        entry = await memory.add("Test content", namespace="ns")
        deleted = await memory.delete(entry.id)
        assert deleted is True
        assert await memory.get(entry.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory):
        deleted = await memory.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_removes_from_fts(self, memory):
        """Deleted entries should not appear in search results."""
        entry = await memory.add("Python programming language", namespace="ns")
        await memory.delete(entry.id)
        results = await memory.search("Python", namespace="ns")
        assert len(results) == 0


class TestClear:
    @pytest.mark.asyncio
    async def test_clear_namespace(self, memory):
        await memory.add("A content here", namespace="ns1")
        await memory.add("B content here", namespace="ns1")
        await memory.add("C content here", namespace="ns2")

        count = await memory.clear(namespace="ns1")
        assert count == 2

        # ns2 untouched
        results = await memory.search("content", namespace="ns2")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_clear_empty_namespace(self, memory):
        count = await memory.clear(namespace="empty")
        assert count == 0


class TestPersistence:
    @pytest.mark.asyncio
    async def test_file_based_persistence(self, tmp_path):
        db_path = tmp_path / "test_memory.db"

        # Write
        mem1 = SQLiteMemory(path=db_path)
        await mem1.add("Persistent fact about Python", namespace="ns")
        await mem1.close()

        # Read from fresh instance
        mem2 = SQLiteMemory(path=db_path)
        results = await mem2.search("Python", namespace="ns")
        assert len(results) == 1
        assert "Persistent" in results[0].entry.content
        await mem2.close()


class TestRecencyFallbackScore:
    @pytest.mark.asyncio
    async def test_recency_fallback_score_is_distinct_from_bm25(self, memory):
        """Recency-fallback entries must have score 0.0, not 1.0 (BM25 max)."""
        await memory.add("User likes Python", namespace="ns")
        # Force recency fallback: zero token overlap with stored content
        results = await memory.search("completelydifferenttokens", namespace="ns")
        assert len(results) == 1
        # Must NOT be 1.0 (indistinguishable from a perfect BM25 match)
        assert results[0].score == 0.0


class TestClose:
    @pytest.mark.asyncio
    async def test_close_is_safe(self, memory):
        await memory.close()
        # Double close should not raise

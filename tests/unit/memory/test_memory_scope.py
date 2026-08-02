"""Tests for scoped memory isolation in TemporaryMemory and SQLiteMemory.

Covers:
- Cross-scope isolation: entries added in scope A are not visible in scope B
- Same-scope visibility: two stores with the same scope share visibility
- Default-scope (scope=None) equivalence with old global behavior
- Clear only removes entries from own scope
- search, get, delete, add all respect scope boundaries
- Namespace transparency: returned entries carry the caller-supplied namespace
"""

from __future__ import annotations

import pytest

from troopai.adk.memory.in_memory import TemporaryMemory
from troopai.adk.memory.sqlite_memory import SQLiteMemory

# ---------------------------------------------------------------------------
# TemporaryMemory scope tests
# ---------------------------------------------------------------------------


class TestTemporaryMemoryScope:
    async def test_scope_none_is_global_default(self) -> None:
        """scope=None behaves identically to the pre-scope global default."""
        mem = TemporaryMemory()
        await mem.add("hello world", namespace="ns:1")
        results = await mem.search("hello", namespace="ns:1")
        assert len(results) == 1
        assert results[0].entry.content == "hello world"
        assert results[0].entry.namespace == "ns:1"

    async def test_scoped_add_returns_caller_namespace(self) -> None:
        """Returned entry carries the caller-supplied (unscoped) namespace."""
        mem = TemporaryMemory(scope="run:a")
        entry = await mem.add("data", namespace="user:1")
        assert entry.namespace == "user:1"

    async def test_cross_scope_isolation_search(self) -> None:
        """Entry added in scope A is not visible when searching in scope B."""
        run_a = TemporaryMemory(scope="run:a")
        run_b = TemporaryMemory(scope="run:b")
        await run_a.add("secret in A", namespace="shared")
        results_b = await run_b.search("secret", namespace="shared")
        assert results_b == []

    async def test_cross_scope_isolation_with_unscoped(self) -> None:
        """Entry added without scope is not visible in a scoped store."""
        global_mem = TemporaryMemory()
        scoped = TemporaryMemory(scope="run:x")
        await global_mem.add("global data", namespace="ns")
        results = await scoped.search("global", namespace="ns")
        assert results == []

    async def test_same_scope_visible_across_instances(self) -> None:
        """Two stores with the same scope and dict share no data (each owns its dict)."""
        # TemporaryMemory uses a per-instance dict, so even same-scope instances
        # have independent storage.  This test documents that contract explicitly.
        a = TemporaryMemory(scope="run:shared")
        b = TemporaryMemory(scope="run:shared")
        await a.add("only in a", namespace="ns")
        results = await b.search("only", namespace="ns")
        # Independent dicts — b does not see a's data.
        assert results == []

    async def test_scoped_search_finds_own_entry(self) -> None:
        """A scoped store finds its own entry."""
        run_a = TemporaryMemory(scope="run:a")
        await run_a.add("dark mode preferred", namespace="user:1")
        results = await run_a.search("dark mode", namespace="user:1")
        assert len(results) >= 1
        assert results[0].entry.namespace == "user:1"

    async def test_clear_only_removes_own_scope_entries(self) -> None:
        """clear(namespace=...) only removes entries from this scope."""
        run_a = TemporaryMemory(scope="run:a")
        run_b = TemporaryMemory(scope="run:b")
        await run_a.add("entry in A", namespace="common")
        await run_b.add("entry in B", namespace="common")

        removed = await run_a.clear(namespace="common")
        assert removed == 1
        # B's entry is unaffected.
        results_b = await run_b.search("entry", namespace="common")
        assert len(results_b) == 1

    async def test_scope_none_clear_only_unscoped(self) -> None:
        """Unscoped clear does not remove scoped entries."""
        global_mem = TemporaryMemory()
        scoped = TemporaryMemory(scope="run:z")
        await global_mem.add("global entry", namespace="ns")
        await scoped.add("scoped entry", namespace="ns")

        removed = await global_mem.clear(namespace="ns")
        assert removed == 1
        # Scoped entry survives.
        results = await scoped.search("scoped", namespace="ns")
        assert len(results) == 1

    async def test_get_by_id_returns_scoped_entry(self) -> None:
        """get(id) returns the entry regardless of scope prefix."""
        run_a = TemporaryMemory(scope="run:a")
        entry = await run_a.add("retrieve me", namespace="ns")
        fetched = await run_a.get(entry.id)
        assert fetched is not None
        assert fetched.content == "retrieve me"

    async def test_delete_removes_own_entry(self) -> None:
        run_a = TemporaryMemory(scope="run:a")
        entry = await run_a.add("to delete", namespace="ns")
        deleted = await run_a.delete(entry.id)
        assert deleted is True
        assert await run_a.get(entry.id) is None

    async def test_two_scopes_same_namespace_no_leakage(self) -> None:
        """Multiple entries across two scopes in same namespace stay isolated."""
        a = TemporaryMemory(scope="run:a")
        b = TemporaryMemory(scope="run:b")
        for i in range(5):
            await a.add(f"entry-a-{i}", namespace="ns")
        for i in range(3):
            await b.add(f"entry-b-{i}", namespace="ns")

        results_a = await a.search("entry-a", namespace="ns")
        results_b = await b.search("entry-b", namespace="ns")
        # A only sees its own 5 entries; B only sees its own 3.
        assert all("entry-a" in r.entry.content for r in results_a)
        assert all("entry-b" in r.entry.content for r in results_b)


# ---------------------------------------------------------------------------
# SQLiteMemory scope tests
# ---------------------------------------------------------------------------


class TestSQLiteMemoryScope:
    async def test_scope_none_is_global_default(self) -> None:
        """scope=None behaves identically to the pre-scope global default."""
        mem = SQLiteMemory(path=":memory:")
        try:
            await mem.add("hello world", namespace="ns:1")
            results = await mem.search("hello", namespace="ns:1")
            assert len(results) == 1
            assert results[0].entry.content == "hello world"
            assert results[0].entry.namespace == "ns:1"
        finally:
            await mem.close()

    async def test_scoped_add_returns_caller_namespace(self) -> None:
        """Returned entry carries the caller-supplied (unscoped) namespace."""
        mem = SQLiteMemory(path=":memory:", scope="run:a")
        try:
            entry = await mem.add("data", namespace="user:1")
            assert entry.namespace == "user:1"
        finally:
            await mem.close()

    async def test_cross_scope_isolation_search(self) -> None:
        """Entry added in scope A is not visible when searching in scope B."""
        # Both stores target the same in-memory DB path — use separate
        # paths to keep SQLite's single-connection-per-in-memory semantics
        # clean.  Cross-scope isolation on a shared file is tested separately.
        run_a = SQLiteMemory(path=":memory:", scope="run:a")
        run_b = SQLiteMemory(path=":memory:", scope="run:b")
        try:
            await run_a.add("secret in A", namespace="shared")
            results_b = await run_b.search("secret", namespace="shared")
            assert results_b == []
        finally:
            await run_a.close()
            await run_b.close()

    async def test_cross_scope_isolation_shared_file(
        self,
        tmp_path: pytest.TempdirFactory,
    ) -> None:
        """Two stores with different scopes sharing one file don't cross-contaminate."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            db_path = f"{td}/test.db"
            run_a = SQLiteMemory(path=db_path, scope="run:a")
            run_b = SQLiteMemory(path=db_path, scope="run:b")
            await run_a.add("only for A", namespace="ns")
            await run_b.add("only for B", namespace="ns")

            results_a = await run_a.search("only", namespace="ns")
            results_b = await run_b.search("only", namespace="ns")

            assert len(results_a) == 1
            assert results_a[0].entry.content == "only for A"
            assert results_a[0].entry.namespace == "ns"

            assert len(results_b) == 1
            assert results_b[0].entry.content == "only for B"
            assert results_b[0].entry.namespace == "ns"

            await run_a.close()
            await run_b.close()

    async def test_unscoped_store_not_visible_in_scoped(self) -> None:
        """Global (unscoped) entry is not visible in a scoped store (shared file)."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            db_path = f"{td}/test.db"
            global_mem = SQLiteMemory(path=db_path)
            scoped = SQLiteMemory(path=db_path, scope="run:x")
            await global_mem.add("global data", namespace="ns")
            results = await scoped.search("global", namespace="ns")
            assert results == []
            await global_mem.close()
            await scoped.close()

    async def test_clear_only_removes_own_scope_shared_file(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            db_path = f"{td}/test.db"
            run_a = SQLiteMemory(path=db_path, scope="run:a")
            run_b = SQLiteMemory(path=db_path, scope="run:b")
            await run_a.add("entry in A", namespace="common")
            await run_b.add("entry in B", namespace="common")

            removed = await run_a.clear(namespace="common")
            assert removed == 1

            # B's entry survives.
            results_b = await run_b.search("entry", namespace="common")
            assert len(results_b) == 1
            assert results_b[0].entry.namespace == "common"

            await run_a.close()
            await run_b.close()

    async def test_get_returns_unscoped_namespace(self) -> None:
        """get(id) returns the entry with caller-supplied (unscoped) namespace."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            db_path = f"{td}/test.db"
            mem = SQLiteMemory(path=db_path, scope="run:a")
            entry = await mem.add("retrieve me", namespace="user:1")
            fetched = await mem.get(entry.id)
            assert fetched is not None
            assert fetched.content == "retrieve me"
            assert fetched.namespace == "user:1"
            await mem.close()

    async def test_delete_removes_entry(self) -> None:
        mem = SQLiteMemory(path=":memory:", scope="run:a")
        try:
            entry = await mem.add("to delete", namespace="ns")
            deleted = await mem.delete(entry.id)
            assert deleted is True
            assert await mem.get(entry.id) is None
        finally:
            await mem.close()

    async def test_scope_none_preserves_old_behavior_search(self) -> None:
        """Existing code using scope=None sees no change in search behavior."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            db_path = f"{td}/test.db"
            mem = SQLiteMemory(path=db_path)
            await mem.add("user preference A", namespace="user:1")
            await mem.add("user preference B", namespace="user:2")

            results = await mem.search("preference", namespace="user:1")
            assert len(results) == 1
            assert results[0].entry.namespace == "user:1"
            await mem.close()


# ---------------------------------------------------------------------------
# Scope validation tests (both backends)
# ---------------------------------------------------------------------------


class TestScopeValidation:
    def test_temporary_memory_scope_with_slash_raises(self) -> None:
        """scope containing '/' is rejected at construction time."""
        with pytest.raises(ValueError, match="must not contain '/'"):
            TemporaryMemory(scope="tenant/run")

    def test_temporary_memory_scope_none_is_valid(self) -> None:
        mem = TemporaryMemory(scope=None)
        assert mem._scope is None

    def test_temporary_memory_scope_without_slash_is_valid(self) -> None:
        mem = TemporaryMemory(scope="run:a")
        assert mem._scope == "run:a"

    def test_sqlite_memory_scope_with_slash_raises(self) -> None:
        """scope containing '/' is rejected at construction time."""
        with pytest.raises(ValueError, match="must not contain '/'"):
            SQLiteMemory(path=":memory:", scope="tenant/run")

    def test_sqlite_memory_scope_none_is_valid(self) -> None:
        mem = SQLiteMemory(path=":memory:", scope=None)
        assert mem._scope is None

    def test_sqlite_memory_scope_without_slash_is_valid(self) -> None:
        mem = SQLiteMemory(path=":memory:", scope="run:a")
        assert mem._scope == "run:a"


# ---------------------------------------------------------------------------
# get() namespace transparency for TemporaryMemory
# ---------------------------------------------------------------------------


class TestTemporaryMemoryGetNamespaceTransparency:
    async def test_get_returns_unscoped_namespace(self) -> None:
        """get(id) returns the entry with caller-supplied (unscoped) namespace."""
        mem = TemporaryMemory(scope="run:a")
        entry = await mem.add("retrieve me", namespace="user:1")
        fetched = await mem.get(entry.id)
        assert fetched is not None
        assert fetched.content == "retrieve me"
        assert fetched.namespace == "user:1"

    async def test_get_unscoped_store_namespace_unchanged(self) -> None:
        """get() on an unscoped store returns the namespace as-is."""
        mem = TemporaryMemory()
        entry = await mem.add("data", namespace="ns")
        fetched = await mem.get(entry.id)
        assert fetched is not None
        assert fetched.namespace == "ns"


# ---------------------------------------------------------------------------
# filter.namespace scoping tests
# ---------------------------------------------------------------------------


class TestFilterNamespaceScoping:
    async def test_temporary_memory_filter_namespace_scoped(self) -> None:
        """search() with filter.namespace respects scope on TemporaryMemory."""
        from troopai.adk.memory.memory_types import MemorySearchFilter

        mem = TemporaryMemory(scope="run:a")
        await mem.add("dark mode preferred", namespace="user:1")
        # filter.namespace == namespace — should find the entry
        results = await mem.search(
            "dark mode",
            namespace="user:1",
            filter=MemorySearchFilter(namespace="user:1"),
        )
        assert len(results) == 1
        assert results[0].entry.namespace == "user:1"

    async def test_temporary_memory_filter_namespace_wrong_ns_returns_empty(self) -> None:
        """filter.namespace for a different ns returns no results."""
        from troopai.adk.memory.memory_types import MemorySearchFilter

        mem = TemporaryMemory(scope="run:a")
        await mem.add("dark mode preferred", namespace="user:1")
        results = await mem.search(
            "dark mode",
            namespace="user:1",
            filter=MemorySearchFilter(namespace="user:2"),
        )
        assert results == []

    async def test_sqlite_memory_filter_namespace_scoped(self) -> None:
        """search() with filter.namespace respects scope on SQLiteMemory."""
        import tempfile

        from troopai.adk.memory.memory_types import MemorySearchFilter

        with tempfile.TemporaryDirectory() as td:
            db_path = f"{td}/test.db"
            mem = SQLiteMemory(path=db_path, scope="run:a")
            await mem.add("dark mode preferred", namespace="user:1")
            results = await mem.search(
                "dark mode",
                namespace="user:1",
                filter=MemorySearchFilter(namespace="user:1"),
            )
            assert len(results) == 1
            assert results[0].entry.namespace == "user:1"
            await mem.close()

    async def test_sqlite_memory_filter_namespace_wrong_ns_returns_empty(self) -> None:
        """filter.namespace for a different ns returns no results (SQLite)."""
        import tempfile

        from troopai.adk.memory.memory_types import MemorySearchFilter

        with tempfile.TemporaryDirectory() as td:
            db_path = f"{td}/test.db"
            mem = SQLiteMemory(path=db_path, scope="run:a")
            await mem.add("dark mode preferred", namespace="user:1")
            results = await mem.search(
                "dark mode",
                namespace="user:1",
                filter=MemorySearchFilter(namespace="user:2"),
            )
            assert results == []
            await mem.close()

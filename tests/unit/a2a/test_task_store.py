"""Tests for :mod:`troopai.adk.a2a.task_store`.

Covers:
* :class:`TaskStore` protocol — ``InMemoryTaskStore`` and
  ``SQLiteTaskStore`` both satisfy it at runtime (``isinstance`` check).
* :class:`InMemoryTaskStore` — get/save/delete/list_by_status, all
  CRUD operations, terminal vs non-terminal filtering.
* :class:`SQLiteTaskStore` — same operations against a real aiosqlite
  in-memory database (no disk I/O, deterministic, no temp files).
* ``recover_on_startup`` — non-terminal rows become FAILED; terminal
  rows are untouched; count returned is accurate.
* Retention sweep — TTL and max-rows bounds trim terminal rows.
* Restart-survival integration — a task saved to a file-based SQLite
  store is queryable from a fresh ``SQLiteTaskStore`` instance on the
  same path (kill-and-restart simulation).
* Serialization round-trip — protobuf ``Task`` serializes and
  deserializes without data loss.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

# Skip the entire module if a2a-sdk is not installed.
pytest.importorskip("a2a.server.agent_execution")

from a2a.types import Task, TaskState, TaskStatus

from troopai.adk.a2a.task_store import (
    InMemoryTaskStore,
    SQLiteTaskStore,
    TaskStore,
    _deserialize,
    _serialize,
)
from troopai.adk.databases.connections.sqlite import SQLiteDatabaseConnection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task(
    task_id: str,
    state: TaskState = TaskState.TASK_STATE_SUBMITTED,
    context_id: str = "ctx1",
) -> Task:
    return Task(
        id=task_id,
        context_id=context_id,
        status=TaskStatus(state=state),
    )


# ---------------------------------------------------------------------------
# TaskStore protocol — both classes satisfy it at runtime
# ---------------------------------------------------------------------------


class TestTaskStoreProtocol:
    def test_in_memory_satisfies_protocol(self) -> None:
        assert isinstance(InMemoryTaskStore(), TaskStore)

    def test_sqlite_satisfies_protocol(self) -> None:
        db = SQLiteDatabaseConnection(path=":memory:")
        store = SQLiteTaskStore(db)
        assert isinstance(store, TaskStore)


# ---------------------------------------------------------------------------
# InMemoryTaskStore
# ---------------------------------------------------------------------------


class TestInMemoryTaskStore:
    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self) -> None:
        store = InMemoryTaskStore()
        assert await store.get("no-such-id") is None

    @pytest.mark.asyncio
    async def test_save_and_get_roundtrip(self) -> None:
        store = InMemoryTaskStore()
        t = _task("t1", TaskState.TASK_STATE_WORKING)
        await store.save(t)
        result = await store.get("t1")
        assert result is not None
        assert result.id == "t1"
        assert result.status.state == TaskState.TASK_STATE_WORKING

    @pytest.mark.asyncio
    async def test_save_overwrites_existing(self) -> None:
        store = InMemoryTaskStore()
        await store.save(_task("t1", TaskState.TASK_STATE_WORKING))
        await store.save(_task("t1", TaskState.TASK_STATE_COMPLETED))
        result = await store.get("t1")
        assert result is not None
        assert result.status.state == TaskState.TASK_STATE_COMPLETED

    @pytest.mark.asyncio
    async def test_delete_removes_task(self) -> None:
        store = InMemoryTaskStore()
        await store.save(_task("t1"))
        await store.delete("t1")
        assert await store.get("t1") is None

    @pytest.mark.asyncio
    async def test_delete_noop_for_unknown(self) -> None:
        store = InMemoryTaskStore()
        # Should not raise.
        await store.delete("ghost")

    @pytest.mark.asyncio
    async def test_list_by_status_terminal_filters_correctly(self) -> None:
        store = InMemoryTaskStore()
        await store.save(_task("t1", TaskState.TASK_STATE_COMPLETED))
        await store.save(_task("t2", TaskState.TASK_STATE_WORKING))
        await store.save(_task("t3", TaskState.TASK_STATE_FAILED))
        await store.save(_task("t4", TaskState.TASK_STATE_SUBMITTED))
        terminal = await store.list_by_status(terminal=True)
        non_terminal = await store.list_by_status(terminal=False)
        assert {t.id for t in terminal} == {"t1", "t3"}
        assert {t.id for t in non_terminal} == {"t2", "t4"}

    @pytest.mark.asyncio
    async def test_list_by_status_empty_store(self) -> None:
        store = InMemoryTaskStore()
        assert await store.list_by_status(terminal=True) == []
        assert await store.list_by_status(terminal=False) == []

    @pytest.mark.asyncio
    async def test_all_terminal_states_classified(self) -> None:
        store = InMemoryTaskStore()
        for state in [
            TaskState.TASK_STATE_COMPLETED,
            TaskState.TASK_STATE_FAILED,
            TaskState.TASK_STATE_CANCELED,
            TaskState.TASK_STATE_REJECTED,
        ]:
            await store.save(_task(f"t-{state}", state))
        terminal = await store.list_by_status(terminal=True)
        assert len(terminal) == 4

    @pytest.mark.asyncio
    async def test_non_terminal_includes_working_and_submitted(self) -> None:
        store = InMemoryTaskStore()
        await store.save(_task("t-w", TaskState.TASK_STATE_WORKING))
        await store.save(_task("t-s", TaskState.TASK_STATE_SUBMITTED))
        await store.save(_task("t-i", TaskState.TASK_STATE_INPUT_REQUIRED))
        non_terminal = await store.list_by_status(terminal=False)
        assert len(non_terminal) == 3

    def test_default_max_tasks_is_bounded(self) -> None:
        # Cost-conservative invariant: the default store is bounded, never
        # unbounded.
        assert InMemoryTaskStore()._max_tasks > 0

    def test_negative_max_tasks_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_tasks"):
            InMemoryTaskStore(max_tasks=-1)

    @pytest.mark.asyncio
    async def test_evicts_least_recently_saved_over_capacity(self) -> None:
        # Regression: the in-memory dict grew without bound. Once the cap is
        # exceeded it must evict the least-recently-saved task.
        store = InMemoryTaskStore(max_tasks=2)
        await store.save(_task("t1"))
        await store.save(_task("t2"))
        await store.save(_task("t3"))  # over the cap -> evict t1
        assert await store.get("t1") is None
        assert await store.get("t2") is not None
        assert await store.get("t3") is not None

    @pytest.mark.asyncio
    async def test_resaving_task_refreshes_its_recency(self) -> None:
        # Re-saving t1 moves it to the most-recently-saved end, so the next
        # eviction drops t2 (now the oldest), not t1.
        store = InMemoryTaskStore(max_tasks=2)
        await store.save(_task("t1"))
        await store.save(_task("t2"))
        await store.save(_task("t1", TaskState.TASK_STATE_WORKING))  # refresh t1
        await store.save(_task("t3"))  # evicts t2, the oldest
        assert await store.get("t2") is None
        assert await store.get("t1") is not None
        assert await store.get("t3") is not None

    @pytest.mark.asyncio
    async def test_max_tasks_zero_disables_bound(self) -> None:
        # Explicit opt-out: max_tasks=0 keeps everything.
        store = InMemoryTaskStore(max_tasks=0)
        for i in range(5):
            await store.save(_task(f"t{i}"))
        assert len(await store.list_by_status(terminal=False)) == 5


# ---------------------------------------------------------------------------
# SQLiteTaskStore — uses aiosqlite in-memory DB (no disk I/O)
# ---------------------------------------------------------------------------


@pytest.fixture
async def sqlite_store() -> SQLiteTaskStore:  # type: ignore[misc]
    db = SQLiteDatabaseConnection(path=":memory:")
    store = SQLiteTaskStore(db, ttl_seconds=0, max_terminal_rows=0)
    # _ensure_ready is called lazily by every public method; no explicit
    # setup call needed. The fixture omits it to verify the lazy path too.
    yield store  # type: ignore[misc]
    await db.close()


class TestSQLiteTaskStore:
    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, sqlite_store: SQLiteTaskStore) -> None:
        assert await sqlite_store.get("no-such-id") is None

    @pytest.mark.asyncio
    async def test_save_and_get_roundtrip(self, sqlite_store: SQLiteTaskStore) -> None:
        t = _task("t1", TaskState.TASK_STATE_WORKING)
        await sqlite_store.save(t)
        result = await sqlite_store.get("t1")
        assert result is not None
        assert result.id == "t1"
        assert result.status.state == TaskState.TASK_STATE_WORKING

    @pytest.mark.asyncio
    async def test_save_overwrites_existing(self, sqlite_store: SQLiteTaskStore) -> None:
        await sqlite_store.save(_task("t1", TaskState.TASK_STATE_WORKING))
        await sqlite_store.save(_task("t1", TaskState.TASK_STATE_COMPLETED))
        result = await sqlite_store.get("t1")
        assert result is not None
        assert result.status.state == TaskState.TASK_STATE_COMPLETED

    @pytest.mark.asyncio
    async def test_delete_removes_task(self, sqlite_store: SQLiteTaskStore) -> None:
        await sqlite_store.save(_task("t1"))
        await sqlite_store.delete("t1")
        assert await sqlite_store.get("t1") is None

    @pytest.mark.asyncio
    async def test_delete_noop_for_unknown(self, sqlite_store: SQLiteTaskStore) -> None:
        await sqlite_store.delete("ghost")

    @pytest.mark.asyncio
    async def test_list_by_status_terminal_filters_correctly(self, sqlite_store: SQLiteTaskStore) -> None:
        await sqlite_store.save(_task("t1", TaskState.TASK_STATE_COMPLETED))
        await sqlite_store.save(_task("t2", TaskState.TASK_STATE_WORKING))
        await sqlite_store.save(_task("t3", TaskState.TASK_STATE_FAILED))
        terminal = await sqlite_store.list_by_status(terminal=True)
        non_terminal = await sqlite_store.list_by_status(terminal=False)
        assert {t.id for t in terminal} == {"t1", "t3"}
        assert {t.id for t in non_terminal} == {"t2"}

    @pytest.mark.asyncio
    async def test_list_by_status_empty(self, sqlite_store: SQLiteTaskStore) -> None:
        assert await sqlite_store.list_by_status(terminal=True) == []
        assert await sqlite_store.list_by_status(terminal=False) == []


# ---------------------------------------------------------------------------
# SQLiteTaskStore — lazy schema initialization (_ensure_ready)
# ---------------------------------------------------------------------------


class TestLazySchemaInit:
    @pytest.mark.asyncio
    async def test_save_without_prior_ensure_schema_succeeds(self) -> None:
        """Every public method initializes the schema lazily on first use."""
        db = SQLiteDatabaseConnection(path=":memory:")
        try:
            store = SQLiteTaskStore(db, ttl_seconds=0, max_terminal_rows=0)
            # No _ensure_schema / _ensure_ready called explicitly.
            await store.save(_task("t1", TaskState.TASK_STATE_WORKING))
            result = await store.get("t1")
            assert result is not None
            assert result.id == "t1"
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_get_without_prior_ensure_schema_returns_none(self) -> None:
        db = SQLiteDatabaseConnection(path=":memory:")
        try:
            store = SQLiteTaskStore(db, ttl_seconds=0, max_terminal_rows=0)
            # get() on a fresh store must not raise OperationalError.
            result = await store.get("missing")
            assert result is None
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_list_by_status_without_prior_ensure_schema(self) -> None:
        db = SQLiteDatabaseConnection(path=":memory:")
        try:
            store = SQLiteTaskStore(db, ttl_seconds=0, max_terminal_rows=0)
            rows = await store.list_by_status(terminal=True)
            assert rows == []
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_delete_without_prior_ensure_schema(self) -> None:
        db = SQLiteDatabaseConnection(path=":memory:")
        try:
            store = SQLiteTaskStore(db, ttl_seconds=0, max_terminal_rows=0)
            # delete() on a fresh store must not raise OperationalError.
            await store.delete("ghost")
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# recover_on_startup
# ---------------------------------------------------------------------------


class TestRecoverOnStartup:
    @pytest.mark.asyncio
    async def test_non_terminal_tasks_become_failed(self) -> None:
        db = SQLiteDatabaseConnection(path=":memory:")
        store = SQLiteTaskStore(db, ttl_seconds=0, max_terminal_rows=0)
        try:
            # Pre-populate with non-terminal rows. save() initializes the
            # schema lazily — no explicit setup call needed.
            for state in [
                TaskState.TASK_STATE_WORKING,
                TaskState.TASK_STATE_SUBMITTED,
                TaskState.TASK_STATE_INPUT_REQUIRED,
            ]:
                await store.save(_task(f"t-{state}", state))
            count = await store.recover_on_startup()
            assert count == 3
            for state in [
                TaskState.TASK_STATE_WORKING,
                TaskState.TASK_STATE_SUBMITTED,
                TaskState.TASK_STATE_INPUT_REQUIRED,
            ]:
                recovered = await store.get(f"t-{state}")
                assert recovered is not None
                assert recovered.status.state == TaskState.TASK_STATE_FAILED, (
                    f"Expected FAILED for state {state}, got {recovered.status.state}"
                )
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_terminal_tasks_untouched(self) -> None:
        db = SQLiteDatabaseConnection(path=":memory:")
        store = SQLiteTaskStore(db, ttl_seconds=0, max_terminal_rows=0)
        try:
            await store.save(_task("t-completed", TaskState.TASK_STATE_COMPLETED))
            await store.save(_task("t-failed", TaskState.TASK_STATE_FAILED))
            count = await store.recover_on_startup()
            assert count == 0
            result = await store.get("t-completed")
            assert result is not None
            assert result.status.state == TaskState.TASK_STATE_COMPLETED
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_empty_store_returns_zero(self) -> None:
        db = SQLiteDatabaseConnection(path=":memory:")
        try:
            store = SQLiteTaskStore(db, ttl_seconds=0, max_terminal_rows=0)
            count = await store.recover_on_startup()
            assert count == 0
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_recover_on_startup_creates_schema(self) -> None:
        # recover_on_startup must call _ensure_schema so callers don't
        # need a separate setup step.
        db = SQLiteDatabaseConnection(path=":memory:")
        try:
            store = SQLiteTaskStore(db, ttl_seconds=0, max_terminal_rows=0)
            # Must not raise even without prior _ensure_schema call.
            count = await store.recover_on_startup()
            assert count == 0
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# Retention sweep
# ---------------------------------------------------------------------------


class TestRetentionSweep:
    @pytest.mark.asyncio
    async def test_ttl_sweep_deletes_old_terminal_rows(self) -> None:
        db = SQLiteDatabaseConnection(path=":memory:")
        store = SQLiteTaskStore(db, ttl_seconds=3600, max_terminal_rows=0)
        try:
            # Prime the lazy schema init by saving a task first, then
            # manually insert a backdated row to test the TTL sweep.
            await store._ensure_ready()
            old_time = time.time() - 7200  # 2 h ago, older than TTL=1 h
            t = _task("old-t", TaskState.TASK_STATE_COMPLETED)
            from troopai.adk.a2a.task_store import _serialize as _s

            task_json = _s(t)
            async with db.connect() as conn:
                await conn.execute(
                    "INSERT INTO a2a_tasks (task_id, status_state, task_json, updated_at) VALUES (?, ?, ?, ?)",
                    ("old-t", TaskState.TASK_STATE_COMPLETED, task_json, old_time),
                )
                await conn.commit()
            # save() a fresh task to trigger the sweep.
            await store.save(_task("new-t", TaskState.TASK_STATE_COMPLETED))
            # Old task should have been swept away.
            assert await store.get("old-t") is None
            # New task must still be there.
            assert await store.get("new-t") is not None
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_max_rows_sweep_keeps_newest(self) -> None:
        db = SQLiteDatabaseConnection(path=":memory:")
        # Allow only 2 terminal rows.
        store = SQLiteTaskStore(db, ttl_seconds=0, max_terminal_rows=2)
        try:
            await store._ensure_ready()
            # Insert 3 completed tasks with staggered timestamps.
            for i in range(3):
                old_time = time.time() - (10 - i)  # t-0 oldest, t-2 newest
                t = _task(f"t-{i}", TaskState.TASK_STATE_COMPLETED)
                from troopai.adk.a2a.task_store import _serialize as _s

                task_json = _s(t)
                async with db.connect() as conn:
                    await conn.execute(
                        "INSERT INTO a2a_tasks (task_id, status_state, task_json, updated_at) VALUES (?, ?, ?, ?)",
                        (f"t-{i}", TaskState.TASK_STATE_COMPLETED, task_json, old_time),
                    )
                    await conn.commit()
            # Trigger sweep by saving a fourth task.
            await store.save(_task("t-3", TaskState.TASK_STATE_COMPLETED))
            terminal = await store.list_by_status(terminal=True)
            assert len(terminal) == 2, f"Expected 2 terminal rows, got {len(terminal)}: {[t.id for t in terminal]}"
            # The oldest (t-0) should have been deleted first.
            assert await store.get("t-0") is None
        finally:
            await db.close()

    @pytest.mark.asyncio
    async def test_sweep_disabled_when_zero(self) -> None:
        db = SQLiteDatabaseConnection(path=":memory:")
        store = SQLiteTaskStore(db, ttl_seconds=0, max_terminal_rows=0)
        try:
            # No setup call — lazy init via save().
            for i in range(5):
                await store.save(_task(f"t-{i}", TaskState.TASK_STATE_COMPLETED))
            terminal = await store.list_by_status(terminal=True)
            assert len(terminal) == 5
        finally:
            await db.close()


# ---------------------------------------------------------------------------
# Kill-and-restart: file-based survival test
# ---------------------------------------------------------------------------


class TestRestartSurvival:
    @pytest.mark.asyncio
    async def test_task_survives_new_store_instance(self, tmp_path: Path) -> None:
        """A task saved to a file DB is queryable from a fresh store instance.

        This simulates a process kill-and-restart: the first SQLiteTaskStore
        is discarded (simulating the old process) and a brand-new instance
        on the same file (simulating the restarted process) can read back
        the persisted task.
        """
        db_path = tmp_path / "a2a_tasks.db"

        # "Process 1": save a task. save() lazily initializes the schema.
        db1 = SQLiteDatabaseConnection(path=db_path)
        store1 = SQLiteTaskStore(db1, ttl_seconds=0, max_terminal_rows=0)
        await store1.save(_task("survivor", TaskState.TASK_STATE_WORKING))
        await db1.close()
        del store1, db1

        # "Process 2": open a fresh store on the same file.
        # get() initializes the schema lazily if needed.
        db2 = SQLiteDatabaseConnection(path=db_path)
        store2 = SQLiteTaskStore(db2, ttl_seconds=0, max_terminal_rows=0)

        result = await store2.get("survivor")
        assert result is not None, "Task did not survive restart"
        assert result.id == "survivor"
        assert result.status.state == TaskState.TASK_STATE_WORKING
        await db2.close()

    @pytest.mark.asyncio
    async def test_recovery_marks_non_terminal_failed_after_restart(self, tmp_path: Path) -> None:
        """recover_on_startup marks tasks FAILED after a simulated restart."""
        db_path = tmp_path / "a2a_recovery.db"

        # "Process 1": save a working task (process was killed mid-run).
        db1 = SQLiteDatabaseConnection(path=db_path)
        store1 = SQLiteTaskStore(db1, ttl_seconds=0, max_terminal_rows=0)
        await store1.save(_task("incomplete", TaskState.TASK_STATE_WORKING))
        await db1.close()
        del store1, db1

        # "Process 2": recover on startup.
        db2 = SQLiteDatabaseConnection(path=db_path)
        store2 = SQLiteTaskStore(db2, ttl_seconds=0, max_terminal_rows=0)
        count = await store2.recover_on_startup()
        assert count == 1

        recovered = await store2.get("incomplete")
        assert recovered is not None
        assert recovered.status.state == TaskState.TASK_STATE_FAILED, (
            "Non-terminal task must be marked FAILED after restart recovery"
        )
        await db2.close()


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    def test_submitted_task_round_trips(self) -> None:
        t = _task("rt1", TaskState.TASK_STATE_SUBMITTED, context_id="ctx-rt")
        restored = _deserialize(_serialize(t))
        assert restored.id == "rt1"
        assert restored.context_id == "ctx-rt"
        assert restored.status.state == TaskState.TASK_STATE_SUBMITTED

    def test_all_terminal_states_survive_serialization(self) -> None:
        for state in [
            TaskState.TASK_STATE_COMPLETED,
            TaskState.TASK_STATE_FAILED,
            TaskState.TASK_STATE_CANCELED,
            TaskState.TASK_STATE_REJECTED,
        ]:
            t = _task(f"s-{state}", state)
            restored = _deserialize(_serialize(t))
            assert restored.status.state == state, f"State {state} did not survive serialization"

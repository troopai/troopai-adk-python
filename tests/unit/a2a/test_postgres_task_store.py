"""Tests for :class:`PostgresTaskStore` against a live ephemeral Postgres."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("a2a.types")
pytest.importorskip("psycopg_pool")
pytest.importorskip("pytest_postgresql")

from a2a.types import Task, TaskState, TaskStatus
from pytest_postgresql.factories import postgresql, postgresql_proc

from troopai.adk.a2a.postgres_task_store import PostgresTaskStore
from troopai.adk.a2a.task_store import TaskStore

pytestmark = pytest.mark.postgres

postgresql_proc_a2a = postgresql_proc()
postgresql_a2a = postgresql("postgresql_proc_a2a")


def _conninfo(pg: Any) -> str:
    info = pg.info
    parts = [f"dbname={info.dbname}", f"user={info.user}", f"host={info.host}", f"port={info.port}"]
    if info.password is not None and len(info.password) > 0:
        parts.append(f"password={info.password}")
    return " ".join(parts)


def _task(task_id: str, state: TaskState = TaskState.TASK_STATE_SUBMITTED) -> Task:
    return Task(id=task_id, context_id="ctx1", status=TaskStatus(state=state))


def test_satisfies_task_store_protocol() -> None:
    assert isinstance(PostgresTaskStore("postgresql://x/y"), TaskStore)


def test_empty_conninfo_rejected() -> None:
    with pytest.raises(ValueError):
        PostgresTaskStore("")


async def test_save_get_round_trip(postgresql_a2a: Any) -> None:
    store = PostgresTaskStore(_conninfo(postgresql_a2a))
    try:
        await store.save(_task("t1"))
        got = await store.get("t1")
        assert got is not None
        assert got.id == "t1"
    finally:
        await store.close()


async def test_get_missing_returns_none(postgresql_a2a: Any) -> None:
    store = PostgresTaskStore(_conninfo(postgresql_a2a))
    try:
        assert await store.get("nope") is None
    finally:
        await store.close()


async def test_delete_removes_task(postgresql_a2a: Any) -> None:
    store = PostgresTaskStore(_conninfo(postgresql_a2a))
    try:
        await store.save(_task("t1"))
        await store.delete("t1")
        assert await store.get("t1") is None
    finally:
        await store.close()


async def test_list_by_status_filters(postgresql_a2a: Any) -> None:
    store = PostgresTaskStore(_conninfo(postgresql_a2a))
    try:
        await store.save(_task("done", state=TaskState.TASK_STATE_COMPLETED))
        await store.save(_task("working", state=TaskState.TASK_STATE_WORKING))
        assert {t.id for t in await store.list_by_status(terminal=True)} == {"done"}
        assert {t.id for t in await store.list_by_status(terminal=False)} == {"working"}
    finally:
        await store.close()


async def test_recover_on_startup_marks_non_terminal_failed(postgresql_a2a: Any) -> None:
    store = PostgresTaskStore(_conninfo(postgresql_a2a))
    try:
        await store.save(_task("running", state=TaskState.TASK_STATE_WORKING))
        await store.save(_task("done", state=TaskState.TASK_STATE_COMPLETED))
        assert await store.recover_on_startup() == 1
        recovered = await store.get("running")
        assert recovered is not None
        assert recovered.status.state == TaskState.TASK_STATE_FAILED
        done = await store.get("done")
        assert done is not None
        assert done.status.state == TaskState.TASK_STATE_COMPLETED
    finally:
        await store.close()


async def test_restart_survival_across_instances(postgresql_a2a: Any) -> None:
    conninfo = _conninfo(postgresql_a2a)
    store = PostgresTaskStore(conninfo)
    await store.save(_task("t1", state=TaskState.TASK_STATE_COMPLETED))
    await store.close()
    fresh = PostgresTaskStore(conninfo)
    try:
        got = await fresh.get("t1")
        assert got is not None
        assert got.id == "t1"
    finally:
        await fresh.close()


async def test_max_rows_retention_caps_terminal_tasks(postgresql_a2a: Any) -> None:
    store = PostgresTaskStore(_conninfo(postgresql_a2a), ttl_seconds=0, max_terminal_rows=2)
    try:
        for index in range(5):
            await store.save(_task(f"t{index}", state=TaskState.TASK_STATE_COMPLETED))
        assert len(await store.list_by_status(terminal=True)) <= 2
    finally:
        await store.close()

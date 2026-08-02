from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("psycopg_pool")

from troopai.adk.a2a.postgres_task_store import PostgresTaskStore
from troopai.adk.graphs.checkpointers.postgres import PostgresCheckpointer
from troopai.adk.session.postgres_multi_sessions import PostgresMultiSessions
from troopai.adk.swarms.checkpointers.postgres import PostgresSwarmCheckpointer


class _FakePool:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _pool_owners() -> list[Any]:
    owners: list[Any] = [
        pytest.param("a2a-task-store", lambda: PostgresTaskStore("postgresql://localhost/test"), id="a2a-task-store"),
        pytest.param(
            "graph-checkpointer",
            lambda: PostgresCheckpointer("postgresql://localhost/test"),
            id="graph-checkpointer",
        ),
        pytest.param(
            "session-manager",
            lambda: PostgresMultiSessions("postgresql://localhost/test"),
            id="session-manager",
        ),
        pytest.param(
            "swarm-checkpointer",
            lambda: PostgresSwarmCheckpointer("postgresql://localhost/test"),
            id="swarm-checkpointer",
        ),
    ]
    try:
        from troopai.adk.memory.stores.pgvector import PgVectorStore
    except ImportError:
        return owners
    owners.append(
        pytest.param(
            "pgvector-store",
            lambda: PgVectorStore(conninfo="postgresql://localhost/test", dimensions=3),
            id="pgvector-store",
        )
    )
    return owners


@pytest.mark.parametrize(("owner_name", "factory"), _pool_owners())
async def test_postgres_pool_owner_close_waits_for_lazy_initialization(
    owner_name: str,
    factory: Callable[[], Any],
) -> None:
    owner = factory()
    pool = _FakePool()

    async with owner._init_lock:
        close_task = asyncio.create_task(owner.close(), name=f"close-{owner_name}")
        await asyncio.sleep(0)
        assert close_task.done() is False
        owner._pool = pool

    await close_task

    assert pool.closed is True
    assert owner._pool is None

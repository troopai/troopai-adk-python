"""Offline unit tests for PgVectorStore (no DB needed)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pgvector")
pytest.importorskip("psycopg")

from troopai.adk.memory import MemoryMetadata, MemorySource
from troopai.adk.memory.stores.pgvector import PgVectorStore
from troopai.adk.memory.vector_store import VectorRecord


@pytest.mark.parametrize("bad", ["", "a" * 65, "bad-name", "123start", "drop;table", "has space"])
def test_rejects_bad_table_name(bad: str) -> None:
    with pytest.raises(ValueError, match="table name"):
        PgVectorStore(conninfo="postgresql://fake", dimensions=2, table=bad)


def test_rejects_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        PgVectorStore(conninfo="postgresql://fake", dimensions=0, table="memory_vectors")


def test_accepts_valid_table_name() -> None:
    # Construction with a valid identifier must NOT raise (no DB connection happens in __init__).
    store = PgVectorStore(conninfo="postgresql://fake", dimensions=2, table="memory_vectors")
    assert store is not None


class _CaptureConn:
    """Async connection stub that records the SQL it is asked to execute."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    async def execute(self, statement: Any, params: Any | None = None) -> None:
        self.statements.append(statement.as_string())

    async def __aenter__(self) -> _CaptureConn:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _CapturePool:
    """Pool stub whose ``connection()`` yields the capturing connection."""

    def __init__(self, conn: _CaptureConn) -> None:
        self._conn = conn

    def connection(self) -> _CaptureConn:
        return self._conn


async def test_upsert_sql_refreshes_namespace_on_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    # Replacing a record under a new namespace must overwrite the stored
    # namespace, so the ON CONFLICT clause has to update it alongside content.
    store = PgVectorStore(conninfo="postgresql://fake", dimensions=2, table="memory_vectors")
    conn = _CaptureConn()

    async def _fake_ensure_ready() -> _CapturePool:
        return _CapturePool(conn)

    monkeypatch.setattr(store, "_ensure_ready", _fake_ensure_ready)
    record = VectorRecord(
        id="a",
        vector=[1.0, 0.0],
        namespace="u1",
        content="a",
        metadata=MemoryMetadata(source=MemorySource.MANUAL),
        created_at=1.0,
        updated_at=1.0,
    )
    await store.upsert([record])

    assert len(conn.statements) == 1
    statement = conn.statements[0]
    assert "ON CONFLICT (id) DO UPDATE" in statement
    assert "namespace = EXCLUDED.namespace" in statement

"""Integration tests for PgVectorStore (skipped unless TROOPAI_TEST_PG_DSN set)."""

from __future__ import annotations

import os

import pytest

from troopai.adk.memory import MemoryKind, MemoryMetadata, MemorySearchFilter, MemorySource
from troopai.adk.memory.vector_store import VectorRecord

pytest.importorskip("pgvector")
pytest.importorskip("psycopg")

_DSN = os.environ.get("TROOPAI_TEST_PG_DSN")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(_DSN is None, reason="TROOPAI_TEST_PG_DSN not set"),
]


def _rec(rid: str, vector: tuple[float, ...], *, kind: MemoryKind = MemoryKind.EPISODIC) -> VectorRecord:
    return VectorRecord(
        id=rid,
        vector=vector,
        namespace="u1",
        content=rid,
        metadata=MemoryMetadata(source=MemorySource.MANUAL, kind=kind),
        created_at=1.0,
        updated_at=1.0,
    )


async def test_pgvector_round_trip() -> None:
    from troopai.adk.memory.stores.pgvector import PgVectorStore

    assert _DSN is not None
    store = PgVectorStore(conninfo=_DSN, dimensions=2, table="test_mega8_vectors")
    try:
        await store.clear(namespace="u1")
        await store.upsert([_rec("a", (1.0, 0.0)), _rec("b", (0.0, 1.0), kind=MemoryKind.SEMANTIC)])
        results = await store.query((1.0, 0.0), namespace="u1", k=2)
        assert results[0].record.id == "a"
        sem = await store.query((0.0, 1.0), namespace="u1", k=5, filter=MemorySearchFilter(kind=MemoryKind.SEMANTIC))
        assert [r.record.id for r in sem] == ["b"]
        fetched = await store.get("a")
        assert fetched is not None
        assert fetched.id == "a"
        assert await store.delete(["a"]) == 1
        assert await store.clear(namespace="u1") == 1
    finally:
        await store.close()

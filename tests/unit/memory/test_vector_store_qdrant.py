"""Tests for QdrantVectorStore (in-process qdrant-client — real, offline)."""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("qdrant_client")

from troopai.adk.memory import MemoryKind, MemoryMetadata, MemorySearchFilter, MemorySource
from troopai.adk.memory.vector_store import VectorRecord


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


async def test_qdrant_namespace_isolation() -> None:
    from troopai.adk.memory.stores.qdrant import QdrantVectorStore

    store = QdrantVectorStore(collection="ns_isolation", dimensions=2, location=":memory:")
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    await store.upsert(
        [
            VectorRecord(
                id=a,
                vector=(1.0, 0.0),
                namespace="u1",
                content="a",
                metadata=MemoryMetadata(source=MemorySource.MANUAL),
                created_at=1.0,
                updated_at=1.0,
            ),
            VectorRecord(
                id=b,
                vector=(1.0, 0.0),
                namespace="u2",
                content="b",
                metadata=MemoryMetadata(source=MemorySource.MANUAL),
                created_at=1.0,
                updated_at=1.0,
            ),
        ]
    )
    results = await store.query((1.0, 0.0), namespace="u2", k=5)
    assert [r.record.id for r in results] == [b]
    await store.close()


def test_qdrant_to_record_raises_on_missing_required_payload() -> None:
    """_to_record must raise RuntimeError (not KeyError) on missing required payload fields."""
    pytest.importorskip("qdrant_client")
    from qdrant_client import models

    from troopai.adk.memory.stores.qdrant import _to_record

    # Point with empty payload — namespace and content are required
    point = models.Record(id="some-id", payload={}, vector=[1.0, 0.0])
    with pytest.raises(RuntimeError, match="namespace"):
        _to_record(point)


async def test_qdrant_round_trip() -> None:
    from troopai.adk.memory.stores.qdrant import QdrantVectorStore

    store = QdrantVectorStore(collection="round_trip", dimensions=2, location=":memory:")
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    await store.upsert(
        [
            _rec(a, (1.0, 0.0)),
            _rec(b, (0.0, 1.0), kind=MemoryKind.SEMANTIC),
        ]
    )
    results = await store.query((1.0, 0.0), namespace="u1", k=2)
    assert results[0].record.id == a
    sem = await store.query((0.0, 1.0), namespace="u1", k=5, filter=MemorySearchFilter(kind=MemoryKind.SEMANTIC))
    assert [r.record.id for r in sem] == [b]
    fetched = await store.get(a)
    assert fetched is not None
    assert fetched.content == a
    assert await store.delete([a]) == 1
    assert await store.clear(namespace="u1") == 1
    await store.close()

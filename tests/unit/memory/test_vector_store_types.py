"""Tests for vector-store value types and Protocol conformance."""

from __future__ import annotations

from troopai.adk.memory import MemoryMetadata, MemorySource
from troopai.adk.memory.vector_store import VectorQueryResult, VectorRecord, VectorStore


def test_vector_record_constructs() -> None:
    rec = VectorRecord(
        id="r1",
        vector=(0.1, 0.2),
        namespace="u1",
        content="hi",
        metadata=MemoryMetadata(source=MemorySource.MANUAL),
        created_at=1.0,
        updated_at=1.0,
    )
    assert rec.vector == (0.1, 0.2)
    assert VectorQueryResult(record=rec, score=0.9).score == 0.9


def test_protocol_is_runtime_checkable() -> None:
    class _Dummy:
        async def upsert(self, records: list[VectorRecord]) -> None: ...
        async def query(self, vector, *, namespace, k=5, filter=None): ...
        async def get(self, record_id): ...
        async def delete(self, ids): ...
        async def clear(self, *, namespace): ...
        async def close(self) -> None: ...

    assert isinstance(_Dummy(), VectorStore)

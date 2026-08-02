"""Tests for the pure-Python InMemoryVectorStore."""

from __future__ import annotations

import pytest

from troopai.adk.memory import (
    MemoryKind,
    MemoryMetadata,
    MemorySearchFilter,
    MemorySource,
)
from troopai.adk.memory.stores.in_memory import InMemoryVectorStore
from troopai.adk.memory.vector_store import VectorRecord


def _rec(
    rid: str, vector: tuple[float, ...], *, ns: str = "u1", kind: MemoryKind = MemoryKind.EPISODIC
) -> VectorRecord:
    return VectorRecord(
        id=rid,
        vector=vector,
        namespace=ns,
        content=rid,
        metadata=MemoryMetadata(source=MemorySource.MANUAL, kind=kind),
        created_at=1.0,
        updated_at=1.0,
    )


async def test_query_orders_by_similarity() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_rec("a", (1.0, 0.0)), _rec("b", (0.0, 1.0))])
    results = await store.query((1.0, 0.0), namespace="u1", k=2)
    assert [r.record.id for r in results] == ["a", "b"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0)


async def test_namespace_isolation() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_rec("a", (1.0, 0.0), ns="u1"), _rec("b", (1.0, 0.0), ns="u2")])
    results = await store.query((1.0, 0.0), namespace="u2", k=5)
    assert [r.record.id for r in results] == ["b"]


async def test_filter_namespace_intersects_positional_namespace() -> None:
    # A filter.namespace that contradicts the positional namespace yields no
    # results (A AND B), matching the keyword backends; a filter.namespace that
    # agrees with it is honoured as a no-op constraint.
    store = InMemoryVectorStore()
    await store.upsert([_rec("a", (1.0, 0.0), ns="u1")])
    contradicting = await store.query((1.0, 0.0), namespace="u1", k=5, filter=MemorySearchFilter(namespace="u2"))
    assert [r.record.id for r in contradicting] == []
    matching = await store.query((1.0, 0.0), namespace="u1", k=5, filter=MemorySearchFilter(namespace="u1"))
    assert [r.record.id for r in matching] == ["a"]


async def test_kind_filter() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            _rec("a", (1.0, 0.0), kind=MemoryKind.EPISODIC),
            _rec("b", (1.0, 0.0), kind=MemoryKind.SEMANTIC),
        ]
    )
    results = await store.query((1.0, 0.0), namespace="u1", k=5, filter=MemorySearchFilter(kind=MemoryKind.SEMANTIC))
    assert [r.record.id for r in results] == ["b"]


async def test_dimension_mismatch_raises() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_rec("a", (1.0, 0.0))])
    with pytest.raises(ValueError, match="dimension"):
        await store.upsert([_rec("b", (1.0, 0.0, 0.0))])


async def test_get_delete_clear() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_rec("a", (1.0, 0.0)), _rec("b", (0.0, 1.0))])
    record = await store.get("a")
    assert record is not None
    assert record.id == "a"
    assert await store.get("missing") is None
    assert await store.delete(["a", "missing"]) == 1
    assert await store.clear(namespace="u1") == 1
    await store.close()


async def test_query_dimension_mismatch_raises() -> None:
    store = InMemoryVectorStore()
    await store.upsert([_rec("a", (1.0, 0.0))])
    with pytest.raises(ValueError, match="dimension"):
        await store.query((1.0, 0.0, 0.0), namespace="u1")


async def test_upsert_mixed_dimension_batch_is_atomic() -> None:
    store = InMemoryVectorStore()
    with pytest.raises(ValueError, match="dimension"):
        await store.upsert([_rec("a", (1.0, 0.0)), _rec("b", (1.0, 0.0, 0.0))])
    assert await store.get("a") is None  # nothing committed from the bad batch


async def test_importance_filter() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            VectorRecord(
                id="lo",
                vector=(1.0, 0.0),
                namespace="u1",
                content="lo",
                metadata=MemoryMetadata(source=MemorySource.MANUAL, importance=1),
                created_at=1.0,
                updated_at=1.0,
            ),
            VectorRecord(
                id="hi",
                vector=(1.0, 0.0),
                namespace="u1",
                content="hi",
                metadata=MemoryMetadata(source=MemorySource.MANUAL, importance=5),
                created_at=1.0,
                updated_at=1.0,
            ),
        ]
    )
    results = await store.query((1.0, 0.0), namespace="u1", k=5, filter=MemorySearchFilter(importance=3))
    assert [r.record.id for r in results] == ["hi"]


async def test_agent_name_filter() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            VectorRecord(
                id="a",
                vector=(1.0, 0.0),
                namespace="u1",
                content="a",
                metadata=MemoryMetadata(source=MemorySource.MANUAL, agent_name="alice"),
                created_at=1.0,
                updated_at=1.0,
            ),
            VectorRecord(
                id="b",
                vector=(1.0, 0.0),
                namespace="u1",
                content="b",
                metadata=MemoryMetadata(source=MemorySource.MANUAL, agent_name="bob"),
                created_at=1.0,
                updated_at=1.0,
            ),
        ]
    )
    results = await store.query((1.0, 0.0), namespace="u1", k=5, filter=MemorySearchFilter(agent_name="bob"))
    assert [r.record.id for r in results] == ["b"]


async def test_categories_filter_any_overlap() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            VectorRecord(
                id="x",
                vector=(1.0, 0.0),
                namespace="u1",
                content="x",
                metadata=MemoryMetadata(source=MemorySource.MANUAL, categories=("red", "blue")),
                created_at=1.0,
                updated_at=1.0,
            ),
            VectorRecord(
                id="y",
                vector=(1.0, 0.0),
                namespace="u1",
                content="y",
                metadata=MemoryMetadata(source=MemorySource.MANUAL, categories=("green",)),
                created_at=1.0,
                updated_at=1.0,
            ),
        ]
    )
    results = await store.query(
        (1.0, 0.0), namespace="u1", k=5, filter=MemorySearchFilter(categories=("blue", "yellow"))
    )
    assert [r.record.id for r in results] == ["x"]


async def test_time_range_filter_exclusive_bounds() -> None:
    store = InMemoryVectorStore()
    await store.upsert(
        [
            VectorRecord(
                id="old",
                vector=(1.0, 0.0),
                namespace="u1",
                content="old",
                metadata=MemoryMetadata(source=MemorySource.MANUAL),
                created_at=10.0,
                updated_at=10.0,
            ),
            VectorRecord(
                id="new",
                vector=(1.0, 0.0),
                namespace="u1",
                content="new",
                metadata=MemoryMetadata(source=MemorySource.MANUAL),
                created_at=20.0,
                updated_at=20.0,
            ),
        ]
    )
    after = await store.query((1.0, 0.0), namespace="u1", k=5, filter=MemorySearchFilter(after=15.0))
    assert [r.record.id for r in after] == ["new"]
    before = await store.query((1.0, 0.0), namespace="u1", k=5, filter=MemorySearchFilter(before=15.0))
    assert [r.record.id for r in before] == ["old"]
    # exclusive boundary: a record exactly at the bound is excluded
    boundary = await store.query((1.0, 0.0), namespace="u1", k=5, filter=MemorySearchFilter(after=20.0))
    assert [r.record.id for r in boundary] == []

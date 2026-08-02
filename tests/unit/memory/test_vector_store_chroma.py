"""Tests for ChromaVectorStore (embedded chromadb — real, offline)."""

from __future__ import annotations

from typing import Any, cast

import pytest

pytest.importorskip("chromadb")

from troopai.adk.memory import MemoryKind, MemoryMetadata, MemorySearchFilter, MemorySource
from troopai.adk.memory.stores.chroma import ChromaVectorStore
from troopai.adk.memory.vector_store import VectorRecord


def _rec(rid: str, vector: tuple[float, ...], *, kind: MemoryKind = MemoryKind.EPISODIC) -> VectorRecord:
    return VectorRecord(
        id=rid,
        vector=vector,
        namespace="u1",
        content=rid,
        metadata=MemoryMetadata(source=MemorySource.MANUAL, kind=kind, categories=("x",)),
        created_at=1.0,
        updated_at=1.0,
    )


async def test_chroma_namespace_isolation() -> None:
    store = ChromaVectorStore(collection="chroma_ns_isolation")
    await store.upsert([_rec("a", (1.0, 0.0))])  # _rec default namespace is "u1"
    await store.upsert(
        [
            VectorRecord(
                id="b",
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
    assert [r.record.id for r in results] == ["b"]
    await store.close()


def test_chroma_build_where_shapes() -> None:
    from troopai.adk.memory.stores.chroma import _build_where

    # single clause (namespace only) -> no $and wrapper
    assert _build_where("u1", None) == {"namespace": {"$eq": "u1"}}
    # multi-clause -> $and wrapper containing namespace + kind + importance
    multi = cast(dict[str, Any], _build_where("u1", MemorySearchFilter(kind=MemoryKind.SEMANTIC, importance=4)))
    assert {"namespace": {"$eq": "u1"}} in multi["$and"]
    assert {"kind": {"$eq": "semantic"}} in multi["$and"]
    assert {"importance": {"$gte": 4}} in multi["$and"]
    # categories are NOT expressible in chroma where -> skipped (no categories clause)
    cats = cast(dict[str, Any], _build_where("u1", MemorySearchFilter(categories=("x",))))
    assert "categories" not in str(cats)


def test_chroma_from_chroma_raises_on_missing_required_fields() -> None:
    """_from_chroma must raise RuntimeError (not KeyError) on missing required fields."""
    from troopai.adk.memory.stores.chroma import _from_chroma

    # Missing 'namespace'
    with pytest.raises(RuntimeError, match="namespace"):
        _from_chroma("id1", (1.0, 0.0), "content", {"created_at": 1.0, "updated_at": 1.0})
    # Missing 'created_at'
    with pytest.raises(RuntimeError, match="created_at"):
        _from_chroma("id1", (1.0, 0.0), "content", {"namespace": "u1", "updated_at": 1.0})


def test_chroma_category_filter_emits_warning(caplog) -> None:
    """Category filter on Chroma must emit a WARNING (not just DEBUG)."""
    import logging

    from troopai.adk.memory.stores.chroma import _build_where

    with caplog.at_level(logging.WARNING, logger="troopai.adk.memory.stores.chroma"):
        _build_where("u1", MemorySearchFilter(categories=("tag",)))
    assert any("category" in r.message.lower() for r in caplog.records if r.levelno >= logging.WARNING)


async def test_chroma_round_trip() -> None:
    store = ChromaVectorStore(collection="chroma_vector_store_test")
    await store.upsert([_rec("a", (1.0, 0.0)), _rec("b", (0.0, 1.0), kind=MemoryKind.SEMANTIC)])
    results = await store.query((1.0, 0.0), namespace="u1", k=2)
    assert results[0].record.id == "a"
    sem = await store.query((0.0, 1.0), namespace="u1", k=5, filter=MemorySearchFilter(kind=MemoryKind.SEMANTIC))
    assert [r.record.id for r in sem] == ["b"]
    fetched = await store.get("a")
    assert fetched is not None
    assert fetched.content == "a"
    assert await store.delete(["a"]) == 1
    assert await store.clear(namespace="u1") == 1
    await store.close()

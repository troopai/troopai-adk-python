"""Tests for PineconeVectorStore (Pinecone client mocked — no network)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pinecone")

from troopai.adk.memory import MemoryKind, MemoryMetadata, MemorySearchFilter, MemorySource
from troopai.adk.memory.vector_store import VectorRecord


class _FakeIndex:
    def __init__(self) -> None:
        self.upserted: list[dict[str, Any]] = []
        self.deleted: list[Any] = []

    def upsert(self, *, vectors: list[dict[str, Any]]) -> dict[str, Any]:
        self.upserted.extend(vectors)
        return {"upserted_count": len(vectors)}

    def query(
        self, *, vector: list[float], top_k: int, filter: dict[str, Any], include_metadata: bool, include_values: bool
    ) -> dict[str, Any]:
        v = self.upserted[0]
        return {"matches": [{"id": v["id"], "score": 0.97, "values": v["values"], "metadata": v["metadata"]}]}

    def fetch(self, *, ids: list[str]) -> dict[str, Any]:
        hit = {x["id"]: x for x in self.upserted}
        return {"vectors": {i: {"values": hit[i]["values"], "metadata": hit[i]["metadata"]} for i in ids if i in hit}}

    def delete(self, **kwargs: Any) -> dict[str, Any]:
        self.deleted.append(kwargs)
        return {}


class _FakePinecone:
    last_index: _FakeIndex

    def __init__(self, *, api_key: str | None = None) -> None: ...

    def Index(self, name: str) -> _FakeIndex:  # noqa: N802 (mirrors Pinecone API)
        _FakePinecone.last_index = _FakeIndex()
        return _FakePinecone.last_index


def _rec(rid: str) -> VectorRecord:
    return VectorRecord(
        id=rid,
        vector=(1.0, 0.0),
        namespace="u1",
        content=rid,
        metadata=MemoryMetadata(source=MemorySource.MANUAL),
        created_at=1.0,
        updated_at=1.0,
    )


async def test_pinecone_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pinecone.Pinecone", _FakePinecone)
    from troopai.adk.memory.stores.pinecone import PineconeVectorStore

    store = PineconeVectorStore(index="testidx", api_key="k")
    await store.upsert([_rec("a")])
    result = await store.clear(namespace="u1")
    assert result == 0  # documented approximation (Pinecone reports no delete count)
    assert len(_FakePinecone.last_index.deleted) == 1
    assert "filter" in _FakePinecone.last_index.deleted[0]


def test_pinecone_build_filter_shapes() -> None:
    from troopai.adk.memory.stores.pinecone import _build_filter

    assert _build_filter("u1", None) == {"namespace": {"$eq": "u1"}}
    multi = _build_filter("u1", MemorySearchFilter(kind=MemoryKind.SEMANTIC, importance=4))
    assert {"namespace": {"$eq": "u1"}} in multi["$and"]
    assert {"kind": {"$eq": "semantic"}} in multi["$and"]
    assert {"importance": {"$gte": 4}} in multi["$and"]
    cats = _build_filter("u1", MemorySearchFilter(categories=("a", "b")))
    cats_clause = next(c for c in cats["$and"] if "categories" in c)
    assert cats_clause == {"categories": {"$in": ["a", "b"]}}


def test_pinecone_from_meta_raises_on_missing_required_fields() -> None:
    """_from_meta must raise RuntimeError (not KeyError) on missing required fields."""
    from troopai.adk.memory.stores.pinecone import _from_meta

    # Missing 'namespace'
    with pytest.raises(RuntimeError, match="namespace"):
        _from_meta("id1", [1.0, 0.0], {"content": "x", "created_at": 1.0, "updated_at": 1.0})
    # Missing 'content'
    with pytest.raises(RuntimeError, match="content"):
        _from_meta("id1", [1.0, 0.0], {"namespace": "u1", "created_at": 1.0, "updated_at": 1.0})


def test_to_meta_omits_empty_categories() -> None:
    """Default (empty) categories must not serialize as an empty list.

    Pinecone rejects an empty-list metadata value, so the key is omitted; it
    round-trips back to an empty tuple via _from_meta.
    """
    from troopai.adk.memory.stores.pinecone import _from_meta, _to_meta

    meta = _to_meta(_rec("a"))
    assert "categories" not in meta
    assert [] not in meta.values()
    restored = _from_meta("a", [1.0, 0.0], meta)
    assert restored.metadata.categories == ()


def test_to_meta_keeps_nonempty_categories() -> None:
    """A non-empty categories tuple is still written as a list of strings."""
    from troopai.adk.memory.stores.pinecone import _to_meta

    record = VectorRecord(
        id="a",
        vector=(1.0, 0.0),
        namespace="u1",
        content="a",
        metadata=MemoryMetadata(source=MemorySource.MANUAL, categories=("x", "y")),
        created_at=1.0,
        updated_at=1.0,
    )
    assert _to_meta(record)["categories"] == ["x", "y"]


async def test_pinecone_upsert_query_get(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pinecone.Pinecone", _FakePinecone)
    from troopai.adk.memory.stores.pinecone import PineconeVectorStore

    store = PineconeVectorStore(index="testidx", api_key="k")
    await store.upsert([_rec("a")])
    assert _FakePinecone.last_index.upserted[0]["id"] == "a"
    results = await store.query((1.0, 0.0), namespace="u1", k=3)
    assert results[0].record.id == "a"
    assert results[0].score == pytest.approx(0.97)
    fetched = await store.get("a")
    assert fetched is not None
    assert fetched.id == "a"
    assert await store.delete(["a"]) == 1
    await store.close()

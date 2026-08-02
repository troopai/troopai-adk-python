"""Tests for the MemoryKind tag and its filtering across lexical backends."""

from __future__ import annotations

from troopai.adk.memory import (
    MemoryKind,
    MemoryMetadata,
    MemorySearchFilter,
    MemorySource,
    SQLiteMemory,
    TemporaryMemory,
)


def test_metadata_defaults_to_episodic() -> None:
    meta = MemoryMetadata(source=MemorySource.MANUAL)
    assert meta.kind is MemoryKind.EPISODIC


async def test_temporary_memory_filters_by_kind() -> None:
    mem = TemporaryMemory()
    await mem.add(
        "raw turn", namespace="u1", metadata=MemoryMetadata(source=MemorySource.MANUAL, kind=MemoryKind.EPISODIC)
    )
    await mem.add(
        "distilled fact",
        namespace="u1",
        metadata=MemoryMetadata(source=MemorySource.EXTRACTION, kind=MemoryKind.SEMANTIC),
    )
    results = await mem.search(
        "fact turn raw distilled", namespace="u1", filter=MemorySearchFilter(kind=MemoryKind.SEMANTIC)
    )
    assert len(results) == 1
    assert results[0].entry.metadata.kind is MemoryKind.SEMANTIC


async def test_sqlite_memory_persists_and_filters_by_kind() -> None:
    mem = SQLiteMemory(path=":memory:")
    await mem.add(
        "raw turn", namespace="u1", metadata=MemoryMetadata(source=MemorySource.MANUAL, kind=MemoryKind.EPISODIC)
    )
    await mem.add(
        "distilled fact",
        namespace="u1",
        metadata=MemoryMetadata(source=MemorySource.EXTRACTION, kind=MemoryKind.SEMANTIC),
    )
    results = await mem.search("fact", namespace="u1", filter=MemorySearchFilter(kind=MemoryKind.SEMANTIC))
    assert all(r.entry.metadata.kind is MemoryKind.SEMANTIC for r in results)
    await mem.close()

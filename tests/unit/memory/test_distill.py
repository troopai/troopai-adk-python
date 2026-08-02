"""Tests for distill_to_semantic (extract -> hash-dedup -> store as SEMANTIC)."""

from __future__ import annotations

from typing import override

from troopai.adk.llms import Embedder, Embedding
from troopai.adk.memory import MemoryEntry, MemoryKind, MemoryMetadata, MemorySource
from troopai.adk.memory.distill import distill_to_semantic
from troopai.adk.memory.extractor import ExtractionResult, MemoryExtractor
from troopai.adk.memory.stores.in_memory import InMemoryVectorStore
from troopai.adk.memory.vector_memory import VectorMemory


class _HashEmbedder(Embedder):
    @override
    async def aembed_documents(self, texts: list[str]) -> list[Embedding]:
        return [Embedding(vector=(float(hash(t) % 97), float(len(t))), model="h") for t in texts]

    @property
    @override
    def dimensions(self) -> int | None:
        return 2


class _FixedExtractor(MemoryExtractor):
    def __init__(self, facts: list[str]) -> None:
        self._facts = facts
        self.seen: list[object] = []

    @override
    async def extract(self, messages: list[object], *, namespace: str) -> list[ExtractionResult]:
        self.seen = list(messages)
        return [ExtractionResult(content=f) for f in self._facts]


def _semantic_mem() -> VectorMemory:
    return VectorMemory(store=InMemoryVectorStore(), embedder=_HashEmbedder())


def _episodic_entry() -> MemoryEntry:
    return MemoryEntry(
        id="e",
        content="chat",
        namespace="u1",
        metadata=MemoryMetadata(source=MemorySource.MANUAL),
        created_at=1.0,
        updated_at=1.0,
    )


async def test_stores_facts_as_semantic_and_dedups_intra_call() -> None:
    mem = _semantic_mem()
    extractor = _FixedExtractor(["user likes tea", "user likes tea", "user is in Paris"])
    stored = await distill_to_semantic([_episodic_entry()], into=mem, extractor=extractor, namespace="u1")
    assert len(stored) == 2  # duplicate dropped
    assert all(e.metadata.kind is MemoryKind.SEMANTIC for e in stored)


async def test_cross_call_dedup_skips_existing() -> None:
    mem = _semantic_mem()
    extractor = _FixedExtractor(["user likes tea"])
    src = [_episodic_entry()]
    first = await distill_to_semantic(src, into=mem, extractor=extractor, namespace="u1")
    second = await distill_to_semantic(src, into=mem, extractor=extractor, namespace="u1")
    assert len(first) == 1
    assert len(second) == 0  # already present


async def test_agent_name_stamped_on_semantic_entries() -> None:
    mem = _semantic_mem()
    stored = await distill_to_semantic(
        [_episodic_entry()],
        into=mem,
        extractor=_FixedExtractor(["some fact"]),
        namespace="u1",
        agent_name="support-agent",
    )
    assert len(stored) == 1
    assert stored[0].metadata.agent_name == "support-agent"


async def test_distill_continues_on_per_fact_failure() -> None:
    """A storage failure for one fact must not abort the rest of the batch."""
    call_count = 0
    good_mem = _semantic_mem()

    class _FailingOnce:
        """Wraps a real Memory but raises on the first add() call."""

        async def search(self, query: str, *, namespace: str, limit: int, filter: object) -> list[object]:
            return await good_mem.search(query, namespace=namespace, limit=limit, filter=filter)  # type: ignore[arg-type]

        async def add(self, content: str, *, namespace: str, metadata: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated store failure on first fact")
            return await good_mem.add(content, namespace=namespace, metadata=metadata)  # type: ignore[arg-type]

    extractor = _FixedExtractor(["fact one", "fact two", "fact three"])
    stored = await distill_to_semantic(
        [_episodic_entry()],
        into=_FailingOnce(),  # type: ignore[arg-type]
        extractor=extractor,
        namespace="u1",
    )
    # Two facts must be stored (the first one failed)
    assert len(stored) == 2


async def test_session_source() -> None:
    class _Event:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeSession:
        id = "s1"

        async def get(self) -> list[object]:
            return [_Event("hello"), _Event("world")]

    mem = _semantic_mem()
    extractor = _FixedExtractor(["fact one"])
    stored = await distill_to_semantic(
        _FakeSession(),  # type: ignore[arg-type]  # structural test double, not a real Session
        into=mem,
        extractor=extractor,
        namespace="u1",
    )
    assert extractor.seen == ["hello", "world"]
    assert len(stored) == 1

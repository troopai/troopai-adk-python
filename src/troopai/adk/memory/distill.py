"""Explicit episodic -> semantic distillation.

Runs a :class:`MemoryExtractor` over raw episodic content and stores the
distilled facts as ``MemoryKind.SEMANTIC`` entries.  Never scheduled by the
framework — the developer calls it, so the extraction + embedding token cost is
always chosen explicitly.

Dedup: extracted facts are deduplicated within a call by normalized-content
hash; cross-call, each fact is compared against the top semantic matches and
exact-hash duplicates are skipped (best-effort, bounded by the search limit).
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from troopai.adk.memory.extractor import MemoryExtractor
from troopai.adk.memory.memory import Memory
from troopai.adk.memory.memory_types import (
    MemoryEntry,
    MemoryKind,
    MemoryMetadata,
    MemorySearchFilter,
    MemorySource,
)

if TYPE_CHECKING:
    from troopai.adk.session.session import Session
    from troopai.adk.types.input import LLMInputContentItem

logger = logging.getLogger(__name__)

# Best-effort cross-call dedup window: how many existing semantic matches to
# check per fact. Larger reduces duplicate stores but costs one embedding +
# query per fact. Dedup beyond this window is not guaranteed.
_DEDUP_SEARCH_LIMIT = 5


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


async def distill_to_semantic(
    source: Session | list[MemoryEntry],
    *,
    into: Memory,
    extractor: MemoryExtractor,
    namespace: str,
    agent_name: str | None = None,
) -> list[MemoryEntry]:
    """Extract facts from ``source`` and store them as semantic memories.

    Args:
        source: A ``Session`` or a list of episodic ``MemoryEntry`` to distill.
        into: The target memory (typically a ``VectorMemory``).
        extractor: The extraction strategy (e.g. ``LLMExtractor``).
        namespace: Target namespace for the semantic entries.
        agent_name: Optional agent name recorded on the entries.

    Returns:
        The semantic ``MemoryEntry`` instances created.
    """
    messages: list[str | LLMInputContentItem]
    if isinstance(source, list):
        messages = [entry.content for entry in source]
    else:
        events = await source.get()
        messages = [event.content for event in events]

    results = await extractor.extract(messages, namespace=namespace)
    logger.info("distill_to_semantic: %d facts extracted (namespace=%s)", len(results), namespace)

    seen: set[str] = set()
    stored: list[MemoryEntry] = []
    for idx, result in enumerate(results):
        fact_hash = _hash(result.content)
        if fact_hash in seen:
            continue
        seen.add(fact_hash)
        try:
            existing = await into.search(
                result.content,
                namespace=namespace,
                limit=_DEDUP_SEARCH_LIMIT,
                filter=MemorySearchFilter(kind=MemoryKind.SEMANTIC),
            )
            if any(_hash(match.entry.content) == fact_hash for match in existing):
                logger.debug("distill_to_semantic: skipping duplicate fact")
                continue
            meta = MemoryMetadata(
                source=MemorySource.EXTRACTION,
                importance=result.importance,
                categories=result.categories,
                kind=MemoryKind.SEMANTIC,
                agent_name=agent_name,
            )
            stored.append(await into.add(content=result.content, namespace=namespace, metadata=meta))
        except Exception as exc:
            logger.warning(
                "distill_to_semantic: failed to store fact %d (%s): %s",
                idx,
                result.content[:50],
                exc,
            )
            continue

    logger.info("distill_to_semantic: stored %d semantic facts", len(stored))
    return stored

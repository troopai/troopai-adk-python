"""Memory type definitions.

Defines the core data structures for the memory module:
entries, metadata, search results, and search filters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class MemorySource(StrEnum):
    """How a memory entry was created."""

    EXTRACTION = "extraction"
    """Extracted from a session by an LLM or rule-based extractor."""

    MANUAL = "manual"
    """Injected directly by the developer."""

    TOOL = "tool"
    """Added by an agent via :class:`MemoryTool`."""


class MemoryKind(StrEnum):
    """Whether a memory is a raw interaction or a distilled fact."""

    EPISODIC = "episodic"
    """Raw or lightly-processed interaction content."""

    SEMANTIC = "semantic"
    """Distilled, durable knowledge (typically produced by distillation)."""


@dataclass(frozen=True)
class MemoryMetadata:
    """Metadata attached to a memory entry.

    Attributes:
        source: How this memory was created.
        importance: Priority from 1 (low) to 5 (critical).
        categories: Semantic tags for filtering.
        session_id: Which session produced this memory, if any.
        agent_name: Which agent produced this memory, if any.
        kind: Episodic (raw) vs semantic (distilled). Defaults to episodic.
        custom: Arbitrary key-value pairs for application-specific data.
    """

    source: MemorySource
    importance: int = 3
    categories: tuple[str, ...] = ()
    session_id: str | None = None
    agent_name: str | None = None
    kind: MemoryKind = MemoryKind.EPISODIC
    custom: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (1 <= self.importance <= 5):
            raise ValueError(f"MemoryMetadata.importance must be 1-5, got {self.importance}")


@dataclass(frozen=True)
class MemoryEntry:
    """A single piece of knowledge stored in memory.

    Attributes:
        id: Unique identifier (UUID).
        content: The knowledge content.
        namespace: Scoping key (e.g. ``"user:123"``, ``"agent:support"``).
        metadata: Source, importance, categories, and other metadata.
        created_at: Unix timestamp when the entry was created.
        updated_at: Unix timestamp when the entry was last updated.
    """

    id: str
    content: str
    namespace: str
    metadata: MemoryMetadata
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class MemorySearchResult:
    """A memory entry with its relevance score from a search.

    Attributes:
        entry: The matching memory entry.
        score: Relevance score from 0.0 (no match) to 1.0 (perfect match).
    """

    entry: MemoryEntry
    score: float


@dataclass(frozen=True)
class MemorySearchFilter:
    """Filters applied during memory search.

    All fields are optional.  Only non-``None`` fields are applied.

    Attributes:
        namespace: Restrict to a specific namespace.
        categories: Restrict to entries with any of these categories.
        importance: Minimum importance level (1-5).
        agent_name: Restrict to entries from a specific agent.
        kind: Restrict to entries of this kind (episodic/semantic).
        after: Only entries created after this Unix timestamp.
        before: Only entries created before this Unix timestamp.
    """

    namespace: str | None = None
    categories: tuple[str, ...] | None = None
    importance: int | None = None
    agent_name: str | None = None
    kind: MemoryKind | None = None
    after: float | None = None
    before: float | None = None

"""Memory ABC — abstract interface for knowledge storage.

The ``Memory`` class defines the contract for all memory backends.
Implementations must provide CRUD operations and search.  The
``add_from_session`` method provides a default pipeline that
delegates to a :class:`MemoryExtractor` then calls ``add()``
for each result.
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from troopai.adk.memory.extractor import MemoryExtractor
from troopai.adk.memory.memory_types import (
    MemoryEntry,
    MemoryMetadata,
    MemorySearchFilter,
    MemorySearchResult,
    MemorySource,
)

if TYPE_CHECKING:
    from troopai.adk.session.session import Session
    from troopai.adk.session.session_event import SessionEvent
    from troopai.adk.types.input import LLMInputContentItem

logger = logging.getLogger(__name__)


class Memory(ABC):
    """Abstract base class for memory backends.

    All memory operations are async.  The ``namespace`` parameter is
    always explicit — the framework never injects a hidden default.

    Subclasses must implement: ``add``, ``search``, ``get``, ``delete``,
    ``clear``.
    """

    @abstractmethod
    async def add(
        self,
        content: str,
        *,
        namespace: str,
        metadata: MemoryMetadata | None = None,
    ) -> MemoryEntry:
        """Store a new memory entry.

        Args:
            content: The knowledge to remember.
            namespace: Scoping key (e.g. ``"user:123"``).
            metadata: Optional metadata.  If ``None``, uses default
                metadata with ``MemorySource.MANUAL``.

        Returns:
            The created :class:`MemoryEntry`.
        """

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        namespace: str,
        limit: int = 5,
        filter: MemorySearchFilter | None = None,
    ) -> list[MemorySearchResult]:
        """Search for relevant memories.

        Args:
            query: Search query text.
            namespace: Namespace to search within.
            limit: Maximum results to return.
            filter: Optional filters on metadata fields.

        Returns:
            List of :class:`MemorySearchResult` ordered by relevance.
        """

    @abstractmethod
    async def get(self, memory_id: str) -> MemoryEntry | None:
        """Retrieve a specific memory entry by ID.

        Args:
            memory_id: The entry's unique identifier.

        Returns:
            The entry, or ``None`` if not found.
        """

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry.

        Args:
            memory_id: The entry's unique identifier.

        Returns:
            ``True`` if deleted, ``False`` if not found.
        """

    @abstractmethod
    async def clear(self, *, namespace: str) -> int:
        """Delete all entries in a namespace.

        Args:
            namespace: The namespace to clear.

        Returns:
            Number of entries deleted.
        """

    async def add_from_session(
        self,
        messages: Sequence[LLMInputContentItem | str],
        *,
        namespace: str,
        extractor: MemoryExtractor,
        session_id: str | None = None,
        agent_name: str | None = None,
    ) -> list[MemoryEntry]:
        """Extract and store memories from conversation messages.

        Default pipeline: calls ``extractor.extract()``, then
        ``self.add()`` for each result.

        Args:
            messages: Conversation messages to analyze.
            namespace: Target namespace for new entries.
            extractor: The extraction strategy to use.
            session_id: Optional session ID to attach to metadata.
            agent_name: Optional agent name to attach to metadata.

        Returns:
            List of created :class:`MemoryEntry` instances.
        """
        logger.info(
            "Memory.add_from_session: extracting from %d messages (namespace=%s)",
            len(messages),
            namespace,
        )

        results = await extractor.extract(list(messages), namespace=namespace)
        entries: list[MemoryEntry] = []

        for result in results:
            metadata = MemoryMetadata(
                source=MemorySource.EXTRACTION,
                importance=result.importance,
                categories=result.categories,
                session_id=session_id,
                agent_name=agent_name,
            )
            entry = await self.add(content=result.content, namespace=namespace, metadata=metadata)
            entries.append(entry)

        logger.info("Memory.add_from_session: stored %d entries", len(entries))
        return entries

    async def add_session(
        self,
        session: Session,
        *,
        namespace: str,
        extractor: MemoryExtractor,
        agent_name: str | None = None,
    ) -> list[MemoryEntry]:
        """Ingest an entire session into memory.

        Loads all events from the session, extracts their content,
        and stores the extracted knowledge.

        Args:
            session: The session to ingest.
            namespace: Target namespace for new entries.
            extractor: The extraction strategy to use.
            agent_name: Optional agent name to attach to metadata.

        Returns:
            List of created :class:`MemoryEntry` instances.
        """
        logger.info(
            "Memory.add_session: ingesting session %s (namespace=%s)",
            session.id,
            namespace,
        )
        events = await session.get()
        messages = [e.content for e in events]
        return await self.add_from_session(
            messages,
            namespace=namespace,
            extractor=extractor,
            session_id=session.id,
            agent_name=agent_name,
        )

    async def add_events(
        self,
        events: list[SessionEvent],
        *,
        namespace: str,
        extractor: MemoryExtractor,
        session_id: str | None = None,
        agent_name: str | None = None,
    ) -> list[MemoryEntry]:
        """Add a batch of session events to memory (incremental update).

        Extracts content from the events and stores the extracted
        knowledge.  Use this for adding just the latest turn without
        re-processing the entire session.

        Args:
            events: Session events to process.
            namespace: Target namespace for new entries.
            extractor: The extraction strategy to use.
            session_id: Optional session ID to attach to metadata.
            agent_name: Optional agent name to attach to metadata.

        Returns:
            List of created :class:`MemoryEntry` instances.
        """
        logger.info(
            "Memory.add_events: processing %d events (namespace=%s)",
            len(events),
            namespace,
        )
        messages = [e.content for e in events]
        return await self.add_from_session(
            messages,
            namespace=namespace,
            extractor=extractor,
            session_id=session_id,
            agent_name=agent_name,
        )

    async def close(self) -> None:
        """Release resources.  No-op by default."""

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique memory entry ID."""
        return str(uuid.uuid4())

    @staticmethod
    def _now() -> float:
        """Current time as Unix timestamp."""
        return time.time()

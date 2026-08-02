"""In-memory memory backend for prototyping.

Stores entries in a plain dict.  Search uses keyword overlap scoring
(case-insensitive word intersection / union).  Thread-safe via
:class:`threading.Lock`.

For production use, prefer :class:`SQLiteMemory` which uses FTS5
for proper full-text search.
"""

from __future__ import annotations

import logging
import threading
from typing import override

from troopai.adk.memory.memory import Memory
from troopai.adk.memory.memory_types import (
    MemoryEntry,
    MemoryMetadata,
    MemorySearchFilter,
    MemorySearchResult,
    MemorySource,
)

logger = logging.getLogger(__name__)


class TemporaryMemory(Memory):
    """Dict-backed memory for prototyping and testing.

    Search uses keyword overlap scoring: the score is the ratio of
    matching words between the query and entry content (case-insensitive
    word intersection / union).

    Thread-safe via :class:`threading.Lock`.

    Two instances with different ``scope`` values store and retrieve
    entries independently even if they share the same underlying dict
    (they don't — each instance owns its own dict, but the scope also
    prefixes every namespace key so two stores with different scopes
    never collide in the rare case a dict is shared externally).
    ``scope=None`` (the default) preserves the global, no-prefix
    behaviour.

    Args:
        scope: Optional run-level isolation token.  When set, every
            namespace is internally stored as ``"<scope>/<namespace>"``
            so two stores with different scopes never see each other's
            entries.  ``None`` (default) keeps the existing global
            behaviour unchanged.

    Example::

        memory = TemporaryMemory()
        entry = await memory.add("User prefers dark mode", namespace="user:1")
        results = await memory.search("dark mode", namespace="user:1")

        # Scoped: two concurrent runs share no data
        run_a = TemporaryMemory(scope="run:a")
        run_b = TemporaryMemory(scope="run:b")
        await run_a.add("secret", namespace="shared")
        assert await run_b.search("secret", namespace="shared") == []
    """

    def __init__(self, *, scope: str | None = None) -> None:
        if scope is not None and "/" in scope:
            raise ValueError(
                f"TemporaryMemory scope must not contain '/'; got {scope!r}. "
                "The '/' character is the internal namespace delimiter.",
            )
        self._scope = scope
        self._entries: dict[str, MemoryEntry] = {}
        self._lock = threading.Lock()

    def _scoped(self, namespace: str) -> str:
        """Return the internal storage key for ``namespace``.

        When :attr:`_scope` is set, the stored key is
        ``"<scope>/<namespace>"``; otherwise the namespace is used as-is.
        The stored :attr:`~troopai.adk.memory.memory_types.MemoryEntry.namespace`
        field always holds the *original* developer-supplied namespace — the
        scope prefix is purely an internal partitioning mechanism.
        """
        if self._scope is None:
            return namespace
        return f"{self._scope}/{namespace}"

    def _unscoped(self, stored_namespace: str) -> str:
        """Strip the scope prefix from a stored namespace value.

        Reverse of :meth:`_scoped`: strips ``"<scope>/"`` from the
        front when :attr:`_scope` is set.  Returns ``stored_namespace``
        unchanged when there is no scope or the prefix is absent (the
        latter prevents corruption on entries written by a different
        scope or by an unscoped store).
        """
        if self._scope is None:
            return stored_namespace
        prefix = f"{self._scope}/"
        if stored_namespace.startswith(prefix):
            return stored_namespace[len(prefix) :]
        return stored_namespace

    @override
    async def add(
        self,
        content: str,
        *,
        namespace: str,
        metadata: MemoryMetadata | None = None,
    ) -> MemoryEntry:
        """Store a new memory entry in the in-process dict.

        The entry is stored under the scoped namespace (see
        :meth:`_scoped`) so two :class:`TemporaryMemory` instances with
        different ``scope`` values never see each other's entries.  The
        returned :class:`MemoryEntry` always carries the
        *caller-supplied* (unscoped) namespace.

        Args:
            content: The knowledge to remember.
            namespace: Scoping key (e.g. ``"user:123"``).
            metadata: Optional metadata.  If ``None``, uses default
                metadata with ``MemorySource.MANUAL``.

        Returns:
            The created :class:`MemoryEntry` with the caller-supplied
            namespace.
        """
        meta = metadata or MemoryMetadata(source=MemorySource.MANUAL)
        now = self._now()
        scoped_ns = self._scoped(namespace)
        entry = MemoryEntry(
            id=self._generate_id(),
            content=content,
            namespace=scoped_ns,
            metadata=meta,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._entries[entry.id] = entry

        logger.debug("TemporaryMemory: added entry %s (namespace=%s)", entry.id, namespace)
        # Return entry with the caller-supplied (unscoped) namespace so
        # the public API remains namespace-transparent.
        return MemoryEntry(
            id=entry.id,
            content=entry.content,
            namespace=namespace,
            metadata=entry.metadata,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
        )

    @override
    async def search(
        self,
        query: str,
        *,
        namespace: str,
        limit: int = 5,
        filter: MemorySearchFilter | None = None,
    ) -> list[MemorySearchResult]:
        """Search entries by keyword overlap (case-insensitive Jaccard score).

        Only entries stored under this instance's scope are visible.

        Args:
            query: Search query text.  Returns empty list if blank.
            namespace: Namespace to search within.
            limit: Maximum results to return.
            filter: Optional filters on metadata fields.

        Returns:
            List of :class:`MemorySearchResult` ordered by descending relevance.
        """
        query_words = set(query.lower().split())
        if len(query_words) == 0:
            return []

        scoped_ns = self._scoped(namespace)
        with self._lock:
            candidates = list(self._entries.values())

        scoped_filter_ns = (
            self._scoped(filter.namespace) if filter is not None and filter.namespace is not None else None
        )
        results: list[MemorySearchResult] = []
        for entry in candidates:
            if not self._matches_filter(entry, scoped_ns, filter, scoped_filter_ns):
                continue

            entry_words = set(entry.content.lower().split())
            if len(entry_words) == 0:
                continue

            intersection = query_words & entry_words
            if len(intersection) == 0:
                continue

            union = query_words | entry_words
            score = len(intersection) / len(union)
            # Surface caller-supplied (unscoped) namespace in results.
            public_entry = MemoryEntry(
                id=entry.id,
                content=entry.content,
                namespace=namespace,
                metadata=entry.metadata,
                created_at=entry.created_at,
                updated_at=entry.updated_at,
            )
            results.append(MemorySearchResult(entry=public_entry, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    @override
    async def get(self, memory_id: str) -> MemoryEntry | None:
        """Retrieve a specific memory entry by ID.

        Returns the entry with the caller-supplied (unscoped) namespace
        so the public API is namespace-transparent regardless of the
        scope prefix stored internally.

        Args:
            memory_id: The entry's unique identifier.

        Returns:
            The entry with the unscoped namespace, or ``None`` if not found.
        """
        with self._lock:
            entry = self._entries.get(memory_id)
        if entry is None:
            return None
        return MemoryEntry(
            id=entry.id,
            content=entry.content,
            namespace=self._unscoped(entry.namespace),
            metadata=entry.metadata,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
        )

    @override
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry.

        Args:
            memory_id: The entry's unique identifier.

        Returns:
            ``True`` if deleted, ``False`` if not found.
        """
        with self._lock:
            if memory_id in self._entries:
                del self._entries[memory_id]
                logger.debug("TemporaryMemory: deleted entry %s", memory_id)
                return True
        return False

    @override
    async def clear(self, *, namespace: str) -> int:
        """Delete all entries in a namespace.

        Args:
            namespace: The namespace to clear.

        Returns:
            Number of entries deleted.
        """
        scoped_ns = self._scoped(namespace)
        with self._lock:
            to_delete = [eid for eid, entry in self._entries.items() if entry.namespace == scoped_ns]
            for eid in to_delete:
                del self._entries[eid]

        logger.info("TemporaryMemory: cleared %d entries (namespace=%s)", len(to_delete), namespace)
        return len(to_delete)

    @staticmethod
    def _matches_filter(
        entry: MemoryEntry,
        namespace: str,
        filter: MemorySearchFilter | None,
        scoped_filter_namespace: str | None = None,
    ) -> bool:
        """Check whether an entry matches the namespace and optional filter.

        Args:
            entry: The candidate entry.
            namespace: Already-scoped namespace; compared directly against
                ``entry.namespace``.
            filter: Optional metadata filter.
            scoped_filter_namespace: Pre-scoped value of
                ``filter.namespace`` (if ``filter.namespace`` is not
                ``None``).  Passed separately so the static method can
                compare the stored (scoped) namespace value correctly
                without needing to know the scope prefix itself.
        """
        # Always filter by the required (already-scoped) namespace
        if entry.namespace != namespace:
            return False

        if filter is None:
            return True

        # Additional namespace constraint — compare against the pre-scoped
        # value so a scoped store's entries (stored as "<scope>/<ns>") match
        # a filter that was expressed in terms of the caller's (unscoped) ns.
        if filter.namespace is not None and entry.namespace != scoped_filter_namespace:
            return False

        if filter.kind is not None and entry.metadata.kind != filter.kind:
            return False

        if filter.importance is not None and entry.metadata.importance < filter.importance:
            return False

        if filter.agent_name is not None and entry.metadata.agent_name != filter.agent_name:
            return False

        if filter.categories is not None:
            entry_cats = set(entry.metadata.categories)
            if len(entry_cats.intersection(filter.categories)) == 0:
                return False

        if filter.after is not None and entry.created_at <= filter.after:
            return False

        return not (filter.before is not None and entry.created_at >= filter.before)

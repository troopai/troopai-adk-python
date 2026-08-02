"""SQLite-backed memory with FTS5 full-text search.

Uses two tables plus triggers:
- ``memories`` — stores entry data and metadata as JSON.
- ``memories_fts`` — FTS5 virtual table for full-text search.
- Triggers keep FTS in sync with the main table.

Uses ``aiosqlite`` for truly async database access.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import override

import aiosqlite

from troopai.adk.databases import SQLiteDatabaseConnection
from troopai.adk.memory.memory import Memory
from troopai.adk.memory.memory_types import (
    MemoryEntry,
    MemoryKind,
    MemoryMetadata,
    MemorySearchFilter,
    MemorySearchResult,
    MemorySource,
)

logger = logging.getLogger(__name__)

# Score assigned to recency-fallback results (no BM25 match).
# Distinct from BM25-normalized scores (0.0-1.0) so callers can
# distinguish "no keyword match, returned by recency" from
# "perfect BM25 match" (score=1.0).
_RECENCY_FALLBACK_SCORE = 0.0

# =====================================================================
# Table schemas — reviewable at a glance, reusable by migrations
# =====================================================================

DEFAULT_MEMORIES_TABLE = "memories"

# Strict SQL-identifier allowlist. Must start with a letter/underscore,
# followed by letters/digits/underscores only. 64-char cap matches the
# conventional RDBMS identifier limit. The table name is interpolated
# into SQL via f-strings (SQLite cannot parameterize identifiers), so
# anything outside this allowlist is a SQL-injection surface.
_TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TABLE_NAME_MAX_LEN = 64


def _validate_table_name(table: str) -> None:
    """Reject any caller-supplied table name that isn't a safe SQL identifier.

    Raises ``ValueError`` on reject. The cost of an unsafe name is silent
    SQL injection on every ``add``/``search``/``delete``, so this validator
    runs at construction time — before any DB I/O — to fail loudly.
    """
    if len(table) == 0:
        raise ValueError("SQLiteMemory table name must be non-empty")
    if len(table) > _TABLE_NAME_MAX_LEN:
        raise ValueError(f"SQLiteMemory table name exceeds {_TABLE_NAME_MAX_LEN} characters: {table!r}")
    if _TABLE_NAME_PATTERN.match(table) is None:
        raise ValueError(f"SQLiteMemory table name must match {_TABLE_NAME_PATTERN.pattern!r} (got {table!r})")


MEMORIES_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS {table} (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    namespace TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{{}}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""

MEMORIES_NAMESPACE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_{table}_namespace
    ON {table}(namespace);
"""

MEMORIES_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table}
    USING fts5(memory_id UNINDEXED, text);
"""

MEMORIES_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS {table}_ai AFTER INSERT ON {table}
BEGIN
    INSERT INTO {fts_table}(memory_id, text)
    VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS {table}_ad AFTER DELETE ON {table}
BEGIN
    DELETE FROM {fts_table} WHERE memory_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS {table}_au AFTER UPDATE ON {table}
BEGIN
    DELETE FROM {fts_table} WHERE memory_id = old.id;
    INSERT INTO {fts_table}(memory_id, text)
    VALUES (new.id, new.content);
END;
"""


def _build_memory_schema_sql(table: str, fts_table: str) -> str:
    """Build the combined DDL for memories + FTS + triggers."""
    return "".join(
        [
            MEMORIES_TABLE_SCHEMA.format(table=table),
            MEMORIES_NAMESPACE_INDEX.format(table=table),
            MEMORIES_FTS_SCHEMA.format(fts_table=fts_table),
            MEMORIES_FTS_TRIGGERS.format(table=table, fts_table=fts_table),
        ]
    )


# =====================================================================
# Async DB connection context manager
# =====================================================================


class SQLiteMemory(Memory):
    """Persistent memory backed by SQLite with FTS5 search.

    Uses ``bm25()`` ranking for full-text search, normalized to
    0.0-1.0 scores.  Filters are applied as SQL WHERE clauses.

    Uses ``aiosqlite`` for truly async database access.

    When keyword search returns no matches, the most recent entries in the
    namespace are returned instead (handles meta-queries and cross-language
    queries).

    Two instances with different ``scope`` values that share the same
    database file store and retrieve entries independently: the scope is
    prepended to every namespace value persisted to the ``namespace``
    column (``"<scope>/<namespace>"``), so rows written by one scope are
    never visible to queries from another scope.  ``scope=None`` (the
    default) preserves the existing global behaviour with no prefix.

    Args:
        path: Path to the SQLite database file, or ``":memory:"``
            for in-memory storage.
        table: Name of the memories table.
        scope: Optional run-level isolation token.  When set, every
            namespace is stored internally as ``"<scope>/<namespace>"``
            so two stores with different scopes never see each other's
            entries.  ``None`` (default) keeps the existing global
            behaviour unchanged.

    Example::

        memory = SQLiteMemory(path="memory.db")
        entry = await memory.add("User prefers dark mode", namespace="user:1")
        results = await memory.search("dark mode", namespace="user:1")

        # Scoped: two concurrent runs share no data
        run_a = SQLiteMemory(path=":memory:", scope="run:a")
        run_b = SQLiteMemory(path=":memory:", scope="run:b")
        await run_a.add("secret", namespace="shared")
        assert await run_b.search("secret", namespace="shared") == []
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        table: str = DEFAULT_MEMORIES_TABLE,
        *,
        scope: str | None = None,
    ) -> None:
        _validate_table_name(table)
        if scope is not None and "/" in scope:
            raise ValueError(
                f"SQLiteMemory scope must not contain '/'; got {scope!r}. "
                "The '/' character is the internal namespace delimiter.",
            )
        self._db_path = str(path)
        self._table = table
        self._fts_table = f"{table}_fts"
        self._db = SQLiteDatabaseConnection(path)
        self._tables_ready = False
        self._init_lock = asyncio.Lock()
        self._scope = scope

    def _scoped(self, namespace: str) -> str:
        """Return the internal storage key for ``namespace``.

        When :attr:`_scope` is set, the stored key is
        ``"<scope>/<namespace>"``; otherwise the namespace is used as-is.
        The value written to the ``namespace`` column in SQLite always
        uses this key, and queries always pass this key as the bind
        parameter.
        """
        if self._scope is None:
            return namespace
        return f"{self._scope}/{namespace}"

    def _unscoped(self, stored_namespace: str) -> str:
        """Strip the scope prefix from a stored namespace column value.

        Reverse of :meth:`_scoped`: strips ``"<scope>/"`` from the
        front when :attr:`_scope` is set.  Returns ``stored_namespace``
        unchanged when there is no scope or the prefix is absent (the
        latter prevents corruption on rows written by a different scope
        or by an unscoped store sharing the same file).
        """
        if self._scope is None:
            return stored_namespace
        prefix = f"{self._scope}/"
        if stored_namespace.startswith(prefix):
            return stored_namespace[len(prefix) :]
        return stored_namespace

    async def _ensure_ready(self) -> None:
        """Create tables if not yet done (lazy, one-shot)."""
        if not self._tables_ready:
            async with self._init_lock:
                if not self._tables_ready:
                    async with self._db.connect() as db:
                        await db.executescript(_build_memory_schema_sql(self._table, self._fts_table))
                        await db.commit()
                    self._tables_ready = True

    @override
    async def add(
        self,
        content: str,
        *,
        namespace: str,
        metadata: MemoryMetadata | None = None,
    ) -> MemoryEntry:
        """Store a new memory entry and index it for full-text search.

        The row is stored with the scoped namespace (see :meth:`_scoped`)
        so two :class:`SQLiteMemory` instances with different ``scope``
        values sharing the same file never see each other's entries.  The
        returned :class:`MemoryEntry` carries the caller-supplied
        (unscoped) namespace.

        Args:
            content: The knowledge to remember.
            namespace: Scoping key (e.g. ``"user:123"``).
            metadata: Optional metadata.  If ``None``, uses default
                metadata with ``MemorySource.MANUAL``.

        Returns:
            The created :class:`MemoryEntry` with the caller-supplied
            namespace.
        """
        await self._ensure_ready()
        meta = metadata or MemoryMetadata(source=MemorySource.MANUAL)
        now = self._now()
        entry_id = self._generate_id()
        scoped_ns = self._scoped(namespace)

        meta_json = json.dumps(
            {
                "source": meta.source.value,
                "importance": meta.importance,
                "categories": list(meta.categories),
                "session_id": meta.session_id,
                "agent_name": meta.agent_name,
                "kind": meta.kind.value,
                "custom": meta.custom,
            },
            separators=(",", ":"),
        )

        async with self._db.connect() as db:
            await db.execute(
                f"INSERT INTO {self._table} "
                f"(id, content, namespace, metadata, created_at, updated_at) "
                f"VALUES (:id, :content, :ns, :meta, :created, :updated)",
                {
                    "id": entry_id,
                    "content": content,
                    "ns": scoped_ns,
                    "meta": meta_json,
                    "created": now,
                    "updated": now,
                },
            )
            await db.commit()

        logger.debug("SQLiteMemory: added entry %s (namespace=%s)", entry_id, namespace)
        return MemoryEntry(
            id=entry_id,
            content=content,
            namespace=namespace,
            metadata=meta,
            created_at=now,
            updated_at=now,
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
        """Search for relevant memories using FTS5 BM25 ranking.

        Falls back to the most recent entries when no keyword matches are found.

        Args:
            query: Search query text.  Returns empty list if blank.
            namespace: Namespace to search within.
            limit: Maximum results to return.
            filter: Optional filters on metadata fields.

        Returns:
            List of :class:`MemorySearchResult` ordered by descending relevance.
        """
        await self._ensure_ready()
        if len(query.strip()) == 0:
            return []

        # OR between all terms — BM25 ranks by relevance (IDF naturally
        # downweights common words).  Language-agnostic: no stopword list.
        words = [cleaned for w in query.strip().split() if (cleaned := w.strip(".,!?;:\"'()[]{}"))]
        if len(words) == 0:
            return []

        # Wrap each token as an FTS5 phrase token, escaping any interior
        # double-quote by doubling it (the FTS5 literal-quote convention).
        # Without this, a token like ``5"display`` yields ``"5"display"`` —
        # a malformed MATCH expression that raises OperationalError.
        fts_query = " OR ".join('"' + w.replace('"', '""') + '"' for w in words)

        scoped_ns = self._scoped(namespace)
        where_clauses = ["m.namespace = :ns"]
        params: dict[str, object] = {"ns": scoped_ns, "query": fts_query, "lim": limit}

        if filter is not None:
            if filter.namespace is not None:
                where_clauses.append("m.namespace = :fns")
                # Scope the filter namespace so it matches the stored (scoped)
                # column value — a plain equality on the unscoped value would
                # always evaluate to FALSE for scoped stores.
                params["fns"] = self._scoped(filter.namespace)
            if filter.categories is not None:
                placeholders = ",".join(f":cat{i}" for i in range(len(filter.categories)))
                where_clauses.append(
                    f"EXISTS (SELECT 1 FROM json_each(m.metadata, '$.categories') WHERE value IN ({placeholders}))"
                )
                for i, category in enumerate(filter.categories):
                    params[f"cat{i}"] = category
            if filter.importance is not None:
                where_clauses.append("CAST(json_extract(m.metadata, '$.importance') AS INTEGER) >= :min_imp")
                params["min_imp"] = filter.importance
            if filter.agent_name is not None:
                where_clauses.append("json_extract(m.metadata, '$.agent_name') = :agent")
                params["agent"] = filter.agent_name
            if filter.kind is not None:
                where_clauses.append("json_extract(m.metadata, '$.kind') = :kind")
                params["kind"] = filter.kind.value
            if filter.after is not None:
                where_clauses.append("m.created_at > :after")
                params["after"] = filter.after
            if filter.before is not None:
                where_clauses.append("m.created_at < :before")
                params["before"] = filter.before

        where_sql = " AND ".join(where_clauses)

        sql = (
            f"SELECT m.id, m.content, m.namespace, m.metadata, "
            f"m.created_at, m.updated_at, bm25({self._fts_table}) AS rank "
            f"FROM {self._fts_table} ft "
            f"JOIN {self._table} m ON m.id = ft.memory_id "
            f"WHERE ft.text MATCH :query AND {where_sql} "
            f"ORDER BY rank "
            f"LIMIT :lim"
        )

        async with self._db.connect() as db:
            cursor = await db.execute(sql, params)
            rows = list(await cursor.fetchall())

        # No keyword matches — fall back to most recent memories.
        # Handles meta-queries ("What do you remember?") and queries
        # with zero token overlap in any language.
        if len(rows) == 0:
            return await self._search_recent(
                namespace=namespace,
                limit=limit,
                filter=filter,
            )

        raw_scores = [abs(row["rank"]) for row in rows]
        max_score = max(raw_scores) if len(raw_scores) > 0 else 1.0
        if max_score == 0:
            max_score = 1.0

        results: list[MemorySearchResult] = []
        for row in rows:
            entry = _row_to_entry(row, public_namespace=namespace)
            normalized_score = abs(row["rank"]) / max_score
            results.append(MemorySearchResult(entry=entry, score=normalized_score))

        return results

    async def _search_recent(
        self,
        *,
        namespace: str,
        limit: int,
        filter: MemorySearchFilter | None,
    ) -> list[MemorySearchResult]:
        """Return most recent memories when keyword search is not viable.

        Used when FTS5 returns no matches — including meta-queries like
        "What do you remember about me?", a query with no token overlap
        with stored content, a query in another language, or an empty
        namespace.
        """
        scoped_ns = self._scoped(namespace)
        where_clauses = ["namespace = :ns"]
        params: dict[str, object] = {"ns": scoped_ns, "lim": limit}

        if filter is not None:
            if filter.namespace is not None:
                where_clauses.append("namespace = :fns")
                # Scope the filter namespace to match the stored (scoped) column value.
                params["fns"] = self._scoped(filter.namespace)
            if filter.categories is not None:
                placeholders = ",".join(f":cat{i}" for i in range(len(filter.categories)))
                where_clauses.append(
                    f"EXISTS (SELECT 1 FROM json_each(metadata, '$.categories') WHERE value IN ({placeholders}))"
                )
                for i, category in enumerate(filter.categories):
                    params[f"cat{i}"] = category
            if filter.importance is not None:
                where_clauses.append("CAST(json_extract(metadata, '$.importance') AS INTEGER) >= :min_imp")
                params["min_imp"] = filter.importance
            if filter.agent_name is not None:
                where_clauses.append("json_extract(metadata, '$.agent_name') = :agent")
                params["agent"] = filter.agent_name
            if filter.kind is not None:
                where_clauses.append("json_extract(metadata, '$.kind') = :kind")
                params["kind"] = filter.kind.value
            if filter.after is not None:
                where_clauses.append("created_at > :after")
                params["after"] = filter.after
            if filter.before is not None:
                where_clauses.append("created_at < :before")
                params["before"] = filter.before

        where_sql = " AND ".join(where_clauses)
        sql = (
            f"SELECT id, content, namespace, metadata, created_at, updated_at "
            f"FROM {self._table} "
            f"WHERE {where_sql} "
            f"ORDER BY created_at DESC "
            f"LIMIT :lim"
        )

        async with self._db.connect() as db:
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()

        return [
            MemorySearchResult(entry=_row_to_entry(row, public_namespace=namespace), score=_RECENCY_FALLBACK_SCORE)
            for row in rows
        ]

    @override
    async def get(self, memory_id: str) -> MemoryEntry | None:
        """Retrieve a specific memory entry by ID.

        Args:
            memory_id: The entry's unique identifier.

        Returns:
            The entry, or ``None`` if not found.
        """
        await self._ensure_ready()
        async with self._db.connect() as db:
            cursor = await db.execute(
                f"SELECT id, content, namespace, metadata, created_at, updated_at FROM {self._table} WHERE id = :id",
                {"id": memory_id},
            )
            row = await cursor.fetchone()

        if row is None:
            return None
        return _row_to_entry(row, public_namespace=self._unscoped(row["namespace"]))

    @override
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry.

        Args:
            memory_id: The entry's unique identifier.

        Returns:
            ``True`` if deleted, ``False`` if not found.
        """
        await self._ensure_ready()
        async with self._db.connect() as db:
            cursor = await db.execute(
                f"DELETE FROM {self._table} WHERE id = :id",
                {"id": memory_id},
            )
            await db.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.debug("SQLiteMemory: deleted entry %s", memory_id)
        return deleted

    @override
    async def clear(self, *, namespace: str) -> int:
        """Delete all entries in a namespace.

        Args:
            namespace: The namespace to clear.

        Returns:
            Number of entries deleted.
        """
        await self._ensure_ready()
        scoped_ns = self._scoped(namespace)
        async with self._db.connect() as db:
            cursor = await db.execute(
                f"DELETE FROM {self._table} WHERE namespace = :ns",
                {"ns": scoped_ns},
            )
            await db.commit()
            count = cursor.rowcount

        logger.info("SQLiteMemory: cleared %d entries (namespace=%s)", count, namespace)
        return count

    @override
    async def close(self) -> None:
        """Close the database connection."""
        await self._db.close()


# =====================================================================
# Helpers
# =====================================================================


def _row_to_entry(row: aiosqlite.Row, *, public_namespace: str | None = None) -> MemoryEntry:
    """Convert a database row to a MemoryEntry.

    Args:
        row: A database row from the memories table.
        public_namespace: When supplied, use this as the returned
            ``MemoryEntry.namespace`` instead of the stored column value.
            Pass the caller-supplied (unscoped) namespace here so the
            public API remains namespace-transparent regardless of any
            scope prefix that was written to the column.
    """
    meta_dict = json.loads(row["metadata"])
    metadata = MemoryMetadata(
        source=MemorySource(meta_dict.get("source", "manual")),
        importance=meta_dict.get("importance", 3),
        categories=tuple(meta_dict.get("categories", [])),
        session_id=meta_dict.get("session_id"),
        agent_name=meta_dict.get("agent_name"),
        kind=MemoryKind(meta_dict.get("kind", "episodic")),
        custom=meta_dict.get("custom", {}),
    )
    return MemoryEntry(
        id=row["id"],
        content=row["content"],
        namespace=public_namespace if public_namespace is not None else row["namespace"],
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )

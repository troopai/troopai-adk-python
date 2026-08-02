"""Postgres + pgvector vector store.

Requires the ``vector`` Postgres extension. Cosine similarity via the ``<=>``
operator (``1 - distance``).

Uses ``psycopg_pool.AsyncConnectionPool`` opened lazily on first use
(``open=False`` then ``open()``), with the schema ensured once under an
init lock.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import cast

try:
    from pgvector import Vector
    from pgvector.psycopg import register_vector_async
    from psycopg import AsyncConnection, sql
    from psycopg.rows import TupleRow
    from psycopg.types.json import Jsonb
    from psycopg_pool import AsyncConnectionPool
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PgVectorStore requires psycopg + pgvector: pip install 'troopai-adk-python[memory-pgvector]'"
    ) from exc

from troopai.adk.memory.memory_types import (
    MemoryKind,
    MemoryMetadata,
    MemorySearchFilter,
    MemorySource,
)
from troopai.adk.memory.vector_store import VectorQueryResult, VectorRecord

logger = logging.getLogger(__name__)

_TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TABLE_NAME_MAX_LEN = 64


def _validate_table_name(table: str) -> None:
    """Reject table names that are not safe SQL identifiers (injection guard)."""
    if len(table) == 0:
        raise ValueError("PgVectorStore table name must be non-empty")
    if len(table) > _TABLE_NAME_MAX_LEN:
        raise ValueError(f"PgVectorStore table name exceeds {_TABLE_NAME_MAX_LEN} characters: {table!r}")
    if _TABLE_NAME_PATTERN.match(table) is None:
        raise ValueError(f"PgVectorStore table name must match {_TABLE_NAME_PATTERN.pattern!r} (got {table!r})")


def _meta_to_dict(meta: MemoryMetadata) -> dict[str, object]:
    return {
        "source": meta.source.value,
        "importance": meta.importance,
        "categories": list(meta.categories),
        "session_id": meta.session_id,
        "agent_name": meta.agent_name,
        "kind": meta.kind.value,
        "custom": meta.custom,
    }


def _dict_to_meta(data: dict[str, object]) -> MemoryMetadata:
    # psycopg JSONB rows surface as dict[str, object] at runtime; the values
    # were written by _meta_to_dict with known field types. cast() narrows each
    # slot without the overhead of isinstance guards on every database read.
    source_raw = cast(str, data.get("source", "manual"))
    importance_raw = cast(int, data.get("importance", 3))
    categories_raw = tuple(cast(list[str], data.get("categories", [])))
    session_raw = data.get("session_id")
    agent_raw = data.get("agent_name")
    kind_raw = cast(str, data.get("kind", "episodic"))
    custom_raw = cast("dict[str, str]", data.get("custom", {}))
    return MemoryMetadata(
        source=MemorySource(source_raw),
        importance=importance_raw,
        categories=categories_raw,
        session_id=cast(str, session_raw) if session_raw is not None else None,
        agent_name=cast(str, agent_raw) if agent_raw is not None else None,
        kind=MemoryKind(kind_raw),
        custom=custom_raw,
    )


def _apply_filter(filter: MemorySearchFilter, where: list[sql.Composed | sql.SQL], params: dict[str, object]) -> None:
    """Translate the non-namespace filter facets into SQL WHERE clauses."""
    if filter.kind is not None:
        where.append(sql.SQL("metadata->>'kind' = %(f_kind)s"))
        params["f_kind"] = filter.kind.value
    if filter.importance is not None:
        where.append(sql.SQL("(metadata->>'importance')::int >= %(f_imp)s"))
        params["f_imp"] = filter.importance
    if filter.agent_name is not None:
        where.append(sql.SQL("metadata->>'agent_name' = %(f_agent)s"))
        params["f_agent"] = filter.agent_name
    if filter.after is not None:
        where.append(sql.SQL("created_at > %(f_after)s"))
        params["f_after"] = filter.after
    if filter.before is not None:
        where.append(sql.SQL("created_at < %(f_before)s"))
        params["f_before"] = filter.before
    if filter.categories is not None:
        where.append(
            sql.SQL(
                "EXISTS (SELECT 1 FROM jsonb_array_elements_text(metadata->'categories') c WHERE c = ANY(%(f_cats)s))"
            )
        )
        params["f_cats"] = list(filter.categories)


def _row_to_record(row: tuple[object, ...]) -> VectorRecord:
    # psycopg returns untyped `object` items from cursor rows; the column order
    # is fixed by every SELECT in this module (id, namespace, content, metadata,
    # embedding, created_at, updated_at).  cast() narrows each slot safely.
    meta_dict = cast("dict[str, object]", row[3])
    embedding = cast("list[float]", row[4])
    return VectorRecord(
        id=str(row[0]),
        vector=tuple(float(x) for x in embedding),
        namespace=str(row[1]),
        content=str(row[2]),
        metadata=_dict_to_meta(meta_dict),
        created_at=cast(float, row[5]),
        updated_at=cast(float, row[6]),
    )


class PgVectorStore:
    """Postgres/pgvector-backed vector store.

    Args:
        conninfo: psycopg connection string.
        dimensions: Embedding dimension (fixes the ``vector(N)`` column).
        table: Table name (validated as a SQL identifier).
    """

    def __init__(self, *, conninfo: str, dimensions: int, table: str = "memory_vectors") -> None:
        _validate_table_name(table)
        if dimensions <= 0:
            raise ValueError(f"PgVectorStore dimensions must be > 0, got {dimensions}")
        self._conninfo = conninfo
        self._dim = dimensions
        self._tbl = sql.Identifier(table)
        self._tbl_idx = sql.Identifier(f"idx_{table}_ns")
        self._pool: AsyncConnectionPool | None = None
        self._init_lock = asyncio.Lock()

    @staticmethod
    async def _configure(conn: AsyncConnection[TupleRow]) -> None:
        await register_vector_async(conn)

    async def _ensure_ready(self) -> AsyncConnectionPool:
        """Open the pool and ensure the schema on first call.

        The init lock serializes concurrent first callers so exactly one pool
        is opened. If schema creation fails the half-opened pool is closed
        before the error propagates, so no connections leak.
        """
        async with self._init_lock:
            if self._pool is None:
                pool: AsyncConnectionPool = AsyncConnectionPool(self._conninfo, open=False, configure=self._configure)
                try:
                    await pool.open()
                    async with pool.connection() as conn:
                        await conn.execute(sql.SQL("CREATE EXTENSION IF NOT EXISTS vector"))
                        await conn.execute(
                            sql.SQL(
                                "CREATE TABLE IF NOT EXISTS {} ("
                                "id TEXT PRIMARY KEY, namespace TEXT NOT NULL, content TEXT NOT NULL, "
                                "metadata JSONB NOT NULL DEFAULT '{{}}', embedding vector({dim}) NOT NULL, "
                                "created_at DOUBLE PRECISION NOT NULL, updated_at DOUBLE PRECISION NOT NULL)"
                            ).format(self._tbl, dim=sql.Literal(self._dim))
                        )
                        await conn.execute(
                            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}(namespace)").format(self._tbl_idx, self._tbl)
                        )
                    self._pool = pool
                except BaseException:
                    await pool.close()
                    raise
            return self._pool

    async def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or update records in the pgvector table.

        Uses ``ON CONFLICT (id) DO UPDATE`` so existing records are replaced.

        Args:
            records: Records to insert or replace.

        Raises:
            ValueError: If any record's vector dimension differs from the
                store's configured dimension.
        """
        pool = await self._ensure_ready()
        async with pool.connection() as conn:
            for record in records:
                if len(record.vector) != self._dim:
                    raise ValueError(
                        f"PgVectorStore: vector dimension {len(record.vector)} does not match "
                        f"store dimension {self._dim} (did the embedding model change?)"
                    )
                await conn.execute(
                    sql.SQL(
                        "INSERT INTO {} "
                        "(id, namespace, content, metadata, embedding, created_at, updated_at) "
                        "VALUES (%(id)s, %(ns)s, %(content)s, %(meta)s, %(emb)s, %(created)s, %(updated)s) "
                        "ON CONFLICT (id) DO UPDATE SET namespace = EXCLUDED.namespace, "
                        "content = EXCLUDED.content, metadata = EXCLUDED.metadata, "
                        "embedding = EXCLUDED.embedding, updated_at = EXCLUDED.updated_at"
                    ).format(self._tbl),
                    {
                        "id": record.id,
                        "ns": record.namespace,
                        "content": record.content,
                        "meta": Jsonb(_meta_to_dict(record.metadata)),
                        "emb": Vector(record.vector),
                        "created": record.created_at,
                        "updated": record.updated_at,
                    },
                )
        logger.debug("PgVectorStore: upserted %d records", len(records))

    async def query(
        self,
        vector: tuple[float, ...],
        *,
        namespace: str,
        k: int = 5,
        filter: MemorySearchFilter | None = None,
    ) -> list[VectorQueryResult]:
        """Return the ``k`` nearest records using pgvector cosine distance.

        Cosine similarity is computed as ``1 - (embedding <=> query)``.

        Args:
            vector: Query embedding to compare against stored records.
            namespace: Namespace to search within.
            k: Maximum number of results to return.
            filter: Optional metadata filters applied as SQL WHERE clauses.

        Returns:
            List of :class:`VectorQueryResult` ordered by descending score.

        Raises:
            ValueError: If the query vector dimension differs from the
                store's configured dimension.
        """
        pool = await self._ensure_ready()
        if len(vector) != self._dim:
            raise ValueError(f"PgVectorStore: query dimension {len(vector)} != store dimension {self._dim}")
        where: list[sql.Composed | sql.SQL] = [sql.SQL("namespace = %(ns)s")]
        params: dict[str, object] = {"ns": namespace, "q": Vector(vector), "k": k}
        if filter is not None:
            _apply_filter(filter, where, params)
        stmt = sql.SQL(
            "SELECT id, namespace, content, metadata, embedding, created_at, updated_at, "
            "1 - (embedding <=> %(q)s) AS score FROM {} "
            "WHERE {} ORDER BY embedding <=> %(q)s LIMIT %(k)s"
        ).format(self._tbl, sql.SQL(" AND ").join(where))
        async with pool.connection() as conn:
            cursor = await conn.execute(stmt, params)
            rows = await cursor.fetchall()
        return [
            VectorQueryResult(record=_row_to_record(row), score=max(0.0, min(1.0, cast(float, row[7])))) for row in rows
        ]

    async def get(self, record_id: str) -> VectorRecord | None:
        """Fetch a record by its global id.

        Args:
            record_id: The unique record identifier (namespace-free).

        Returns:
            The matching :class:`VectorRecord`, or ``None`` if not found.
        """
        pool = await self._ensure_ready()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                sql.SQL(
                    "SELECT id, namespace, content, metadata, embedding, created_at, updated_at "
                    "FROM {} WHERE id = %(id)s"
                ).format(self._tbl),
                {"id": record_id},
            )
            row = await cursor.fetchone()
        return _row_to_record(row) if row is not None else None

    async def delete(self, ids: list[str]) -> int:
        """Delete records by id and return the exact count removed.

        Args:
            ids: Record identifiers to delete.

        Returns:
            Number of records actually removed (exact via ``rowcount``).
        """
        if len(ids) == 0:
            return 0
        pool = await self._ensure_ready()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                sql.SQL("DELETE FROM {} WHERE id = ANY(%(ids)s)").format(self._tbl),
                {"ids": list(ids)},
            )
            count = cursor.rowcount
            return count if count is not None else 0

    async def clear(self, *, namespace: str) -> int:
        """Delete all records in a namespace.

        Args:
            namespace: The namespace whose records should be deleted.

        Returns:
            Number of records removed (exact via ``rowcount``).
        """
        pool = await self._ensure_ready()
        async with pool.connection() as conn:
            cursor = await conn.execute(
                sql.SQL("DELETE FROM {} WHERE namespace = %(ns)s").format(self._tbl),
                {"ns": namespace},
            )
            count = cursor.rowcount
            return count if count is not None else 0

    async def close(self) -> None:
        """Close the connection pool and release Postgres connections."""
        async with self._init_lock:
            if self._pool is not None:
                pool, self._pool = self._pool, None
                await pool.close()

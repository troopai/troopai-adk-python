"""Concrete :class:`Checkpointer` implementations.

:class:`InMemoryCheckpointer` — a dict-backed, process-local store for
tests, notebooks, and single-process demos.

:class:`SQLiteCheckpointer` — a durable, single-file store (one row per
``thread_id``, latest wins) for multi-process or crash-recoverable runs.

:class:`TieredCheckpointer` — a hot/cold composite that writes to a fast
hot store and falls back to a cold store on reads, re-warming the hot
tier automatically. :meth:`~TieredCheckpointer.archive` migrates
aged hot entries to cold based on an in-process save timestamp.

:class:`~troopai.adk.graphs.checkpointers.postgres.PostgresCheckpointer` —
an ACID, JSONB-backed store with optimistic locking for shared / distributed
environments. Requires ``psycopg[binary,pool]>=3.2``; import directly from
``troopai.adk.graphs.checkpointers.postgres`` (not re-exported here to keep
the optional dependency out of the default import path).

:class:`~troopai.adk.graphs.checkpointers.redis.RedisCheckpointer` — a fast,
TTL-aware store with optimistic locking (atomic Lua compare-and-set) for
short-lived in-flight runs. Requires ``redis>=5.0``; import directly from
``troopai.adk.graphs.checkpointers.redis`` (not re-exported here to keep the
optional dependency out of the default import path).

:class:`~troopai.adk.graphs.checkpointers.s3.S3Checkpointer` — an archival,
last-write-wins store backed by AWS S3 (one JSON object per ``thread_id``).
No optimistic locking — suited for single-writer audit and long-term
archival workloads. Requires ``boto3>=1.34.0``; import directly from
``troopai.adk.graphs.checkpointers.s3`` (not re-exported here to keep the
optional dependency out of the default import path).
"""

from __future__ import annotations

from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
from troopai.adk.graphs.checkpointers.sqlite import SQLiteCheckpointer
from troopai.adk.graphs.checkpointers.tiered import TieredCheckpointer

__all__ = ["InMemoryCheckpointer", "SQLiteCheckpointer", "TieredCheckpointer"]

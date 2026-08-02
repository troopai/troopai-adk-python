"""Concrete :class:`~troopai.adk.swarms.checkpointer.SwarmCheckpointer` implementations.

:class:`InMemorySwarmCheckpointer` — a dict-backed, process-local store for
tests, notebooks, and single-process demos.

:class:`TieredSwarmCheckpointer` — a hot/cold composite that writes to a fast
hot store and falls back to a cold store on reads, re-warming the hot
tier automatically. :meth:`~TieredSwarmCheckpointer.archive` migrates
aged hot entries to cold based on an in-process save timestamp.

:class:`~troopai.adk.swarms.checkpointers.postgres.PostgresSwarmCheckpointer` —
an ACID, JSONB-backed store with optimistic locking for shared / distributed
environments. Requires ``psycopg[binary,pool]>=3.2``; import directly from
``troopai.adk.swarms.checkpointers.postgres`` (not re-exported here to keep
the optional dependency out of the default import path).

:class:`~troopai.adk.swarms.checkpointers.redis.RedisSwarmCheckpointer` —
a fast, TTL-aware store backed by Redis hashes with Lua compare-and-set
optimistic locking. Requires ``redis>=5.0``; import directly from
``troopai.adk.swarms.checkpointers.redis`` (not re-exported here to keep
the optional dependency out of the default import path).

:class:`~troopai.adk.swarms.checkpointers.s3.S3SwarmCheckpointer` —
an archival, last-write-wins object store backed by AWS S3. No optimistic
locking — S3 is the right choice for durable single-writer audit workloads.
Requires ``boto3>=1.34.0``; import directly from
``troopai.adk.swarms.checkpointers.s3`` (not re-exported here to keep
the optional dependency out of the default import path).

:class:`SwarmCheckpointerHooks` — the shared auto-save bridge each backend
registers to persist on ``on_swarm_turn_end`` / ``on_swarm_turn_interrupt``.
"""

from troopai.adk.swarms.checkpointers.hooks import SwarmCheckpointerHooks
from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer
from troopai.adk.swarms.checkpointers.tiered import TieredSwarmCheckpointer

__all__ = ["InMemorySwarmCheckpointer", "SwarmCheckpointerHooks", "TieredSwarmCheckpointer"]

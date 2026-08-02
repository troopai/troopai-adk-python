"""Vector-store backends.

``InMemoryVectorStore`` (zero-dep baseline) is exported here.  External
backends are imported from their own modules (e.g.
``from troopai.adk.memory.stores.pgvector import PgVectorStore``) so a missing
optional dependency raises a clear error only when that backend is used.
"""

from .in_memory import InMemoryVectorStore

__all__ = ["InMemoryVectorStore"]

# Graph Production State Stores

When to reach for a network-backed checkpointer, and how each backend
behaves.

## Overview

`InMemoryCheckpointer` and `SQLiteCheckpointer` cover most development and
single-host production scenarios. Switch to a network-backed backend when
your deployment needs:

- **Multi-process or multi-host resume.** A Kubernetes pod restart, a
  rolling deploy, or a lambda-style invocation that picks up a run started
  by a different process all require the checkpoint to live outside the
  originating process.
- **Horizontal scaling.** Multiple runner instances processing the same
  pipeline pool can coordinate through a shared network store. The
  optimistic-locking contract (Postgres and Redis) prevents two instances
  from overwriting each other's progress.
- **Long-term archival.** Completed or paused runs kept for audit,
  replay-analysis, or compliance belong in a durable object store (S3) rather
  than a transient Redis instance.

All checkpointers implement the same `Checkpointer` Protocol
(`save` / `load` / `list_checkpoints` / `delete` / `register`). Swapping
backends requires only changing the import and the constructor call — the
runner and the graph loop never touch the backend directly.

## Backends

### Postgres (`PostgresCheckpointer`)

ACID semantics via PostgreSQL JSONB. Each `thread_id` maps to one row;
saves are upserts guarded by a rotating fencing token (`lock_token` UUID
column). A concurrent writer that loads a stale token raises
`CheckpointConflictError` on its next save.

The table (`graph_checkpoints`) is created automatically on first
connection. PostgreSQL 13+ is required (`gen_random_uuid()` built-in).

**Install:**

```bash
pip install 'troopai-adk-python[checkpointer-postgres]'
```

**Construct:**

```python
from troopai.adk.graphs.checkpointers.postgres import PostgresCheckpointer

# conninfo is a libpq connection string.
checkpointer = PostgresCheckpointer("host=db port=5432 dbname=runs user=app")
```

Pass `conninfo` as a libpq connection string. The caller owns the lifecycle:
call `await checkpointer.close()` at application shutdown. The runner does
not close a caller-supplied checkpointer.

Use Postgres when you need full ACID guarantees, want the checkpoint schema
to live alongside your application database, or need point-in-time recovery
via WAL.

### Redis (`RedisCheckpointer`)

Fast, atomic operations via Redis hashes. Each `thread_id` maps to one
hash key (`graph:ckpt:<thread_id>`) holding a JSON payload and a fencing
token. Saves use an atomic Lua compare-and-set script so no two writers
can silently overwrite each other.

TTL is opt-in. The default keeps checkpoints until explicitly deleted.
Set `ttl_seconds` to evict stale entries automatically; keep the TTL
comfortably longer than a run's maximum expected duration to avoid a
mid-run eviction causing a spurious `CheckpointConflictError`.

**Install:**

```bash
pip install 'troopai-adk-python[checkpointer-redis]'
```

**Construct — pre-configured client:**

```python
from redis.asyncio import Redis
from troopai.adk.graphs.checkpointers.redis import RedisCheckpointer

client = Redis.from_url("redis://cache:6379/0")
checkpointer = RedisCheckpointer(client=client)
```

**Construct — URL shorthand:**

```python
checkpointer = RedisCheckpointer(url="redis://cache:6379/0", ttl_seconds=86400)
```

Supply `client=` or `url=`, not both. A `client=`-supplied instance is not
closed by the checkpointer; a `url=`-constructed client is closed on
`await checkpointer.close()`.

Use Redis when low write latency matters (sub-millisecond saves per node
completion) and eviction-based lifecycle management is acceptable.

### S3 (`S3Checkpointer`)

Archival, last-write-wins object storage. Each `thread_id` maps to one
JSON object at `s3://{bucket}/{prefix}{thread_id}.json`. S3 writes are
unconditional — there are no fencing tokens and no `CheckpointConflictError`.
Design for a single writer per `thread_id`.

The boto3 client is synchronous; all S3 calls are wrapped in
`asyncio.to_thread` so they do not block the event loop.

**Install:**

```bash
pip install 'troopai-adk-python[checkpointer-s3]'
```

**Construct:**

```python
from troopai.adk.graphs.checkpointers.s3 import S3Checkpointer

checkpointer = S3Checkpointer(
    bucket="my-graph-checkpoints",
    prefix="graph-checkpoints/",   # default; must end with "/"
    region="us-east-1",            # None delegates to boto3 resolution chain
)
```

AWS credentials are resolved through the standard boto3 chain (environment
variables, `~/.aws/credentials`, instance metadata, etc.).

Use S3 for compliance or audit workloads where runs must be retained for
months, cold-start latency on `load` is acceptable, and a single writer
per run is the operational norm.

### Tiered (`TieredCheckpointer`)

Hot-plus-cold composite that layers two backends. All writes go to the hot
tier. Reads check the hot tier first; a miss falls through to the cold tier,
and the loaded entry is re-warmed into the hot tier for subsequent reads.

`archive(graph)` migrates hot entries that were last written or re-warmed
more than `archive_after_seconds` ago to the cold tier, then removes them
from hot. The age is tracked in-memory by the composite instance — it resets
on process restart.

**Construct (Redis hot, S3 cold):**

```python
from troopai.adk.graphs.checkpointers.tiered import TieredCheckpointer

checkpointer = TieredCheckpointer(
    hot=redis_checkpointer,
    cold=s3_checkpointer,
    archive_after_seconds=3600,   # archive entries idle for 1 hour
)
```

Hook-driven saves (via `register`) write to the hot tier through the
composite's own `save`, so the archive-eligibility timestamp is updated
correctly on every hook-triggered write.

Use Tiered when you want fast in-flight writes (Redis) with automatic
long-term retention (S3), and you want a single checkpointer handle for
both concerns.

## Concurrency and Conflict Handling

Postgres and Redis both implement optimistic locking via a fencing token.
The protocol:

1. The first `save` for a `thread_id` inserts the row and caches the
   returned token.
2. Each subsequent `save` supplies the cached token in a conditional
   `UPDATE`. If the token has been rotated by a concurrent writer, the
   update matches zero rows and `CheckpointConflictError` is raised.
3. A successful `save` rotates the token and caches the new value.
4. `load` also caches the token it reads so the next `save` from that
   instance can use it.

**Reload-and-retry pattern:**

```python
from troopai.adk.exceptions import CheckpointConflictError

MAX_RETRIES = 3

for attempt in range(MAX_RETRIES):
    try:
        await checkpointer.save(checkpoint)
        break
    except CheckpointConflictError:
        if attempt == MAX_RETRIES - 1:
            raise
        # Reload to acquire the current token before retrying.
        fresh = await checkpointer.load(thread_id, graph)
        if fresh is None:
            raise
        # Rebuild the checkpoint from the fresh state before the next attempt.
```

S3 is exempt — it uses last-write-wins semantics and never raises
`CheckpointConflictError`.

## Wiring to a Graph Run

All network-backed checkpointers attach to a graph run the same way as
`SQLiteCheckpointer`. Pass the checkpointer in the `hooks=` list (or
directly to `arun_graph_from_checkpoint`). See
[`docs/graphs/checkpointing.md`](checkpointing.md) for the full
checkpoint/resume contract, the `Runner.arun_graph_from_checkpoint` API,
and the selective re-fire semantics.

```python
from troopai.adk.graphs.checkpointers.postgres import PostgresCheckpointer
from troopai.adk.run.runner import Runner

checkpointer = PostgresCheckpointer("host=db dbname=runs user=app")

# Initial run — checkpoints after every node.
result = await Runner.arun_graph(
    pipeline,
    "Process the quarterly report.",
    hooks=[checkpointer],
    thread_id="run-q4-2025",
)

# Resume after a crash or pod restart.
result = await Runner.arun_graph_from_checkpoint(
    pipeline,
    checkpointer=checkpointer,
    thread_id="run-q4-2025",
)

await checkpointer.close()
```

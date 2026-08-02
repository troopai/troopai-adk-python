# Swarm Production State Stores

When to reach for a network-backed checkpointer, and how each backend
behaves.

## Overview

`InMemorySwarmCheckpointer` covers single-process runs and tests.
Switch to a network-backed backend when your deployment needs:

- **Multi-process or multi-host resume.** A pod restart, a rolling deploy,
  or a serverless invocation that picks up a swarm run started by a different
  process requires the checkpoint to live outside the originating process.
- **Horizontal scaling.** Multiple runner instances sharing a swarm pipeline
  pool can coordinate through a shared network store. The optimistic-locking
  contract (Postgres and Redis) prevents two instances from overwriting each
  other's state.
- **Long-term archival.** Completed or paused runs kept for audit, replay,
  or compliance belong in a durable object store (S3) rather than a
  transient Redis instance.

All checkpointers expose the same surface (`save` / `load` /
`list_checkpoints` / `delete` / `register`). Swapping backends requires
only changing the import and the constructor call — the runner and the
swarm loop never touch the backend directly.

## Backends

### Postgres (`PostgresSwarmCheckpointer`)

ACID semantics via PostgreSQL JSONB. Each `thread_id` maps to one row;
saves are upserts guarded by a rotating fencing token (`lock_token` UUID
column). A concurrent writer that loads a stale token raises
`CheckpointConflictError` on its next save.

The table (`swarm_checkpoints`) is created automatically on first
connection. PostgreSQL 13+ is required (`gen_random_uuid()` built-in).

**Install:**

```bash
pip install 'troopai-adk-python[checkpointer-postgres]'
```

**Construct:**

```python
from troopai.adk.swarms.checkpointers.postgres import PostgresSwarmCheckpointer

# conninfo is a libpq connection string.
checkpointer = PostgresSwarmCheckpointer("host=db port=5432 dbname=runs user=app")
```

The caller owns the lifecycle: call `await checkpointer.close()` at
application shutdown. The runner does not close a caller-supplied checkpointer.

Use Postgres when you need full ACID guarantees, want the checkpoint schema
alongside your application database, or need point-in-time recovery via WAL.

### Redis (`RedisSwarmCheckpointer`)

Fast, atomic operations via Redis hashes. Each `thread_id` maps to one
hash key (`swarm:ckpt:<thread_id>`) holding a JSON payload and a fencing
token. Saves use an atomic Lua compare-and-set script; two concurrent writers
cannot silently overwrite each other.

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
from troopai.adk.swarms.checkpointers.redis import RedisSwarmCheckpointer

client = Redis.from_url("redis://cache:6379/0")
checkpointer = RedisSwarmCheckpointer(client=client)
```

**Construct — URL shorthand:**

```python
checkpointer = RedisSwarmCheckpointer(url="redis://cache:6379/0", ttl_seconds=86400)
```

Supply `client=` or `url=`, not both. A `client=`-supplied instance is not
closed by the checkpointer; a `url=`-constructed client is closed on
`await checkpointer.close()`.

Use Redis when low write latency matters and eviction-based lifecycle
management is acceptable.

### S3 (`S3SwarmCheckpointer`)

Archival, last-write-wins object storage. Each `thread_id` maps to one
JSON object at `s3://{bucket}/{prefix}{thread_id}.json`. S3 writes are
unconditional — there are no fencing tokens and no `CheckpointConflictError`.
Design for a single writer per `thread_id`.

The boto3 client is synchronous; all S3 calls are wrapped in
`asyncio.to_thread` to avoid blocking the event loop.

**Install:**

```bash
pip install 'troopai-adk-python[checkpointer-s3]'
```

**Construct:**

```python
from troopai.adk.swarms.checkpointers.s3 import S3SwarmCheckpointer

checkpointer = S3SwarmCheckpointer(
    bucket="my-swarm-checkpoints",
    prefix="swarm-checkpoints/",   # must end with "/"
    region="us-east-1",            # None uses the boto3 resolution chain
)
```

Use S3 for compliance or audit workloads where runs must be retained for
months, cold-start load latency is acceptable, and a single writer per run
is the operational norm.

### Tiered (`TieredSwarmCheckpointer`)

Hot-plus-cold composite that layers two backends. All writes go to the hot
tier. Reads check the hot tier first; a miss falls through to the cold tier
and the loaded entry is re-warmed into hot.

`archive(swarm)` migrates hot entries that were last written or re-warmed
more than `archive_after_seconds` ago to the cold tier, then removes them
from hot. The age is tracked in-memory by the composite instance — it resets
on process restart.

**Construct (Redis hot, S3 cold):**

```python
from troopai.adk.swarms.checkpointers.tiered import TieredSwarmCheckpointer

checkpointer = TieredSwarmCheckpointer(
    hot=redis_checkpointer,
    cold=s3_checkpointer,
    archive_after_seconds=3600,   # archive entries idle for 1 hour
    thread_id="default",           # used by register()'s auto-save hook
)
```

Hook-driven saves (via `register`) write to the hot tier through the
composite's own `save`, keeping the archive-eligibility timestamp current
on every hook-triggered write.

Use Tiered when you want fast in-flight writes (Redis) with automatic
long-term retention (S3), and a single checkpointer handle for both concerns.

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
        fresh = await checkpointer.load(thread_id, swarm)
        if fresh is None:
            raise
        # Rebuild the checkpoint from the fresh state before the next attempt.
```

S3 is exempt — it uses last-write-wins semantics and never raises
`CheckpointConflictError`.

## Wiring to a Swarm Run

Pass the checkpointer directly to the runner. The auto-save hooks fire on
`on_swarm_turn_end` and `on_swarm_turn_interrupt`, the latter ensuring that
HITL-parked state reaches the checkpoint store even when a turn suspends
before completing. No manual `swarm.hooks` wiring is needed.

```python
from troopai.adk.swarms.checkpointers.postgres import PostgresSwarmCheckpointer
from troopai.adk.run.runner import Runner

checkpointer = PostgresSwarmCheckpointer(
    "host=db dbname=runs user=app",
    thread_id="swarm-q4-2025",
)

pipeline_swarm = Swarm(
    members=(...),
    entry=entry_agent,
    policy=policy,
    termination=termination,
)

# Initial run — checkpoints after every member turn via the hook registry.
result = await Runner.arun_swarm(
    pipeline_swarm,
    "Collaborate on the quarterly report.",
    checkpointer=checkpointer,
)

# Resume after a crash or pod restart — auto-saving continues for the
# duration of the resumed run (the checkpointer is not dropped after load).
result = await Runner.arun_swarm_from_checkpoint(
    pipeline_swarm,
    checkpointer=checkpointer,
    thread_id="swarm-q4-2025",
)

await checkpointer.close()
```

The same checkpointer is available via a profile runner:

```python
result = await (
    Runner.configure()
    .swarm(pipeline_swarm)
    .checkpointer(checkpointer)
    .arun("Collaborate on the quarterly report.")
)
```

`arun_swarm_from_checkpoint` loads the `SwarmState` from the checkpoint,
rehydrates it against the supplied `Swarm` (resolving member names),
clears any parked interrupts, and re-enters the swarm loop with the
carried-over `total_turns`, `shared_history`, and `per_agent_scratch`.
The checkpointer continues auto-saving for the duration of the resumed run.

> **thread_id consistency:** The `checkpointer` passed to
> `arun_swarm_from_checkpoint` MUST have been constructed with the same
> `thread_id` as the `thread_id` argument — the load reads from the
> `thread_id` argument, but resume auto-saves write under the checkpointer's
> own `thread_id` (set at construction), so a mismatch would load from one
> key and save to a different one.

See [`docs/swarms/hitl.md`](hitl.md) for the full interrupt/resume surface.

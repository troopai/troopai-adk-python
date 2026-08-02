# Graph Node Reliability: Per-Node Timeout and Retry

Bound runaway nodes and recover from transient failures — without changing
the behaviour of nodes that opt into neither feature.

## Why

A graph node that calls an external API or runs a long agent turn can block a
superstep indefinitely. A transient network hiccup should not kill an entire
pipeline. Per-node timeout and retry address both concerns.

Both features are **opt-in and default-off**:

- `NodeRetryPolicy()` defaults to `max_attempts=1` — one attempt, no retries.
- `GraphConfig.per_node_timeout` defaults to `None` — no timeout.

This means a graph that does not configure these fields behaves exactly as it
did before the features existed. The framework never adds cost the developer did
not choose. See [Parity Guarantee](#parity-guarantee).

## `per_node_timeout` and `NodeRetryPolicy`

Both live in `troopai.adk.graphs.config`:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class NodeRetryPolicy:
    max_attempts: int = 1        # 1 = no retries; N = up to N attempts
    initial_backoff: float = 1.0 # seconds before first retry
    max_backoff: float = 30.0    # cap on backoff duration
    retry_on: tuple[type[Exception], ...] = ()
    # empty = retry on every Exception; non-empty = only those types
```

`GraphConfig` carries the graph-level defaults:

```python
@dataclass(frozen=True)
class GraphConfig:
    default_retry: NodeRetryPolicy = field(default_factory=NodeRetryPolicy)
    per_node_timeout: float | None = None
    fail_fast: bool = True
    # ... other fields
```

### Per-attempt timeout

When `per_node_timeout` is set (or overridden on a node), every attempt gets the
full timeout — each retry starts a fresh `asyncio.timeout(timeout)` context.
A timeout that fires raises Python's built-in `TimeoutError` internally and is
then translated to `GraphNodeTimeoutError` by the reliability wrapper (see
[Failure-Boundary Contract](#failure-boundary-contract)).

### Retry backoff

Between attempts, the wrapper sleeps for `backoff` seconds, then doubles it,
capped at `max_backoff`:

```
attempt 1 fails → sleep initial_backoff
attempt 2 fails → sleep min(initial_backoff * 2, max_backoff)
attempt 3 fails → sleep min(initial_backoff * 4, max_backoff)
...
```

### `retry_on` semantics

- **Empty tuple (default)**: every `Exception` is retryable (subject to
  `max_attempts`). `asyncio.CancelledError` is `BaseException`, not `Exception`,
  and is never caught — a fail-fast sibling cancel propagates cleanly.
- **Non-empty tuple**: only instances of those exception types trigger a retry;
  any other exception propagates immediately without retrying.

A timeout (`TimeoutError`) is retried only when `TimeoutError` is in `retry_on`
(or `retry_on` is empty). However, a timeout on the **final attempt** always
surfaces as `GraphNodeTimeoutError` regardless — see
[Failure-Boundary Contract](#failure-boundary-contract).

## Per-Node Override

`GraphNode` carries two optional fields that override the graph-level defaults
when set to a non-`None` value:

```python
@dataclass(frozen=True)
class GraphNode:
    retry: NodeRetryPolicy | None = None
    # None ⇒ inherit GraphConfig.default_retry

    timeout: float | None = None
    # None ⇒ inherit GraphConfig.per_node_timeout
```

The effective policy is resolved by `resolve_node_reliability` in
`run/node_reliability.py`:

```python
def resolve_node_reliability(
    graph: Graph[Any],
    node: GraphNode,
) -> tuple[NodeRetryPolicy, float | None]:
    policy = node.retry if node.retry is not None else graph.config.default_retry
    timeout = node.timeout if node.timeout is not None else graph.config.per_node_timeout
    return policy, timeout
```

### Setting per-node overrides

`GraphNode` and `Graph` are frozen dataclasses. For graph-level defaults,
pass `default_retry` and `per_node_timeout` to `GraphConfig`. For a per-node
override, use `dataclasses.replace` on a compiled graph's node — the mechanism
used by `examples/graphs/node_reliability.py`:

```python
import asyncio
import dataclasses
import logging
from troopai.adk.graphs import Graph, GraphConfig
from troopai.adk.graphs.config import NodeRetryPolicy
from troopai.adk.run.runner import Runner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def fetch_data(text: str) -> str:
    return f"fetched:{text}"


async def summarize(text: str) -> str:
    return f"summary:{text}"


# Graph-level default: 3 attempts, 30 s timeout per attempt.
default_policy = NodeRetryPolicy(
    max_attempts=3,
    initial_backoff=1.0,
    max_backoff=10.0,
    retry_on=(IOError, TimeoutError),
)

# Build and compile with graph-level defaults.
pipeline = (
    Graph.new("reliability-demo", description="Fetch → summarize with retry/timeout")
    .node("fetch", fetch_data)
    .node("summarize", summarize)
    .pipe("fetch", "summarize")
    .entry("fetch")
    .terminal("summarize")
    .with_config(GraphConfig(
        default_retry=default_policy,
        per_node_timeout=30.0,
    ))
    .compile()
)

# Per-node override: give "fetch" a tighter policy than the graph default.
fetch_node = pipeline.get_node("fetch")
fetch_override = dataclasses.replace(
    fetch_node,
    retry=NodeRetryPolicy(max_attempts=5, initial_backoff=0.5, max_backoff=5.0),
    timeout=10.0,
)
pipeline = dataclasses.replace(
    pipeline,
    nodes=tuple(fetch_override if n.id == "fetch" else n for n in pipeline.nodes),
)


async def main() -> None:
    result = await Runner.arun_graph(pipeline, "quarterly report")
    logger.info("status=%s output=%s", result.status, result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
```

The override fields (`retry`, `timeout`) on `GraphNode` take effect automatically
— no other wiring is needed. `resolve_node_reliability` is called by
`_invoke_node` in `run/graph_loop.py` before every node execution.

## Exceptions

Two exceptions are raised by the reliability wrapper and reach the normal graph
error path (`GraphRunResult.error`, `on_node_error` hook, `fail_fast`
interaction):

### `GraphNodeTimeoutError`

```python
class GraphNodeTimeoutError(TroopAIError):
    node_id: str    # id of the node that timed out
    timeout: float  # per-attempt timeout that was configured
    attempts: int   # number of attempts made before giving up
```

`.__cause__` is the underlying `TimeoutError`. Raised when a node's **final**
attempt hits the per-attempt timeout — regardless of the retry configuration.

### `NodeRetriesExhaustedError`

```python
class NodeRetriesExhaustedError(TroopAIError):
    node_id: str        # id of the node that exhausted its budget
    attempts: int       # == policy.max_attempts
    last_error: Exception  # the exception from the final attempt
```

`.__cause__` is also set to `last_error` (`raise ... from last_error`). Raised
when a retryable exception occurs on the final attempt of a multi-attempt policy
that did not time out.

Both exceptions surface through the same error path as any other node exception:

- `GraphHooks.on_node_error(context, state, node_id, exc)` is called.
- Under `fail_fast=True` (default) the run is marked `FAILED` and sibling tasks
  are cancelled. See [fail_fast Interaction](#fail_fast-interaction).
- `GraphRunResult.error` carries the serialised message.

## Failure-Boundary Contract

The exact decision order in `run_node_with_reliability`:

1. **Timeout on the final attempt**: always raises `GraphNodeTimeoutError`,
   regardless of `max_attempts`, `retry_on`, or retryability. A single-attempt
   node that times out raises `GraphNodeTimeoutError`, not the raw `TimeoutError`.
2. **Retryable exception on the final attempt of a multi-attempt policy**:
   raises `NodeRetriesExhaustedError` chained from the original exception.
3. **All other cases** (non-retryable exception, or single-attempt policy with a
   non-timeout exception): the original exception is re-raised unchanged. No
   wrapping occurs.

The priority of rule 1 over rule 2 means: if you configure both a timeout and
`max_attempts > 1`, and the last attempt times out, you get
`GraphNodeTimeoutError` (not `NodeRetriesExhaustedError`), even though a
non-timeout failure on the same last attempt would have produced
`NodeRetriesExhaustedError`.

## Parity Guarantee

A node that configures neither `retry` nor `timeout` (and whose graph uses the
default `GraphConfig()`) is executed exactly once, and any exception it raises
propagates unchanged — no `GraphNodeTimeoutError`, no `NodeRetriesExhaustedError`,
no extra wrapping. This is rule 3 of the failure-boundary contract and is
enforced unconditionally. Existing graphs require no migration.

## `fail_fast` Interaction

Timeout and retry exceptions enter the same error path as any other node
exception. The `fail_fast` field on `GraphConfig` governs what happens next:

- **`fail_fast=True` (default)**: the first node error in a superstep cancels
  all sibling tasks immediately via `asyncio.wait(FIRST_COMPLETED)`. The run
  exits with `GraphRunStatus.FAILED` and the exception message is in
  `GraphRunResult.error`. `asyncio.CancelledError` from a cancelled sibling is
  `BaseException` and is not caught by the reliability wrapper — cancellation
  propagates cleanly.
- **`fail_fast=False`**: sibling tasks in the same superstep are allowed to
  finish. Downstream nodes that depended on the failed node's output do not fire
  (their `JoinBarrier` never becomes ready). Unaffected parallel branches
  complete normally.

For `fail_fast` basics and the full error-handling model, see
`docs/graphs/graphs.md`.

## See Also

- `docs/graphs/graphs.md` — profile runner API, `fail_fast`, error handling,
  decision tree.
- `docs/graphs/checkpointing.md` — crash recovery, selective re-fire, cumulative
  budgets.
- `src/troopai/adk/run/node_reliability.py` — `resolve_node_reliability`,
  `run_node_with_reliability`.
- `src/troopai/adk/graphs/config.py` — `NodeRetryPolicy`, `GraphConfig`.
- `src/troopai/adk/exceptions/exceptions.py` — `GraphNodeTimeoutError`,
  `NodeRetriesExhaustedError`.

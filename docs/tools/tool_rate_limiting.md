# Per-Tool Rate Limiting — `ToolRateLimit`

Cap the rate at which a `FunctionTool` may be invoked. The framework
enforces a sliding 60-second window per tool and either sleeps until a
slot opens (`"wait"`) or returns a clear error to the LLM (`"error"`).

## When to use

| Situation | Use |
|-----------|-----|
| Tool calls a third-party API with an RPM budget (search, weather, payments) | `rate_limit=ToolRateLimit(rpm=N)` |
| Tool hits an internal database / queue you want to protect | Same |
| You want the LLM to know throttling happened (so it can adapt) | `behavior="error"` |
| You want throttling to be invisible to the LLM | `behavior="wait"` (default) |
| Quick declaration, default behaviour | `max_calls_per_minute=N` shorthand |

## Quick example

```python
from troopai.adk.tools import ToolRateLimit, function_tool


@function_tool(
    name="search_api",
    description="Search the public knowledge base.",
    max_calls_per_minute=30,   # shorthand for ToolRateLimit(rpm=30)
)
def search_api(query: str) -> str:
    ...
```

Or with explicit behaviour control:

```python
@function_tool(
    name="payment_api",
    description="Process a payment.",
    rate_limit=ToolRateLimit(rpm=10, behavior="error"),
)
def payment_api(amount: int) -> str:
    ...
```

## Algorithm

The executor maintains a `deque[float]` of `time.monotonic()` timestamps
on the tool instance. Before each invocation:

1. Drop timestamps older than 60 seconds (sliding window).
2. If the deque length is below `rpm`, append `now` and admit the call.
3. Otherwise compute `retry_after = 60 - (now - oldest_timestamp)`.
   - `behavior="wait"`: `await asyncio.sleep(retry_after)` and retry.
   - `behavior="error"`: return a rate-limit error result; the underlying
     handler is **not** invoked.

`time.monotonic()` is the right clock for this: NTP corrections, DST
transitions, and clock drift cannot accidentally let extra calls
through (or freeze the tool).

## Behaviour comparison

| | `wait` (default) | `error` |
|---|---|---|
| Underlying handler runs | Always (after sleep) | Only if window open |
| LLM sees throttling | No | Yes — gets a clear error |
| Cost when saturated | Sleep time, no token cost | One tool result, model can adapt |
| Best for | Background work, "fire and forget" | Interactive flows where the model should decide what to do |

## Capping the wait — `max_wait_seconds`

`behavior="wait"` without a cap can hold the event loop indefinitely
in the worst case. With `rpm=1` and a tight tool-call loop across
`max_turns=20`, the cumulative wall-clock hold can reach ~10–20
minutes. Under parallel tool execution, N concurrent waiters can each
sleep one window, multiplying the hold by N.

In multi-tenant deployments (or anywhere a long sleep would starve
other coroutines), set `max_wait_seconds`:

```python
ToolRateLimit(rpm=10, behavior="wait", max_wait_seconds=5.0)
```

If a single `acquire_rate_slot()` call would accumulate more than
`max_wait_seconds` of sleep, it falls back to `behavior="error"`
semantics — returns `False` so the LLM sees a clear rate-limit
message instead of the run blocking. Default `None` preserves the
unbounded-wait behaviour.

## Concurrency

The rate limiter uses an `asyncio.Lock` to serialise window updates so
parallel tool calls (e.g. when `tool_execution_mode=PARALLEL`) cannot
overshoot the limit. The lock is created lazily on the first call so
it binds to the event loop the tool is running in — never to a loop
that happened to be active at construction time.

## State location

State is per-`FunctionTool` instance:

- `_rate_state: deque[float]` — timestamp log
- `_rate_lock: asyncio.Lock | None` — created on first acquire

The state lives on the instance and persists across runs in the same
process. Two tools that should share a backend budget must share a
single `FunctionTool` instance — declaring the same `ToolRateLimit`
on two different tools gives them independent windows.

If you need to clear the rate-limit state (e.g. between independent
test cases), construct a fresh `FunctionTool` instance.

## Mutual exclusion at the decorator

`rate_limit=` and `max_calls_per_minute=` are mutually exclusive on
`@function_tool`:

```python
# ValueError at decoration time
@function_tool(name="x", rate_limit=ToolRateLimit(rpm=5), max_calls_per_minute=10)
def x() -> str:
    return ""
```

Use `max_calls_per_minute` for the common case; reach for `rate_limit`
only when you need non-default behaviour.

## What this does NOT do

- It does not coordinate across processes. A two-process deployment
  with `rpm=30` per process can hit the upstream API at 60 rpm.
  Cross-process coordination requires a shared store (Redis, a custom
  middleware) — out of scope for this framework feature.
- It does not retry on upstream rate-limit responses (HTTP 429). If
  the upstream rate-limits you despite this guard, the tool's own
  error handler decides what to do.
- It does not retroactively adjust the window when a call fails. A
  failed invocation still counts toward the rpm budget — the time
  has been spent, and the upstream may have charged for it.

## See also

- `tests/unit/tools/test_tool_rate_limiting.py` — full behaviour tests
- `examples/tools/tool_rate_limiting.py` — runnable example

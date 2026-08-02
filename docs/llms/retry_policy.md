# LLM Retry Policy

Framework-level retry with exponential backoff and jitter for transient LLM
failures. Complements (rather than replaces) `LLMConfig.num_retries`, which
is the SDK-side retry hint forwarded to the provider library.

## Why two retry knobs?

| Knob | Runs where | Knobs exposed |
|------|------------|---------------|
| `LLMConfig.num_retries` | Inside the provider library (litellm, anthropic-sdk) | Count only |
| `LLMConfig.retry_policy` | **Outside** the SDK, in the framework | Count, delay, backoff multiplier, cap, jitter, category filter |

Use `num_retries` for cheap transient-error recovery that the SDK can handle
itself. Use `retry_policy` when you need:

- Explicit backoff and jitter control (avoid thundering herds)
- A category filter (retry timeouts but not rate limits, or vice versa)
- Observability — retry attempts emit framework logs instead of disappearing into the SDK

## Error categories

Each `LLM` implementation classifies provider-specific exceptions into one of:

- `"rate_limit"` — HTTP 429, upstream throttling
- `"server_error"` — HTTP 5xx, provider-side faults
- `"timeout"` — connect/read timeouts

Authentication errors, invalid-request errors, and other permanent failures
are deliberately **not** retried — no category covers them.

## Basic usage

```python
from troopai.adk.llms import LLMConfig
from troopai.adk.types.llms import LLMRetryPolicy

config = LLMConfig(
    retry_policy=LLMRetryPolicy(
        max_retries=5,
        initial_delay=0.5,
        max_delay=30.0,
        multiplier=2.0,
        jitter=True,
    )
)
```

Delay progression (no jitter, `initial_delay=0.5`, `multiplier=2.0`, `max_delay=30`):

```
attempt 0 → 0.5s
attempt 1 → 1.0s
attempt 2 → 2.0s
attempt 3 → 4.0s
attempt 4 → 8.0s
attempt 5 → 16.0s
attempt 6 → 30.0s  (capped)
```

With `jitter=True` (default), each delay is randomized to
`[0, computed_delay]` — avoiding the thundering-herd failure mode where many
workers retry in lock-step after a shared upstream outage.

## Category filtering

```python
from troopai.adk.types.llms import LLMRetryPolicy

# Retry server errors and timeouts, but give up immediately on rate limits
# (useful when upstream has a hard quota and retries are pointless).
policy = LLMRetryPolicy(
    retry_on=frozenset(["server_error", "timeout"]),
)
```

## Interaction with streaming

Retries apply to **non-streaming** calls only. Reconnecting mid-stream would
silently lose tokens or double-emit events, so streaming failures surface
immediately. If you need retry semantics around a streamed call, wrap it in
your own loop that discards the partial output and starts fresh.

## See also

- `src/troopai/adk/types/llms/retry_policy.py` — dataclass definition
- `src/troopai/adk/llms/litellm/litellm_retry.py` — litellm exception classifier and retry loop
- `tests/unit/llms/test_retry_policy.py` — tests for the policy and loop

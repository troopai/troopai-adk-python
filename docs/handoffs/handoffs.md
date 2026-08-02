# Handoffs

Agent-to-agent routing with temporal slicing, filters, and cost optimization.

## HandoffInputData

When a handoff occurs, `HandoffInputData` captures the conversation split into
temporal slices:

| Field | Description |
|-------|-------------|
| `intent` | What triggered the handoff (Intent, Pydantic model, or raw string) |
| `context` | Messages that existed **before** the agent's turn started |
| `output` | Messages generated **during** the agent's turn |
| `forwarded` | Filtered subset for the next agent (`None` = use `context + output`) |

The `.messages` property returns `context + output` as a flat tuple.

### Why temporal slicing?

Without it, filters and callbacks see a flat list with no way to distinguish
prior context from the current agent's work. With temporal slicing:

- Filters can trim old context while preserving recent tool results
- Audit callbacks can analyze what the agent did vs what it inherited
- Cost optimization: summarize old history, keep fresh output intact

### RunItem type

Fields use `RunItem` (from `troopai.adk.types.items`) — a Union of typed
`@dataclass(frozen=True)` item classes that wrap Layer 1 (provider-agnostic)
types. Each item provides `to_param()` for Layer 1 replay plus dict-like
access (`.get()`, `[]`, `.items()`, `in`) for mapping-style consumers.

Item classes: `SystemItem`, `UserItem`, `MessageOutputItem`, `ToolCallItem`,
`ToolCallOutputItem`, `ReasoningItem`, `HandoffOutputItem`, `CompactionItem`.

Filters should use `isinstance` checks for type-safe dispatch:
```python
from troopai.adk.types.items import ToolCallItem, ToolCallOutputItem

for item in data.output:
    if isinstance(item, ToolCallItem):
        print(f"Tool call: {item.name}")
```

## Callback Signatures

`on_handoff` supports three signatures, auto-detected from parameter annotations:

**`(ctx)`** — Side effects only:
```python
def log_handoff(ctx: RunContext) -> None:
    print("Handoff occurred")
```

**`(ctx, intent)`** — Access the validated intent:
```python
def log_refund(ctx: RunContext, intent: RefundIntent) -> None:
    print(f"Refund for order {intent.order_id}")
```

**`(ctx, data: HandoffInputData)`** — Access temporal slices:
```python
def audit(ctx: RunContext, data: HandoffInputData) -> None:
    print(f"Context: {len(data.context)} msgs, Output: {len(data.output)} msgs")
    if data.forwarded is not None:
        print(f"Forwarded {len(data.forwarded)} of {len(data.messages)} msgs")
```

The third variant is detected by the `HandoffInputData` type annotation on the
second parameter. All three support both sync and async callbacks.

## Filters

Filters transform what gets forwarded to the next agent without modifying
the audit trail (context + output are never mutated).

### How filters work

1. A filter reads from `forwarded` (if set by a prior filter) or `messages`
2. It sets `forwarded` on the result via `data.clone(forwarded=...)`
3. `prepare_handoff_input()` uses `forwarded` if set, else `messages`

### Built-in filters

| Filter | Effect |
|--------|--------|
| `remove_tool_calls` | Strip tool call/result messages |
| `remove_system_messages` | Strip system messages |
| `keep_last_n(n)` | Keep only last N messages |
| `forward_intent` | Append classified Intent as user message |
| `intent_only` | Set `forwarded=()` — next agent sees only system prompt + intent |
| `compose(*filters)` | Chain multiple filters left-to-right |

### Custom filter example

```python
def keep_recent_context(n: int = 3):
    """Trim old context, keep full output."""
    def _filter(data: HandoffInputData) -> HandoffInputData:
        trimmed = data.context[-n:] if len(data.context) > n else data.context
        return data.clone(forwarded=trimmed + data.output)
    return _filter
```

### Composing filters

```python
from troopai.adk.handoffs.handoff_filters import compose, remove_tool_calls

pipeline = compose(remove_tool_calls, keep_recent_context(3))
Handoff(target=agent, input_filter=pipeline)
```

In a compose chain, each filter reads `forwarded` from the previous filter's
output, so the chain builds incrementally.

## Examples

- `examples/handoffs/temporal_slicing.py` — Temporal-aware callbacks and filters
- `examples/handoffs/message_filters.py` — Built-in, custom, and composed filters
- `examples/handoffs/cost_optimized.py` — HandoffConfig strategies and budget caps

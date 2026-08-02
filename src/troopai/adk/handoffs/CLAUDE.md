# Handoffs Module

Agent-to-agent routing with two strategies: code-orchestrated and LLM-orchestrated.

## Architecture

```
handoffs/
├── handoff.py           # Handoff — per-agent LLM handoff config + handoff() factory
├── handoff_route.py     # HandoffRoute — deterministic Intent-based routing DSL + handoff_route() factory
├── handoff_target.py    # HandoffTarget — code-orchestrated target (used by HandoffRoute)
├── handoff_config.py    # HandoffConfig — strategy, window, etc.
├── handoff_input_data.py # HandoffInputData — temporal-sliced data passed between agents
├── handoff_prompt.py    # Opt-in system prompt prefix for LLM-orchestrated handoffs
├── handoff_filters.py           # Built-in input filters (remove_tool_calls, keep_last_n, etc.)
├── handoff_helpers.py           # Internal helpers for Runner (normalize, build_tools, find_target)
└── __init__.py          # Public API exports
```

## Two Strategies

### Code-Orchestrated (HandoffRoute)

Deterministic routing via Intent type matching. Zero LLM routing tokens.

**Key classes:** `HandoffRoute` → `HandoffTarget`
**Factory:** `handoff_route()` builds from `(Intent, Agent)` tuples.

### LLM-Orchestrated (list)

Agents exposed as `transfer_to_<name>` tools. LLM decides transfers.

**Key classes:** `Handoff` (data object), `handoff_helpers.py` (Runner helpers)
**Factory:** `handoff()` builds a single `Handoff` with named params.

## HandoffInputData (Temporal Slicing)

Separates messages into temporal slices so filters and callbacks can distinguish
what happened before vs during the agent's turn.

| Field | Type | Description |
|-------|------|-------------|
| `intent` | Any | What triggered the handoff (Intent, Pydantic model, or raw string) |
| `context` | `tuple[RunItem, ...]` | Messages BEFORE the agent's turn |
| `output` | `tuple[RunItem, ...]` | Messages DURING the agent's turn |
| `forwarded` | `Optional[tuple[RunItem, ...]]` | Filtered subset for next agent (None = use messages) |

**Properties:** `.messages` returns `context + output` (full view, backward-compatible).

**`RunItem`** (`types/items.py`): Union of typed `@dataclass(frozen=True)` item classes
(`SystemItem`, `UserItem`, `MessageOutputItem`, `ToolCallItem`, `ToolCallOutputItem`,
`ReasoningItem`, `HandoffOutputItem`, `CompactionItem`). Each wraps Layer 1 types and
provides `to_param()` for Layer 2 conversion. Filters use `isinstance` checks for dispatch.

### Callback Signatures (three supported)

| Signature | Use case |
|-----------|----------|
| `(ctx)` | Side effects only — logging, metrics |
| `(ctx, intent)` | Access validated typed input or raw Intent |
| `(ctx, data: HandoffInputData)` | Access temporal slices, full audit trail |

Detection: 1 param = no-input, 2 params + annotation is `HandoffInputData` = full data, else = intent.

### How Filters Work

Filters operate on `forwarded` (if set by a prior filter in a compose chain) or `messages`.
They set `forwarded` on the result — context/output are never mutated (audit trail preserved).
`prepare_handoff_input()` uses `forwarded` if set, else `messages`.

## Key Classes

### Handoff (handoff.py)

Per-agent config for LLM-orchestrated handoffs. Frozen dataclass.

| Field | Type | Description |
|-------|------|-------------|
| `target` | Agent | Target agent (required) |
| `name` | str? | Custom tool name (default: `transfer_to_{agent_name}`) |
| `description` | str? | Tool description for LLM |
| `on_handoff` | callback? | Invoked when handoff occurs (3 signature variants) |
| `input_type` | type? | Pydantic model → schema + validation |
| `schema` | dict? | JSON Schema for tool args (auto-generated from input_type) |
| `schema_enforcement` | SchemaEnforcement | Schema enforcement level (default: STRICT) |
| `input_filter` | callable? | Transform handoff data |
| `enabled` | bool/callable | Whether target is active |
| `config` | HandoffConfig | Strategy, window, budget |

Methods: `get_name()`, `get_description()`, `to_tool_definition()`, `invoke()`

### HandoffConfig

| Field | Type | Description |
|-------|------|-------------|
| `strategy` | Literal | "full", "last_n", "intent_only", "summary" |
| `window` | int? | Number of messages for `last_n` strategy |
| `budget` | int? | Max tokens of history to transfer (oldest dropped via truncation if exceeded — no LLM call; default `20_000`) |
| `collapse` | bool | Collapse history into a single system message (default: False) |

### HandoffRoute (handoff_route.py)

Fluent DSL for Intent-based routing. `.when(*intents).to(agent)` chain.

### HandoffTarget (handoff_target.py)

Internal target for code-orchestrated routing. Created by `HandoffRoute.when().to()`.

## How the Runner Processes Handoffs

The loop tracks `context_end` — an index into `messages` marking where context ends
and output begins. At handoff time: `context = messages[:context_end]`,
`output = messages[context_end:]`. After handoff, `context_end` resets.

1. **Code-orchestrated**: `agent.handoffs.resolve(intent, ctx)` → `target.invoke(intent, context, output, run_context)`
2. **LLM-orchestrated**: `find_handoff_target()` → `target.invoke(tool_args, context, output, run_context)`

**System prompt injection:** After any handoff, `inject_system_prompt()` replaces the source agent's system prompt with the target agent's.

## Filters

Built-in filters in `handoff_filters.py`:
- `forward_intent` — Append the classified Intent as a user message
- `remove_tool_calls` — Strip tool call/result messages
- `remove_system_messages` — Strip system messages
- `keep_last_n(n)` — Keep only last N messages
- `intent_only` — Remove all messages, keep only intent
- `compose(*filters)` — Chain multiple filters into a pipeline

See `examples/handoffs/message_filters.py` and `examples/handoffs/temporal_slicing.py`.

## Cost Optimization

**`SchemaEnforcement.COMPACT`** strips Pydantic metadata from schemas.

See `docs/handoffs/handoffs.md` for detailed documentation.

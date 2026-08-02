(guides/handoffs)=

# 🔀 Handoffs

A **handoff** is directed routing from one agent to another. When a handoff
fires, execution leaves the current agent permanently and continues inside the
target agent with a (potentially filtered) slice of the conversation history.
Unlike `as_tool()`, which delegates and returns, a handoff is a one-way
transfer.

The mechanic underlying all handoffs is a tool call. Each handoff target is
registered as a `transfer_to_<name>` function tool on the calling agent; the
LLM (or Python code) selects one; the Runner intercepts the call and routes
execution. The interception keeps history unambiguous: every handoff leaves a
`HandoffCallItem` + `HandoffOutputItem` pair in `RunResult.new_items`. See
[Architecture: Handoffs & Swarms](../architecture/handoffs-and-swarms.md) for the rationale.

---

## Two orchestration modes

The ADK supports two distinct modes for deciding *which* agent to hand off to.
Choosing the right one is primarily a cost and predictability trade-off.

### LLM-orchestrated

Pass a list of agents (or `Handoff` objects) to `Agent.handoffs`. The Runner
synthesises a `transfer_to_<name>` tool per target and adds it to the calling
agent's tool list. The LLM calls the appropriate tool when it decides a
handoff is needed.

```python
triage = Agent(
    name="Triage",
    system_prompt="Route the user to the right specialist.",
    handoffs=[refunds_agent, billing_agent, technical_agent],
)
```

Use LLM-orchestrated handoffs when:

- The triage decision requires reasoning over conversation content.
- The set of agents is small (each adds a tool to the context window).
- You want the LLM to handle edge cases and ambiguity naturally.

### Code-orchestrated

Set `Agent.output_schema` to a union of `Intent` types and assign a
`HandoffRoute` to `Agent.handoffs`. The LLM outputs a structured intent;
`HandoffRoute.resolve()` maps intent type → agent in pure Python. Zero LLM
routing tokens are spent on the routing decision itself.

```python
from troopai.adk.types.intents import Intent, Respond
from troopai.adk.handoffs import HandoffRoute

class RefundIntent(Intent):
    kind: Literal["refund"] = "refund"
    order_id: str | None = None

TriageOutput = Union[RefundIntent, BillingIntent, Respond]

triage = Agent(
    name="Triage",
    system_prompt="Classify the user's request into an intent type.",
    output_schema=TriageOutput,
    handoffs=(
        HandoffRoute("triage")
        .when(RefundIntent).to(refunds_agent)
        .when(BillingIntent).to(billing_agent)
        .otherwise(general_agent)
    ),
)
```

Use code-orchestrated handoffs when:

- Classification is the triage agent's only job; specialists do the reasoning.
- You want deterministic, auditable routing.
- Cost matters and the routing set is large.

The `handoff_route()` factory provides a more compact spelling when you do not
need per-route callbacks or filters:

```python
from troopai.adk.handoffs import handoff_route

triage = Agent(
    handoffs=handoff_route(
        (RefundIntent, refunds_agent),
        (BillingIntent, billing_agent),
        otherwise=general_agent,
    ),
)
```

---

## `Handoff` and `HandoffConfig`

### `Handoff` — per-target configuration

`Handoff` is a frozen dataclass that configures one LLM-orchestrated handoff
target. Bare agents in `Agent.handoffs` are auto-wrapped in `Handoff` with
defaults; construct `Handoff` explicitly when you need any of its knobs.

| Field | Type | Purpose |
|---|---|---|
| `target` | `Agent` | The agent to hand off to (required). |
| `name` | `str \| None` | Custom tool name. Default: `transfer_to_{agent_name_snake_case}`. |
| `description` | `str \| None` | Tool description for the LLM. Default: auto-generated from `target.description` or agent name. |
| `on_handoff` | callback | Invoked when the handoff fires. Supports three signatures (see [Callback signatures](#callback-signatures)). |
| `input_type` | Pydantic model type | Typed tool arguments. Auto-generates JSON schema; validates and parses tool call args into the model. |
| `input_filter` | `HandoffInputFilter \| None` | Transforms `HandoffInputData` before passing it to the next agent. |
| `enabled` | `bool \| callable` | Whether this target is active. Disabled targets are hidden from the LLM. |
| `config` | `HandoffConfig` | Strategy, window, budget, collapse, and error policy. |
| `metadata` | `Mapping[str, str]` | Arbitrary labels for tracing and telemetry. Not shown to the LLM. |

### `HandoffConfig` — context transfer policy

`HandoffConfig` controls how much of the conversation history the target agent
receives and what happens when callbacks fail.

| Field | Type | Default | Purpose |
|---|---|---|---|
| `strategy` | `HandoffStrategy` | `FULL` | Which messages to include: `FULL`, `LAST_N`, `INTENT_ONLY`, `SUMMARY`. |
| `window` | `int \| None` | `None` | Number of messages when `strategy=LAST_N`. |
| `budget` | `TokenBudget \| int \| None` | `20_000` | Token cap on transferred history. Oldest messages are dropped (no LLM call) when exceeded. `None` disables the cap. |
| `collapse` | `HandoffCollapseMode \| bool` | `OFF` | Wrap transferred history into a single system or user message to save tokens. |
| `on_error` | `"halt" \| "reject_with_message"` | `"halt"` | What to do when `input_filter` or `on_handoff` raises. `"reject_with_message"` surfaces the error to the LLM as a tool result; the LLM can retry. |
| `error_message_builder` | `Callable[[Exception], str] \| None` | `None` | Custom formatter for the rejection message when `on_error="reject_with_message"`. |

```python
from troopai.adk.handoffs import Handoff
from troopai.adk.handoffs.handoff_config import HandoffConfig

Handoff(
    target=billing_agent,
    description="Transfer for invoice or payment questions.",
    config=HandoffConfig(
        budget=5_000,          # at most 5,000 tokens of prior history
        collapse=True,         # wrap history into one system message
        on_error="reject_with_message",
    ),
)
```

---

## `input_filter` and `HandoffInputFilter`

`Handoff.input_filter` is a callable `(HandoffInputData) -> HandoffInputData`
that transforms what the next agent sees before the handoff executes. Filters
are applied after `HandoffConfig.strategy` selects messages but before the
target agent's first turn.

Filters operate on the `forwarded` field: each filter reads from `forwarded`
(if set by a prior filter) or from `messages` (`context + output`), then sets
`forwarded` on the result via `data.clone(forwarded=...)`. The `context` and
`output` fields are never mutated — they form an audit trail.

### Built-in filters

| Filter | Effect |
|---|---|
| `keep_last_n(n)` | Keep only the last `n` messages. |
| `remove_tool_calls` | Strip `ToolCallItem`, `ToolCallOutputItem`, and `HandoffOutputItem`. |
| `remove_system_messages` | Strip `SystemItem` messages. |
| `forward_intent` | Append the classified `Intent` as a user message (code-orchestrated). |
| `intent_only` | Set `forwarded=()` — next agent sees only its system prompt and the intent. |
| `compose(*filters)` | Chain multiple filters left-to-right. |

### Example — sliding window

```python
from troopai.adk.handoffs import Handoff
from troopai.adk.handoffs.handoff_filters import keep_last_n

Handoff(
    target=specialist_agent,
    description="Transfer when the user needs a specialist.",
    input_filter=keep_last_n(6),  # forward only the 6 most recent items
)
```

### Example — composed pipeline

```python
from troopai.adk.handoffs.handoff_filters import (
    compose,
    remove_tool_calls,
    keep_last_n,
)

concise_handoff = Handoff(
    target=specialist_agent,
    input_filter=compose(remove_tool_calls, keep_last_n(4)),
)
```

---

## `HandoffInputData`

`HandoffInputData` is the frozen dataclass passed to `input_filter` callbacks
and to `on_handoff` when the callback uses the full-data signature. It
separates the conversation into temporal slices so filters can distinguish
what the agent inherited from what it produced.

| Field | Type | Contents |
|---|---|---|
| `intent` | `Any` | What triggered the handoff: a validated Pydantic model (when `input_type` is set), a raw `Intent` subclass (code-orchestrated), or the raw tool args string. |
| `context` | `tuple[RunItem, ...]` | Messages that existed **before** the current agent's turn. For the first agent this is `[system, user]`. For subsequent agents it is the filtered history forwarded from the prior agent. |
| `output` | `tuple[RunItem, ...]` | Messages generated **during** the current agent's turn — LLM responses, tool calls, tool results, and the handoff trigger. Empty for code-orchestrated handoffs that fire immediately after classification. |
| `forwarded` | `tuple[RunItem, ...] \| None` | The filtered subset to forward. When `None`, the Runner uses `context + output`. Set by `input_filter` to decouple what the next agent sees from the full audit trail. |

The `.messages` property returns `context + output` as a flat tuple and is
the full pre-filter view.

### Callback signatures

`on_handoff` auto-detects its signature from parameter annotations:

```python
# (ctx) — side effects only
def log_handoff(ctx: RunContext[Any]) -> None:
    logger.info("Handoff occurred.")

# (ctx, intent) — access the validated typed input
def log_refund(ctx: RunContext[Any], intent: RefundIntent) -> None:
    logger.info("Refund for order %s", intent.order_id)

# (ctx, data: HandoffInputData) — full temporal access
async def audit(ctx: RunContext[Any], data: HandoffInputData) -> None:
    logger.info(
        "context=%d output=%d forwarded=%s",
        len(data.context),
        len(data.output),
        "None" if data.forwarded is None else len(data.forwarded),
    )
```

---

## Conditional handoffs

### LLM-orchestrated — `enabled` callable

Pass a callable to `Handoff.enabled`. The callable receives `(ctx)` or
`(ctx, agent)` (arity is detected automatically). Disabled targets are
removed from the LLM's tool list before each turn.

```python
def premium_only(ctx: RunContext[Any]) -> bool:
    return ctx.context is not None and ctx.context.get("tier") == "premium"

Handoff(
    target=priority_support,
    description="Priority support — premium users only.",
    enabled=premium_only,
)
```

### Code-orchestrated — `.otherwise()` and per-route `enabled`

`HandoffRoute` supports an `enabled` callable on each `.to()` call and a
`.otherwise()` fallback for unmatched intents:

```python
HandoffRoute("triage")
    .when(TechnicalIssue).to(technical_agent, enabled=premium_only)
    .when(RefundIntent).to(refunds_agent)
    .otherwise(general_agent)
```

When the `enabled` check fails for a matched route, `HandoffRoute.resolve()`
falls through to the next matching rule or `.otherwise()`. If no rule and no
fallback is active, `UnhandledIntentError` is raised.

---

## History items — `HandoffCallItem` and `HandoffOutputItem`

Every handoff produces a pair of items in `RunResult.new_items`. These items
are part of Layer 3 (`RunItem`) and appear in the conversation history
alongside tool calls and LLM messages.

| Item | Discriminator | What it represents |
|---|---|---|
| `HandoffCallItem` | `"handoff_call"` | The `transfer_to_<name>` tool call that triggered the handoff. `raw` wraps `LLMResponseFunctionToolCall`; `target_agent` names the destination. |
| `HandoffOutputItem` | `"handoff_output"` | The synthetic tool result injected after the call. `source` is the calling agent; `target` is the destination agent. |

You can inspect these in a post-run callback or in trajectory evaluations:

```python
from troopai.adk.types.items import HandoffCallItem, HandoffOutputItem

result = await Runner.arun(triage_agent, "I need a refund for order #123.")

for item in result.new_items:
    if isinstance(item, HandoffCallItem):
        logger.info("Handoff call → %s", item.target_agent)
    elif isinstance(item, HandoffOutputItem):
        logger.info("Handoff output: %s → %s", item.source, item.target)
```

The pair is also the basis for trajectory evaluation graders such as
`handoff_occurred()`. See [Architecture: Handoffs & Swarms](../architecture/handoffs-and-swarms.md)
for why the tool-call model was chosen over custom message types.

---

## Common patterns

### Triage → specialist

The most common pattern. A lightweight triage agent classifies the request
and immediately hands off to a specialist. Use code-orchestrated routing to
minimise routing cost; use LLM-orchestrated routing when the triage decision
requires reasoning.

```python
# LLM-orchestrated triage
triage = Agent(
    name="Customer Support Triage",
    system_prompt=(
        "Route the user to the right specialist:\n"
        "- Refund requests → Refunds Specialist\n"
        "- Billing questions → Billing Specialist\n"
    ),
    handoffs=[refunds_agent, billing_agent],
)

result = await Runner.arun(triage, "I was charged twice this month.")
logger.info("Handled by: %s", result.last_agent.name)
```

### Multi-step pipeline

Chain agents in a linear sequence where each specialist finishes and hands
off to the next stage. Use `Handoff.input_filter` to strip noise between
stages.

```python
from troopai.adk.handoffs.handoff_filters import keep_last_n, remove_tool_calls, compose

collector = Agent(
    name="Collector",
    system_prompt="Gather the necessary information from the user.",
    handoffs=[
        Handoff(
            target=resolver_agent,
            description="Transfer once all information is collected.",
            input_filter=compose(remove_tool_calls, keep_last_n(8)),
        )
    ],
)
```

### Typed handoff input — structured arguments

When the triage agent should pass structured data to a specialist, set
`input_type` to a Pydantic model. The tool gets a JSON schema; the Runner
validates and parses the LLM's tool call arguments before `on_handoff` fires.

```python
from pydantic import BaseModel, Field

class EscalationInput(BaseModel):
    reason: str = Field(description="Why this needs escalation.")
    priority: int = Field(description="Priority 1–5.", ge=1, le=5)

def handle_escalation(ctx: RunContext[Any], input: EscalationInput) -> None:
    logger.info("Escalation: %s (priority %d)", input.reason, input.priority)

Handoff(
    target=escalation_agent,
    input_type=EscalationInput,
    on_handoff=handle_escalation,
    description="Escalate complex issues with a reason and priority.",
)
```

---

## See also

- [Architecture: Handoffs & Swarms](../architecture/handoffs-and-swarms.md) —
  deeper treatment of the handoff execution model and comparison with swarms.
- [Concepts: Multi-agent patterns](../concepts/index.md) —
  when to use handoffs vs swarms vs graphs.
- `examples/handoffs/llm_orchestrated.py` — bare agent list, `Handoff` objects,
  dynamic enablement, typed input.
- `examples/handoffs/code_orchestrated.py` — `HandoffRoute`, `handoff_route()`,
  multi-intent matching, `Respond` fallback.
- `examples/handoffs/message_filters.py` — built-in, custom, and composed filters.
- `examples/handoffs/temporal_slicing.py` — `on_handoff` with `HandoffInputData`,
  temporal-aware filters, `forwarded` decoupling.
- `examples/handoffs/cost_optimized.py` — `HandoffConfig` strategies and budget caps.

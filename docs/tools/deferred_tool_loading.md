# Deferred Tool Loading & `build_tool_search()`

Hide rarely-used tools from the LLM's per-turn tool list until the
LLM searches for them. Saves tokens on every turn for every tool the
model isn't currently using.

## When to use

| Situation | Use |
|-----------|-----|
| Agent has 30+ tools, most unused per turn | `defer_loading=True` + `build_tool_search()` |
| One specialised tool used in 5% of turns | Same |
| All tools are core path | Skip — defaults are best |
| You want a fixed reveal-on-trigger | Use `enabled=callable` instead |

A typical 100-token tool definition × 50 unused tools × 30 turns is
**150,000 tokens** of system-prompt cost the model never reads. Hiding
them flips the cost from "always paid" to "paid when relevant".

## Multi-tenant warning — read this first

The reveal state lives on the search tool's closure. The search tool
is part of `agent.tools`, and `Agent` is configuration shared across
concurrent `Runner.arun()` calls. If you reuse one `Agent` instance
across users, **a tool revealed in user A's session stays revealed
for user B's session**. For multi-tenant deployments, construct the
search tool (and the agent) per request:

```python
def make_agent() -> Agent:
    search = build_tool_search([payment_processor, schema_migrator])
    return Agent(
        name="Worker",
        system_prompt="Use tools as needed. Use tool_search when no tool fits.",
        tools=[echo, payment_processor, schema_migrator, search],
    )

# One fresh agent per request:
result = await Runner.arun(make_agent(), user_input)
```

For single-user / single-process deployments, a module-scope `agent`
is fine.

## Quick example (single-user)

```python
from troopai.adk.agents import Agent
from troopai.adk.tools import build_tool_search, function_tool


@function_tool(name="echo", description="Echo a message.")
def echo(msg: str) -> str:
    return msg


@function_tool(
    name="payment_processor",
    description="Charge a credit card via the payment gateway.",
    defer_loading=True,
)
def payment_processor(amount_cents: int) -> str:
    return f"charged {amount_cents}"


@function_tool(
    name="schema_migrator",
    description="Run a SQL schema migration.",
    defer_loading=True,
)
def schema_migrator(sql: str) -> str:
    return "migration applied"


search = build_tool_search([payment_processor, schema_migrator])

agent = Agent(
    name="Worker",
    system_prompt="Use tools as needed. Use tool_search when no current tool fits.",
    tools=[echo, payment_processor, schema_migrator, search],
)
# LLM initially sees: echo, tool_search.
# After it calls tool_search("charge"), payment_processor appears on
# the next turn.
```

## How it works

`FunctionTool.defer_loading=True` flips the tool from "visible" to
"hidden by default". `build_tools()` filters such tools out of the
LLM's tool list every turn — unless their name is in the per-run
`revealed` set.

`build_tool_search()` returns a plain `FunctionTool` (not a
provider-specific wrapper). Its `on_invoke`:

1. Parses `{ "query": ..., "top_k": 5 }` from the LLM.
2. Ranks the configured deferred tools by query relevance (substring
   match over name + description; name hits weighted higher).
3. Adds the matched tool names to the search tool's `revealed` set
   (mutated in place; per-run scope).
4. Returns a JSON list of `[{"name", "description"}, ...]` so the LLM
   knows which tools just became available.

`build_tools()` consults the search tool via
`FunctionTool.get_search_state().revealed` — fresh every turn, so a
revealed tool flips into the visible list on the very next call to
`build_tools()`.

## Reveal scope is per-run

Each `Runner.arun()` constructs the agent's tool list anew. The
`revealed` set lives on the search tool's closure state, so:

- Within a single run: revealed tools persist across turns.
- Across runs of the **same** agent + search-tool instance: revealed
  state persists. If you re-use the same `Agent(tools=[..., search])`
  for a second run, the second run starts with whatever was revealed
  in the first.
- Across runs of fresh agent instances: each `build_tool_search()`
  call creates a new closure with an empty set.

For strict per-run isolation, construct the search tool inside the
function that builds the agent each run:

```python
def make_agent() -> Agent:
    search = build_tool_search([payment_processor])
    return Agent(name="Worker", system_prompt="...", tools=[..., search])

result = await Runner.arun(make_agent(), "task")
```

## Decorator field

```python
@function_tool(name="rare", defer_loading=True)
def rare(x: str) -> str: ...
```

The flag is also a constructor kwarg on `FunctionTool` directly, for
authors who don't use the decorator.

## Customising the search tool

`build_tool_search()` supports name + description override:

```python
search = build_tool_search(
    deferred_tools,
    name="discover_tool",
    description=(
        "When you need a tool you don't see, describe what you want "
        "and I'll surface candidates from the catalogue."
    ),
)
```

## Custom rankers

The default substring ranker is intentionally simple. To plug in a
better matcher (embeddings, LLM-as-ranker, BM25), construct your own
search tool by calling `build_tool_search()` and treating the
returned `FunctionTool` as the entry point — there is no need to
reach into the framework's private state. If you need a fundamentally
different shape (e.g. a paginated catalogue tool), wrap a fresh
`@function_tool` and have it call `tool_search`'s `on_invoke` from
within: the reveal mechanism is encapsulated in the closure
`build_tool_search()` returns.

## What this does NOT do

- It does not unload a tool once revealed. Once a name enters the
  reveal set, the tool stays visible for the rest of the run. If you
  need to re-hide a tool, use `enabled=callable` with your own gating
  logic.
- It does not coordinate across processes. Reveal state is in-process.
- An empty query returns NO matches. This is deliberate — without
  this rule, an empty query plus a large `top_k` would enumerate the
  full catalogue.
- The framework refuses to **execute** unrevealed deferred tools, not
  just to expose their schemas. A prompt-injected LLM that emits a
  function-call to a deferred tool name without searching first will
  receive a "tool not found" result rather than triggering the
  underlying handler. (See the `TestExecutorGate` tests.)

## See also

- `tests/unit/tools/test_deferred_tool_loading.py` — full behaviour
  tests (filter, reveal, search ranking, top_k cap).
- `examples/tools/deferred_tool_loading.py` — runnable example.

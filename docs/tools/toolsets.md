# Toolsets — Composable Tool Collections

A **Toolset** is a live abstraction over a group of tools. Instead of
listing 20 individual `FunctionTool` instances on `Agent.tools`,
group them into toolsets that namespace, filter, or rename their
contents per turn.

The shipped variants:

| Variant | Purpose |
|---|---|
| `FunctionToolset` | Wrap a flat list of `FunctionTool` instances — the leaf primitive |
| `PrefixedToolset` | Add a namespace prefix to every tool name |
| `RenamedToolset` | Apply a `{old: new}` rename map |
| `FilteredToolset` | Drop tools whose predicate returns `False` (per-turn, context-aware) |
| `CombinedToolset` | Materialise multiple toolsets into one merged dict |
| `WrapperToolset` | Base for user subclasses that override `get_tools` (also the surface for toolset-scoped middleware) |

## Why use toolsets

- **Avoid name collisions** when combining MCP servers, sub-agent
  delegations, and ad-hoc Python tools.
- **Filter dynamically** per `RunContext` — expose admin tools only
  when `ctx.context["role"] == "admin"`.
- **Organise large registries** into named domains (`weather_*`,
  `db_*`, `slack_*`).
- **Keep `Agent.tools` short** — instead of 30 entries, pass 3 toolsets.

Toolsets sit alongside individual tools in `Agent.tools`. The two
forms compose freely:

```python
from troopai.adk.tools import FunctionToolset, function_tool

agent = Agent(
    name="Operations",
    system_prompt="Help operators.",
    tools=[
        ad_hoc_tool,                                       # standalone
        FunctionToolset(tools=db_tools).prefixed("db"),   # namespaced
        FunctionToolset(tools=weather_tools).prefixed("weather"),
    ],
)
```

## Quickstart

```python
from troopai.adk.agents.agent import Agent
from troopai.adk.tools import FunctionToolset, function_tool


@function_tool(name="get_temp", description="Get temperature")
def get_temp(city: str) -> str:
    return f"{city}: 22C"


@function_tool(name="get_conditions", description="Get conditions")
def get_conditions(city: str) -> str:
    return f"{city}: sunny"


# A toolset, namespaced
weather = FunctionToolset(tools=[get_temp, get_conditions]).prefixed("weather")

agent = Agent(
    name="Forecaster",
    system_prompt="Help users with weather.",
    tools=[weather],
)
# The LLM sees: weather_get_temp, weather_get_conditions
```

## PrefixedToolset

Adds a fixed prefix to every wrapped tool name. The default `_`
separator is the Pydantic-AI convention and is compatible with every
provider's tool-name regex.

```python
ts = FunctionToolset(tools=[query]).prefixed("db")
# tool name visible to LLM: db_query

ts = FunctionToolset(tools=[query]).prefixed("db", separator="-")
# tool name visible to LLM: db-query
```

## RenamedToolset

Apply a per-name rename map. Names not in the map pass through
unchanged.

```python
ts = FunctionToolset(tools=[execute_query, drop_table]).renamed(
    {"execute_query": "sql"}
)
# tool names visible to LLM: sql, drop_table
```

## FilteredToolset

Predicate is evaluated **each turn** against the live `RunContext`,
so the visible tool set can change as the run progresses.

```python
def is_admin(ctx, tool):
    return ctx.context.get("role") == "admin"


sensitive = FunctionToolset(tools=[delete_user, restart_db]).filtered(is_admin)

agent = Agent(
    name="Ops",
    system_prompt="...",
    tools=[sensitive],
)
# When run with context={"role": "admin"}: delete_user, restart_db visible.
# When run with context={"role": "user"}: nothing visible.
```

The predicate may be sync or async. Receives `(ctx, tool)`; returns
`True` to keep, `False` to drop.

## CombinedToolset

Materialise multiple toolsets in declaration order and merge their
results. Conflicts are surfaced by `build_tools()` with the full list
of contributing sources.

```python
weather = FunctionToolset(tools=[...]).prefixed("weather")
db = FunctionToolset(tools=[...]).prefixed("db")

combo = CombinedToolset(toolsets=[weather, db])
agent = Agent(name="Ops", system_prompt="...", tools=[combo])

# Or use the builder shorthand for two:
combo = weather.combined_with(db)
```

There is no `+` operator on `Toolset` — the ordered-sequence form
makes conflict-error messages clearer ("contributed by toolsets[0],
toolsets[2]") and avoids surprising right-associativity.

## WrapperToolset

Base for user subclasses. Override `get_tools()` to add custom
transformation:

```python
from troopai.adk.tools import WrapperToolset


class TaggedToolset(WrapperToolset):
    """Append a tag to every tool's description."""

    tag: str = "[experimental]"

    async def get_tools(self, ctx=None):
        inner = await self.wrapped.get_tools(ctx)
        from dataclasses import replace
        return {
            name: replace(tool, description=f"{self.tag} {tool.description}")
            for name, tool in inner.items()
        }
```

## Name conflict detection

When two or more toolsets contribute the same name, `build_tools()`
raises `ToolsetNameConflictError` with every contributing source
named in one error message:

```
ToolsetNameConflictError: Agent 'Ops' has toolset name conflicts:
  - 'shared_query' contributed by: PrefixedToolset#0, PrefixedToolset#2
```

A standalone-vs-toolset collision is also raised (e.g.
`agent.tools[0]` collides with `FunctionToolset#1`). Pure
standalone-vs-standalone collisions are NOT raised by `build_tools`
— that path predates toolsets and may be load-bearing for patterns
like multiple `build_tool_search()` instances.

## Cloning preserves internal state

When `PrefixedToolset` or `RenamedToolset` materialises a renamed
clone, the internal slots (`_agent` from `as_tool()`, `_cache`,
`_rate_state`, `_search_state`) are preserved by reference. This
means:

- A renamed delegate tool still reports the original agent via
  `get_delegate_agent()`.
- Cache hits work identically — the cache dict is shared.
- Rate limits apply to the underlying tool, not the surface name.

## Composition examples

```python
# Filtered then prefixed: only certain admin tools, namespaced.
admin_tools = FunctionToolset(tools=admin_actions).filtered(is_admin).prefixed("admin")

# Renamed then combined: rename one tool out of an MCP toolset, then merge with locals.
mcp_renamed = mcp_tools.renamed({"execute_query": "sql"})
all_tools = CombinedToolset(toolsets=[mcp_renamed, FunctionToolset(tools=local_tools)])

# Three-way combination
agent = Agent(
    name="Hub",
    system_prompt="...",
    tools=[
        CombinedToolset(toolsets=[weather, db, slack]),
    ],
)
```

## See also

- `examples/tools/toolsets/` — runnable examples.
- `src/troopai/adk/tools/toolsets/abstract.py` — the `Toolset` ABC and
  builder methods.
- `src/troopai/adk/exceptions/exceptions.py` — `ToolsetNameConflictError`.

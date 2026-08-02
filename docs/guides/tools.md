(guides/tools)=

# 🔧 Tools

Tools give agents the ability to act — query databases, call APIs, run code,
search the web, or invoke another agent. The ADK supports three complementary
kinds of tools that appear side-by-side in the same `Agent.tools` list.

## Three tool kinds at a glance

| Kind | Class | Who executes | Best for |
|---|---|---|---|
| **Function tool** | `FunctionTool` | Your Python process | Custom logic, APIs, DB calls |
| **Hosted tool** | `HostedTool` subclass | LLM provider server | Web search, code execution, file search, image generation |
| **MCP tool** | `MCPToolset` (toolset) | External MCP server | Third-party tool servers, extensible ecosystems |

All three sit in `Agent.tools`; the runner dispatches each type through
its appropriate execution path at runtime.

---

## Function tools — the default

A **function tool** wraps any Python callable. Prefer `@function_tool` for
the common case; construct `FunctionTool` directly when you need to share an
`on_invoke` implementation or build tools programmatically.

### Decorator form

```python
from troopai.adk.tools import function_tool

@function_tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Return the current weather for a city.

    Args:
        city: Name of the city.
        unit: Temperature unit — 'celsius' or 'fahrenheit'.
    """
    return f"Weather in {city}: 22 {unit}"
```

`@function_tool` without parentheses auto-extracts `name`, `description`,
and parameter docs from the function's signature and docstring (Google,
NumPy, and Sphinx styles are all auto-detected). Pass keyword arguments for
overrides:

```python
@function_tool(
    name="weather_api",
    description="Fetch current weather from the weather API.",
    max_result_tokens=200,
    max_retries=3,
)
def get_weather(city: str) -> str: ...
```

### Direct construction

For advanced cases — dynamic dispatch, shared invocation callbacks — pass
an explicit `on_invoke`:

```python
from troopai.adk.tools import FunctionTool
from troopai.adk.tools.tool_context import ToolContext
import json

async def _invoke(ctx: ToolContext, raw_args: str) -> str:
    args = json.loads(raw_args)
    return do_work(args["query"])

search = FunctionTool(
    name="search",
    description="Search the knowledge base.",
    schema=SearchParams,      # Pydantic BaseModel
    on_invoke=_invoke,
)
```

### Schema generation from type hints

`@function_tool` passes the function through `generate_function_schema`,
which builds a Pydantic model from the type-annotated parameters and stores
it on `FunctionTool.schema`. The runner serialises this to JSON before
sending the tool definition to the LLM. The `schema_enforcement` field
controls the output:

| Value | Behaviour |
|---|---|
| `SchemaEnforcement.NORMALIZED` (default) | Provider-agnostic defaults applied; safe for all providers |
| `SchemaEnforcement.STRICT` | Full strict-mode (all props required, no `additionalProperties`) for providers that support it |
| `SchemaEnforcement.NONE` | Raw schema forwarded as-is |

### `FunctionTool` fields reference

The key fields on `FunctionTool` (all keyword-only, all available in
`@function_tool`):

| Field | Default | Purpose |
|---|---|---|
| `name` | function name | Identifier the LLM calls |
| `description` | from docstring | Shown to LLM |
| `schema` | from type hints | Pydantic model or JSON schema dict |
| `enabled` | `True` | Bool or `(RunContext) → bool`; dynamic enable/disable |
| `requires_approval` | `False` | Bool or `(ToolContext) → bool`; HITL gate |
| `max_result_tokens` | `None` | Truncate result before adding to history |
| `max_retries` | `None` | LLM retry budget before tool is removed |
| `timeout` | `None` | Per-call timeout (seconds) via `asyncio.wait_for` |
| `timeout_behavior` | `"error_as_result"` | `"error_as_result"` or `"raise_exception"` |
| `cache` | `False` | Cache results keyed on raw JSON args |
| `response_format` | `"text"` | `"text"` or `"content_and_artifact"` (dual return) |
| `return_direct` | `False` | Skip post-processing; result becomes final output |
| `requires_env` | `()` | Env vars validated at agent construction |
| `requires_packages` | `()` | PEP 508 package specs validated at agent construction |
| `rate_limit` | `None` | `ToolRateLimit(rpm=N)` sliding-window throttle |
| `defer_loading` | `False` | Hide from LLM until revealed by `build_tool_search()` |
| `streaming` | `False` | Yield `AsyncIterator[ToolStreamEvent]` incrementally |
| `prepare` | `None` | `(RunContext, FunctionTool) → FunctionTool | None` per-step modifier |

---

## Token-budget fields

Two fields control token costs at the individual tool level.

**`max_result_tokens`** caps the number of tokens the runner inserts into
the conversation history for this tool's result. Without a cap, a single
call returning thousands of tokens re-enters the context on every subsequent
turn until a context editor clears it (itself an opt-in). For tools that
return variable-length content — RAG retrievers, web readers, shell commands
— set a bound:

```python
@function_tool(name="rag_search", max_result_tokens=400)
def rag_search(query: str) -> str:
    return retrieve_documents(query)
```

**`max_retries`** is the LLM's retry budget for this tool. When the tool
fails `max_retries` times the runner removes it from the LLM's tool list for
the rest of the run:

```python
@function_tool(name="flaky_api", max_retries=2)
def flaky_api(input: str) -> str: ...
```

`max_retries=None` (the default) means the tool may be retried freely,
bounded only by `RunConfig.max_turns`. A tool inside a governed skill defers
to the skill's governance when `max_retries=None`; a non-`None` value
silently overrides governance, which is why the default is `None`.

---

## Tool guardrails

Tool guardrails run orthogonally to agent-level guardrails. They validate
each tool's inputs before `on_invoke` and outputs after — the rest of the
agent pipeline is unaffected.

```python
from troopai.adk.tools.tool_guardrails import (
    ToolGuardrails,
    ToolInputGuardrail,
    ToolOutputGuardrail,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    tool_input_guardrail,
    tool_output_guardrail,
)

@tool_input_guardrail
def block_pii(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    if contains_pii(data.context.tool_arguments):
        return ToolGuardrailFunctionOutput.reject_content("PII detected")
    return ToolGuardrailFunctionOutput.allow()

@tool_output_guardrail
def sanitize_output(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    if is_sensitive(data.output):
        return ToolGuardrailFunctionOutput.reject_content("[redacted]")
    return ToolGuardrailFunctionOutput.allow()

@function_tool(
    name="user_lookup",
    guardrails=ToolGuardrails(
        input=[block_pii],
        output=[sanitize_output],
    ),
)
def user_lookup(user_id: str) -> str: ...
```

Three verdict methods are available on `ToolGuardrailFunctionOutput`:

| Method | Effect |
|---|---|
| `.allow()` | Execution continues normally |
| `.reject_content(msg)` | LLM sees the message instead of the result; agent continues |
| `.raise_exception()` | Raises `ToolGuardrailTripwireTriggered`; halts the run |

Tool guardrails are distinct from agent-level input/output guardrails —
they are tightly scoped to a single tool and accumulate results in
`RunResult.guardrail_results`.

---

## Hosted tools

Hosted tools are provider-native capabilities. The ADK forwards typed
configuration to each provider's wire format; the provider executes the tool
server-side and returns the result. Your Python process never runs them.

Every hosted tool is a `HostedTool` subclass — a `@dataclass(kw_only=True)`
that declares `SUPPORTED_PROVIDERS`. Passing a hosted tool to a provider that
does not support it raises `UnsupportedHostedToolError`; silent drops are
forbidden.

### Available hosted tools

| Class | Supported providers | Purpose |
|---|---|---|
| `WebSearchTool` | Anthropic, OpenAI Responses, Gemini | Live web search |
| `CodeExecutionTool` | OpenAI Responses, Gemini | Server-side Python interpreter |
| `FileSearchTool` | OpenAI Responses | Vector-store file search |
| `ImageGenerationTool` | OpenAI Responses | Image generation |
| `URLContextTool` | Gemini | Fetch and ground on a URL |
| `ComputerTool` | Anthropic (computer use) | Desktop computer control |
| `HostedMCPTool` | Anthropic | Anthropic-hosted MCP server |

Import from `troopai.adk.tools`:

```python
from troopai.adk.tools import WebSearchTool, CodeExecutionTool, FileSearchTool

from troopai.adk.llms import AnthropicLLM
from troopai.adk.agents import Agent

agent = Agent(
    llm=AnthropicLLM(model="claude-sonnet-4-5"),
    tools=[
        WebSearchTool(
            max_uses=5,
            allowed_domains=["arxiv.org", "nature.com"],
        ),
    ],
)
```

Per-provider attribute support is documented on each class with
`**<Provider> only.**` tags. Attributes unsupported by the active provider
are silently dropped at the converter boundary (with a `logger.debug` line).

**`WebSearchTool` knobs:**

| Attribute | Providers | Purpose |
|---|---|---|
| `max_uses` | Anthropic | Max searches per turn |
| `allowed_domains` / `blocked_domains` | Anthropic | Domain allow/block lists |
| `user_location` | Anthropic + OpenAI Responses | Geolocation hint |
| `search_context_size` | OpenAI Responses | `"low"` / `"medium"` / `"high"` |

**`CodeExecutionTool` knobs:**

| Attribute | Providers | Purpose |
|---|---|---|
| `container` | OpenAI Responses | Bind to a specific container `cntr_...` |

**`FileSearchTool` (OpenAI Responses only):**

```python
FileSearchTool(
    vector_store_ids=["vs_abc123"],
    max_num_results=5,
)
```

:::{note}
Anthropic's code execution is in beta and not represented as a typed class.
Use `LLMConfig.extra_body` to pass beta-format tool definitions for
capabilities not yet covered by a typed `HostedTool` subclass.
:::

---

## MCP tools

The Model Context Protocol (MCP) lets agents use tools from external servers
— local processes, remote services, or community tool registries. The ADK
connects to any MCP server and exposes its tools as ordinary `FunctionTool`
instances alongside your own.

Add an `MCPToolset` to `Agent.tools`:

```python
from troopai.adk.tools.toolsets.mcp_toolset import MCPToolset
from troopai.adk.mcp.stdio import MCPServerStdio

agent = Agent(
    name="ResearchAgent",
    tools=[
        MCPToolset(
            server=MCPServerStdio(
                command="uvx",
                args=["mcp-server-fetch"],
            ),
        ),
    ],
)
```

`MCPToolset` lazy-connects on the first `get_tools()` call and disposes the
server when the run ends. The server's tool list is materialised as
`FunctionTool` instances each turn. Composition with existing toolset
wrappers works transparently:

```python
MCPToolset(server=server).prefixed("fetch").filtered(my_predicate)
```

:::{tip}
See the MCP guide (`guides/mcp`) for server types (`MCPServerStdio`,
`MCPServerStreamableHttp`, `MCPServerSSE`), `HostedMCPTool` for
Anthropic-hosted MCP servers, and the `MCPServerManager` for sharing a
connection across agents.
:::

---

## `Agent.as_tool()` — agents as tools

Any agent can be wrapped as a tool and placed in another agent's `tools`
list. This enables hierarchical delegation where the parent LLM decides when
to invoke the sub-agent and the sub-agent runs independently, returning its
final output as a tool result.

```python
from troopai.adk.agents import Agent

researcher = Agent(name="Researcher", system_prompt="...")
writer = Agent(name="Writer", system_prompt="...")

supervisor = Agent(
    name="Supervisor",
    system_prompt="Coordinate research and writing.",
    tools=[
        researcher.as_tool(),
        writer.as_tool(),
    ],
)
```

`as_tool()` returns a standard `FunctionTool`. It accepts the same budget
and behaviour controls as any other tool:

| Parameter | Purpose |
|---|---|
| `tool_name` / `tool_description` | Override what the parent LLM sees |
| `max_turns` | Sub-agent loop limit (default `10`) |
| `timeout` | Seconds; on expiry returns error string, does not raise |
| `budget` | `LLMUsageLimits` — token/request cap for the sub-agent |
| `max_result_tokens` | Truncate result before inserting into parent history |
| `extractor` | `(RunResult) → str` to customise what the parent sees |
| `on_stream` | Callback to observe sub-agent streaming events in real time |

Sub-agent intermediate steps never enter the parent context. The parent
sees exactly two messages: the tool call and the final result string.

:::{tip}
See the Agents guide (`guides/agents`) for the full `as_tool()` parameter
reference and the nested HITL pattern.
:::

---

## Common patterns

### Dynamic enable / disable

```python
@function_tool(
    name="admin_action",
    enabled=lambda ctx: ctx.context.get("role") == "admin",
)
def admin_action(command: str) -> str: ...
```

### Conditional human approval

```python
async def need_approval(ctx) -> bool:
    return ctx.context.get("environment") == "production"

@function_tool(name="deploy", requires_approval=need_approval)
def deploy(service: str) -> str: ...
```

### Dual return (LLM summary + app data)

```python
@function_tool(name="rag", response_format="content_and_artifact")
def rag_search(query: str) -> tuple[str, list[dict]]:
    docs = retrieve(query)
    return f"Found {len(docs)} results", docs  # LLM sees summary; app gets docs
```

Access `result.new_items[-1].artifact` to retrieve the full payload.

### Deferred tool loading for large registries

When an agent has 50+ tools, the per-turn token cost of listing every tool
definition is significant. Mark specialist tools with `defer_loading=True`
and expose a `build_tool_search()` discovery tool so the LLM reveals only
what it needs:

```python
@function_tool(name="specialist_tool", defer_loading=True)
def specialist_tool(param: str) -> str: ...
```

### Per-tool sandboxing

To execute a tool inside an isolated environment (Docker, E2B, Vercel, etc.),
attach a `ShellTool` or `ApplyPatchTool` with the appropriate `executor` or
`editor`. For finer control, see the Sandbox guide (`guides/sandbox`).

---

## See also

- **Type layers** — `FunctionToolCallResultParam` is a Layer 1 type (tool-result
  replay, sent to the LLM). Tool call results are `FunctionToolCallResult`
  (Layer 3, developer-facing). Wire conversion lives inside each provider's
  converter. See [Type layers](../architecture/type-layers.md) for the layer contract.
- **MCP guide** — `guides/mcp` — MCP server types, `MCPServerManager`,
  `HostedMCPTool`, per-tool filters.
- **Guardrails guide** — `guides/guardrails` — user-authored agent-level
  input/output guardrails; decorator and config `ref` patterns.
- **Sandbox guide** — `guides/sandbox` — sandboxed tool execution via Docker,
  E2B, Modal, and other isolators.
- **Toolsets** — `docs/tools/toolsets.md` — `FunctionToolset`,
  `PrefixedToolset`, `FilteredToolset`, `RenamedToolset`, `CombinedToolset`.
- **Middleware** — `docs/tools/middleware.md` — logging, metrics, and tracing
  wrappers. Verdict logic belongs in guardrails, not middleware.
- **Streaming tools** — `docs/tools/streaming_tool_results.md` — `streaming=True`
  async-generator mode and `ToolStreamEvent`.

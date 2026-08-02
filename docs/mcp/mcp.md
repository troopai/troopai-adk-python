# Model Context Protocol (MCP)

Model Context Protocol is Anthropic's JSON-RPC standard for letting
LLM agents discover and call tools, prompts, and resources hosted by
external servers. The `troopai.adk.mcp` package wraps the upstream
`mcp` Python SDK and exposes one Toolset adapter so MCP servers slot
into agents like any other tool collection.

This page covers the full MCP surface:

- **Core** — stdio + streamable HTTP transports, `MCPToolset`,
  filters, caching, header injection, error handling, OTel.
- **Advanced** — `HostedMCPTool` (provider-side execution),
  `structuredContent` artifact channel, sampling (server → host
  LLM), MCP resources, prompts-as-tools, `$ref` schema resolution,
  HITL approval, ref-counted `MCPServerManager`, verbose event
  wiring.
- **Additional transports** — WebSocket transport, SSE transport
  (MCP-spec-deprecated), elicitation handler.

## Install

```bash
pip install 'troopai-adk-python[mcp]'
```

Without the `mcp` extra, every `troopai.adk.mcp.*` export is bound to
`None`; callers can compare against `None` to detect availability.

## Quick start — stdio

```python
import asyncio

from troopai.adk.agents.agent import Agent
from troopai.adk.mcp import MCPServerStdio, MCPServerStdioParams
from troopai.adk.run.runner import Runner
from troopai.adk.tools.toolsets import MCPToolset


async def main() -> None:
    server = MCPServerStdio(
        name="filesystem",
        params=MCPServerStdioParams(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/scratch"],
        ),
    )
    agent = Agent(
        name="x",
        system_prompt="Use filesystem tools.",
        tools=[MCPToolset(server=server)],
    )
    result = await Runner.arun(agent, "List /tmp/scratch")
    print(result.final_output)


asyncio.run(main())
```

The `MCPToolset` lazy-connects on the first turn's `get_tools()`
call and the runner's `finally` block invokes `MCPToolset.adispose()`
to terminate the subprocess.

## Quick start — streamable HTTP

```python
from troopai.adk.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams

server = MCPServerStreamableHttp(
    name="github",
    params=MCPServerStreamableHttpParams(url="https://api.example.com/mcp"),
)
agent = Agent(name="x", system_prompt="...", tools=[MCPToolset(server=server)])
```

## Toolset composition

`MCPToolset` is a `Toolset` subclass, so the standard composition
builders work out of the box:

```python
toolset = (
    MCPToolset(server=stdio_server)
    .prefixed("svc")            # Namespace tools as svc_*
    .filtered(my_predicate)     # Drop tools the predicate rejects
)
```

For multi-server agents, prefix each toolset to avoid
`ToolsetNameConflictError`:

```python
agent = Agent(
    name="x",
    system_prompt="...",
    tools=[
        MCPToolset(server=server_a).prefixed("a"),
        MCPToolset(server=server_b).prefixed("b"),
    ],
)
```

## Filtering

`MCPToolset` accepts a server-aware `tool_filter` predicate. The
predicate receives a `ToolFilterContext` carrying the originating
server's name plus the live `RunContext`, plus the converted
`FunctionTool`. Sync or async; exceptions are fail-closed (tool
excluded with a WARNING log). The filter gates **every** tool surface the
toolset exposes — the converted server tools and, when enabled, the
`read_<server>_resource` and prompt-as-tool surfaces — so denying a
server hides all of them.

```python
from troopai.adk.mcp import ToolFilter, ToolFilterContext


def only_read_tools(ctx: ToolFilterContext, tool: FunctionTool) -> bool:
    return tool.name.startswith("read_") or tool.name.startswith("list_")


toolset = MCPToolset(server=server, tool_filter=only_read_tools)
```

For static allow/deny lists, prefer the existing `Toolset.filtered`
builder. Reach for `MCPToolset.tool_filter` only when you need
server-name awareness or per-turn `RunContext` access that the
generic `ToolsetFilter` cannot see.

## Caching

`cache_tools_list=True` (the default on `MCPServerStdio` and
`MCPServerStreamableHttp`) keeps the result of the most recent
`list_tools` until either:

1. The MCP server pushes a `notifications/tools/list_changed`
   notification — the framework's subscriber flips the dirty flag
   automatically.
2. The caller invokes `server.invalidate_tools_cache()`.

Cache reads and writes are serialised by an `asyncio.Lock`, so
concurrent agent turns do not race.

```python
server = MCPServerStdio(name="x", params=MCPServerStdioParams(...), cache_tools_list=True)
async with server:
    first = await server.list_tools()   # Real round-trip
    cached = await server.list_tools()  # Cached
    server.invalidate_tools_cache()
    refreshed = await server.list_tools()  # Round-trip again
```

## Per-request authentication (HTTP)

`MCPServerStreamableHttpParams.header_provider` is a callable returning fresh
headers for every outbound request, enabling token rotation without
re-creating the HTTP session:

```python
def fresh_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_current_token()}"}


server = MCPServerStreamableHttp(
    name="x",
    params=MCPServerStreamableHttpParams(url="...", header_provider=fresh_headers),
)
```

Async providers are also supported:

```python
async def fresh_headers() -> dict[str, str]:
    token = await sts_client.exchange()
    return {"Authorization": f"Bearer {token}"}
```

The provider is invoked from an httpx request event hook reading
from a `ContextVar`; concurrent agent runs each see their own
provider with no shared state.

## Lifecycle

Three modes:

1. **Auto-managed (default)** — `MCPToolset(server=...,
   auto_connect=True)`. Connects on first `get_tools()`; the
   runner's `finally` block calls `adispose()` to clean up. No
   ceremony.
2. **Explicit `async with` on the server** — for code that wants
   exact connect/cleanup placement:
   ```python
   async with server:
       toolset = MCPToolset(server=server, auto_connect=False)
       await Runner.arun(Agent(name="x", system_prompt="...", tools=[toolset]), "hi")
   ```
3. **`MCPServerManager`** — for agents using several servers under
   one explicit lifecycle:
   ```python
   manager = MCPServerManager(servers=[server_a, server_b])
   async with manager:
       agent = Agent(
           name="x",
           system_prompt="...",
           tools=[
               MCPToolset(server=server_a, auto_connect=False),
               MCPToolset(server=server_b, auto_connect=False),
           ],
       )
       await Runner.arun(agent, "hi")
   ```

## Error handling

| Exception | Raised when |
|---|---|
| `MCPConnectionError` | Server is unreachable, fails to spawn, or fails to initialise. |
| `MCPToolCallError` | Server returned `isError=True` for a tool call. Flows through the standard tool-error path: the executor converts it to a model-visible string when `RunConfig.fail_on_tool_error=False` (default), re-raises when `True`. |
| `MCPToolNotFoundError` | Tool requested by the LLM no longer exists on the server. |
| `MCPSchemaConversionError` | Malformed `inputSchema`. |

All inherit from `MCPError`, which inherits from `TroopAIError`.

## Observability

- **Logging** — connect/disconnect lifecycle is logged at `INFO`;
  cache hits and ignored content parts at `DEBUG`; cleanup
  exceptions at `WARNING`.
- **OpenTelemetry** — when an OTel span is active on the calling
  task, every outbound MCP `call_tool` request carries the W3C
  Trace Context (`traceparent`, `tracestate`) in its `_meta`
  field. MCP servers honouring the protocol-level trace context
  inherit the parent trace span automatically. No configuration
  needed; soft-imports OTel.

## Cost levers

Every cost lever on `FunctionTool` works transparently on converted
MCP tools — they ARE `FunctionTool` instances:

| Lever | What it saves |
|---|---|
| `cache_tools_list=True` | Skips `list_tools` round-trip per turn after the first. |
| `notifications/tools/list_changed` push invalidation | No polling for tool catalogue changes. |
| `max_result_tokens` (per-tool, post-conversion) | Truncates large tool results before they enter history. |
| `max_retries` | Removes a broken MCP tool from the LLM's view after N failures. |
| `cache_function` | Selective tool-result caching by argument string. |
| `prepare` | Per-step description / arg modification. |
| `rate_limit` | Sliding-window RPM cap. |
| `defer_loading` | Hide a tool until explicit reveal via `build_tool_search`. |

All work on MCP-derived tools without further configuration.

## Advanced surfaces

### Hosted MCP (OpenAI Responses)

```python
from troopai.adk.tools.hosted import HostedMCPTool

agent = Agent(
    name="x",
    system_prompt="...",
    tools=[
        HostedMCPTool(
            server_label="github",
            server_url="https://api.example.com/mcp",
            require_approval="never",
            allowed_tools=["search", "fetch"],
        ),
    ],
    llm=OpenAIResponsesLLM(model="gpt-4o-mini"),
)
```

The Responses API runs the MCP loop server-side; no Python-side
connection is opened. Anthropic / Gemini / Chat Completions raise
`UnsupportedHostedToolError` because they don't ship hosted MCP.

### structuredContent artifact

```python
toolset = MCPToolset(server=server, use_structured_content=True)
```

Tools surface their `CallToolResult.structuredContent` via
`FunctionToolCallResult.artifact` — the LLM still sees the textual
content, but the application can read the structured dict.

### Sampling (server → host LLM)

```python
server = MCPServerStdio(name="x", params=..., llm=my_llm)
```

The MCP server can call back into `my_llm.acomplete()` via
`sampling/createMessage`. Useful for servers that want chained
reasoning during a tool call.

### MCP resources + prompts

```python
toolset = MCPToolset(
    server=server,
    use_mcp_resources=True,        # Adds read_<server>_resource tool
    expose_prompts_as_tools=True,  # Each prompt → callable FunctionTool
)
```

### `$ref` resolution

```python
toolset = MCPToolset(server=server, inline_refs=True)
```

Inlines intra-document `$ref` pointers in `inputSchema`; needed for
providers that don't honour `$ref`.

### HITL approval

```python
toolset = MCPToolset(server=server, requires_approval=True)
```

Every converted MCP tool flows through the framework's HITL
deferral pipeline — calls produce `ToolApprovalItem`s the
application must approve via `RunState.approve()`.

### Reference-counted server sharing

```python
manager = MCPServerManager(servers=[server])
await manager.acquire(server)  # +1
# ... use the server ...
await manager.release(server)  # -1; cleans up at zero
```

### Verbose events

`RunHooks` gains `on_mcp_connect / on_mcp_connected / on_mcp_error`.
`VerboseHooks` renders them via the existing `EVENT_MCP_*` style
entries — enable with `RunConfig(verbose=VerboseConfig())`.

## Additional transports

### WebSocket transport

```python
server = MCPServerWebsocket(
    name="ws-demo",
    params=MCPServerWebsocketParams(url="wss://api.example.com/mcp"),
)
```

Requires the optional `websockets` package.

### SSE transport (MCP-spec-deprecated)

```python
server = MCPServerSse(
    name="sse-demo",
    params=MCPServerSseParams(url="https://api.example.com/sse"),
)
```

Deprecated by the MCP spec; prefer `MCPServerStreamableHttp`.

### Elicitation handler

```python
async def ask_user(params):
    ans = input(params.message)
    return {"answer": ans}

server = MCPServerStdio(name="x", params=..., elicitation_callback=ask_user)
```

The handler receives the server's elicitation request and returns
the user's response.

## Limitations (still apply)

- Disposal covers only the entry-point agent's `tools`. Toolsets
  contributed by handoff targets must be managed via
  `MCPServerManager` (`auto_connect=False`) or explicit `async with`.
- MCP Tasks API for long-running tools is not yet wrapped.

## See also

- `examples/mcp/` — runnable examples (stdio, streamable HTTP,
  multi-server, auth headers, cached).
- Upstream MCP spec: <https://modelcontextprotocol.io/>.

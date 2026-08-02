(guides/mcp)=

# 🔌 MCP (Model Context Protocol)

**Model Context Protocol (MCP)** is an open standard that lets an
LLM host discover and invoke tools, prompts, and resources served by
external processes. An MCP server is a separate process (or remote
service) that advertises a catalogue of capabilities; an MCP client
connects, fetches that catalogue, and calls tools as the LLM requests
them.

The ADK implements the client side fully — any MCP server on any
transport plugs into an agent's `tools` list without changes to the
agent itself. The ADK can also expose its own tools *as* an MCP server
so other hosts can consume them.

```{admonition} Prerequisite
:class: note
MCP support is an optional extra.

    pip install 'troopai-adk-python[mcp]'

Without the extra, every `troopai.adk.mcp.*` name is bound to `None`;
compare against `None` to detect availability at runtime.
```

---

## MCP vs function tools vs A2A

Three extension points extend what an agent can do beyond its own Python
code. They are not interchangeable:

| Kind | What supplies it | Defined in your code? | Protocol |
|---|---|---|---|
| `FunctionTool` | A Python callable you write. | Yes. | None — direct call. |
| MCP tool | An external MCP server advertises it. | No — discovered at runtime. | JSON-RPC over stdio / HTTP. |
| A2A | A separate agent process. | No — delegated at runtime. | HTTP + SSE. |

**Rule:** function tools are the default unit of behaviour. MCP extends
the *tool surface* from external servers. A2A delegates to *other
agents* that happen to run as HTTP services. See
{doc}`../concepts/index` for the side-by-side comparison table.

---

## Client side — consuming MCP server tools

### The `MCPToolset` adapter

`MCPToolset` is a `Toolset` subclass. Drop one into `Agent.tools` and
the runner handles connection, tool discovery, and disposal
automatically:

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
        name="fs-agent",
        system_prompt="Use filesystem tools.",
        tools=[MCPToolset(server=server)],
        llm="claude-haiku-4-5",
    )
    result = await Runner.arun(agent, "List /tmp/scratch")
    print(result.final_output)


asyncio.run(main())
```

`MCPToolset` lazy-connects on the first `get_tools()` call (the start
of the run) and the runner's `finally` block calls `adispose()` to
clean up the server.

### Toolset composition

Because `MCPToolset` is a `Toolset`, the standard builders work on it
without any special-casing:

```python
toolset = (
    MCPToolset(server=server)
    .prefixed("fs")        # all tools become fs_read_file, fs_list_dir, …
    .filtered(my_pred)     # drop tools the predicate rejects
)
```

For agents using more than one MCP server, prefix each toolset to
avoid `ToolsetNameConflictError`:

```python
agent = Agent(
    name="multi",
    system_prompt="Two MCP servers attached.",
    tools=[
        MCPToolset(server=server_a).prefixed("a"),
        MCPToolset(server=server_b).prefixed("b"),
    ],
    llm="claude-haiku-4-5",
)
```

See `examples/mcp/multi_server/main.py` for a runnable version.

---

## Transports

The ADK ships four transports. Pick by deployment topology:

### Stdio (subprocess)

`MCPServerStdio` spawns a child process and communicates over its
`stdin`/`stdout`. Ideal for local tools (Node.js servers, Python
scripts, compiled binaries).

```python
from troopai.adk.mcp import MCPServerStdio, MCPServerStdioParams

server = MCPServerStdio(
    name="everything",
    params=MCPServerStdioParams(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything"],
        env={"NODE_ENV": "production"},   # extra env vars (optional)
        cwd="/tmp/work",                  # subprocess working dir (optional)
    ),
)
```

The subprocess is reliably terminated when `cleanup()` runs, even if
the run raised mid-call.

### Streamable HTTP

`MCPServerStreamableHttp` connects to a remote MCP server over HTTP
POST + SSE. This is the modern production transport: stateless
horizontal scaling, standard auth headers, load-balancer friendly.

```python
from troopai.adk.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams

server = MCPServerStreamableHttp(
    name="github",
    params=MCPServerStreamableHttpParams(
        url="https://api.example.com/mcp",
        headers={"X-Api-Key": "static-key"},          # static headers
        header_provider=lambda: {"Authorization": f"Bearer {rotate()}"},  # per-request
        timeout_seconds=30.0,
        sse_read_timeout_seconds=300.0,
    ),
)
```

`header_provider` is called fresh on every outbound HTTP request via an
httpx event hook reading from a `ContextVar`. Concurrent agent turns
each see their own provider with no cross-contamination. Async
providers are supported:

```python
async def fresh_token() -> dict[str, str]:
    token = await sts_client.exchange()
    return {"Authorization": f"Bearer {token}"}

params = MCPServerStreamableHttpParams(url="...", header_provider=fresh_token)
```

### WebSocket

```python
from troopai.adk.mcp import MCPServerWebsocket, MCPServerWebsocketParams

server = MCPServerWebsocket(
    name="ws-server",
    params=MCPServerWebsocketParams(url="wss://api.example.com/mcp"),
)
```

Requires the optional `websockets` package.

### SSE (deprecated)

`MCPServerSse` connects over SSE. The MCP spec deprecated this
transport in favour of streamable HTTP; prefer `MCPServerStreamableHttp`
for new deployments.

---

## Server side — hosting an MCP server

The ADK can expose its own tools as an MCP server so other hosts
(other ADK processes, Claude Desktop, any MCP client) can call them.

`src/troopai/adk/mcp/mcp_server.py` exposes the `MCPServer` abstract
base class. Implement `connect`, `cleanup`, `list_tools`, `call_tool`,
`list_prompts`, `get_prompt`, and `capabilities` to create a custom
server. The ADK's `MCPServerWithClientSession` shared base supplies a
production-ready implementation of caching, locking, and notification
handling that concrete transports (stdio, HTTP) inherit.

For simple cases the hosted-MCP route (OpenAI Responses API) may be
more convenient — see the *Hosted MCP* section below.

### Auth

For HTTP-transport servers, inject auth via `header_provider` on the
params object. `HeaderProvider` is a `Callable[[], dict[str, str] |
Awaitable[dict[str, str]]]`. The `active_header_provider` `ContextVar`
carries the current provider into each request hook; concurrent runs
see isolated providers.

```python
from troopai.adk.mcp import HeaderProvider

def my_provider() -> dict[str, str]:
    return {"Authorization": f"Bearer {vault.get_token()}"}

params = MCPServerStreamableHttpParams(url="...", header_provider=my_provider)
```

---

## Approval flow (HITL)

MCP defines a human-in-the-loop approval round-trip. When an MCP tool
call requires human sign-off before it executes, the framework emits
an `MCPApprovalRequestItem` and suspends. The application inspects the
request, decides, and resumes with an `MCPApprovalResponseItem`.

Both items are defined in `src/troopai/adk/types/items/items.py` and
exported from `troopai.adk.types`. They carry the server name, tool
name, and JSON-encoded arguments so the human reviewer has full
context.

The simplest way to enable approval for all tools on a server is:

```python
toolset = MCPToolset(server=server, requires_approval=True)
```

Every converted MCP tool then flows through the standard HITL deferral
pipeline — calls produce `ToolApprovalItem`s that the application must
approve or reject via `RunState.approve()` / `RunState.reject()` before
the run continues.

For per-tool granularity, write a `tool_filter` that wraps selected
`FunctionTool` instances with `requires_approval=True` after
conversion.

```{admonition} Layer 3 items
:class: note
`MCPApprovalRequestItem` and `MCPApprovalResponseItem` are Layer 3
`RunItem` types — they appear in `RunResult.new_items` and in the run's
conversation history. See {doc}`../architecture/type-layers`
for the full layer contract.
```

---

## Listing tools — `MCPListToolsItem`

When the runner queries a server's tool catalogue, it produces an
`MCPListToolsItem` in the run's history. The item captures a snapshot
of the tools discovered:

```python
from troopai.adk.types import MCPListToolsItem

for item in result.new_items:
    if isinstance(item, MCPListToolsItem):
        print(f"Server: {item.raw.server}")
        for tool in item.raw.tools:
            print(f"  {tool.name}: {tool.description}")
```

`MCPListToolsItem.raw` is an `MCPListTools` dataclass with fields
`server` (server name), `tools` (list of `MCPListToolsTool`), and
`error` (set when listing failed).

---

## Composition with function tools

MCP tools and function tools coexist naturally in one agent — they are
all `FunctionTool` instances from the runner's perspective:

```python
from troopai.adk.tools.function_tool import function_tool


@function_tool
def local_lookup(key: str) -> str:
    """Look up a value in the local cache."""
    return cache.get(key, "not found")


agent = Agent(
    name="composer",
    system_prompt="Use local cache or MCP filesystem as needed.",
    tools=[
        local_lookup,
        MCPToolset(server=filesystem_server).prefixed("fs"),
    ],
    llm="claude-haiku-4-5",
)
```

Every `FunctionTool` cost lever — `max_result_tokens`, `max_retries`,
`cache_function`, `prepare`, `rate_limit`, `defer_loading` — works on
MCP-derived tools without further configuration because they are
`FunctionTool` instances produced by `mcp_tool_to_function_tool`.

---

## Common patterns

### Local development with stdio

Run an MCP server locally via `npx`, a Python script, or any compiled
binary. Stdio is zero-config: no port management, no network, no TLS.
The subprocess inherits the parent's PATH and environment by default.

```python
server = MCPServerStdio(
    name="dev-tools",
    params=MCPServerStdioParams(command="python", args=["my_mcp_server.py"]),
)
```

Use the reference test server (`@modelcontextprotocol/server-everything`)
to verify integration before writing your own.

### Production with HTTP MCP

Deploy your MCP server as a standalone HTTP service behind a load
balancer. Use `MCPServerStreamableHttp` with static bearer tokens or a
rotating `header_provider`. Set `sse_read_timeout_seconds` to cover
the worst-case duration of your longest-running tool call.

```python
server = MCPServerStreamableHttp(
    name="prod-api",
    params=MCPServerStreamableHttpParams(
        url="https://mcp.internal.example.com/mcp",
        header_provider=lambda: {"Authorization": f"Bearer {vault.token()}"},
        sse_read_timeout_seconds=600.0,  # 10 minutes for long jobs
    ),
)
```

### Multi-tenant MCP gateway

When one server handles many tenants, inject per-tenant credentials
via a `header_provider` bound to the current request context:

```python
import contextvars

_tenant_token: contextvars.ContextVar[str] = contextvars.ContextVar("tenant_token")


def tenant_provider() -> dict[str, str]:
    return {"X-Tenant-Token": _tenant_token.get()}


server = MCPServerStreamableHttp(
    name="gateway",
    params=MCPServerStreamableHttpParams(
        url="https://mcp.example.com/mcp",
        header_provider=tenant_provider,
    ),
)


async def handle_request(tenant_token: str, user_message: str) -> str:
    token = _tenant_token.set(tenant_token)
    try:
        result = await Runner.arun(agent, user_message)
        return result.final_output
    finally:
        _tenant_token.reset(token)
```

Concurrent `Runner.arun` calls each see their own `ContextVar` value;
the single `MCPServerStreamableHttp` instance reuses the same HTTP
connection while injecting different headers per call.

### Ref-counted sharing with `MCPServerManager`

For agents that span multiple MCP servers with a shared lifecycle:

```python
from troopai.adk.mcp import MCPServerManager

manager = MCPServerManager(servers=[server_a, server_b])
async with manager:
    agent = Agent(
        name="x",
        system_prompt="...",
        tools=[
            MCPToolset(server=server_a, auto_connect=False),
            MCPToolset(server=server_b, auto_connect=False),
        ],
        llm="claude-haiku-4-5",
    )
    await Runner.arun(agent, "Do something.")
```

`MCPServerManager` ref-counts `acquire` / `release` calls and cleans
up each server only when its count reaches zero.

---

## Hosted MCP (OpenAI Responses API)

The OpenAI Responses API can run the MCP loop server-side. Use
`HostedMCPTool` — no Python-side connection is opened:

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
    llm=OpenAIResponsesLLM(model="gpt-4o"),
)
```

Anthropic, Gemini, and Chat Completions raise `UnsupportedHostedToolError`
because they do not ship hosted MCP server-side.

---

## Limits

- **Handoff target disposal** — the auto-disposal path covers only
  the entry-point agent's `tools`. Toolsets contributed by handoff
  target agents must be managed via `MCPServerManager`
  (`auto_connect=False`) or an explicit `async with` on the server.
- **MCP Tasks API** — long-running tool calls via the MCP Tasks API
  are not yet wrapped.
- **`MCPApprovalRequestItem` / `MCPApprovalResponseItem` emission** —
  these Layer 3 items are defined and exported but not yet emitted by
  the runner's hosted-tool loop. Wiring is a runner-loop change; the
  type definitions are stable.
- **Streamable HTTP transport** — the ADK imports
  `mcp.client.streamable_http.streamablehttp_client`. This import
  works correctly with `mcp` 1.x (current). An older memory note
  referenced a rename issue on a pre-1.0 SDK snapshot; that breakage
  no longer applies.

---

## See also

- {doc}`../concepts/index` — MCP vs A2A vs function tools side-by-side.
- {doc}`../architecture/type-layers` — the three-layer type architecture,
  including how `MCPListToolsItem`, `MCPApprovalRequestItem`, and
  `MCPApprovalResponseItem` fit into the Layer 3 `RunItem` contract.
- `examples/mcp/` — runnable examples (stdio, streamable HTTP,
  multi-server, auth headers, cached, SSE, WebSocket).
- Upstream MCP spec: <https://modelcontextprotocol.io/>.

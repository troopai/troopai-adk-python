# MCP Module

Model Context Protocol integration. Optional extra: `pip install
'troopai-adk-python[mcp]'`. Every export degrades to ``None`` when
the underlying ``mcp`` package is missing.

## Files

| File | Purpose |
|---|---|
| `__init__.py` | Optional-import gate; re-exports public surface. |
| `mcp_server.py` | `MCPServer` ABC + `MCPServerWithClientSession` shared base (cache, locks, ClientSession lifecycle, sampling/elicitation wiring). |
| `stdio.py` | `MCPServerStdio` + `MCPServerStdioParams` — subprocess transport. |
| `http.py` | `MCPServerStreamableHttp` + `MCPServerStreamableHttpParams` — modern streamable-HTTP transport with per-request header injection. |
| `websocket.py` | `MCPServerWebsocket` + params — WebSocket transport. |
| `sse.py` | `MCPServerSse` + params — SSE transport (MCP-spec-deprecated). |
| `manager.py` | `MCPServerManager` — multi-server lifecycle holder with ref-counted `acquire` / `release`. |
| `conversion.py` | `mcp_tool_to_function_tool`, `call_tool_result_to_str`. The only module that imports `mcp.types`. |
| `extras.py` | `build_resource_tool` + `build_prompt_tools` — opt-in surfaces (resources, prompts-as-tools). |
| `schema_resolver.py` | `inline_intra_document_refs` — opt-in `$ref` resolver for `inputSchema`. |
| `sampling.py` | `make_sampling_callback` — bridges MCP `sampling/createMessage` to the host `LLM`. |
| `elicitation.py` | `ElicitationHandler` + `make_elicitation_callback` — server-driven user input. |
| `filters.py` | `ToolFilter` and `ToolFilterContext` — server-aware per-tool predicates. |
| `auth.py` | `HeaderProvider` callable type and `active_header_provider` ContextVar. |
| `notifications.py` | `make_message_handler` — `tools/list_changed` push subscriber. |
| `otel.py` | `build_mcp_meta` — W3C Trace Context injection into MCP `_meta`. |
| `run_hooks_bridge.py` | ContextVar bridge: `Runner.arun` → `MCPServer` → `RunHooks.on_mcp_*`. |
| `exceptions.py` | `MCPError` hierarchy under `TroopAIError`. |

## Architectural Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | MCP servers compose into `agent.tools` via `MCPToolset(Toolset)` (lives at `tools/toolsets/mcp_toolset.py`) | Reuses existing toolset infrastructure (`prefixed`, `filtered`, `combined_with`, `ToolsetNameConflictError`). One slot in `agent.tools` for "everything the agent can do" — no separate `mcp_servers` field. |
| 2 | Auto-managed lifecycle by default | `MCPToolset.auto_connect=True` lazy-connects on first `get_tools()`; `Toolset.adispose()` (added to the ABC) is called from `Runner.arun` and `_run_streamed_impl` `finally` blocks. `auto_connect=False` is the escape hatch for externally-managed connections (`MCPServerManager`, explicit `async with`). |
| 3 | `mcp.types.*` confined to this package | Same discipline as `llms/litellm/` confines litellm wire types. The framework outside `mcp/` sees only `FunctionTool` (Layer 1) and the existing `MCPListToolsItem` / `MCPApprovalRequestItem` / `MCPApprovalResponseItem` Layer 3 RunItems (not yet emitted by the runner). |
| 4 | `ToolFilter` is MCP-specific (server-aware), not the generic `ToolsetFilter` | A multi-server agent often needs to filter tools by *origin* — e.g. allow read tools only from `prod-api`. `ToolFilterContext.server_name` exposes that; the generic `ToolsetFilter` does not. |
| 5 | Per-request header injection via `ContextVar` | One HTTP client serves rotating tokens. Concurrent agent turns each see their own provider with no shared state. Adapted from Microsoft Agent Framework's pattern. |
| 6 | Push-driven cache invalidation via `tools/list_changed` | Zero-cost on the steady state, immediate on changes. Invalidation flips a flag (`_cache_dirty`); the next `list_tools` call re-fetches under `_cache_lock`. |
| 7 | OTel context propagation via MCP `_meta` | The MCP protocol reserves `_meta` for transport-agnostic request metadata; W3C Trace Context headers (`traceparent` / `tracestate`) are injected via the global propagator. Soft-import: zero overhead and no error when OTel is missing. |
| 8 | `MCPServer.cleanup()` runs in the same task that called `connect()` | Avoids the AnyIO "cancel scope must end in same task" pitfall. The runner's auto-disposal path preserves this invariant because `arun()` and `_run_streamed_impl()` each run in a single task. |
| 9 | `isError=True` flows through standard tool-error path | `call_tool_result_to_str` raises `MCPToolCallError`; the executor converts to model-visible string when `RunConfig.fail_on_tool_error=False` (default), re-raises when `True`. No bespoke logic. |
| 10 | `inputSchema` passed through unchanged | Most providers handle intra-document `$ref` natively. Strict-mode normalisation would silently mutate the MCP server's contract; keep the raw schema and let the LLM provider decide. |

## Type-Layer Compliance

- **Layer 2 (wire)** — `mcp.types.Tool`, `CallToolResult`,
  `ServerCapabilities`, `ListPromptsResult`, `GetPromptResult` —
  consumed only inside this package. They never appear in `agents/`,
  `run/`, the framework's developer-facing tools layer, or in
  toolsets *except* at the boundary of `tools/toolsets/mcp_toolset.py`
  which performs the conversion.
- **Layer 1 (framework)** — `FunctionTool` is the surface produced
  by `mcp_tool_to_function_tool`.
- **Layer 3 (items)** — `MCPListToolsItem`,
  `MCPApprovalRequestItem`, `MCPApprovalResponseItem` are defined in
  `types/items/items.py` but are not yet emitted by the runner. They
  are reserved for future hosted-MCP approval flow integration; the
  Layer 1 wire types and Layer 3 RunItems are already correct so
  wiring is purely a runner-loop change.

## Cost / Performance Levers

- `cache_tools_list=True` (default) — saves `list_tools` round-trip per turn.
- `notifications/tools/list_changed` push invalidation — no polling.
- All `FunctionTool` cost levers (`max_result_tokens`, `max_retries`,
  `cache_function`, `prepare`, `rate_limit`, `defer_loading`) work
  transparently on converted MCP tools.
- `header_provider` ContextVar avoids per-call session re-creation
  when only headers change.

## Future work (not yet shipped)

- MCP Tasks API for long-running tools (Strands pattern).
- Pickling support for Temporal compatibility (Google ADK pattern).
- Surfacing `MCPApprovalRequestItem` / `MCPApprovalResponseItem` for
  the **hosted** MCP approval flow (the Layer 3 RunItems exist;
  their wiring belongs to the OpenAI Responses run-loop integration,
  not to this module).

See `docs/mcp/mcp.md` for usage. See `examples/mcp/` for runnable examples.

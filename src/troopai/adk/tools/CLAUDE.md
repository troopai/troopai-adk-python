# Tools Module

Tool system for wrapping Python functions as agent capabilities.

## Layout

Split by **who executes the tool**, plus framework support files at root.

| Path | Who executes |
|---|---|
| `function_tool.py` — `FunctionTool` + `@function_tool` | Framework's tool executor (user's Python function) |
| `builtin/` — `MemoryTool`, `JITContextAwareTool`, `DocumentSearchTool` | Framework (built-in capabilities) |
| `local/` — `ShellTool`, `ApplyPatchTool` | Developer's environment (user-supplied executor/editor) |
| `hosted/` — `WebSearchTool`, `CodeExecutionTool`, … | LLM provider, server-side |

Support files at root:

- `builtin_tool.py` — `BuiltinTool` / `ExecutableBuiltinTool` ABCs
- `tool_context.py` — `ToolContext` and execution-/history-aware subclasses
- `tool_guardrails.py` — per-tool input/output guardrail decorators
- `tool_output_trimmer.py` — `trim_tool_output` wrapper
- `deferred_tool.py` — HITL deferred-execution machinery
- `tool_middleware.py` — `ToolMiddleware` Protocol (plumbing only — see below)

See `tools/{builtin,hosted,local}/CLAUDE.md` for per-package detail.
See `docs/tools/tools.md` and `examples/tools/`.

## Tool Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | str | Tool identifier |
| `description` | Optional[str] | Purpose (shown to LLM) |
| `schema` | BaseModel \| dict | Inputs |
| `schema_enforcement` | SchemaEnforcement | NONE / NORMALIZED / STRICT |
| `input_guardrails` / `output_guardrails` | list | Pre/post checks |
| `enabled` | bool \| Callable | Dynamic enable/disable |
| `requires_approval` | bool \| Callable | HITL approval before execution |
| `max_result_tokens` | Optional[int] | Result truncation (Runner) |
| `max_retries` | Optional[int] | LLM retry budget (None=∞, 0=no retries, N=N retries) |
| `timeout` / `timeout_behavior` / `on_timeout` | float \| str \| Callable | Per-tool timeout |
| `on_invoke` | ToolInvokeFunction \| None | Invocation callback |
| `execution_aware` | bool | Whether tool gets `ExecutionAwareToolContext` |
| `cache_function` | Callable \| None | `(args, result) -> bool` selective cache |
| `response_format` | str | `"text"` (default) or `"content_and_artifact"` |
| `return_direct` | bool | Skip LLM rewrite, result becomes final output |
| `prepare` | Callable \| None | Dynamic tool definition modifier per LLM step |
| `requires_env` / `requires_packages` | tuple[str, ...] | Validated at agent construction |
| `rate_limit` | ToolRateLimit \| None | Sliding-window (rpm + wait/error behavior) |
| `defer_loading` | bool | Hide from LLM until revealed via `build_tool_search()` |

`as_tool()` delegation: observed via `get_delegate_agent()` method (no public field).

## Tool Execution

Results: `FunctionToolCallResult` (`@dataclass(frozen=True)`, in
`types/tools/tool_types.py`). Converted to `ToolResultMessage` before
adding to history.

**Streaming parity:** input guardrails, output guardrails, and
`ExecutionAwareToolContext` work in both modes via shared
`_execute_single_tool_call()`.

## Tool Timeout

`asyncio.wait_for()` wrapper. Set `FunctionTool.timeout`.

- `"error_as_result"` (default): LLM sees error, can retry.
- `"raise_exception"`: Halts with `ToolTimeoutError`.
- `on_timeout`: Custom error message `(ctx, error) -> str`.

## LLM Retry Budget

Per-tool failure budget; tools removed from LLM's view after exhaustion.
Failures = exceptions (when `fail_on_tool_error=False`) + timeouts (when
`timeout_behavior="error_as_result"`). Does NOT count guardrail
rejections, HITL deferrals, halt-execution errors.

## Tool Guardrails

- `@tool_input_guardrail()` → `ToolInputGuardrailData` → `ToolGuardrailFunctionOutput`
- `@tool_output_guardrail()` → `ToolOutputGuardrailData` → `ToolGuardrailFunctionOutput`
- Verdicts: `.allow()`, `.reject_content(...)`, `.raise_exception()`
- `resolved_action()` maps each verdict onto the shared `GuardrailAction`
  (`reject_content`→`TRANSFORM`, `raise_exception`→`RAISE`, `allow`→`PASS`).

## Built-in Tools (Framework-Local)

ABCs for tools the framework runs (vs provider-hosted). All sit in
`Agent.tools` alongside `FunctionTool`.

- `BuiltinTool` — `name` + `description` only (run-loop handled).
- `ExecutableBuiltinTool(BuiltinTool)` — adds `schema` + `on_invoke`.

Concrete: `ShellTool`, `ApplyPatchTool` (skipped with warning if no
executor/editor); `JITContextAwareTool` (see below); `MemoryTool`
family from `troopai.adk.memory`; `DocumentSearchTool` family (RAG search
over a document corpus) built on `troopai.adk.rag` — `document_search_tool.py`
holds the core tool plus typed `PDFSearchTool` / `DOCXSearchTool` /
`WebsiteSearchTool` / … wrappers that each pin a loader.

Provider-hosted (web search, code exec, file search, image gen, URL
context) → typed `HostedTool` subclasses in `tools/hosted/`. See
`tools/hosted/CLAUDE.md` and `tools-guardrails` rule.
`LLMConfig.extra_body` is escape hatch for beta/esoteric shapes.

## Tool Context

- `ToolContext` — `ctx.context` (user dict).
- `ExecutionAwareToolContext` — read-only snapshots: `usage`, `turns`,
  `messages`, `tokens`. Opt-in via type annotation.
- `HistoryAwareToolContext(ExecutionAwareToolContext)` — adds
  `history: tuple[RunItem, ...]` (Layer 3 RunItems).

## JITContextAwareTool

Active context management — LLM manages its own context window via
tools. Subclasses `BuiltinTool`, expands at runtime into `FunctionTool`
instances. Tools: `save_note`, `recall_notes`, `manage_context`,
`search_history` (history-aware), `context_stats` (execution-aware).

`manage_context` emits compact/drop directives; Runner consumes via
`apply_directives()` before next LLM call. Storage: pluggable
`NoteStore` Protocol (default `InMemoryNoteStore`).

See `docs/tools/jit_context_aware.md`.

## Toolsets

`tools/toolsets/` — live, materialise-per-turn collections. `Toolset`
ABC + `FunctionToolset`, `PrefixedToolset`, `RenamedToolset`,
`FilteredToolset` (per-turn predicate against `RunContext`),
`CombinedToolset`, `WrapperToolset`. `Agent.tools` accepts toolset
entries; `build_tools()` flattens with `ToolsetNameConflictError` on
collisions. Renamed clones preserve internal state via
`FunctionTool.clone()`.

See `docs/tools/toolsets.md`, `examples/tools/toolsets/`.

## Tool Middleware

`tool_middleware.py` — `ToolMiddleware` Protocol with
`(ctx, tool, args, next) -> result`. Wraps `tool.on_invoke`.
**Plumbing only** — see `middleware-vs-guardrails` rule for the
forbidden-vs-allowed contract. Verdicts (PII, jailbreak, content
filtering, rate limiting, approval, schema validation) belong in
guardrails or typed surfaces.

Registration: `Agent.middleware.tools` (agent-global) and
`WrapperToolset.middleware` (toolset-scoped, composes inside agent
chain). Standard middleware: `ToolLoggingMiddleware`,
`ToolMetricsMiddleware`.

See `docs/tools/middleware.md`, `examples/tools/middleware/`.

## Streaming Tool Results

`function_tool(..., streaming=True)` → async-generator mode. Wrapped
function yields `ToolStreamEvent` instead of returning a value. The
executor drains inside the innermost middleware terminal, surfaces
non-`"done"` events as `RunItemType.TOOL_PARTIAL_OUTPUT` to consumers
of `Runner.arun(stream=True)`. LLM sees exactly one tool-result
message — the value on the terminal `"done"` event.

Drain placement is load-bearing: lives inside the terminal closure
(in `tools_executor.py` and `tool_middleware.py`) so middleware
observes the final accumulated value, not chunks. A module-level
`_TOOL_STREAM_SINK: ContextVar` carries the active sink across the
chain — set by `execute_tool_calls_streamed`, default `None` for
non-streaming (silent drain + `logger.warning`). `requires_approval ×
streaming` defers iterator start until approval granted.

Mutually incoherent with: `cache`, `cache_function`,
`response_format="content_and_artifact"`, `return_direct=True`. All
four raise `ValueError` at construction.

See `docs/tools/streaming_tool_results.md`.

## Cost Optimization Features

Framework-wide levers in `architecture.md`. Tool-specific:

| Feature | Description |
|---|---|
| Tool artifact | `response_format="content_and_artifact"` — LLM gets summary, app gets data |
| `return_direct` | Skip LLM post-processing, result becomes final output |
| `prepare` | Dynamic tool def modifier per LLM step (more powerful than `enabled`) |
| `cache_function` | Selective caching — cache successes but not errors |
| `trim_tool_output` | Wrap existing FunctionTool in token/char budget |

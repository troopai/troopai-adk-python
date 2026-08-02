# Tools System

The tools system in TroopAI Agents wraps Python functions as agent capabilities.

## Tool Types

### Function Tools (`FunctionTool`)

Regular Python functions exposed to the LLM via `@function_tool` decorator or direct `FunctionTool` construction.

```python
from troopai.adk.tools import function_tool

@function_tool(name="search", description="Search the database")
def search(query: str) -> str:
    return f"Results for {query}"
```

### Framework-Local Built-in Tools

Framework-local tools live in `Agent.tools` alongside `FunctionTool` — there
is no separate `builtin_tools` list. These are tools whose behavior is
implemented inside this codebase rather than dispatched to the provider:

- `ShellTool` — shell-command execution (requires `executor`)
- `ApplyPatchTool` — file-patch editor (requires `editor`)
- `JITContextAwareTool` — active context management via notes / directives
- `MemoryTool` (and sub-tools) — local memory management

```python
from troopai.adk.tools import JITContextAwareTool, function_tool

@function_tool(name="search", description="Search the database")
def search(query: str) -> str: ...

agent = Agent(
    name="Researcher",
    tools=[search, JITContextAwareTool()],
)
```

### Provider-Native Capabilities

Provider-native tools (web search, file search, computer use, image
generation, code interpreter) are NOT wrapped as framework tool classes.
Pass the raw provider tool JSON via `LLMConfig.extra_body` / `extra_args`.
Keeping provider-specific schemas out of the framework preserves provider
neutrality and avoids the partial-coverage trap where a stale wrapper
silently masks newer provider tool versions.

The response-side types below remain available for users who want to
parse provider-native tool results coming back through the LLM response.

## Type System

### Definition Types (sent to LLM)

| Type | Description |
|------|-------------|
| `FunctionToolDefinition` | Flat frozen dataclass with name, description, schema, type |

### Result Types (from execution)

| Type | Description |
|------|-------------|
| `FunctionToolCallResult` | BaseModel with `call_id`, `output` (multimodal), `artifact`, `type`, `id`, `status` |
| `FunctionToolFailureCounts` | `dict[str, int]` alias for retry budget tracking |

### Tool Output Types (structured returns)

| Type | Description |
|------|-------------|
| `ToolOutputText` | Text output: `{type: "text", text: str}` |
| `ToolOutputImage` | Image output: `{type: "image", image_url: str, detail: ...}` |
| `ToolOutputFileContent` | File output: `{type: "file", file_data: str, filename: ...}` |

### Built-in Tool Types (from LLM responses)

Structured types for built-in tool calls and results:

| Category | Types |
|----------|-------|
| Web Search | `WebSearchToolCall`, `WebSearchToolCallResult`, `WebSearchResult` |
| File Search | `FileSearchToolCall`, `FileSearchToolCallResult`, `FileSearchResult` |
| Computer Use | `ComputerToolCall`, `ComputerToolCallResult`, `ComputerAction` |

## Tool Guardrails

### Input Guardrails (before execution)

```python
@tool_input_guardrail()
def check_input(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    if not valid(data.context.tool_arguments):
        return ToolGuardrailFunctionOutput.reject_content("Invalid")
    return ToolGuardrailFunctionOutput.allow()
```

### Output Guardrails (after execution)

```python
@tool_output_guardrail()
def check_output(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    return ToolGuardrailFunctionOutput.allow()
```

Three behaviors: `allow()`, `reject_content(msg)`, `raise_exception()`.

### Guardrail Exception Handling

When a tool guardrail raises an unexpected exception (e.g., an external validation service is down):

- **`fail_on_tool_error=True` (default)** — The exception propagates. This is **fail-closed**: a broken guardrail halts execution rather than silently allowing the tool call through.
- **`fail_on_tool_error=False`** — The exception is logged and the guardrail is skipped.

This applies to both input and output guardrails, and in both the executor and HITL resumption paths.

## HITL (Human-in-the-Loop)

```python
@function_tool(name="deploy", description="Deploy", requires_approval=True)
def deploy(service: str) -> str: ...

# Conditional approval
async def require_in_prod(ctx: ToolContext) -> bool:
    return ctx.context.get("env") == "production"

@function_tool(name="restart", description="Restart", requires_approval=require_in_prod)
def restart(service: str) -> str: ...
```

### Resumption Flow

When a tool requires approval, execution pauses and returns a `RunResult` with `requires_action=True`:

```python
result = await Runner.arun(agent, "Deploy to production")

while result.requires_action:
    for req in result.deferred_requests.approvals:
        if await confirm(f"Approve {req.tool_name}?"):
            result.state.approve(req)
        else:
            result.state.reject(req, "Not authorized")

    result = await Runner.arun(agent, result.state)
```

### Resumption Security

On resumption, the executor re-runs security checks before executing approved tools:

- **Layer 0 (can_use_tool)** — Permission callback runs again. A tool whose permissions changed between deferral and approval is blocked.
- **Enabled check** — A tool disabled between deferral and approval is blocked.
- **Output guardrails** — Run on the tool's result after execution.

See `examples/agent_patterns/human_in_the_loop.py` for the full pattern and `examples/agent_patterns/nested_human_in_the_loop.py` for nested HITL through `as_tool()` boundaries.

## Streaming Parity

Streaming mode has full feature parity with non-streaming. Both paths use the shared `_execute_single_tool_call()` coroutine:

- Input guardrails
- Output guardrails
- `ExecutionAwareToolContext`
- HITL approval events (`TOOL_APPROVAL_REQUESTED`)

## Advanced Features

### Tool Artifact (Dual Return)

Tools with `response_format="content_and_artifact"` return `(content_for_llm, artifact_for_app)`:

```python
@function_tool(name="rag", response_format="content_and_artifact")
def rag_search(query: str) -> tuple[str, list[Document]]:
    docs = retrieve(query)
    return f"Found {len(docs)} results", docs  # LLM gets summary, app gets docs

# Access: result.new_items[-1].artifact -> list[Document]
```

### return_direct (Skip LLM Rewrite)

Tool result becomes `final_output` immediately, skipping one LLM round-trip:

```python
@function_tool(name="report", return_direct=True)
def generate_report(data: str) -> str:
    return build_formatted_report(data)  # Goes straight to user
```

### Tool Prepare (Dynamic Modification)

Modify or exclude tool definitions per LLM step based on runtime context:

```python
def prepare_search(ctx, tool_def):
    remaining = ctx.context.get("api_calls_remaining", 0)
    if remaining <= 0:
        return None  # Exclude tool this step
    return FunctionToolDefinition(
        name=tool_def.name,
        description=f"Search ({remaining} calls remaining)",
        schema=tool_def.schema,
    )

@function_tool(name="search", prepare=prepare_search)
def search(query: str) -> str: ...
```

### Conditional Cache

Cache selectively — success only, not errors:

```python
@function_tool(
    name="api",
    cache=True,
    cache_function=lambda args, result: "error" not in result.lower(),
)
def api_call(query: str) -> str: ...
```

## Cost Optimization

- `max_result_tokens` — Cap tool output size
- `max_retries` — Remove broken tools from LLM's view
- `response_format="content_and_artifact"` — LLM gets summary, app gets full data
- `return_direct=True` — Skip one LLM round-trip for polished outputs
- JSON minification — Automatic compact serialization
- `CacheStrategy.STABLE` — Preserve prompt cache when tools change
- `cache=True` + `cache_function` — Intelligent result caching

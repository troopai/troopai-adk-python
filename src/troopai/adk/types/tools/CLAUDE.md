# Tool Types

Tool configuration, behavior, and built-in tool call/result types.

## Files

- `tool_config.py` - `ToolChoice`, `ToolExecutionMode`
- `tool_use_behavior.py` - `FunctionToolResult`, `StopAtTools`, `ToolUseBehavior`
- `builtin_tool_types.py` - Built-in tool call/result types

## ToolChoice

`ToolChoice` is a TypeAlias: `Union[Literal["auto", "required", "none"], str]`. Controls how the LLM selects tools. Configured via `LLMConfig.tool_choice`.

- `"auto"` -- LLM decides whether to call a tool
- `"required"` -- LLM must call a tool
- `"none"` -- text only, no tool calls
- Any other `str` -- force a specific tool by name

## ToolExecutionMode

`ToolExecutionMode` (`StrEnum`): `SEQUENTIAL` (one tool per turn, default), `PARALLEL` (multiple concurrent tool calls). Configured via `LLMConfig.tool_execution_mode`.

These are batch-level settings for all tools in a single LLM request. Per-tool settings (timeout, max_retries) live on `FunctionTool`.

## Tool Use Behavior

`ToolUseBehavior` is a union type controlling what happens after tool execution, set via `Agent.tool_use_behavior`:

- `"run_llm_again"` (default) -- tool results go back to the LLM
- `"stop_on_first_tool"` -- first tool's output becomes the final result
- `StopAtTools(stop_at_tool_names=[...])` -- stop on specific named tools
- Custom function: `(RunContext, list[FunctionToolResult]) -> ToolsToFinalOutputResult`

Supporting types:

- `FunctionToolResult` -- result from a single tool execution (name, call_id, output)
- `ToolsToFinalOutputResult` -- return type for custom behavior functions (is_final_output, final_output)
- `ToolsToFinalOutputFunction` -- callable type alias for custom behavior functions

## Built-in Tool Response Types

Provider-agnostic `@dataclass(frozen=True)` types for structured access to provider-native tool calls (from LLM responses) and their results. These type the **response side** only — what comes BACK when a developer enabled a provider-native capability via a typed `HostedTool` subclass (`tools/hosted/`) or `LLMConfig.extra_body` / `extra_args`. Request-side authoring contract: `tools-guardrails` rule.

Each category follows the same pattern: a result entry type, a tool call type, and a tool call result type.

### Web Search

- `WebSearchResult` -- single result entry (url, title, optional snippet)
- `WebSearchToolCall` -- search call from LLM (id, query, optional status)
- `WebSearchToolCallResult` -- result container (call_id, list of `WebSearchResult`)

### File Search

- `FileSearchResult` -- single result entry (optional file_id, filename, score, text)
- `FileSearchToolCall` -- search call from LLM (id, list of queries, optional status)
- `FileSearchToolCallResult` -- result container (call_id, list of `FileSearchResult`)

### Computer Use

- `ComputerAction` -- action requested by the tool (type + optional coordinates, text, keys, button, scroll amounts)
- `ComputerToolCall` -- computer use call from LLM (id, action, call_id, optional status)
- `ComputerToolCallResult` -- result container (call_id, optional output text, optional base64 screenshot)

These are definition types. Integration with the LLM response parsing layer is out of scope for this iteration.

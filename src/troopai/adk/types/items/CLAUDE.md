# Items Module

Conversation history entries accumulated during agent runs (Layer 3 of the type system).

## Design

`RunItemBase[T]` is the generic base. Every item has `raw: T` (required),
`type` discriminator, and `agent_name`. Two categories: wrapped items
(`raw` is a Layer 1 output BaseModel) and synthetic items (`raw` is a
TypedDict, dataclass, or dedicated type).

See `.claude/rules/items.md` for design constraints.

## Item Classes

| Item | Type Discriminator | Raw Type | Extra Fields |
|------|--------------------|----------|--------------|
| `SystemItem` | `"system"` | `LLMInputEasyMessage` | — |
| `UserItem` | `"user"` | `LLMInputEasyMessage` | — |
| `MessageOutputItem` | `"message_output"` | `list[LLMResponseText \| LLMResponseRefusal]` | `id`, `status` |
| `ToolCallItem` | `"tool_call"` | `LLMResponseFunctionToolCall` | `description` |
| `ToolCallOutputItem` | `"tool_call_output"` | `FunctionToolCallResult` | — |
| `ReasoningItem` | `"reasoning"` | `LLMResponseReasoning` | — |
| `HandoffCallItem` | `"handoff_call"` | `LLMResponseFunctionToolCall` | `target_agent` |
| `HandoffOutputItem` | `"handoff_output"` | `FunctionToolCallResultParam` | `source`, `target` |
| `CompactionItem` | `"compaction"` | `LLMInputEasyMessage` | — |
| `MCPListToolsItem` | `"mcp_list_tools"` | `MCPListTools` | — |
| `MCPApprovalRequestItem` | `"mcp_approval_request"` | `MCPApprovalRequest` | — |
| `MCPApprovalResponseItem` | `"mcp_approval_response"` | `MCPApprovalResponse` | — |
| `ToolApprovalItem` | `"tool_approval"` | `DeferredToolCall` | `approved`, `message` |
| `ToolSearchCallItem` | `"tool_search_call"` | `ToolSearchToolCall` | — |
| `ToolSearchOutputItem` | `"tool_search_output"` | `ToolSearchToolCallResult` | — |

## ItemHelpers

Static utilities for working with RunItems (data extraction lives here, not on item classes):

| Method | Purpose |
|--------|---------|
| `extract_last_content(item)` | Last text or refusal from a MessageOutputItem |
| `input_to_new_input_list(input)` | Normalize string or item list to `list[LLMInputContentItem]` |
| `text_message_output(item)` | Concatenate all text parts from a MessageOutputItem |
| `text_message_outputs(items)` | Concatenate text from all MessageOutputItems in a sequence |
| `extract_last_text(items)` | Text from the last MessageOutputItem in a sequence |
| `refusal_message_output(item)` | Extract refusal text from a MessageOutputItem |
| `tool_call_output_str(item)` | Coerce a ToolCallOutputItem's output to string |
| `reasoning_summary_text(item)` | Concatenate summary text from a ReasoningItem |
| `reasoning_content_text(item)` | Concatenate reasoning content from a ReasoningItem |
| `tool_call_output_item(tool_call, output)` | Create a ToolCallOutputItem from a ToolCallItem |
| `response_to_run_items(response, agent_name)` | Convert LLMResponse directly to RunItems (no intermediate dict) |
| `message_to_run_items(msg, agent_name)` | Convert a single message dict to RunItems |
| `messages_to_run_items(messages)` | Convert a sequence of message dicts to RunItems |
| `run_items_to_params(items)` | Convert RunItems to Layer 1 params via `to_param()` |

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| `raw: T` on base, required | Single source of truth, matches OpenAI pattern |
| `type` discriminator on every item | Serialization/deserialization without `isinstance` |
| `agent_name: Optional[str]` | Multi-agent observability without weak-ref complexity |
| Frozen dataclasses | Immutable items for handoff filter pipelines |

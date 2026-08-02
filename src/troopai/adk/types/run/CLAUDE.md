# Result Module

Outcome of agent execution, including HITL interruption support.

## RunResult

`RunResult[T]` is a `@dataclass` returned by `Runner.run()` / `Runner.arun()`.

| Field | Type | Description |
|-------|------|-------------|
| `final_output` | `Any` | Agent's final output (str or structured), None if interrupted |
| `user_prompt` | `UserPrompt` | Original user input |
| `new_items` | `list[RunItem]` | Layer 3 RunItems generated during execution |
| `context` | `RunContext[T]` | Run context with usage tracking |
| `last_agent` | `Optional[Agent[T]]` | Final active agent (after handoffs) |
| `deferred_requests` | `Optional[DeferredToolRequests]` | Pending HITL approvals |
| `state` | `Optional[RunState]` | Serializable state for resumption |
| `guardrail_results.input` | `tuple[AgentInputGuardrailResult, ...]` | Input guardrail audit trail |
| `guardrail_results.output` | `tuple[AgentOutputGuardrailResult, ...]` | Output guardrail audit trail |

## Key Methods

- `requires_action` — `True` if deferred requests are pending
- `to_input_list()` — Convert to message list for continued conversation
- `final_output_as(type)` — Type-safe cast of final output

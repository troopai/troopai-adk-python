# Agents Module

Agent implementation with guardrails and handoffs.

## Files

- `agent.py` — `Agent` and `BaseAgent` dataclasses
- `guardrails.py` — Input/output guardrail decorators
- `middleware.py` — `Middleware` config dataclass (per-layer middleware lists)

## Agent Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | str | Required identifier |
| `description` | str? | What this agent does (flows to as_tool, handoffs) |
| `system_prompt` | str \| SystemPrompt \| DynamicSystemPrompt | See `prompts/` |
| `tools` | list[Tool \| Toolset] | FunctionTool, skill tool, local-execution framework tool, or `Toolset` (incl. `MCPToolset` for MCP servers) |
| `input_guardrails` / `output_guardrails` | list | Pre/post validation |
| `handoffs` | list[Agent] \| HandoffRoute | Delegation targets |
| `llm` | str | Model override (e.g., "gpt-4o") |
| `llm_config` | LLMConfig | Temperature, max_tokens, etc. |
| `middleware` | Middleware | Per-layer (`tools` shipped; `agents`/`llms` reserved). Plumbing-only |
| `output_schema` | BaseModel | Pydantic for structured output |
| `tool_use_behavior` | ToolUseBehavior | Post-tool-execution behavior |

See `docs/agents/`, `examples/agent_patterns/`.

## Tool Use Behavior

`Agent.tool_use_behavior` controls whether the loop runs another LLM
call after tool execution. Applies only to FunctionTools (handoff
tools and HITL deferrals take priority). Variants and
`ToolsToFinalOutputResult` contract live in `types/tools/CLAUDE.md`.

## Middleware

`Agent.middleware: Middleware` holds per-layer lists. Three slots,
one per execution layer:

| Slot | Status | Wraps |
|---|---|---|
| `tools` | shipped | `tool.on_invoke` (Protocol in `tools/tool_middleware.py`) |
| `agents` | shipped | Per-agent block in run loop, re-fires per handoff/swarm transition (Protocol in `run/agent_middleware.py`). Non-streaming only |
| `llms` | shipped | `LLM.acomplete()` inside `call_llm` (Protocol in `llms/llm_middleware.py`). Non-streaming only — `call_llm_streamed` warning-skips |

**Plumbing-only contract.** Logging/metrics/tracing/retries/arg
injection/caching only. Verdicts (PII, jailbreak, content filtering,
schema validation, rate limiting, approval gates) belong in guardrails
or typed surfaces (`requires_approval`, `tool.rate_limit`,
`tool.schema_enforcement`). Enforced by `middleware-vs-guardrails` rule.

## Guardrails

Agent-level guardrails validate input before / output after execution.

- `@agent_input_guardrail` — receives `AgentInputGuardrailData`,
  returns `AgentGuardrailFunctionOutput`. Default `run_in_parallel=True`;
  blocking mode saves tokens if tripwire triggers.
- `@agent_output_guardrail` — receives `AgentOutputGuardrailData`,
  returns `AgentGuardrailFunctionOutput`. Supports `remediation`
  (re-prompt agent on failure instead of crashing).

### Action vocabulary

Every verdict maps onto a framework-owned `GuardrailAction`
(`PASS`/`RAISE`/`TRANSFORM`) via `resolved_action()`, shared with tool and flow
guardrails. An output verdict may carry `transformed_output` (+ `changed_spans`):
the runner substitutes it for the agent output instead of halting — opt-in,
bounded, text-only. Built-ins live in `troopai.adk.guardrails`. See
`docs/guardrails/`.

### Severity (`AgentGuardrailSeverity`)

- `INFO` — DEBUG log, no action
- `WARNING` — WARNING log, does NOT halt
- `ERROR` — halts (same as `tripwire_triggered=True`)

When `severity` is set, it overrides `tripwire_triggered`. When
`None`, `tripwire_triggered` is authoritative.

### Timeout

`timeout` (seconds), `AgentTimeoutPolicy.FAIL` (default, trips wire)
or `PASS` (continues silently). Optional `on_timeout` async callback
for metrics.

### Remediation (output only)

When `remediation` is set and the guardrail trips, the runner injects
the message as feedback and re-runs. After `max_retries` exhausted →
`AgentOutputGuardrailTripwireTriggered`.

### Config Guardrails

Global guardrails via `RunConfig`. Run before agent-level guardrails.

### Attributes

| Attribute | Input | Output | Description |
|---|:-:|:-:|---|
| `guardrail_function` | ✓ | ✓ | The validation function |
| `name` | ✓ | ✓ | Optional (defaults to function name) |
| `run_in_parallel` | ✓ | — | Run alongside agent (default True) |
| `timeout` / `timeout_policy` / `on_timeout` | ✓ | ✓ | Optional |
| `remediation` / `max_retries` | — | ✓ | Self-correction feedback |

Results: `result.guardrail_results.input` (list of
`AgentInputGuardrailResult`), `.output` (list of
`AgentOutputGuardrailResult`).

## Handoffs

`handoffs=[…]` (LLM-orchestrated list) or `handoffs=HandoffRoute(...)`
(deterministic code-orchestrated). Both strategies, `Handoff`
wrapper, `HandoffConfig.budget`/`collapse` token control, and
`HandoffRoute` builder live in `handoffs/CLAUDE.md`.

## Agent as Tool (`as_tool()`)

Wrap an agent as `FunctionTool` for LLM-orchestrated delegation.
Parent LLM calls naturally; sub-agent runs; result returns. Control
stays with parent.

| Parameter | Description |
|---|---|
| `tool_name` / `tool_description` | Override defaults |
| `input_schema` | Custom Pydantic input (default `AgentToolInput`) |
| `input_builder` | Transform parsed input before agent |
| `extractor` | Post-process RunResult before returning |
| `on_stream` | Receive sub-agent streaming events real-time |
| `max_turns` | Sub-agent loop limit (default 10) |
| `timeout` | Seconds via `asyncio.wait_for` |
| `budget` | `LLMUsageLimits` for sub-agent |
| `max_result_tokens` | Truncate result before parent sees it |
| `run_config` | Inherits parent's if `None` |

**Governance:** timeout via `wait_for` (parent gets error string, no
raise); budget merged into sub-agent's `RunConfig.usage_limits`
(explicit overrides inherited); result truncation via
`FunctionTool.max_result_tokens` before insertion into parent history.

**Introspection:** `agent.get_delegate_tools()` (FunctionTools wrapping
delegates only); `agent.get_agent_graph()` (recursive topology dict
with delegates + handoff targets, handles cycles).

**Context isolation:** sub-agent intermediate steps NEVER enter
parent context. Parent sees only 2 messages: tool call + final
result. Matches LangChain Deep Agents.

**Nested HITL:** sub-agent deferrals propagate transparently via
`AgentToolDeferral` (internal exception); sub-agent `RunState` stored
in `DeferredToolCall.metadata`.

**Other:** no history sharing; parent's `TContext` flows via
`ctx.context`; tool identity has `delegate=True` (accessed via
`get_delegate_agent()` method).

See `docs/agents/`, `examples/agent_patterns/`.

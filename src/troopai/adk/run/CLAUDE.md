# Run Module

Core execution engine.

## Files

| File | Purpose |
|---|---|
| `runner.py` | `Runner` class — direct `run()` / `arun()` entry points |
| `profile.py` | Immutable `RunnerProfile` fluent API and target runner handles |
| `config.py` | `RunConfig` (limits, history processors, max_total_turns, salvage hook) |
| `context.py` | `RunContext` with usage tracking |
| `stream.py` | `StreamEvent` types, HITL streaming |
| `state.py` | `RunState` for HITL resumption |
| `hooks.py` | `RunHooks` lifecycle callbacks |
| `loop.py` | Agent loop: LLM → tools → handoffs |
| `llm_calls.py` | LLM communication (streaming + non-streaming) |
| `tools_executor.py` | Tool dispatch, retry budget, guardrails |
| `handoffs_executor.py` | Handoff resolution |
| `guardrails_executor.py` | Input/output guardrail orchestration |
| `resumption.py` | HITL state resumption |
| `demo.py` | `run_demo_loop` interactive REPL |

## Runner

Orchestrates execution. Delegates LLM communication to the `LLM` ABC
(default: `LiteLLM`). Runner: tool enablement, handoff conversion,
message construction, agent loop. LLM: parameter mapping, structured
output, response parsing.

Internal methods: `_call_llm()`, `_call_llm_streamed()`,
`_resolve_llm()`, `_resolve_model_name()`, `_build_tools()`,
`_resolve_output_schema()`, `_resolve_llm_config()`.

## RunnerProfile (Fluent)

`Runner.configure()` creates an immutable `RunnerProfile`. Chain defaults such
as `model()`, `context()`, `limits()`, `context_management()`,
`history_processors()`, `max_total_turns()`, `verbose()`,
`fail_on_tool_error()`, and `with_config()`, then bind a target:
`agent(agent)`, `swarm(swarm)`, `graph(graph)`, `task(task)`,
`pipeline(pipeline)`, `task_group(group)`, or `flow(flow)`.

Target runners add target-specific options such as `AgentRunner.max_turns()`,
`SwarmRunner.checkpointer()`, and `GraphRunner.resume_from()`. Terminal
methods are `run()` (sync) and `arun()` (async), with `stream=True` where the
target supports streamed execution.

## RunConfig

| Field | Purpose |
|---|---|
| `model` | Default override |
| `verbose` | Debug output |
| `fail_on_tool_error` | Raise on tool errors |
| `usage_limits` | Token limits (per response) |
| `history_processors` | Pre-LLM Layer 3 RunItem transforms (sync, no agent/context) |
| `call_model_input_filter` | Pre-LLM Layer 1 rewrite hook (sync/async, has agent + context) |
| `max_total_turns` | Cross-agent swarm safety |
| `context_management` | Compaction + editing |
| `compaction_llm` | Explicit `LLM` for compaction; falls back to agent's `LLM` |
| `on_max_turns` | Salvage handler when per-agent `max_turns` exhausted |
| `tracing_enabled` | Framework span emission (default `False`) |
| `tracing_metadata` | Per-run tags surfaced on root `AgentSpanData.metadata` |
| `metrics_enabled` | Emit OTel metric instruments (independent of span export, default `False`) |
| `tenant_budget` | Per-tenant dollar budget (per-run + per-period); default None |
| `cost_ledger` | Cross-run cost store for per-period budgets; default None |
| `ledger_fail_open` | On a ledger outage, fail-closed (treat period as spent, apply `kill_on_exceed`) vs. proceed as if zero spent; default `False` (closed) |
| `router` | Optional LLM router; tries candidates in order, escalating on failure; default None |
| `tenant_tool_allowlist` | Per-tenant allowed tool names; off when None |
| `tenant_allowlist_default_deny` | Deny tenants absent from the allowlist |
| `tenant_allowlist_soft_deny` | Return a denial message instead of raising |
| `audit_sink` | Append-only tool-call audit sink; off when None |
| `audit_strict` | Re-raise on audit-sink failure (default best-effort) |

## RunResult

| Field | Description |
|---|---|
| `final_output` | Final response (str or structured) |
| `user_prompt` | Original prompt |
| `new_items` | Layer 3 RunItems |
| `context` | RunContext with usage stats |
| `last_agent` | Final agent (after handoffs) |

Helpers: `last_response_id` (provider chaining), `release_agents(*,
release_new_items=True)` (drop refs so caches don't pin agent graph),
`to_input_list()`.

A `TRANSFORM`-mode output guardrail substitutes `final_output` and rewrites the
trailing message via `apply_output_transform` instead of halting (opt-in,
text-only). `guardrail_audit` records every guardrail action across all levels
as hashes, never raw payloads. See `docs/guardrails/`.

## Stream Events

| Type | Name | Description |
|---|---|---|
| `raw_response_event` | — | Raw LLM token |
| `run_item_stream_event` | `MESSAGE_OUTPUT_CREATED` | Message generated |
| `run_item_stream_event` | `TOOL_CALLED` | Tool invoked |
| `run_item_stream_event` | `TOOL_OUTPUT` | Tool result |
| `run_item_stream_event` | `HANDOFF_OCCURRED` | Agent switch |
| `agent_updated_stream_event` | — | New agent active |

## Streaming Cancel

`RunResultStreaming.cancel(mode=...)`:

- `"immediate"` — drains pending events, cancels producer task
  synchronously, enqueues sentinel. Blocked `stream_events()`
  consumer wakes on next receive (no polling). In-flight tools
  finish; between-tool check in `execute_tool_calls_streamed`
  prevents further launches.
- `"after_turn"` — cooperative: flag flipped, current LLM response
  + tool batch finish, loop breaks at top of next turn.

See `docs/run/streaming_cancel.md`.

## System Prompt Resolution

`Runner._resolve_system_prompt(agent, ctx_wrapper)`:

1. If callable → call with `DynamicSystemPromptData(context=ctx_wrapper, agent=agent)`.
2. If result is `SystemPrompt` → `.generate()`.
3. Otherwise → `str(...)`.

`_build_initial_messages()` is async to support this.

After handoff: `inject_system_prompt()` replaces source agent's
system prompt with target's. Applied at all 4 handoff sites.

## Tool Retry Budget

`tool_failure_counts: dict[str, int]`:

- `_build_tools()`: filters out tools whose count exceeds
  `max_retries` (saves tokens).
- `execute_tool_calls()`: pre-checks budget; increments on failure.
- `max_retries` is per-tool; `max_turns` is per-run. A broken tool
  with `max_retries=2` stops wasting turns after 3 failures.

## Reset Tool Choice

`LLMConfig.reset_tool_choice` (default `True`) prevents infinite loops
with `"required"` + `"run_llm_again"`. After tools execute, agent
loop overrides `tool_choice` to `"auto"` for the next call via
`tool_choice_override` — Agent is never mutated.

`loop.py` tracks the override; `llm_calls.py` accepts it on
`call_llm()`/`call_llm_streamed()`; `resumption.py` computes it after
HITL approval/rejection.

## HITL Resumption

Both `run()`/`arun()` accept `RunState`, optional `stream=True`.

**Persistence:** `RunState.to_json()` is `json.dumps(to_dict())`;
`from_json()` is `from_dict(json.loads(...))`. `from_dict` reads
every field with a tolerant `dict.get(key, default)`, so an older
persisted payload (missing later-added keys, or carrying keys this
build no longer recognises) loads to safe defaults — no version
key, no mismatch raise.

**Structured approvals:** `state.approve(call, approver_id=..., reason=...)`
and `state.reject(call, message=..., approver_id=..., reason=...)`.
`message` shown to LLM (retry guidance); `reason` is internal audit only.

See `docs/run/runstate_serialization.md`.

## Usage Limits

`LLMUsageLimits` checked after each response. Fields: `request_limit`,
`total_tokens_limit`. Raises `UsageLimitExceeded` when exceeded.

## History Processors vs `call_model_input_filter`

History processors transform Layer 3 RunItems (no agent/context, sync
only). For Layer 1 (message-level) rewrites with agent + context
access, use `call_model_input_filter` (sync or async, runs every turn
after history processors and context management).

## Swarm Safety

`max_total_turns` = cross-agent cumulative limit. Distinct from
per-agent `max_turns` (resets on handoff).

## Swarm Execution

`Runner.arun_swarm(swarm, input, *, context, run_config, hooks)` /
`arun_swarm_streamed(...)` / `Runner.configure().swarm(swarm)`. Driver lives in
`run/swarm_loop.py` and reuses `run_agent_loop` verbatim. Seams:

- `run/next_step.py` — `NextStepSwarmYield` variant
- `run/loop.py` — match arm surfaces yield to driver
- `run/turn_resolution.py` — detects `swarm_done` / `transfer_to_<n>` calls
- `run/stream.py` — `SwarmEvent` variants on `StreamEvent`

See `swarms/CLAUDE.md`, `docs/swarms/swarms.md`.

## Hooks

- **`RunHooks`** — run-level via `Runner.arun(hooks=...)`. Full
  lifecycle: agent/llm/tool start/end, handoff, guardrail/skill/session.
- **`AgentHooks`** — per-agent via `Agent(hooks=...)`. Fires after
  matching `RunHooks` call, scoped to one agent. Useful in swarms.
  `on_handoff` fires on **incoming** agent with `source=from_agent`.
  Omits guardrail/skill/session (run-level concerns).

Both can run simultaneously. See `docs/hooks/agent_hooks.md`.

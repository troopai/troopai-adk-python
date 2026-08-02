# Workflows Module

Bridge layer that makes any TroopAI `Agent`, `Swarm`, `Graph`, or `Flow` run
durably inside a **Temporal** or **Restate** execution engine.  Intercepts
LLM calls and tool calls at their boundaries and re-routes them through the
engine's journaling / activity primitive, making the run crash-recoverable
and replay-safe.

## Files

### Core (no external deps)

| File | Purpose |
|---|---|
| `engine.py` | `DurableEngine` Protocol, `ModelActivityConfig`, `ToolActivityConfig` |
| `__init__.py` | Re-exports from `engine.py` |

### `temporal/` — Temporal.io backend

| File | Purpose |
|---|---|
| `llm.py` | `TemporalLLM` — shim that routes `acomplete` through `execute_activity` |
| `streaming.py` | `TemporalStreamingLLM` — extends `TemporalLLM` with `acomplete_streamed` |
| `tools.py` | `activity_tool()` factory; `TemporalToolWrapper` per-tool config registry |
| `workflow.py` | `TroopAIWorkflow` base class with HITL signals/queries/updates; `HumanReply`, `ToolApprovalDecision` |
| `plugin.py` | `TroopAITemporalPlugin` — bundles sandbox + data-converter kwargs for the worker |
| `activity.py` | `invoke_model_activity` Temporal activity; `ModelActivityInput`; shared model registry |
| `mcp.py` | `TemporalMCPToolSet` — routes MCP tool calls through named activities |
| `tracing.py` | `should_emit_span`, `deterministic_timestamp`, `deterministic_uuid` (replay-safe) |
| `determinism.py` | `build_sandbox_restrictions`, `DEFAULT_PASSTHROUGH_MODULES` |
| `serialization.py` | `build_troopai_data_converter` — custom Temporal `DataConverter` |
| `routing.py` | `TenantTaskQueueRouter` + `MappingTaskQueueRouter` + `start_tenant_workflow` — per-tenant task-queue dispatch |

### `restate/` — Restate backend

| File | Purpose |
|---|---|
| `llm.py` | `RestateLLM` — shim that routes `acomplete` through `ctx.run()` |
| `tools.py` | `restate_tool()` — wraps a callable to journal calls via `ctx.run()` |
| `service.py` | `TroopAIRestateService` mixin + `RestateHumanReply` (HITL over promises) |
| `activity.py` | `invoke_model_handler` — standalone Restate handler for durable LLM calls |

## Key Architectural Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **`TemporalLLM` shim, not per-primitive wrappers** | A single shim at the `LLM.acomplete` boundary covers all primitives (Agent, Swarm, Graph, Flow) without modifying their runners. Each wrapper checks `workflow.in_workflow()` lazily, so the same object works in durable and non-durable contexts. |
| 2 | **Configurable tool wrapping — opt-in + auto** | `TemporalToolWrapper` + `activity_tool()` let callers choose which tools become activities (`False` = keep in-workflow, `ToolActivityConfig` = custom policy, absent = default policy). No tools are wrapped unless the developer explicitly opts in. |
| 3 | **`TroopAIWorkflow` base with HITL signals/queries/updates** | Maps Temporal's signal → `send_human_reply` (enqueue), query → `get_state` (snapshot), update → `approve_tool_call` (decision). `InterruptException` raised by graph HITL nodes maps to `wait_for_condition` + `consume_replies` in the subclass `run()`. |
| 4 | **Shared `DurableEngine` Protocol** | Both backends satisfy the same `wrap_llm` / `wrap_tool` / `in_durable_context` Protocol in `engine.py`. Switching backends is a one-line change on the caller side. |
| 5 | **Restate adapter mirrors Temporal API surface** | `RestateLLM` / `restate_tool` mirror `TemporalLLM` / `activity_tool` signatures so the developer experience is consistent across backends. HITL uses Restate durable promises in place of Temporal signals. |
| 6 | **Checkpointers optional under Temporal** | Temporal's event history IS the durable state for agent-as-workflow runs. `GraphCheckpointer` is still composable (e.g. for mid-superstep snapshots), but there is no automatic coupling. |
| 7 | **No `TemporalRunner` — `Runner.arun()` called inside `@workflow.run`** | Avoids a parallel runner hierarchy. Concrete subclasses of `TroopAIWorkflow` import and call `Runner` directly; the framework adds no hidden entry point. Matches the "Agent = config, Runner = execution" invariant. |

## Upstream References

| Upstream | What we consulted |
|---|---|
| [Temporal Python SDK](https://docs.temporal.io/develop/python) | `execute_activity`, retry policies, sandbox restrictions, data converters, signals/queries/updates |
| [Restate Python SDK](https://docs.restate.dev/develop/python) | `ctx.run` journaling, durable promises, service/handler patterns |
| [LangGraph](https://langchain-ai.github.io/langgraph/) | Interrupt/resume contract, checkpoint-based HITL |
| [Google ADK](https://google.github.io/adk-docs/) | Activity boundary placement relative to LLM calls |

See `docs/workflows/workflows.md` for usage. See `examples/workflows/` for
runnable code.

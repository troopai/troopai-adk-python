# Permissions Module

Fine-grained tool execution control for agents.

> **Current state**: this module currently re-exports the reserved permission
> types from `types/permissions/` for preview only. The **runner uses a simpler
> bool callback** (`RunConfig.can_use_tool`). The rich types (`PermissionMode`,
> `PermissionResult*`, `ToolPermissionContext`, `PermissionUpdate`) are **not
> yet integrated** — see the Planned section below.

## Current Runner Integration (What Actually Runs)

The runner's permission check is a single synchronous (or awaitable-returning)
bool callback invoked in `run/tools_executor.py` for every tool call:

```
RunConfig.can_use_tool:
    Optional[Callable[[Agent, str, ToolContext], Union[bool, Awaitable[bool]]]]
```

### Flow

```
call site in run/tools_executor.py
  - config.can_use_tool(agent, tool_name, tool_context)
  - returns bool (optionally via await)
        |
        v
  False  ->  msgs.tool_permission_denied(tool_name) replaces tool output
  True   ->  proceeds to input guardrails -> HITL check -> execute tool
```

- **`Agent.can_use_tool` / `PermissionMode`**: not consulted. Only
  `RunConfig.can_use_tool` is read.
- **Rich results**: denial is binary. There is no `PermissionResultDeny(interrupt=True)`
  hard-stop; denial is always "soft" (the LLM sees the denial message and may
  retry with different arguments).
- **Input mutation**: not supported. `PermissionResultAllow.updated_input` has
  no call site.
- **Dynamic updates**: `PermissionUpdate` is not applied anywhere.
- **Resumption**: on deferred-tool resumption, the same bool callback is
  re-evaluated via the same path.

## Resolution Order (Current)

**`can_use_tool`**: `RunConfig.can_use_tool` > `None`. `Agent.can_use_tool` is
not currently read by the runner.

## Interaction with Other Systems

### With `tool.enabled`
- `enabled` filters tools BEFORE the LLM sees them (tool schema filtering).
- `can_use_tool` runs DURING execution (after the LLM decides to call).
- Orthogonal: `enabled=False` → never offered to LLM; `can_use_tool` can still
  deny even when `enabled=True`.

### With `requires_approval` (HITL)
- The bool callback runs before the HITL check.
- `PermissionMode="acceptEdits"` / `"bypassPermissions"` are **not** implemented.
  HITL is gated solely by the tool's `requires_approval` field.

## Planned: Rich Permission System (Not Yet Implemented)

The reserved types in `src/troopai/adk/types/permissions/` model Anthropic's
Claude Code permission system. Landing this work will:

1. Introduce `PermissionMode` coarse-grained control
   (`"default"` / `"acceptEdits"` / `"plan"` / `"bypassPermissions"`).
2. Replace (or complement) the bool callback with
   `async (tool_name, tool_arguments, ToolPermissionContext) -> PermissionResult`.
3. Distinguish soft deny (synthetic tool result → LLM adapts) from hard deny
   (`PermissionDeniedError` → run terminates).
4. Support `PermissionResultAllow.updated_input` for sanitization / redirection.
5. Apply `PermissionUpdate` rules dynamically.
6. Re-run rich permission checks on HITL resumption.

This work is tracked as a separate plan (structured tool-approval /
`RunState` serialization). Until it ships, application code should use the
simple bool callback and **must not** construct `PermissionResult*` values
expecting the runner to honor them.

## Types Reference

See `src/troopai/adk/types/permissions/CLAUDE.md` for the reserved type catalog
and intended semantics.

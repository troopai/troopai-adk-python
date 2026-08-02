# Permission System Types

> **Status: Reserved**. The rich types in this module are defined for a future
> permission callback system and are **not yet wired into the runner**.
>
> The current runner uses a simpler synchronous bool callback:
> `RunConfig.can_use_tool: Callable[[Agent, str, ToolContext], bool | Awaitable[bool]]`
> (see `src/troopai/adk/run/config.py`).
>
> **Do not use these types in application code.** Use `RunConfig.can_use_tool`
> instead. They are kept as a scaffolding target for the structured-approval
> follow-up (see Phase F4 in the type-cleanup plan).

## Key Files

- `permission_types.py` — Reserved type definitions (not yet consumed by runtime code)

## Reserved Types

| Type | Intended purpose |
|------|------------------|
| `PermissionMode` | Coarse-grained execution mode (`"default"`, `"acceptEdits"`, `"plan"`, `"bypassPermissions"`) |
| `CanUseTool` | Async callback producing a structured `PermissionResult` |
| `ToolPermissionContext` | Rich decision context (run context, agent, suggestions) |
| `PermissionResultAllow` | Allow decision with optional input mutation |
| `PermissionResultDeny` | Deny decision with soft/hard interrupt semantics |
| `PermissionResult` | Discriminated union of the two results |
| `PermissionUpdate` | Dynamic permission rule adjustments |

## Why They Exist

These types model Anthropic's Claude Code permission system (mode + structured
result + rule updates) and are the target API for the structured tool-approval
feature. Landing that feature is tracked as a separate plan — until then, the
types are intentionally inert.

## When These Become Live

The structured-approval feature will:

1. Replace `RunConfig.can_use_tool: Callable[..., bool]` with the richer
   `CanUseTool` signature (or add it alongside as a second, opt-in hook).
2. Wire `PermissionMode` into `RunConfig` / `Agent` and apply it inside
   `run/tools_executor.py`.
3. Expose `PermissionUpdate` through session / run state.

Until that ships, `src/troopai/adk/permissions/__init__.py` only re-exports these
types for downstream preview — no runtime code path constructs or consumes them.

## Types Reference

See `permission_types.py` for the full definitions. See the "Permissions Module"
CLAUDE.md under `src/troopai/adk/permissions/` for runner integration status.

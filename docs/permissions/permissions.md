# Permissions Usage

## Per-Agent Permission Mode

```python
from troopai.adk import Agent, PermissionMode

# Plan mode: block all tools, LLM produces text only
agent = Agent(
    name="Planner",
    system_prompt="Plan the task",
    permission_mode="plan",
)

# Auto-approve mode: skip HITL checks
agent = Agent(
    name="AutoExecutor",
    system_prompt="Execute tasks",
    permission_mode="acceptEdits",
)
```

## Per-Agent Permission Callback

```python
from troopai.adk import Agent, PermissionResultAllow, PermissionResultDeny

async def require_admin(tool_name, args, ctx):
    """Only allow admin to call delete tools."""
    if tool_name.startswith("delete_"):
        is_admin = ctx.run_context.context.get("is_admin", False)
        if not is_admin:
            return PermissionResultDeny(
                message="Admin required for delete operations",
                interrupt=False,  # Soft deny: inform LLM, allow retry
            )
    return PermissionResultAllow()

agent = Agent(
    name="SecureAgent",
    system_prompt="...",
    can_use_tool=require_admin,
)
```

## Input Mutation

```python
async def sanitize_paths(tool_name, args, ctx):
    """Redirect all writes to sandbox directory."""
    if tool_name in ("write", "edit"):
        file_path = args.get("file_path", "")
        if file_path.startswith("/etc/"):
            return PermissionResultDeny(
                message="System paths not allowed",
                interrupt=False,
            )
        # Optionally redirect to sandbox
        if not file_path.startswith("/sandbox/"):
            sanitized = f"/sandbox/{file_path}"
            return PermissionResultAllow(
                updated_input={**args, "file_path": sanitized}
            )
    return PermissionResultAllow()
```

## Global Permission Mode

```python
from troopai.adk import RunConfig

config = RunConfig(
    permission_mode="bypassPermissions",  # Skip all checks (CI/automated)
)
```

## Deferred Tool Permission Tracking

```python
# Permission callback that escalates to hard deny if too many attempts
attempt_count = 0

async def track_attempts(tool_name, args, ctx):
    global attempt_count
    if tool_name == "delete_user":
        attempt_count += 1
        if attempt_count > 3:
            return PermissionResultDeny(
                message="Too many delete attempts.",
                interrupt=True  # Hard deny even if human keeps approving
            )
    return PermissionResultAllow()
```

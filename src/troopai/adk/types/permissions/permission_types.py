"""Permission system types for fine-grained tool execution control.

Adapted from Anthropic's Claude Agent SDK permission model, this module provides:
- PermissionMode: Coarse-grained execution mode ("default", "acceptEdits", "plan", "bypassPermissions")
- CanUseTool: Async callback for per-tool permission decisions
- ToolPermissionContext: Rich context passed to permission callbacks
- PermissionResult types: Allow/Deny decisions with optional input mutation
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

TContext = TypeVar("TContext")

PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]
"""Coarse-grained permission mode for tool execution.

Modes:
- "default": Standard behavior - uses requires_approval and guardrails as configured.
- "acceptEdits": Auto-approve all requires_approval requests (skip HITL for all tools).
- "plan": Block ALL tool execution; LLM can only produce text responses.
- "bypassPermissions": Skip requires_approval AND can_use_tool checks (for CI/automated environments).
"""


@dataclass
class PermissionUpdate:
    """A dynamic permission rule change to apply to the system.

    Allows the can_use_tool callback to signal that permissions should be
    updated for future tool calls in the same session (e.g., caching a rule).
    This enables reactive permission adjustments based on runtime context
    without restarting the agent.

    Attributes:
        type: Type of update operation. Determines which other fields are used:

            - ``"addRules"``: Add new permission rules to the current rules list.
            - ``"replaceRules"``: Replace existing rules entirely.
            - ``"removeRules"``: Remove specific rules from the list.
            - ``"setMode"``: Change the global PermissionMode for this and future calls.
            - ``"addDirectories"``: Grant access to additional filesystem directories.
            - ``"removeDirectories"``: Revoke access to filesystem directories.

        rules: Permission rule values to add/replace/remove.
            Used with ``type`` in ``("addRules", "replaceRules", "removeRules")``.

        behavior: Default behavior for new rules.
            Options: ``"allow"``, ``"deny"``, ``"ask"``.
            Used with ``type`` in ``("addRules", "replaceRules")``.

        mode: New ``PermissionMode`` to switch to.
            Used with ``type="setMode"``.

        directories: Filesystem paths for directory operations.
            Used with ``type`` in ``("addDirectories", "removeDirectories")``.

        destination: Where to persist this update.
            Options: ``"session"``, ``"localSettings"``, ``"projectSettings"``,
            ``"userSettings"``. Defaults to ``"session"``.

    Example:
        >>> update = PermissionUpdate(type="addDirectories", directories=["/sandbox/data"], destination="session")
        >>> # Grants temporary access to sandbox directory for this session
    """

    type: Literal[
        "addRules",  # Add new permission rules to the current rules list
        "replaceRules",  # Replace existing rules entirely
        "removeRules",  # Remove specific rules from the list
        "setMode",  # Change the global PermissionMode for this and future calls
        "addDirectories",  # Grant access to additional filesystem directories
        "removeDirectories",  # Revoke access to filesystem directories
    ]
    """str: Type of update operation.

    Options:
        - "addRules": Adds to existing rules without removing them
        - "replaceRules": Replaces all rules with new ones
        - "removeRules": Removes specific rules from the list
        - "setMode": Switches to a new PermissionMode for subsequent calls
        - "addDirectories": Allows access to new filesystem paths
        - "removeDirectories": Revokes access to filesystem paths
    """

    rules: list[Any] | None = None
    """Optional[list]: Permission rule values to add/replace/remove.

    Used for: type in ("addRules", "replaceRules", "removeRules")
    Ignored for: type in ("setMode", "addDirectories", "removeDirectories")

    Type and content depends on the 'type' field.
    Example: ["rule1", "rule2"] for rule types.
    """

    behavior: Literal["allow", "deny", "ask"] | None = None
    """Optional[str]: Default behavior for new rules.

    Options:
        - "allow": New rules automatically approve matching operations
        - "deny": New rules automatically block matching operations
        - "ask": New rules prompt user for decision (interactive)

    Used for: type in ("addRules", "replaceRules")
    Ignored for: other types
    """

    mode: PermissionMode | None = None
    """Optional[PermissionMode]: New PermissionMode to switch to.

    Used for: type="setMode"
    Must be None for: all other types

    When set, overrides the current permission_mode for subsequent calls.
    Example: "plan" to switch to planning mode mid-session.
    """

    directories: list[str] | None = None
    """Optional[list[str]]: Filesystem paths for directory operations.

    Used for: type in ("addDirectories", "removeDirectories")
    Must be None for: all other types

    Can be absolute or relative paths.
    Example: ["/home/user/sandbox", "/tmp/allowed/"]
    """

    destination: Literal["userSettings", "projectSettings", "localSettings", "session"] | None = None
    """Optional[str]: Where to persist this update.

    Options:
        - "session": In-memory only; lost when session ends (default, fastest)
        - "localSettings": Local-machine scope; persisted but not shared with the repository
        - "projectSettings": Project scope; persisted and shared with the repository
        - "userSettings": User scope; persisted globally across projects for the current user

    Default is "session" for temporary runtime changes.
    """


@dataclass
class PermissionResultAllow:
    """Permission decision: allow the tool call to proceed.

    Optionally provides a mutated input to replace the original tool arguments.
    This enables sanitization, normalization, or enrichment of tool inputs
    before execution. All downstream layers (guardrails, on_invoke_tool) will
    see the mutated input transparently.

    When the Runner receives this decision:

    1. If ``updated_input`` is ``None``: uses original tool_arguments.
    2. If ``updated_input`` is set: replaces tool_arguments with updated values.
    3. Tool execution proceeds with (possibly mutated) arguments.
    4. If ``updated_permissions`` is set: applies those permission updates.

    Attributes:
        behavior: Discriminator field. Always ``"allow"`` for this class.
        updated_input: Optional dict replacing original tool_arguments.
            Pass ``None`` to keep original arguments unchanged.
        updated_permissions: Optional list of ``PermissionUpdate`` objects
            applied after this decision. Used to cache, escalate, or
            restrict permissions for future calls in the same session.

    Example:
        >>> async def sanitize_and_escalate(tool_name, args, ctx):
        ...     if tool_name == "write":
        ...         safe_path = f"/sandbox/{args['file_path']}"
        ...         return PermissionResultAllow(
        ...             updated_input={**args, "file_path": safe_path},
        ...             updated_permissions=[PermissionUpdate(type="setMode", mode="plan", destination="session")],
        ...         )
        ...     return PermissionResultAllow()
    """

    behavior: Literal["allow"] = "allow"
    updated_input: dict[str, Any] | None = None
    updated_permissions: list[PermissionUpdate] | None = None


@dataclass
class PermissionResultDeny:
    """Permission decision: deny the tool call.

    Controls whether the denial causes a hard interrupt (raises exception) or
    a soft skip (tool result sent to LLM, execution continues).

    Two denial modes:

    - **Soft Deny** (``interrupt=False``): tool call is skipped; a synthetic
      result is sent to the LLM so it can adapt without halting the run.
    - **Hard Deny** (``interrupt=True``): ``PermissionDeniedError`` is raised
      immediately; the entire run terminates.

    Attributes:
        behavior: Discriminator. Always ``"deny"`` for this class.
        message: Human-readable reason for the denial. Sent to the LLM as a
            synthetic tool result (soft deny) or as the exception message
            (hard deny).
        interrupt: When ``False`` (default), a soft deny is performed and
            the LLM can adapt. When ``True``, a hard deny raises
            ``PermissionDeniedError`` and terminates the run immediately.

    Example (soft deny for validation):
        >>> async def validate_email(tool_name, args, ctx):
        ...     if tool_name == "send_email":
        ...         email = args.get("to", "")
        ...         if not "@" in email:
        ...             return PermissionResultDeny(
        ...                 message="Invalid email address. Please use a valid email.",
        ...                 interrupt=False,  # Let LLM retry
        ...             )
        ...     return PermissionResultAllow()

    Example (hard deny for security):
        >>> async def security_check(tool_name, args, ctx):
        ...     is_authenticated = ctx.run_context.context.get("authenticated", False)
        ...     if not is_authenticated:
        ...         return PermissionResultDeny(
        ...             message="Authentication required. This is a critical security violation.",
        ...             interrupt=True,  # Stop immediately
        ...         )
        ...     return PermissionResultAllow()
    """

    behavior: Literal["deny"] = "deny"
    """str: Always "deny" (discriminator field for Union type)."""

    message: str = ""
    """str: Human-readable reason for the denial.

    Sent to LLM as synthetic tool result (if interrupt=False).
    Sent to exception message (if interrupt=True).

    Guidelines:
        - Be specific: "Admin role required" not just "denied"
        - Be constructive: "Try with admin account" not just "no"
        - Be polite: Users (via LLM) might read this
        - Keep concise: 1-2 sentences

    Examples:
        - "Admin-only operation. Use an admin account to proceed."
        - "Files in /etc/ are protected. Use /sandbox/ instead."
        - "This operation requires 2FA verification."
        - "Daily limit exceeded. Try again tomorrow."
    """

    interrupt: bool = False
    """bool: Whether this is a soft or hard denial.

    Values:
        - False (default): Soft deny
          * Tool call is skipped
          * LLM receives message and can adapt
          * Allows LLM to recover gracefully
          * Use for security policies, rate limits, validation

        - True: Hard deny
          * Raises PermissionDeniedError immediately
          * Entire run terminates with exception
          * No recovery possible within run
          * Use only for critical security violations

    Soft Deny Flow:
        Tool Call → Permission Check → DENIED
        → Synthetic result to LLM
        → LLM continues (might ask user, try different tool, etc.)

    Hard Deny Flow:
        Tool Call → Permission Check → DENIED
        → PermissionDeniedError raised
        → Entire run terminates
    """


# Union discriminated type for the return value of CanUseTool callbacks
PermissionResult = PermissionResultAllow | PermissionResultDeny


@dataclass
class ToolPermissionContext[TContext]:
    """Rich context passed to can_use_tool permission callbacks (runs before guardrails).

    This decision-making context gives permission callbacks access to the
    full ``RunContext`` and the current ``Agent``, unlike ``ToolContext``
    which provides only user-provided context (intentional isolation).

    Generic over ``TContext`` — the type of user-defined context data passed
    to the agent (e.g., a dict with keys like ``"user_id"``, ``"is_admin"``).

    Attributes:
        tool_name: Name of the tool being requested.
            Guaranteed to match one of the agent's registered tools.
        tool_arguments: Parsed arguments dict provided by the LLM for this
            call. Keys are parameter names; values are the arguments.
            If ``PermissionResultAllow(updated_input={...})`` is returned,
            the Runner replaces these with the updated version.
        tool_call_id: LLM-generated unique identifier for this tool call.
        run_context: Full execution context with usage tracking and
            user-provided context accessible via ``run_context.context``.
        agent: The agent that requested tool execution. Provides access to
            ``agent.name``, ``agent.permission_mode``, ``agent.tools``, and
            ``agent.system_prompt`` for agent-specific decisions.
        signal: Reserved for future abort/cancellation signals.
            Currently always ``None``.
        suggestions: Accumulated permission suggestions from a CLI runtime
            context, if present. Typically empty in standard ADK usage.

    Example (comprehensive permission check):
        >>> async def permission_callback(tool_name, args, ctx):
        ...     # Extract user info
        ...     user_id = ctx.run_context.context.get("user_id")
        ...     is_admin = ctx.run_context.context.get("is_admin", False)
        ...
        ...     # Rule 1: Only admins can delete
        ...     if tool_name.startswith("delete_") and not is_admin:
        ...         return PermissionResultDeny(message="Only admins can perform delete operations.", interrupt=False)
        ...
        ...     # Rule 2: Sanitize file paths
        ...     if tool_name == "read_file":
        ...         path = args.get("path", "")
        ...         if path.startswith("/etc/"):
        ...             return PermissionResultDeny(message="Cannot access system files.", interrupt=False)
        ...         # Redirect to sandbox
        ...         return PermissionResultAllow(updated_input={**args, "path": f"/sandbox/{path}"})
        ...
        ...     # Rule 3: Rate limiting (hypothetical)
        ...     if ctx.run_context.usage.total_tokens > 90000:
        ...         return PermissionResultDeny(
        ...             message="Token limit approaching. Blocking tool execution.", interrupt=False
        ...         )
        ...
        ...     # Rule 4: Agent-specific rules
        ...     if ctx.agent.name == "DataProcessor":
        ...         if tool_name not in ["read_data", "analyze_data", "export_data"]:
        ...             return PermissionResultDeny(message=f"DataProcessor cannot use {tool_name}.", interrupt=False)
        ...
        ...     return PermissionResultAllow()
    """

    tool_name: str
    """str: The name of the tool being requested.

    Examples: "read_file", "delete_user", "send_email"

    Use to make tool-specific decisions:
        - "Only allow delete_* for admins"
        - "Block write_* for read-only users"
        - Different rules per tool
    """

    tool_arguments: dict[str, Any]
    """dict: Parsed arguments dict for the tool call.

    Contains exact arguments the LLM provided.
    Keys are parameter names; values are argument values.

    Example:
        For tool.read_file(path: str, mode: str):
        tool_arguments = {"path": "/home/user/file.txt", "mode": "r"}

    Use to:
        - Validate argument values
        - Check for sensitive data
        - Sanitize paths or inputs
        - Make argument-specific decisions

    If you return PermissionResultAllow(updated_input={...}),
    the Runner replaces tool_arguments with your version.
    """

    tool_call_id: str
    """str: Unique identifier for this tool call.

    Generated by LLM and guaranteed unique within session.

    Use for:
        - Logging/auditing which specific call was denied
        - Linking to external systems
        - Deduplication if call is retried

    Example: "call_abc123def456"
    """

    run_context: Any
    """RunContext[TContext]: The full execution context.

    Contains all execution state and user-provided context.

    Key attributes:
        - run_context.context: User-provided context dict (TContext)
          Access: ctx.run_context.context.get("user_id")
          Access: ctx.run_context.context.get("is_admin")
          Access: ctx.run_context.context["tenant"]

        - run_context.usage: LLMUsage object with token counts
          Access: ctx.run_context.usage.prompt_tokens
          Access: ctx.run_context.usage.completion_tokens
          Access: ctx.run_context.usage.total_tokens

    Use to check:
        - User identity/roles from context
        - Token budget remaining
        - Session metadata
        - Tenant/organization info
    """

    agent: Any
    """Agent[TContext]: The agent requesting tool execution.

    Gives access to agent metadata for decision-making.

    Key attributes:
        - agent.name: Agent identifier string
        - agent.permission_mode: Current PermissionMode
        - agent.tools: List of available tools
        - agent.system_prompt: Agent's system prompt

    Use to:
        - Make agent-specific rules ("VizAgent can only call viz_*")
        - Check what tools are available in context
        - Understand the agent's role/purpose
    """

    signal: Any | None = None
    """Optional[Any]: Future abort/cancellation signal (reserved).

    Currently always None. Intended for graceful shutdown mechanisms
    in future versions. Do not use in current implementation.
    """

    suggestions: list[PermissionUpdate] = field(default_factory=list)
    """list[PermissionUpdate]: Accumulated permission suggestions from CLI.

    Contains suggestions from Claude Code CLI runtime context.

    Typically empty in standard ADK usage. Only populated when running
    within Claude Code with MCP server context that provides suggestions.
    """


CanUseTool = Callable[[str, dict[str, Any], ToolPermissionContext[Any]], Awaitable[PermissionResult]]
"""Async callback for fine-grained per-tool permission decisions.

The callback receives:
- tool_name (str): The name of the tool being requested.
- tool_arguments (dict): The parsed arguments dict for the tool call.
- context (ToolPermissionContext): Rich context about the invocation.

Returns a PermissionResult (either PermissionResultAllow or PermissionResultDeny).

The callback runs in Layer 0 (before all guardrails) for each tool call during
agent execution. It has access to the full RunContext and Agent, allowing it to
make permission decisions based on user identity, roles, token usage, etc.

If the callback raises an exception, the system fails open (allows the tool call)
and logs the error if verbose mode is enabled.

Example:
    async def require_admin_for_delete(
        tool_name: str,
        tool_arguments: dict[str, Any],
        context: ToolPermissionContext,
    ) -> PermissionResult:
        if tool_name.startswith("delete_"):
            is_admin = context.run_context.context.get("is_admin", False)
            if not is_admin:
                return PermissionResultDeny(
                    message="Only admins can delete resources.",
                    interrupt=False,  # Soft deny - inform LLM
                )
        return PermissionResultAllow()

    agent = Agent(
        name="SecureAgent",
        system_prompt="...",
        can_use_tool=require_admin_for_delete,
    )
"""

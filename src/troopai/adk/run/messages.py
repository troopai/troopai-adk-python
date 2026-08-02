"""Configurable messages used by the Runner and execution engine.

All user-facing and LLM-facing strings are centralized here so they
can be customized per-run via ``RunConfig.messages``.  Each field is
a callable that formats the message with context-specific arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


def _tool_not_found(name: str) -> str:
    return f"Error: Tool '{name}' not found"


def _tool_not_found_on_resumption(name: str) -> str:
    return f"Error: Tool '{name}' not found or has no implementation"


def _tool_permission_denied(name: str) -> str:
    return f"Error: Permission denied for tool '{name}'"


def _tool_permission_check_error(name: str) -> str:
    return f"Error: Permission check for tool '{name}' failed with an error; tool call suppressed"


def _tool_disabled(name: str) -> str:
    return f"Error: Tool '{name}' is currently disabled"


def _tool_retry_exhausted(name: str, n: int) -> str:
    return f"Tool '{name}' is disabled after {n} failed attempt(s)."


def _tool_timeout(name: str, t: float) -> str:
    return f"Tool '{name}' timed out after {t}s. Please try a simpler request."


def _tool_execution_error(name: str, err: str) -> str:
    return f"Error executing tool '{name}': {err}"


def _nested_agent_rejected(name: str) -> str:
    return f"The {name} agent's operation was denied by the user."


def _handoff_transferred(name: str) -> str:
    return f"Transferred to {name}."


@dataclass
class RunMessages:
    """Configurable messages for runner, tool execution, and handoffs.

    Override individual fields to customize the messages sent to the LLM
    or surfaced to users during agent execution.  Pass an instance via
    ``RunConfig(messages=RunMessages(...))``.

    Attributes:
        tool_not_found: Message when a tool call references a
            non-existent tool. Receives the tool name.
        tool_not_found_on_resumption: Message when an approved tool is
            not found during HITL resumption. Receives the tool name.
        tool_permission_denied: Message when ``can_use_tool`` denies a
            tool invocation. Receives the tool name.
        tool_disabled: Message when a tool is disabled via its
            ``enabled`` callback. Receives the tool name.
        tool_retry_exhausted: Message when a tool exceeds its
            ``max_retries`` budget. Receives the tool name and attempt
            count.
        tool_timeout: Default timeout error message (used when
            ``on_timeout`` is not set). Receives the tool name and
            timeout duration in seconds.
        tool_execution_error: Message when a tool raises an exception
            and ``fail_on_tool_error=False``. Receives the tool name
            and error string.
        tool_rejected: Default rejection message for HITL-denied tool
            calls.
        nested_agent_rejected: Rejection message for nested agent tool
            calls denied via HITL. Receives the agent name.
        handoff_transferred: Synthetic ``tool_result`` content when a
            handoff is executed. Receives the target agent name.
        handoff_skipped: ``tool_result`` content for non-handoff tool
            calls in a batch that also contains a handoff.
        handoff_collapse_prefix: Prefix for collapsed handoff history
            (``strategy='summary'`` + ``collapse=True``).
        memory_injection_header: Header prepended to injected memories
            before the user prompt.
        tool_unavailable_prefix: Prefix prepended to disabled tool
            descriptions when using the ``STABLE`` cache strategy.

    Example::

        messages = RunMessages(
            tool_not_found=lambda name: f"Outil '{name}' introuvable",
            tool_rejected="Appel d'outil refusé.",
        )
        config = RunConfig(messages=messages)
    """

    # --- Tool execution messages ---

    tool_not_found: Callable[[str], str] = field(default=_tool_not_found)
    """Message when a tool call references a non-existent tool."""

    tool_not_found_on_resumption: Callable[[str], str] = field(default=_tool_not_found_on_resumption)
    """Message when an approved tool is not found during HITL resumption."""

    tool_permission_denied: Callable[[str], str] = field(default=_tool_permission_denied)
    """Message when ``can_use_tool`` denies a tool invocation."""

    tool_permission_check_error: Callable[[str], str] = field(default=_tool_permission_check_error)
    """Message when ``can_use_tool`` raises an unexpected exception.

    Distinct from ``tool_permission_denied`` so the LLM (and developers)
    can distinguish a deliberate policy refusal from a transient callback
    error.
    """

    tool_disabled: Callable[[str], str] = field(default=_tool_disabled)
    """Message when a tool is disabled via its ``enabled`` callback."""

    tool_retry_exhausted: Callable[[str, int], str] = field(default=_tool_retry_exhausted)
    """Message when a tool exceeds its ``max_retries`` budget."""

    tool_timeout: Callable[[str, float], str] = field(default=_tool_timeout)
    """Default timeout error message (used when ``on_timeout`` is not set)."""

    tool_execution_error: Callable[[str, str], str] = field(default=_tool_execution_error)
    """Message when a tool raises an exception (and ``fail_on_tool_error=False``)."""

    tool_rejected: str = "Tool call was denied."
    """Default rejection message for HITL-denied tool calls."""

    nested_agent_rejected: Callable[[str], str] = field(default=_nested_agent_rejected)
    """Rejection message for nested agent tool calls denied via HITL."""

    # --- Handoff messages ---

    handoff_transferred: Callable[[str], str] = field(default=_handoff_transferred)
    """Synthetic tool_result content when a handoff is executed."""

    handoff_skipped: str = "Tool call skipped due to agent handoff."
    """Tool_result content for non-handoff tool_calls in a batch with a handoff."""

    handoff_collapse_prefix: str = "Previous conversation:\n"
    """Prefix for collapsed handoff history (strategy='summary' + collapse=True)."""

    # --- Memory messages ---

    memory_injection_header: str = "[Relevant memories from previous interactions]"
    """Header prepended to injected memories before the user prompt."""

    # --- Tool availability ---

    tool_unavailable_prefix: str = "[UNAVAILABLE] "
    """Prefix prepended to disabled tool descriptions (STABLE cache strategy)."""

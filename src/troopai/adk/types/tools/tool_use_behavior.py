"""Types for controlling what happens after tool execution.

Defines the ``ToolUseBehavior`` union type used by ``Agent.tool_use_behavior``
to determine whether tool results go back to the LLM, become final output
directly, or are processed by a custom function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from troopai.adk.utils import MaybeAwaitable

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext


@dataclass
class FunctionToolResult:
    """Result from a single FunctionTool execution, passed to custom behavior functions.

    Attributes:
        name: Name of the tool that was called.
        call_id: Unique ID from the LLM for this tool call.
        output: The raw output from the tool execution.
    """

    name: str
    """Name of the tool that was called."""

    call_id: str
    """Unique ID from the LLM for this tool call."""

    output: Any
    """The raw output from the tool execution."""


@dataclass
class ToolsToFinalOutputResult:
    """Return type for custom tool_use_behavior functions.

    Attributes:
        is_final_output: Whether the tool results should be treated as final output.
        final_output: The final output value. Required when ``is_final_output`` is ``True``.
    """

    is_final_output: bool
    """Whether the tool results should be treated as final output."""

    final_output: Any | None = None
    """The final output value. Required when is_final_output is True."""


@dataclass
class StopAtTools:
    """Stop execution when any of the named tools are called.

    When the LLM calls one of the named tools, its output becomes
    the final result and no further LLM call is made.

    Attributes:
        stop_at_tool_names: List of tool names that trigger early stopping.
    """

    stop_at_tool_names: list[str]
    """List of tool names that trigger early stopping."""


ToolsToFinalOutputFunction = Callable[
    ["RunContext[Any]", list[FunctionToolResult]],
    MaybeAwaitable[ToolsToFinalOutputResult],
]
"""Custom function type for tool_use_behavior.

Receives the RunContext and a list of FunctionToolResult, returns a
ToolsToFinalOutputResult indicating whether to treat tool output as
final or continue the loop.
"""

ToolUseBehavior = Literal["run_llm_again"] | Literal["stop_on_first_tool"] | StopAtTools | ToolsToFinalOutputFunction
"""Controls what happens after tool execution on an Agent.

- ``"run_llm_again"`` (default): Tool results go back to the LLM.
- ``"stop_on_first_tool"``: First tool's output becomes the final result.
- ``StopAtTools(stop_at_tool_names=[...])``: Stop on specific tools only.
- Custom function: ``(RunContext, list[FunctionToolResult]) -> ToolsToFinalOutputResult``
"""

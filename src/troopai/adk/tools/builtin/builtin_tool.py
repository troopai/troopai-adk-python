from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from troopai.adk.schemas import SchemaEnforcement
from troopai.adk.tools.tool_guardrails import ToolGuardrails
from troopai.adk.types.tools import ApprovalPolicy
from troopai.adk.utils import MaybeAwaitable

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext
    from troopai.adk.tools.function_tool import FunctionTool

BuiltinToolTimeoutBehavior = Literal["error_as_result", "raise_exception"]
BuiltinToolErrorFunction = Callable[[Any, Exception], MaybeAwaitable[str]]


@dataclass(kw_only=True)
class BuiltinTool(ABC):
    """Base class for built-in tools provided by the agent framework.

    Built-in tools are pre-defined tools that come with the framework
    and do framework-level local work (memory, shell, patch editing,
    JIT context management). Provider-native capabilities are deliberately
    NOT wrapped as ``BuiltinTool`` subclasses — pass raw provider JSON
    through ``LLMConfig.extra_body`` / ``extra_args`` instead.

    Concrete subclasses are either:

    - Plain ``BuiltinTool`` — wraps framework behaviour that the loop
      converts into a ``FunctionTool`` via a dedicated builder before
      reaching the LLM layer (``ShellTool``, ``ApplyPatchTool``).
    - :class:`ExecutableBuiltinTool` — has ``description``, ``schema``,
      and ``on_invoke`` so the LLM layer can expose it directly as a
      function tool (``MemoryTool``, ``JITContextAwareTool``).
    """

    name: str
    """The name of the built-in tool."""


@dataclass(kw_only=True)
class ExecutableBuiltinTool(BuiltinTool):
    """A built-in tool with local execution capability.

    Unlike provider-native ``BuiltinTool`` subclasses, executable
    builtins have a ``description``, ``schema``, and ``on_invoke``
    callback.  The LLM layer converts them to function-call format.

    Attributes:
        description: Tool description shown to the LLM.
        schema: Pydantic model or JSON schema dict for input validation.
        schema_enforcement: Provider-agnostic schema normalization policy.
        guardrails: Input and output guardrails for local execution.
        enabled: Static or dynamic availability policy.
        requires_approval: Human-approval policy.
        max_result_tokens: Maximum result size returned to the model.
        max_retries: Per-run failure budget.
        timeout: Maximum execution time in seconds.
        timeout_behavior: Whether timeout becomes a result or an exception.
        timeout_error: Optional timeout-result formatter.
        on_invoke: Callback that executes the tool.
        execution_aware: Whether invocation receives execution state.
        history_aware: Whether invocation receives conversation history.
        response_format: Text-only or content-and-artifact result handling.
        return_direct: Whether the result becomes the final run output.
        prepare: Per-turn dynamic FunctionTool modifier.
        defer_loading: Whether the tool starts hidden from the model.
        metadata: String labels surfaced through observability.
    """

    description: str = ""
    """Tool description shown to the LLM."""

    schema: type[BaseModel] | dict[str, Any] = field(default_factory=dict)
    """Pydantic model or JSON schema dict for input parameters."""

    schema_enforcement: SchemaEnforcement = SchemaEnforcement.NORMALIZED
    """Provider-agnostic schema normalization policy."""

    guardrails: ToolGuardrails | None = None
    """Input and output guardrails applied by the shared executor."""

    enabled: bool | Callable[[RunContext[Any]], Any] | MaybeAwaitable[bool] = True
    """Static or dynamic availability policy."""

    requires_approval: ApprovalPolicy = False
    """Human-approval policy evaluated before invocation."""

    max_result_tokens: int | None = None
    """Maximum result size returned to the model, in tokens."""

    max_retries: int | None = None
    """Per-run failure budget; ``None`` defers to skill governance."""

    timeout: float | None = None
    """Maximum execution time in seconds."""

    timeout_behavior: BuiltinToolTimeoutBehavior = "error_as_result"
    """Whether a timeout becomes a tool result or an exception."""

    timeout_error: BuiltinToolErrorFunction | None = None
    """Optional formatter for timeout results."""

    on_invoke: Callable | None = None
    """Callback that executes the tool.

    Set by subclasses in ``__post_init__`` or passed directly.
    """

    execution_aware: bool = False
    """Whether invocation receives an execution-aware context."""

    history_aware: bool = False
    """Whether invocation receives a history-aware context."""

    response_format: str = "text"
    """How the shared executor interprets the callback result."""

    return_direct: bool = False
    """Whether the tool result becomes the final run output directly."""

    prepare: Callable | None = None
    """Optional per-turn modifier applied to the adapted FunctionTool."""

    defer_loading: bool = False
    """Whether the tool starts hidden until explicitly revealed."""

    metadata: Mapping[str, str] = field(default_factory=dict)
    """String labels surfaced through tracing and telemetry."""

    def to_function_tool(self) -> FunctionTool:
        """Adapt this built-in to the canonical local execution surface.

        The returned wrapper delegates to this instance's callback rather than
        cloning the built-in, so stateful stores and indexes remain attached to
        the original object.

        Returns:
            A transient FunctionTool governed by the shared executor.
        """
        from troopai.adk.tools.function_tool import FunctionTool

        return FunctionTool(
            name=self.name,
            description=self.description,
            schema=self.schema,
            schema_enforcement=self.schema_enforcement,
            guardrails=self.guardrails,
            enabled=self.enabled,
            requires_approval=self.requires_approval,
            max_result_tokens=self.max_result_tokens,
            max_retries=self.max_retries,
            timeout=self.timeout,
            timeout_behavior=self.timeout_behavior,
            timeout_error=self.timeout_error,
            on_invoke=self.on_invoke,
            execution_aware=self.execution_aware,
            history_aware=self.history_aware,
            response_format=self.response_format,
            return_direct=self.return_direct,
            prepare=self.prepare,
            defer_loading=self.defer_loading,
            metadata=self.metadata,
        )

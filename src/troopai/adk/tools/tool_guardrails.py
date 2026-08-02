from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, overload

from typing_extensions import TypedDict, TypeVar

from troopai.adk.exceptions import UserError
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.types.guardrails.action import GuardrailAction
from troopai.adk.utils.typedef import MaybeAwaitable

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent


@dataclass
class ToolInputGuardrailResult:
    """The result of a tool input guardrail run.

    Attributes:
        guardrail: The guardrail that was run.
        output: The output of the guardrail function.
    """

    guardrail: ToolInputGuardrail[Any]
    """The guardrail that was run."""

    output: ToolGuardrailFunctionOutput
    """The output of the guardrail function."""


@dataclass
class ToolOutputGuardrailResult:
    """The result of a tool output guardrail run.

    Attributes:
        guardrail: The guardrail that was run.
        output: The output of the guardrail function.
    """

    guardrail: ToolOutputGuardrail[Any]
    """The guardrail that was run."""

    output: ToolGuardrailFunctionOutput
    """The output of the guardrail function."""


class RejectContentBehavior(TypedDict):
    """Rejects the tool call/output but continues execution with a message to the model."""

    type: Literal["reject_content"]
    message: str


class RaiseExceptionBehavior(TypedDict):
    """Raises an exception to halt execution."""

    type: Literal["raise_exception"]


class AllowBehavior(TypedDict):
    """Allows normal tool execution to continue."""

    type: Literal["allow"]


@dataclass
class ToolGuardrailFunctionOutput:
    """The output of a tool guardrail function.

    Attributes:
        output_info: Optional data about checks performed.
        behavior: Defines how the system responds when this guardrail
            result is processed — ``allow``, ``reject_content``, or
            ``raise_exception``.
    """

    output_info: Any
    """
    Optional data about checks performed. For example, the guardrail could include
    information about the checks it performed and granular results.
    """

    behavior: RejectContentBehavior | RaiseExceptionBehavior | AllowBehavior = field(
        default_factory=lambda: AllowBehavior(type="allow")
    )
    """
    Defines how the system should respond when this guardrail result is processed.
    - allow: Allow normal tool execution to continue without interference (default)
    - reject_content: Reject the tool call/output but continue execution with a message to the model
    - raise_exception: Halt execution by raising a ToolGuardrailTripwireTriggered exception
    """

    @classmethod
    def allow(cls, output_info: Any = None) -> ToolGuardrailFunctionOutput:
        """Create a guardrail output that allows the tool execution to continue normally.

        Args:
            output_info: Optional data about checks performed.

        Returns:
            ToolGuardrailFunctionOutput configured to allow normal execution.
        """
        return cls(output_info=output_info, behavior=AllowBehavior(type="allow"))

    @classmethod
    def reject_content(cls, message: str, output_info: Any = None) -> ToolGuardrailFunctionOutput:
        """Create a guardrail output that rejects the tool call/output but continues execution.

        Args:
            message: Message to send to the model instead of the tool result.
            output_info: Optional data about checks performed.

        Returns:
            ToolGuardrailFunctionOutput configured to reject the content.
        """
        return cls(
            output_info=output_info,
            behavior=RejectContentBehavior(type="reject_content", message=message),
        )

    @classmethod
    def raise_exception(cls, output_info: Any = None) -> ToolGuardrailFunctionOutput:
        """Create a guardrail output that raises an exception to halt execution.

        Args:
            output_info: Optional data about checks performed.

        Returns:
            ToolGuardrailFunctionOutput configured to raise an exception.
        """
        return cls(output_info=output_info, behavior=RaiseExceptionBehavior(type="raise_exception"))

    def resolved_action(self) -> GuardrailAction:
        """Map this verdict onto the shared guardrail action vocabulary.

        ``reject_content`` substitutes the result the model sees, so it resolves
        to ``TRANSFORM``; ``raise_exception`` halts (``RAISE``); ``allow`` passes.
        """
        behavior_type = self.behavior["type"]
        if behavior_type == "reject_content":
            return GuardrailAction.TRANSFORM
        if behavior_type == "raise_exception":
            return GuardrailAction.RAISE
        return GuardrailAction.PASS


@dataclass
class ToolInputGuardrailData:
    """Input data passed to a tool input guardrail function.

    Attributes:
        context: The tool context containing information about the
            current tool execution.
        agent: The agent that is executing the tool.
    """

    context: ToolContext[Any]
    """
    The tool context containing information about the current tool execution.
    """

    agent: Agent[Any]  # Will be typed as Agent[Any] when agent module is implemented
    """
    The agent that is executing the tool.
    """


@dataclass
class ToolOutputGuardrailData(ToolInputGuardrailData):
    """Input data passed to a tool output guardrail function.

    Extends input data with the tool's output.

    Attributes:
        output: The output produced by the tool function.
        context: The tool context containing information about the
            current tool execution (inherited).
        agent: The agent that is executing the tool (inherited).
    """

    output: Any
    """
    The output produced by the tool function.
    """


TContext_co = TypeVar("TContext_co", bound=Any, covariant=True)


@dataclass
class ToolInputGuardrail[TContext_co: Any]:
    """A guardrail that runs before a function tool is invoked.

    Attributes:
        guardrail_function: The callable that implements the guardrail
            logic. Receives a :class:`ToolInputGuardrailData` and
            returns a :class:`ToolGuardrailFunctionOutput` (sync or
            async).
        name: Optional name for the guardrail. When ``None``, the
            function's ``__name__`` is used.
    """

    guardrail_function: Callable[[ToolInputGuardrailData], MaybeAwaitable[ToolGuardrailFunctionOutput]]
    """
    The function that implements the guardrail logic.
    """

    name: str | None = None
    """
    Optional name for the guardrail. If not provided, uses the function name.
    """

    def get_name(self) -> str:
        """Return the guardrail's name, falling back to the function name.

        Returns:
            The explicit ``name`` when set, otherwise
            ``guardrail_function.__name__``.
        """
        return self.name or self.guardrail_function.__name__

    async def run(self, data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        """Execute the guardrail function.

        Args:
            data: Input data carrying the tool context and agent.

        Returns:
            The guardrail function's output describing the verdict.

        Raises:
            UserError: If ``guardrail_function`` is not callable.
        """
        if not callable(self.guardrail_function):
            raise UserError(f"Guardrail function must be callable, got {self.guardrail_function}")

        result = self.guardrail_function(data)
        if inspect.isawaitable(result):
            return await result
        return result


@dataclass
class ToolOutputGuardrail[TContext_co: Any]:
    """A guardrail that runs after a function tool is invoked.

    Attributes:
        guardrail_function: The callable that implements the guardrail
            logic. Receives a :class:`ToolOutputGuardrailData` and
            returns a :class:`ToolGuardrailFunctionOutput` (sync or
            async).
        name: Optional name for the guardrail. When ``None``, the
            function's ``__name__`` is used.
    """

    guardrail_function: Callable[[ToolOutputGuardrailData], MaybeAwaitable[ToolGuardrailFunctionOutput]]
    """
    The function that implements the guardrail logic.
    """

    name: str | None = None
    """
    Optional name for the guardrail. If not provided, uses the function name.
    """

    def get_name(self) -> str:
        """Return the guardrail's name, falling back to the function name.

        Returns:
            The explicit ``name`` when set, otherwise
            ``guardrail_function.__name__``.
        """
        return self.name or self.guardrail_function.__name__

    async def run(self, data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
        """Execute the guardrail function.

        Args:
            data: Input data carrying the tool context, agent, and
                tool output.

        Returns:
            The guardrail function's output describing the verdict.

        Raises:
            UserError: If ``guardrail_function`` is not callable.
        """
        if not callable(self.guardrail_function):
            raise UserError(f"Guardrail function must be callable, got {self.guardrail_function}")

        result = self.guardrail_function(data)
        if inspect.isawaitable(result):
            return await result
        return result


# Decorators
_ToolInputFuncSync = Callable[[ToolInputGuardrailData], ToolGuardrailFunctionOutput]
_ToolInputFuncAsync = Callable[[ToolInputGuardrailData], Awaitable[ToolGuardrailFunctionOutput]]


@overload
def tool_input_guardrail(func: _ToolInputFuncSync): ...


@overload
def tool_input_guardrail(func: _ToolInputFuncAsync): ...


@overload
def tool_input_guardrail(
    *, name: str | None = None
) -> Callable[[_ToolInputFuncSync | _ToolInputFuncAsync], ToolInputGuardrail[Any]]: ...


def tool_input_guardrail(
    func: _ToolInputFuncSync | _ToolInputFuncAsync | None = None, *, name: str | None = None
) -> ToolInputGuardrail[Any] | Callable[[_ToolInputFuncSync | _ToolInputFuncAsync], ToolInputGuardrail[Any]]:
    """Decorator to create a :class:`ToolInputGuardrail` from a function.

    Can be used bare (``@tool_input_guardrail``) or with a name keyword
    (``@tool_input_guardrail(name="my_guardrail")``).

    Args:
        func: The guardrail function when used as a bare decorator.
            ``None`` when called with keyword arguments.
        name: Optional name for the guardrail. Falls back to
            ``func.__name__`` when ``None``.

    Returns:
        A :class:`ToolInputGuardrail` when used as a bare decorator, or
        a decorator callable when called with keyword arguments.
    """

    def decorator(f: _ToolInputFuncSync | _ToolInputFuncAsync) -> ToolInputGuardrail[Any]:
        return ToolInputGuardrail(guardrail_function=f, name=name or f.__name__)

    if func is not None:
        return decorator(func)
    return decorator


_ToolOutputFuncSync = Callable[[ToolOutputGuardrailData], ToolGuardrailFunctionOutput]
_ToolOutputFuncAsync = Callable[[ToolOutputGuardrailData], Awaitable[ToolGuardrailFunctionOutput]]


@overload
def tool_output_guardrail(func: _ToolOutputFuncSync): ...


@overload
def tool_output_guardrail(func: _ToolOutputFuncAsync): ...


@overload
def tool_output_guardrail(
    *, name: str | None = None
) -> Callable[[_ToolOutputFuncSync | _ToolOutputFuncAsync], ToolOutputGuardrail[Any]]: ...


def tool_output_guardrail(
    func: _ToolOutputFuncSync | _ToolOutputFuncAsync | None = None, *, name: str | None = None
) -> ToolOutputGuardrail[Any] | Callable[[_ToolOutputFuncSync | _ToolOutputFuncAsync], ToolOutputGuardrail[Any]]:
    """Decorator to create a :class:`ToolOutputGuardrail` from a function.

    Can be used bare (``@tool_output_guardrail``) or with a name keyword
    (``@tool_output_guardrail(name="my_guardrail")``).

    Args:
        func: The guardrail function when used as a bare decorator.
            ``None`` when called with keyword arguments.
        name: Optional name for the guardrail. Falls back to
            ``func.__name__`` when ``None``.

    Returns:
        A :class:`ToolOutputGuardrail` when used as a bare decorator, or
        a decorator callable when called with keyword arguments.
    """

    def decorator(f: _ToolOutputFuncSync | _ToolOutputFuncAsync) -> ToolOutputGuardrail[Any]:
        return ToolOutputGuardrail(guardrail_function=f, name=name or f.__name__)

    if func is not None:
        return decorator(func)
    return decorator


@dataclass
class ToolGuardrails:
    """Per-phase tool-level guardrail lists registered on a FunctionTool.

    Each slot is typed against its phase-specific Protocol so the type
    checker rejects mixing input and output guardrails at registration.
    Both slots default to ``None`` (rather than empty list) to preserve
    the prior FunctionTool semantics where the absence of a guardrail
    list signaled "no guardrails configured" — this lets the executor's
    fast-path skip the iteration entirely.

    Attributes:
        input: Guardrails that run before the tool's ``on_invoke``.
            They validate parsed arguments and can detect PII, jailbreak
            attempts, schema violations, and so on. Each entry returns a
            :class:`ToolGuardrailFunctionOutput` verdict
            (``allow`` / ``reject_content`` / ``raise_exception``).
        output: Guardrails that run after the tool's ``on_invoke`` returns.
            They validate the result and can mask PII, enforce output
            schemas, or reject suspicious return values. Same verdict
            shape as input guardrails.
    """

    input: list[ToolInputGuardrail[Any]] | None = None
    """Input-phase guardrails. ``None`` means no input guardrails configured."""

    output: list[ToolOutputGuardrail[Any]] | None = None
    """Output-phase guardrails. ``None`` means no output guardrails configured."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, overload

from troopai.adk.exceptions import UserError
from troopai.adk.utils.typedef import MaybeAwaitable

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.run import RunContext
    from troopai.adk.types.guardrails.action import GuardrailAction, GuardrailSpan
    from troopai.adk.types.input import LLMInputContentItem

logger = logging.getLogger(__name__)


class AgentGuardrailSeverity(StrEnum):
    """Severity level for a guardrail verdict.

    Controls whether a guardrail violation halts execution or is recorded
    as a non-blocking signal for audit/monitoring.

    When ``severity`` is set on a ``AgentGuardrailFunctionOutput``, it takes
    precedence over ``tripwire_triggered`` for the halt decision:

    - ``INFO`` — logged at DEBUG, included in results, never halts.
    - ``WARNING`` — logged at WARNING, included in results, never halts.
    - ``ERROR`` — halts execution (equivalent to ``tripwire_triggered=True``).

    When ``severity`` is ``None`` (the default), only ``tripwire_triggered``
    determines whether execution halts.
    """

    INFO = "info"
    """Logged, no action. Use for low-confidence detections or telemetry."""

    WARNING = "warning"
    """Logged, included in results, no halt. Use for medium-confidence
    detections that should be auditable but not block execution."""

    ERROR = "error"
    """Halts execution. Equivalent to ``tripwire_triggered=True``."""


class AgentTimeoutPolicy(StrEnum):
    """Policy for handling guardrail execution timeouts.

    When a guardrail's ``timeout`` is exceeded, this policy determines
    whether the guardrail is treated as having tripped (``FAIL``) or
    passed silently (``PASS``).
    """

    FAIL = "fail"
    """Treat timeout as a tripwire trigger — halt execution."""

    PASS = "pass"
    """Treat timeout as a pass — continue execution silently."""


@dataclass
class AgentGuardrailTimeoutInfo:
    """Information passed to a guardrail timeout callback.

    Provides all context a production system needs for metrics, alerting,
    and audit logging when a guardrail times out.

    Attributes:
        guardrail_name: Name of the guardrail that timed out.
        agent_name: Name of the agent whose guardrail timed out.
        timeout: The timeout duration in seconds that was exceeded.
        policy: The ``AgentTimeoutPolicy`` that was applied (``FAIL`` or ``PASS``).
    """

    guardrail_name: str
    """Name of the guardrail that timed out."""

    agent_name: str
    """Name of the agent whose guardrail timed out."""

    timeout: float
    """The timeout duration in seconds that was exceeded."""

    policy: AgentTimeoutPolicy
    """The ``AgentTimeoutPolicy`` applied — ``FAIL`` (halt) or ``PASS`` (continue)."""


@dataclass
class AgentGuardrailFunctionOutput:
    """The output of an agent guardrail function.

    This is used for both input and output guardrails at the agent level.

    Attributes:
        output_info: Optional metadata about the checks performed.
        tripwire_triggered: If True, a violation was detected. When ``severity``
            is ``None``, this alone controls whether execution halts.
        severity: Optional severity level. When set, overrides
            ``tripwire_triggered`` for the halt decision: only ``ERROR``
            halts, while ``WARNING`` and ``INFO`` are recorded without
            halting. When ``None``, ``tripwire_triggered`` is authoritative.
        transformed_output: Optional complete replacement for the checked
            output (output guardrails, text only). When set, the runner
            substitutes it wholesale; ``None`` means no transform.
        changed_spans: Optional ranges flagged for audit/tracing only; never
            used to construct or apply a transform.
    """

    output_info: Any = None
    """
    Optional data about checks performed. For example, the guardrail could include
    information about the checks it performed and granular results.
    """

    tripwire_triggered: bool = False
    """
    If True, the guardrail has detected a violation and execution should halt.
    This will raise an AgentInputGuardrailTripwireTriggered or AgentOutputGuardrailTripwireTriggered
    exception depending on the guardrail type.

    When ``severity`` is set, only ``ERROR`` halts execution regardless of this
    field's value.
    """

    severity: AgentGuardrailSeverity | None = None
    """Optional severity level that overrides ``tripwire_triggered`` for halt decisions.

    When ``None`` (default), ``tripwire_triggered`` alone controls halting.
    When set:
        - ``INFO``: logged, included in results, never halts.
        - ``WARNING``: logged at WARNING level, included in results, never halts.
        - ``ERROR``: halts execution (equivalent to ``tripwire_triggered=True``).
    """

    transformed_output: Any = None
    """Complete replacement for the checked output. ``None`` (the default) means
    no transform. When set on an output guardrail, the runner substitutes it
    wholesale for the agent output — both the returned final output and the
    trailing history message — and never splices ``changed_spans``. Text outputs
    only; a transform cannot target ``None``. A transforming guardrail MUST also
    set ``tripwire_triggered=True`` and leave ``severity`` unset, so the verdict
    still halts when the runner cannot apply the substitution.
    """

    changed_spans: list[GuardrailSpan] | None = None
    """Ranges the guardrail flagged or moved, for audit and tracing only. Never
    read by the runner to construct or apply a transform."""

    def resolved_action(self) -> GuardrailAction:
        """Map this verdict onto the shared guardrail action vocabulary.

        A transform takes precedence; otherwise ``severity`` (when set) decides,
        and finally ``tripwire_triggered``. ``severity`` still gates halting via
        the runner's existing severity check, so it is not independent of the
        resolved action.
        """
        # Imported here, not at module load: this module is imported while the
        # types package is still initialising, so a top-level import would cycle.
        from troopai.adk.types.guardrails.action import GuardrailAction

        if self.transformed_output is not None:
            return GuardrailAction.TRANSFORM
        if self.severity is not None:
            if self.severity == AgentGuardrailSeverity.ERROR:
                return GuardrailAction.RAISE
            return GuardrailAction.PASS
        return GuardrailAction.RAISE if self.tripwire_triggered else GuardrailAction.PASS


@dataclass
class AgentInputGuardrailResult:
    """The result of an input guardrail run.

    Attributes:
        guardrail: The guardrail that was run.
        agent: The agent whose input was checked by the guardrail.
        guardrail_output: The guardrail function's verdict (pass/fail + metadata).
    """

    guardrail: AgentInputGuardrail[Any]
    """The guardrail that was run."""

    agent: Agent
    """The agent whose input was checked by the guardrail."""

    guardrail_output: AgentGuardrailFunctionOutput
    """The guardrail function's verdict (pass/fail + metadata)."""


@dataclass
class AgentOutputGuardrailResult:
    """The result of an output guardrail run.

    Attributes:
        guardrail: The guardrail that was run.
        agent: The agent that produced the checked output.
        agent_output: The agent's output that was validated.
        guardrail_output: The guardrail function's verdict (pass/fail + metadata).
    """

    guardrail: AgentOutputGuardrail[Any]
    """The guardrail that was run."""

    agent: Agent
    """The agent that produced the checked output."""

    agent_output: Any
    """The agent's output that was validated by this guardrail."""

    guardrail_output: AgentGuardrailFunctionOutput
    """The guardrail function's verdict (pass/fail + metadata)."""


@dataclass
class AgentInputGuardrailData:
    """Input data passed to an input guardrail function.

    This follows the OpenAI agent SDK pattern where guardrail functions
    receive context, agent, and input as a single data object.

    Attributes:
        context: The run context wrapper containing user context and
            usage tracking.
        agent: The agent that is being executed.
        user_prompt: The user prompt passed to the agent, either as a
            string or a list of input items.

    Example:
        @agent_input_guardrail
        async def my_guardrail(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            # Access the context
            user_id = data.context.context.get("user_id") if data.context.context else None

            # Check the user prompt
            if "bad_word" in str(data.user_prompt):
                return AgentGuardrailFunctionOutput(tripwire_triggered=True)

            return AgentGuardrailFunctionOutput(tripwire_triggered=False)
    """

    context: RunContext[Any]
    """
    The run context wrapper containing user context and usage tracking.
    Access user context via `context.context` and usage via `context.usage`.
    """

    agent: Agent
    """
    The agent that is being executed.
    """

    user_prompt: str | list[LLMInputContentItem]
    """
    The user prompt passed to the agent, either as a string or a list of input items.
    """


@dataclass
class AgentOutputGuardrailData:
    """Input data passed to an output guardrail function.

    This follows the OpenAI agent SDK pattern where guardrail functions
    receive context, agent, and output as a single data object.

    Attributes:
        context: The run context wrapper containing user context and
            usage tracking.
        agent: The agent that was executed.
        output: The output produced by the agent.

    Example:
        @agent_output_guardrail
        async def my_guardrail(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            # Check the output for sensitive content
            output_text = str(data.output)
            if "secret" in output_text.lower():
                return AgentGuardrailFunctionOutput(
                    tripwire_triggered=True,
                    output_info={"reason": "Output contains sensitive content"}
                )

            return AgentGuardrailFunctionOutput(tripwire_triggered=False)
    """

    context: RunContext[Any]
    """
    The run context wrapper containing user context and usage tracking.
    Access user context via `context.context` and usage via `context.usage`.
    """

    agent: Agent
    """
    The agent that was executed.
    """

    output: Any
    """
    The output produced by the agent.
    """


@dataclass
class AgentInputGuardrail[TContext_co: Any]:
    """A guardrail that runs before or in parallel with agent execution.

    Input guardrails validate the input to an agent before (or while) the agent
    processes it. They can be used to detect PII, jailbreak attempts, prompt
    injection, off-topic requests, and more.

    Attributes:
        guardrail_function: The function that implements the guardrail logic.
        name: Optional name for the guardrail. If not provided, uses the function name.
        run_in_parallel: If True (default), the guardrail runs in parallel with agent
            execution for better latency. If False, the guardrail blocks before the
            agent starts, which saves tokens if the tripwire triggers.
        timeout: Optional timeout in seconds for guardrail execution.
            When exceeded, ``timeout_policy`` determines the behavior.
        timeout_policy: Policy when timeout is exceeded. ``FAIL`` trips
            the wire, ``PASS`` continues silently. Defaults to ``FAIL``.
        on_timeout: Optional async callback invoked when the guardrail
            times out. Receives a ``AgentGuardrailTimeoutInfo`` with name, agent,
            duration, and policy. Use for metrics, alerting, or audit logging.
    """

    guardrail_function: Callable[[AgentInputGuardrailData], MaybeAwaitable[AgentGuardrailFunctionOutput]]
    """
    The function that implements the guardrail logic.
    """

    name: str | None = None
    """
    Optional name for the guardrail. If not provided, uses the function name.
    """

    run_in_parallel: bool = True
    """
    If True, the guardrail runs in parallel with agent execution.
    If False, the guardrail blocks before the agent starts.
    """

    timeout: float | None = None
    """Optional timeout in seconds for guardrail execution.

    When set, the guardrail's ``run()`` is wrapped in ``asyncio.wait_for()``.
    If the timeout is exceeded, ``timeout_policy`` determines whether the
    guardrail trips or passes silently.
    """

    timeout_policy: AgentTimeoutPolicy = AgentTimeoutPolicy.FAIL
    """Policy when ``timeout`` is exceeded.

    - ``FAIL``: treat timeout as a tripwire trigger — halt execution.
    - ``PASS``: treat timeout as a pass — continue execution silently.

    Defaults to ``FAIL`` (safe default).
    """

    on_timeout: Callable[[AgentGuardrailTimeoutInfo], Awaitable[None]] | None = None
    """Optional async callback invoked when the guardrail times out.

    Receives a ``AgentGuardrailTimeoutInfo`` dataclass with ``guardrail_name``,
    ``agent_name``, ``timeout``, and ``policy``. The callback runs *after*
    the policy decision is made — it cannot change the outcome, only observe it.

    Example::

        async def alert_on_timeout(info: AgentGuardrailTimeoutInfo) -> None:
            await metrics.increment("guardrail.timeout", tags={
                "guardrail": info.guardrail_name,
                "agent": info.agent_name,
                "policy": info.policy.value,
            })

        AgentInputGuardrail(
            guardrail_function=my_fn,
            timeout=5.0,
            timeout_policy=AgentTimeoutPolicy.PASS,
            on_timeout=alert_on_timeout,
        )
    """

    def get_name(self) -> str:
        """Get the name of this guardrail.

        Returns:
            The explicit ``name`` if set, otherwise the guardrail
            function's ``__name__``.
        """
        return self.name or self.guardrail_function.__name__

    async def run(self, data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
        """Run the guardrail on the given input data.

        Args:
            data: The input data to validate.

        Returns:
            AgentGuardrailFunctionOutput with tripwire_triggered indicating if validation failed.

        Raises:
            UserError: If the guardrail function is not callable.
        """
        if not callable(self.guardrail_function):
            raise UserError(f"Guardrail function must be callable, got {self.guardrail_function}")

        result = self.guardrail_function(data)
        if inspect.isawaitable(result):
            return await result
        return result


@dataclass
class AgentOutputGuardrail[TContext_co: Any]:
    """A guardrail that runs after agent execution completes.

    Output guardrails validate the output from an agent before it's returned.
    They can be used to detect PII in responses, check for hallucinations,
    ensure compliance, and more.

    Output guardrails always run in blocking mode (after the agent completes).

    Attributes:
        guardrail_function: The function that implements the guardrail logic.
        name: Optional name for the guardrail. If not provided, uses the function name.
        remediation: Optional feedback message to inject when the guardrail trips.
            When set, the runner re-prompts the agent with this feedback instead
            of raising immediately, giving the agent a chance to self-correct.
        max_retries: Maximum remediation attempts before raising. Only meaningful
            when ``remediation`` is set. Defaults to 1.
        timeout: Optional timeout in seconds for guardrail execution.
        timeout_policy: Policy when timeout is exceeded. Defaults to ``FAIL``.
        on_timeout: Optional async callback invoked on timeout. Receives
            ``AgentGuardrailTimeoutInfo`` for metrics/alerting.
    """

    guardrail_function: Callable[[AgentOutputGuardrailData], MaybeAwaitable[AgentGuardrailFunctionOutput]]
    """
    The function that implements the guardrail logic.
    """

    name: str | None = None
    """
    Optional name for the guardrail. If not provided, uses the function name.
    """

    remediation: str | None = None
    """Optional feedback message for agent self-correction.

    When set and the guardrail trips, the runner injects this message as
    user feedback and re-runs the agent instead of raising immediately.
    After ``max_retries`` failed attempts, raises normally.

    Example: ``"Your response contained PII. Please regenerate without
    including personal information."``
    """

    max_retries: int = 1
    """Maximum remediation attempts before raising.

    Only meaningful when ``remediation`` is set. After this many re-runs
    still trip the guardrail, ``AgentOutputGuardrailTripwireTriggered`` is raised.
    Defaults to 1 (one retry attempt).
    """

    timeout: float | None = None
    """Optional timeout in seconds for guardrail execution.

    When set, the guardrail's ``run()`` is wrapped in ``asyncio.wait_for()``.
    If the timeout is exceeded, ``timeout_policy`` determines whether the
    guardrail trips or passes silently.
    """

    timeout_policy: AgentTimeoutPolicy = AgentTimeoutPolicy.FAIL
    """Policy when ``timeout`` is exceeded.

    - ``FAIL``: treat timeout as a tripwire trigger — halt execution.
    - ``PASS``: treat timeout as a pass — continue execution silently.

    Defaults to ``FAIL`` (safe default).
    """

    on_timeout: Callable[[AgentGuardrailTimeoutInfo], Awaitable[None]] | None = None
    """Optional async callback invoked when the guardrail times out.

    Receives a ``AgentGuardrailTimeoutInfo`` dataclass with ``guardrail_name``,
    ``agent_name``, ``timeout``, and ``policy``. The callback runs *after*
    the policy decision is made — it cannot change the outcome, only observe it.
    """

    def __post_init__(self) -> None:
        """Validate AgentOutputGuardrail configuration after initialization."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.remediation is not None and self.max_retries == 0:
            # A remediation message with zero retries can never fire: the loop
            # gates on ``count < max_retries`` (0 < 0 is False), so the guardrail
            # raises on first trip exactly as if no remediation were set. This is
            # a legal but inert combination — warn rather than reject, since
            # max_retries=0 is an otherwise-valid "no retries" value.
            logger.warning(
                "AgentOutputGuardrail %r sets remediation but max_retries=0; the "
                "remediation message will never be applied. Set max_retries >= 1 "
                "to enable self-correction.",
                self.get_name(),
            )

    def get_name(self) -> str:
        """Get the name of this guardrail.

        Returns:
            The explicit ``name`` if set, otherwise the guardrail
            function's ``__name__``.
        """
        return self.name or self.guardrail_function.__name__

    async def run(self, data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
        """Run the guardrail on the given output data.

        Args:
            data: The output data to validate.

        Returns:
            AgentGuardrailFunctionOutput with tripwire_triggered indicating if validation failed.

        Raises:
            UserError: If the guardrail function is not callable.
        """
        if not callable(self.guardrail_function):
            raise UserError(f"Guardrail function must be callable, got {self.guardrail_function}")

        result = self.guardrail_function(data)
        if inspect.isawaitable(result):
            return await result
        return result


# Type aliases for guardrail functions
_InputGuardrailFuncSync = Callable[[AgentInputGuardrailData], AgentGuardrailFunctionOutput]
_InputGuardrailFuncAsync = Callable[[AgentInputGuardrailData], Awaitable[AgentGuardrailFunctionOutput]]

_OutputGuardrailFuncSync = Callable[[AgentOutputGuardrailData], AgentGuardrailFunctionOutput]
_OutputGuardrailFuncAsync = Callable[[AgentOutputGuardrailData], Awaitable[AgentGuardrailFunctionOutput]]


# Input guardrail decorator overloads
@overload
def agent_input_guardrail(func: _InputGuardrailFuncSync) -> AgentInputGuardrail[Any]: ...


@overload
def agent_input_guardrail(func: _InputGuardrailFuncAsync) -> AgentInputGuardrail[Any]: ...


@overload
def agent_input_guardrail(
    *,
    name: str | None = None,
    run_in_parallel: bool = True,
    timeout: float | None = None,
    timeout_policy: AgentTimeoutPolicy = AgentTimeoutPolicy.FAIL,
    on_timeout: Callable[[AgentGuardrailTimeoutInfo], Awaitable[None]] | None = None,
) -> Callable[[_InputGuardrailFuncSync | _InputGuardrailFuncAsync], AgentInputGuardrail[Any]]: ...


def agent_input_guardrail(
    func: _InputGuardrailFuncSync | _InputGuardrailFuncAsync | None = None,
    *,
    name: str | None = None,
    run_in_parallel: bool = True,
    timeout: float | None = None,
    timeout_policy: AgentTimeoutPolicy = AgentTimeoutPolicy.FAIL,
    on_timeout: Callable[[AgentGuardrailTimeoutInfo], Awaitable[None]] | None = None,
) -> (
    AgentInputGuardrail[Any] | Callable[[_InputGuardrailFuncSync | _InputGuardrailFuncAsync], AgentInputGuardrail[Any]]
):
    """Decorator to create an AgentInputGuardrail from a function.

    Can be used with or without parentheses:

        @agent_input_guardrail
        async def my_guardrail(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            ...

        @agent_input_guardrail(name="custom_name", run_in_parallel=False)
        async def blocking_guardrail(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            ...

        @agent_input_guardrail(timeout=5.0, timeout_policy=AgentTimeoutPolicy.PASS)
        async def slow_guardrail(data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            ...

    Args:
        func: The guardrail function (when used without parentheses).
        name: Optional name for the guardrail.
        run_in_parallel: Whether to run in parallel with agent execution (default True).
        timeout: Optional timeout in seconds.
        timeout_policy: Policy when timeout is exceeded (default FAIL).
        on_timeout: Optional async callback for timeout side effects.

    Returns:
        An AgentInputGuardrail instance or a decorator function.
    """

    def decorator(f: _InputGuardrailFuncSync | _InputGuardrailFuncAsync) -> AgentInputGuardrail[Any]:
        return AgentInputGuardrail(
            guardrail_function=f,
            name=name or f.__name__,
            run_in_parallel=run_in_parallel,
            timeout=timeout,
            timeout_policy=timeout_policy,
            on_timeout=on_timeout,
        )

    if func is not None:
        return decorator(func)
    return decorator


# Output guardrail decorator overloads
@overload
def agent_output_guardrail(func: _OutputGuardrailFuncSync) -> AgentOutputGuardrail[Any]: ...


@overload
def agent_output_guardrail(func: _OutputGuardrailFuncAsync) -> AgentOutputGuardrail[Any]: ...


@overload
def agent_output_guardrail(
    *,
    name: str | None = None,
    remediation: str | None = None,
    max_retries: int = 1,
    timeout: float | None = None,
    timeout_policy: AgentTimeoutPolicy = AgentTimeoutPolicy.FAIL,
    on_timeout: Callable[[AgentGuardrailTimeoutInfo], Awaitable[None]] | None = None,
) -> Callable[[_OutputGuardrailFuncSync | _OutputGuardrailFuncAsync], AgentOutputGuardrail[Any]]: ...


def agent_output_guardrail(
    func: _OutputGuardrailFuncSync | _OutputGuardrailFuncAsync | None = None,
    *,
    name: str | None = None,
    remediation: str | None = None,
    max_retries: int = 1,
    timeout: float | None = None,
    timeout_policy: AgentTimeoutPolicy = AgentTimeoutPolicy.FAIL,
    on_timeout: Callable[[AgentGuardrailTimeoutInfo], Awaitable[None]] | None = None,
) -> (
    AgentOutputGuardrail[Any]
    | Callable[[_OutputGuardrailFuncSync | _OutputGuardrailFuncAsync], AgentOutputGuardrail[Any]]
):
    """Decorator to create an AgentOutputGuardrail from a function.

    Can be used with or without parentheses:

        @agent_output_guardrail
        async def my_guardrail(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            ...

        @agent_output_guardrail(name="custom_name")
        async def named_guardrail(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            ...

        @agent_output_guardrail(remediation="Remove PII and try again.", max_retries=2)
        async def pii_guardrail(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            ...

    Args:
        func: The guardrail function (when used without parentheses).
        name: Optional name for the guardrail.
        remediation: Optional feedback for agent self-correction on trip.
        max_retries: Max remediation attempts (default 1).
        timeout: Optional timeout in seconds.
        timeout_policy: Policy when timeout is exceeded (default FAIL).
        on_timeout: Optional async callback for timeout side effects.

    Returns:
        An AgentOutputGuardrail instance or a decorator function.
    """

    def decorator(f: _OutputGuardrailFuncSync | _OutputGuardrailFuncAsync) -> AgentOutputGuardrail[Any]:
        return AgentOutputGuardrail(
            guardrail_function=f,
            name=name or f.__name__,
            remediation=remediation,
            max_retries=max_retries,
            timeout=timeout,
            timeout_policy=timeout_policy,
            on_timeout=on_timeout,
        )

    if func is not None:
        return decorator(func)
    return decorator


@dataclass
class AgentGuardrails:
    """Per-phase agent-level guardrail lists registered on an Agent.

    Each slot is typed against its phase-specific Protocol so the type
    checker rejects mixing input and output guardrails at registration.

    Attributes:
        input: Guardrails that run before (or in parallel with) agent
            execution. They validate the user prompt and can detect PII,
            jailbreak attempts, prompt injection, off-topic requests,
            and so on. Each entry returns an
            :class:`AgentGuardrailFunctionOutput` verdict.
        output: Guardrails that run after agent execution completes.
            They validate the final output and can detect PII in the
            response, hallucinations, schema violations, and so on.
            Output guardrails support the ``remediation`` field so the
            runner can re-prompt the agent on trip instead of raising
            immediately.
    """

    input: list[AgentInputGuardrail[Any]] = field(default_factory=list)
    """Input-phase guardrails. See class docstring."""

    output: list[AgentOutputGuardrail[Any]] = field(default_factory=list)
    """Output-phase guardrails. See class docstring."""


@dataclass
class AgentGuardrailResults:
    """Per-phase audit trail of agent-level guardrail verdicts for a run.

    Each slot is an immutable tuple of result objects produced by the
    matching guardrail phase. Populated by the runner after each
    guardrail phase completes; immutable once written.

    Attributes:
        input: Verdicts from every input guardrail that ran (blocking
            and parallel) for this run.
        output: Verdicts from every output guardrail that ran for this
            run.
    """

    input: tuple[AgentInputGuardrailResult, ...] = ()
    """Input guardrail verdicts. See class docstring."""

    output: tuple[AgentOutputGuardrailResult, ...] = ()
    """Output guardrail verdicts. See class docstring."""

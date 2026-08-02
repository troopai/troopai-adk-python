"""FlowStepGuardrails — typed pre/post verdict surface for Flow steps.

Step-level analogue of :class:`troopai.adk.tools.tool_guardrails.ToolGuardrails`
but operating on :class:`FlowStepContext` (pre) and the developer's
typed flow state (post). Same verdict shape as the tool guardrail
layer:

- ``allow()`` — proceed (the post body / next listener fires).
- ``reject_content(message)`` — short-circuit with a routed explanation
  (lands on a :class:`FlowStepRejectedEvent` and routes through
  ``FlowConfig.error_policy``).
- ``raise_exception(exc)`` — surface a typed framework exception
  (routes through ``error_policy`` as any step body exception).

Strictly a verdict surface. "Should this proceed?" decisions
always go through the typed guardrail channel; the middleware
surface is plumbing-only (logging, metrics, retries). This module
exists so step-level safety policies (PII filters, content checks,
schema validators, role enforcement) can attach to a Flow step the
same way they attach to a tool.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from troopai.adk.types.guardrails.action import GuardrailAction

if TYPE_CHECKING:
    from troopai.adk.flows.step_context import FlowStepContext


@dataclass(frozen=True, kw_only=True)
class FlowStepGuardrailVerdict:
    """Typed verdict returned by a :data:`FlowStepGuardrailFn`.

    Mirrors the verdict shape used by
    :class:`troopai.adk.tools.tool_guardrails.ToolGuardrailFunctionOutput`:
    callers construct one of the three factory variants
    (:meth:`allow`, :meth:`reject_content`,
    :meth:`raise_exception`) rather than wiring booleans.

    Attributes:
        allowed: ``True`` to proceed; ``False`` to short-circuit.
        message: When ``allowed=False``, the routed explanation
            surfaced through events / error handlers.
        exception: When set, the verdict raises this exception
            instead of routing through the normal rejection path.
            Used by the :meth:`raise_exception` factory.
    """

    allowed: bool
    """``True`` to proceed; ``False`` to short-circuit."""

    message: str | None = None
    """Routed rejection explanation when ``allowed=False``."""

    exception: BaseException | None = None
    """Typed framework exception when the verdict raises rather than rejects."""

    @classmethod
    def allow(cls) -> FlowStepGuardrailVerdict:
        """Construct an allow verdict (proceed)."""
        return cls(allowed=True)

    @classmethod
    def reject_content(cls, message: str) -> FlowStepGuardrailVerdict:
        """Construct a reject verdict carrying a routed ``message``."""
        return cls(allowed=False, message=message)

    @classmethod
    def raise_exception(cls, exception: BaseException) -> FlowStepGuardrailVerdict:
        """Construct a verdict that raises ``exception`` at the boundary."""
        return cls(allowed=False, exception=exception)

    def resolved_action(self) -> GuardrailAction:
        """Map this verdict onto the shared guardrail action vocabulary.

        A flow step has no replaceable return value, so it is ``PASS`` or
        ``RAISE`` only: an allowed verdict passes, and both rejection variants
        (routed message and raised exception) resolve to ``RAISE``. The audit
        record disambiguates the two by their recorded fields.
        """
        return GuardrailAction.PASS if self.allowed else GuardrailAction.RAISE


FlowStepGuardrailFn = Callable[
    ["FlowStepContext[Any]"],
    "FlowStepGuardrailVerdict | Awaitable[FlowStepGuardrailVerdict]",
]
"""One step-level guardrail callable.

Receives a :class:`FlowStepContext` for the step about to run
(pre-guardrail) or that just ran (post-guardrail) and returns a
:class:`FlowStepGuardrailVerdict`. Sync or async — the executor
awaits the result when awaitable.

Pre-guardrails see the step's incoming context; post-guardrails
see the same context augmented with the step's completed state.
"""


@dataclass(frozen=True, kw_only=True)
class FlowStepGuardrails:
    """Bundle of pre/post guardrails attached to one Flow step.

    Mirrors the bundling pattern of
    :class:`troopai.adk.tools.tool_guardrails.ToolGuardrails`. Pre runs after
    the full gate chain (``enabled``, resume-decision,
    ``requires_approval``) and the rate-limit acquire, immediately before
    the step body; post runs after the body completes successfully
    (skipped on body exception).

    Attributes:
        pre: Tuple of pre-step guardrail callables, evaluated in
            order. First non-allow verdict short-circuits the chain.
            Empty tuple ⇒ no pre checks.
        post: Tuple of post-step guardrail callables, evaluated in
            order. First non-allow verdict short-circuits the chain.
            Empty tuple ⇒ no post checks.
        metadata: Open-ended developer payload — never read by the
            framework. Useful for tagging the guardrail bundle with
            compliance references, policy identifiers, audit hints.
    """

    pre: tuple[FlowStepGuardrailFn, ...] = ()
    """Pre-step guardrails (evaluated in order, first non-allow short-circuits)."""

    post: tuple[FlowStepGuardrailFn, ...] = ()
    """Post-step guardrails (evaluated in order, first non-allow short-circuits)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Open-ended developer payload — never read by the framework."""

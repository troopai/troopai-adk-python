from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Generic

from typing_extensions import TypeVar

from troopai.adk.llms.llm_usage import LLMUsage

if TYPE_CHECKING:
    from troopai.adk.types.run.guardrail_audit import GuardrailAuditRecord

TContext = TypeVar("TContext", default=Any)


_MISSING: Final[object] = object()
"""Sentinel for ``RunContext._swarm_resume_reply``.

Distinguishes "no reply seeded" (the default) from an explicit ``None``
reply value (a valid abstain answer). Identity comparison only — never
exported; consumers use the accessor methods on :class:`RunContext`.
"""


@dataclass(eq=False)
class RunContext(Generic[TContext]):
    """Context that flows through agent execution.

    RunContext carries user-provided context and tracks usage metrics
    throughout the execution of an agent. It is created at the start
    of a run and passed through all components.

    In our design, RunContext serves as both the user-facing context
    container and the internal wrapper (unlike OpenAI's SDK which has
    a separate RunContextWrapper).

    Type Parameters:
        TContext: The type of the user-provided context.

    Attributes:
        context: User-provided context data (e.g., user_id, session info).
        usage: Token usage tracking across all LLM calls.
        tenant_id: Opaque tenant identifier for this run; ``None`` = untenanted.
            Threaded to status records/quotas, spans, the tenant metric
            dimension, and logs.
        cost_usd: Running total USD cost across this run's LLM calls
            (best-effort; ``0.0`` when unavailable).
    """

    context: TContext
    """User-provided context data."""

    usage: LLMUsage = field(default_factory=LLMUsage)
    """The token usage tracking across all LLM calls. This is updated as the agent makes calls to LLMs.
    """

    tenant_id: str | None = None
    """Opaque tenant identifier for this run (``None`` = untenanted).

    Threaded to status records/quotas, spans (``troopai.tenant.id``), the
    ``tenant`` metric dimension, and logs.
    """

    cost_usd: float = 0.0
    """Running total USD cost across this run's LLM calls. Accumulated by
    the agent loop from the resolved LLM's cost lookup; best-effort
    (``0.0`` when cost is unavailable)."""

    _swarm_resume_reply: Any = field(default=_MISSING, repr=False, compare=False)
    """Reserved framework slot for the swarm HITL-pure resume channel.

    The swarm driver seeds this with the caller-supplied reply before
    re-firing a member parked on a pure-HITL interrupt;
    :func:`troopai.adk.swarms.interrupt.request_human_input_in_swarm`
    consumes it and clears the slot. Default is the ``_MISSING``
    sentinel so an explicit ``None`` reply (abstain) is distinguishable
    from "no reply seeded". External code should NOT read or write this
    field directly — use the accessor methods.
    """

    _guardrail_audit: list[GuardrailAuditRecord] = field(default_factory=list, repr=False, compare=False)
    """Accumulated guardrail audit records for this run.

    Guardrail executors append a record after each verdict; the runner drains
    the list onto ``RunResult.guardrail_audit`` when the run completes. External
    code uses the accessor methods, never this field directly.
    """

    _tool_caches: dict[int, Any] = field(default_factory=dict, repr=False, compare=False)
    """Run-scoped tool caches keyed by a tool-owned namespace."""

    def has_swarm_resume_reply(self) -> bool:
        """Return whether a swarm resume reply has been seeded."""
        return self._swarm_resume_reply is not _MISSING

    def consume_swarm_resume_reply(self) -> Any:
        """Return and clear the seeded swarm resume reply.

        Raises:
            LookupError: When no reply has been seeded. Callers must
                check :meth:`has_swarm_resume_reply` first.
        """
        if self._swarm_resume_reply is _MISSING:
            raise LookupError(
                "RunContext.consume_swarm_resume_reply: no reply seeded — check has_swarm_resume_reply() first"
            )
        reply = self._swarm_resume_reply
        self._swarm_resume_reply = _MISSING
        return reply

    def seed_swarm_resume_reply(self, reply: Any) -> None:
        """Seed a swarm resume reply on this context.

        Idempotent overwrite; the swarm driver calls this once per
        resumed turn. Callers should consume the seeded reply by
        invoking :meth:`consume_swarm_resume_reply` from inside the
        member's tool body (typically via
        :func:`troopai.adk.swarms.interrupt.request_human_input_in_swarm`).
        """
        self._swarm_resume_reply = reply

    def clear_swarm_resume_reply(self) -> None:
        """Reset the swarm resume reply slot to the unseeded sentinel.

        Used by the swarm driver to clean up after a resumed turn in
        case the member's tool didn't consume the reply (defensive —
        prevents the slot from leaking into a subsequent turn).
        """
        self._swarm_resume_reply = _MISSING

    def record_guardrail_audit(self, record: GuardrailAuditRecord) -> None:
        """Append one guardrail audit record to this run's trail.

        Called by the guardrail executors (which hold the run context) after each
        verdict. Tool guardrail functions receive only a ``ToolContext`` and never
        reach this — recording stays with the executor.
        """
        self._guardrail_audit.append(record)

    def collect_guardrail_audit(self) -> tuple[GuardrailAuditRecord, ...]:
        """Return the accumulated guardrail audit records as an immutable tuple."""
        return tuple(self._guardrail_audit)

    def get_tool_cache(self, namespace: int) -> Any | None:
        """Return the run-scoped cache for ``namespace``, or ``None``."""
        return self._tool_caches.get(namespace)

    def set_tool_cache(self, namespace: int, cache: Any) -> None:
        """Attach a run-scoped cache for ``namespace``."""
        self._tool_caches[namespace] = cache

    @classmethod
    def make(cls, context: TContext | None, *, tenant_id: str | None = None) -> RunContext[TContext]:
        """Build a RunContext from a caller-supplied optional context slot.

        The Runner's public entry points accept ``context: TContext | None``
        (``None`` meaning "no user context"). ``RunContext.context`` is
        declared as ``TContext`` under invariant generics, so the direct
        call ``RunContext(context=context)`` infers the wider
        ``RunContext[TContext | None]`` and would require a ``cast``. This
        helper absorbs the one narrowing step in a single place so call
        sites stay cast-free.

        Args:
            context: User-provided context (or ``None``).
            tenant_id: Opaque tenant identifier for this run.
                Threaded to status records, spans, metric dimensions, and
                logs. ``None`` = untenanted (default).

        Returns:
            A freshly constructed ``RunContext[TContext]``.
        """
        # ``TContext`` itself carries ``default=Any``, so at the framework
        # boundary a ``None`` runtime value is acceptable; the ignore is
        # scoped to this single construction.
        return cls(context=context, tenant_id=tenant_id)  # type: ignore[arg-type]

    @classmethod
    def from_run_context(cls, run_context: RunContext[TContext]) -> RunContext[TContext]:
        """Return the given RunContext as-is.

        In our architecture, RunContext combines both user context and
        framework state (usage), so no wrapping is needed.  This method
        exists for API compatibility with call sites that expect a
        factory pattern.

        Args:
            run_context: An existing RunContext instance.

        Returns:
            The same RunContext instance, unchanged.
        """
        return run_context

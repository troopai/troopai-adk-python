"""``SandboxObservability`` — run-scoped emission handle for sandbox capabilities.

Built once per run inside the sandbox lifecycle bracket and bound onto
each capability (the same way the live session is bound). The
``run_command`` tool calls these methods around each command so per-
command spans, hooks, usage, and audit events all fire from one place.

Carries run-scoped state (``RunContext`` / ``Agent`` / ``RunHooks``)
injected at bind time — NOT reached through ``ToolContext``. The
capability tools are framework built-ins, distinct from developer
``@function_tool``s.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from troopai.adk.sandbox.observability.audit_sink import SandboxAuditEvent
from troopai.adk.tracing.spans import sandbox_span
from troopai.adk.types.sandbox.usage import SandboxSingleExecUsage

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.run.context import RunContext
    from troopai.adk.sandbox.observability.audit_sink import AuditSink
    from troopai.adk.types.sandbox.cost import SandboxCostDescriptor
    from troopai.adk.types.sandbox.exec_result import ExecResult
    from troopai.adk.types.sandbox.usage import SandboxUsage

logger = logging.getLogger(__name__)

__all__ = ["SandboxObservability"]

_COMMAND_TRACE_LIMIT = 1024


@dataclasses.dataclass
class SandboxObservability:
    """Run-scoped sandbox emission handle bound onto capabilities.

    Attributes:
        backend_id: Resolved backend identifier (or ``"injected"``).
        tracing_enabled: Mirror of ``RunConfig.tracing_enabled``.
        usage: The per-session accumulator (mutated in place).
        session_id: Backend session id for audit events.
        audit_sink: Optional audit sink; events are best-effort.
        cost: Optional rate card for per-command computed cost.
        hooks: Composed run hooks (or ``None``).
        context: Active ``RunContext`` (for hook calls).
        agent: Active agent (for hook calls + audit ``agent_name``).
    """

    backend_id: str
    """Resolved backend identifier (or ``"injected"``)."""

    tracing_enabled: bool
    """Mirror of ``RunConfig.tracing_enabled``."""

    usage: SandboxUsage
    """The per-session accumulator (mutated in place)."""

    session_id: str | None = None
    """Backend session id stamped on audit events."""

    audit_sink: AuditSink | None = None
    """Optional audit sink; emission is best-effort (errors suppressed)."""

    cost: SandboxCostDescriptor | None = None
    """Optional rate card for per-command computed cost."""

    hooks: RunHooks[Any] | None = None
    """Composed run hooks fan-out, or ``None``."""

    context: RunContext[Any] | None = None
    """Active ``RunContext`` passed to lifecycle hooks."""

    agent: Agent[Any] | None = None
    """Active agent (for hook calls + audit ``agent_name``)."""

    async def emit_audit(self, event: SandboxAuditEvent) -> None:
        """Emit one audit event, best-effort (logs + suppresses sink errors)."""
        if self.audit_sink is None:
            return
        try:
            await self.audit_sink.emit(event)
        except Exception:
            logger.debug("sandbox audit sink.emit failed; suppressed (event_type=%s)", event.event_type, exc_info=True)

    def _agent_name(self) -> str:
        if self.agent is not None:
            return self.agent.name
        return "<unknown>"

    def _hook_args(self) -> tuple[RunHooks[Any], RunContext[Any], Agent[Any]] | None:
        """Return the run-scoped hook triple, or ``None`` when any is absent.

        Hooks fire only when ``hooks`` + ``context`` + ``agent`` are all
        present. Returning the narrowed triple (rather than a bool) keeps
        this the single definition of that precondition while preserving
        type narrowing at each call site.
        """
        if self.hooks is not None and self.context is not None and self.agent is not None:
            return self.hooks, self.context, self.agent
        return None

    async def before_exec(self, command: str) -> None:
        """Fire the exec-start hook (when hooks + context + agent present)."""
        args = self._hook_args()
        if args is not None:
            hooks, context, agent = args
            await hooks.on_sandbox_exec_start(context, agent, command[:_COMMAND_TRACE_LIMIT])

    async def after_exec(self, command: str, result: ExecResult) -> None:
        """Record usage, emit the span + exec audit, fire the exec-end hook."""
        traced = command[:_COMMAND_TRACE_LIMIT]
        duration_ms = result.duration_ms if result.duration_ms is not None else 0
        cost_usd = self.cost.cost_for_ms(duration_ms) if self.cost is not None else None
        self.usage.add_exec(
            SandboxSingleExecUsage(
                command=traced,
                exit_code=result.exit_code,
                duration_ms=duration_ms,
                cost_usd=cost_usd,
            )
        )
        with sandbox_span(
            backend_id=self.backend_id,
            command=traced,
            exit_code=result.exit_code,
            duration_ms=result.duration_ms,
            disabled=not self.tracing_enabled,
        ):
            pass
        await self.emit_audit(
            SandboxAuditEvent(
                event_type="exec",
                agent_name=self._agent_name(),
                backend_id=self.backend_id,
                session_id=self.session_id,
                command=traced,
                exit_code=result.exit_code,
            )
        )
        args = self._hook_args()
        if args is not None:
            hooks, context, agent = args
            await hooks.on_sandbox_exec_end(context, agent, traced, result)

    async def on_violation(self, command: str, reason: str) -> None:
        """Emit a ``violation`` audit event for a guardrail rejection."""
        await self.emit_audit(
            SandboxAuditEvent(
                event_type="violation",
                agent_name=self._agent_name(),
                backend_id=self.backend_id,
                session_id=self.session_id,
                command=command[:_COMMAND_TRACE_LIMIT],
                error=reason,
            )
        )

    async def on_start(self, session: Any) -> None:
        """Fire the sandbox-start lifecycle hook (when hooks/context/agent present)."""
        args = self._hook_args()
        if args is not None:
            hooks, context, agent = args
            await hooks.on_sandbox_start(context, agent, session)

    async def on_stop(self, session: Any) -> None:
        """Fire the sandbox-stop lifecycle hook with the accumulated usage."""
        args = self._hook_args()
        if args is not None:
            hooks, context, agent = args
            await hooks.on_sandbox_stop(context, agent, session, self.usage)

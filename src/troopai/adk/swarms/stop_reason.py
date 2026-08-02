"""Swarm stop-reason taxonomy.

When a ``TerminationCondition`` fires, it produces a ``StopReason``
that names which condition stopped the swarm and attaches a
human-readable detail. Surfaced on ``SwarmRunResult.stop_reason`` and
``SwarmDoneEvent.reason``.

Kept as a frozen dataclass (not an enum) so that custom termination
conditions can supply their own ``kind`` strings without extending a
closed enum.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StopReason:
    """Why the swarm stopped.

    Examples::

        StopReason(kind="explicit_done", detail="Task completed: answer produced.")
        StopReason(kind="max_turns", detail="Hit the 20-turn cap.")
        StopReason(kind="token_budget", detail="Consumed 100000/100000 tokens.")
        StopReason(kind="handoff_to", detail="Swarm handed off to 'user' for HITL.")
        StopReason(kind="max_handoffs", detail="Hit the 20-handoff hard guard.")

    Attributes:
        kind: Short machine-readable identifier of the stop reason.
            Built-in values: ``"explicit_done"``, ``"max_turns"``,
            ``"token_budget"``, ``"handoff_to"``, ``"text_mention"``,
            ``"max_handoffs"``, ``"max_total_tokens"``,
            ``"interrupted"``, ``"policy_error"``. Custom
            :class:`~troopai.adk.swarms.termination.TerminationCondition`
            subclasses may supply their own.
        detail: Human-readable explanation. Included in logs, tracing
            spans, and ``SwarmDoneEvent.reason``.
    """

    kind: str
    """Short machine-readable identifier of the stop reason.

    Ships with these kinds:

    - ``"explicit_done"`` — agent called ``swarm_done``
    - ``"max_turns"`` — ``MaxTurnsTermination`` fired
    - ``"token_budget"`` — ``TokenBudgetTermination`` fired
    - ``"handoff_to"`` — ``HandoffToTermination`` matched a target
    - ``"text_mention"`` — ``TextMentionTermination`` matched a phrase
    - ``"max_handoffs"`` — ``SwarmConfig.max_handoffs`` hard guard
    - ``"max_total_tokens"`` — ``SwarmConfig.max_total_tokens`` guard
    - ``"interrupted"`` — a member parked on a human-input interrupt
    - ``"policy_error"`` — the routing policy raised; the driver
      converts it to a clean stop rather than crashing mid-run

    Custom ``TerminationCondition`` subclasses MAY supply their own.
    """

    detail: str
    """Human-readable explanation. Included in logs, tracing spans,
    and ``SwarmDoneEvent.reason``."""

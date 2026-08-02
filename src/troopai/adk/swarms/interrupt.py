"""Swarm-level pause/resume primitives.

Mirrors :mod:`troopai.adk.graphs.interrupt`'s ``GraphResume`` for the
swarms substrate. The ``Interrupt`` / ``NestedAgentInterrupt`` types
themselves are reused from the graphs module (cross-substrate by
design — same human-decision payload regardless of orchestration
substrate).

Usage:

    >>> first = await Runner.arun_swarm(swarm, "go", hooks=[checkpointer])
    >>> assert first.stop_reason.kind == "interrupted"
    >>> # caller composes reply, then resumes:
    >>> second = await Runner.arun_swarm_from_checkpoint(
    ...     swarm,
    ...     checkpointer=checkpointer,
    ...     thread_id="thr-1",
    ...     resume=SwarmResume(replies={"member_a": NestedAgentReply(...)}),
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from troopai.adk.graphs.interrupt import Interrupt, InterruptException

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext


@dataclass(frozen=True)
class SwarmResume:
    """Human replies for resuming a paused swarm run.

    Pass an instance to :meth:`Runner.arun_swarm_from_checkpoint` so the
    swarm driver can unblock members that raised an
    :class:`~troopai.adk.graphs.interrupt.InterruptException` or whose
    tool deferred via :class:`AgentToolDeferral`.

    Attributes:
        replies: Human-supplied approval values keyed by member name.
            For ``NestedAgentInterrupt`` resumes, the value MUST be a
            :class:`~troopai.adk.graphs.interrupt.NestedAgentReply`. For
            plain HITL interrupts, any JSON-safe value the originating
            tool understands.
        rejected: Model-visible decline messages keyed by member name.
            Analogous to ``state.reject(message=...)`` for each
            declined interrupt. Mutually exclusive with ``replies`` for
            a given member.
    """

    replies: dict[str, Any] = field(default_factory=dict)
    """Per-member reply values, keyed by member name."""

    rejected: dict[str, str] = field(default_factory=dict)
    """Per-member decline messages, keyed by member name."""


def request_human_input_in_swarm(
    ctx_wrapper: RunContext[Any],
    member_name: str,
    question: str,
    *,
    kind: str = "generic",
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Return the seeded reply, or raise :class:`InterruptException`.

    Swarm-substrate companion to
    :func:`troopai.adk.graphs.interrupt.request_human_input`. Called
    from inside a swarm member's tool body. When the swarm driver
    has seeded a reply on the run context (because the caller
    re-entered via :meth:`Runner.arun_swarm_from_checkpoint` with a
    matching ``SwarmResume.replies`` entry), this function consumes
    and returns it. Otherwise it raises ``InterruptException`` so the
    swarm loop captures the interrupt under ``member_name`` and parks
    the run.

    Key-presence — not value truthiness — is authoritative: a seeded
    ``None`` reply (abstain) is a valid result, distinct from "no
    reply".

    Args:
        ctx_wrapper: The :class:`RunContext` flowing through the
            current run. The seeded reply lives on this object.
        member_name: The swarm member's name. Becomes the parked
            interrupt's ``node_id`` so the swarm loop can match it
            against ``SwarmState.pending_interrupts`` and the caller
            can address replies by member name.
        question: Short human-readable question explaining what
            decision is needed.
        kind: Discriminator (``"tool_approval"``, ``"route_choice"``,
            ``"input_request"``, …). Defaults to ``"generic"``.
        metadata: Kind-specific payload forwarded onto
            :attr:`Interrupt.metadata`. Pass a plain dict; ``None``
            is treated as an empty dict.

    Returns:
        The seeded reply value when the swarm driver is re-firing
        this member after a resume.

    Raises:
        InterruptException: When no reply has been seeded — the
            member is on its first invocation of this tool and the
            swarm loop should park the run.
    """
    if ctx_wrapper.has_swarm_resume_reply():
        return ctx_wrapper.consume_swarm_resume_reply()
    raise InterruptException(
        Interrupt(
            node_id=member_name,
            question=question,
            kind=kind,
            metadata=metadata or {},
        )
    )


__all__ = ["SwarmResume", "request_human_input_in_swarm"]

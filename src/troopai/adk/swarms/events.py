"""Swarm stream events — typed-event vocabulary for ``arun_swarm_streamed``.

Events multiplex on the existing stream (user-confirmed design):
``Runner.arun_swarm_streamed`` yields the same ``StreamEvent`` union
as ``arun_streamed`` plus these swarm-scoped variants. Per-agent
events (``raw_response_event``, ``run_item_stream_event``,
``agent_updated_stream_event``) continue to flow unchanged between
``SwarmTurnStartEvent`` and ``SwarmTurnEndEvent`` boundaries.

Consumers pattern-match on ``isinstance`` (or on the ``type`` field
for serialization)::

    async for ev in result.stream_events():
        if isinstance(ev, SwarmHandoffEvent):
            log.info("%s -> %s: %s", ev.from_agent, ev.to_agent, ev.message)
        elif isinstance(ev, SwarmDoneEvent):
            break
        # else: per-agent or per-run event — handle as before.

All events are frozen dataclasses with required fields — the driver
constructs them with full data; there are no placeholder defaults.
``type`` is a ``Literal`` discriminator with a default constant so
the consumer can switch on it in serialized form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from troopai.adk.graphs.interrupt import Interrupt
    from troopai.adk.swarms.stop_reason import StopReason
    from troopai.adk.types.items.items import RunItem


@dataclass(frozen=True)
class SwarmStartEvent:
    """Emitted once at the beginning of a swarm run.

    Follows the existing stream pattern where run-level lifecycle
    events (e.g. ``agent_updated_stream_event``) are emitted on the
    shared stream before per-turn events begin.

    Attributes:
        entry_agent: Name of the agent that takes the first turn.
        member_names: Names of all member agents in the swarm roster
            (for UI/logging).
        type: Discriminator constant. Always ``"swarm_start"``.
    """

    entry_agent: str
    """Name of the agent that takes the first turn."""

    member_names: tuple[str, ...]
    """Names of all member agents in the swarm roster (for UI/logging)."""

    type: Literal["swarm_start"] = "swarm_start"
    """Discriminator constant. Always ``"swarm_start"``."""


@dataclass(frozen=True)
class SwarmTurnStartEvent:
    """Emitted at the top of each member turn, before the LLM call.

    Paired 1:1 with ``SwarmTurnEndEvent``.

    Attributes:
        agent: Name of the agent whose turn is starting.
        turn: 1-indexed swarm turn number (monotonically increasing).
        type: Discriminator constant. Always ``"swarm_turn_start"``.
    """

    agent: str
    """Name of the agent whose turn is starting."""

    turn: int
    """1-indexed swarm turn number (monotonically increasing)."""

    type: Literal["swarm_turn_start"] = "swarm_turn_start"
    """Discriminator constant."""


@dataclass(frozen=True)
class SwarmHandoffEvent:
    """Emitted when a ``SwarmHandoff`` signal is resolved.

    Fires after the current turn ends and before the next
    ``SwarmTurnStartEvent``. Surfaces the explicit handoff payload so
    observability tools can trace routing decisions without replaying
    the full conversation.

    Attributes:
        from_agent: Name of the agent that emitted the handoff.
        to_agent: Name of the target agent.
        message: Explicit handoff content (``SwarmHandoff.message``).
        type: Discriminator constant. Always ``"swarm_handoff"``.
    """

    from_agent: str
    """Name of the agent that emitted the handoff."""

    to_agent: str
    """Name of the target agent."""

    message: str
    """Explicit handoff content (``SwarmHandoff.message``)."""

    type: Literal["swarm_handoff"] = "swarm_handoff"
    """Discriminator constant."""


@dataclass(frozen=True)
class SwarmTurnEndEvent:
    """Emitted at the end of each member turn, after all per-turn
    ``run_item_stream_event`` items have been emitted.

    Paired 1:1 with ``SwarmTurnStartEvent``.

    Attributes:
        agent: Name of the agent whose turn is ending.
        items: Layer 3 items produced during this turn (frozen tuple
            for event-object immutability).
        type: Discriminator constant. Always ``"swarm_turn_end"``.
    """

    agent: str
    """Name of the agent whose turn is ending."""

    items: tuple[RunItem, ...]
    """Layer 3 items produced during this turn (frozen tuple for
    event-object immutability)."""

    type: Literal["swarm_turn_end"] = "swarm_turn_end"
    """Discriminator constant."""


@dataclass(frozen=True)
class SwarmTurnInterruptEvent:
    """Emitted in place of ``SwarmTurnEndEvent`` when a turn suspends.

    Replaces ``SwarmTurnEndEvent`` for any turn that exits via
    ``InterruptException`` (pure HITL) or ``AgentToolDeferral``
    (nested-agent-defer, lifted to ``NestedAgentInterrupt``). The
    parked :class:`Interrupt` carries the kind, question, and
    metadata the consumer needs to prompt the human; resuming uses
    the same ``SwarmResume`` flow as ``arun_swarm_from_checkpoint``
    via ``arun_swarm_streamed(..., initial_state=..., resume=...)``.

    Attributes:
        agent: Name of the suspended member.
        turn: One-indexed turn number (matches ``state.total_turns``).
        interrupt: The parked interrupt (or
            :class:`~troopai.adk.graphs.interrupt.NestedAgentInterrupt`
            subtype).
        type: Discriminator constant. Always
            ``"swarm_turn_interrupt"``.
    """

    agent: str
    """Name of the suspended member."""

    turn: int
    """One-indexed turn number (matches ``state.total_turns``)."""

    interrupt: Interrupt
    """The parked interrupt (or :class:`NestedAgentInterrupt` subtype)."""

    type: Literal["swarm_turn_interrupt"] = "swarm_turn_interrupt"
    """Discriminator constant."""


@dataclass(frozen=True)
class SwarmDoneEvent:
    """Emitted exactly once, at the end of a swarm run.

    Follows the last ``SwarmTurnEndEvent``. Carries the same
    ``StopReason`` surfaced on ``SwarmRunResult.stop_reason``.

    Attributes:
        reason: Why the swarm stopped.
        final_output: The terminal agent's final output. ``None`` if
            the swarm ended via a non-``ExplicitDoneTermination``
            condition (e.g. max_turns).
        type: Discriminator constant. Always ``"swarm_done"``.
    """

    reason: StopReason
    """Why the swarm stopped."""

    final_output: Any
    """The terminal agent's final output. ``None`` if the swarm ended
    via a non-``ExplicitDoneTermination`` condition (e.g. max_turns)."""

    type: Literal["swarm_done"] = "swarm_done"
    """Discriminator constant."""


SwarmEvent = (
    SwarmStartEvent
    | SwarmTurnStartEvent
    | SwarmHandoffEvent
    | SwarmTurnEndEvent
    | SwarmTurnInterruptEvent
    | SwarmDoneEvent
)
"""Discriminated union of all swarm-scoped stream events."""

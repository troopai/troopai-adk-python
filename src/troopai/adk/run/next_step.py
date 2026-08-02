"""NextStep discriminated union for agent loop control.

Inspired by OpenAI's agents SDK, this module replaces inline if/else
loop control with a clean discriminated union.  Each variant represents
a distinct outcome of processing an LLM response:

- ``NextStepFinalOutput`` — the agent produced its answer.
- ``NextStepHandoff`` — the agent requested a handoff to another agent.
- ``NextStepRunAgain`` — tools executed, send results back to the LLM.
- ``NextStepInterruption`` — a tool requires human approval (HITL).
- ``NextStepSwarmYield`` — the agent called a swarm-injected tool
  (``transfer_to_<member>`` or ``swarm_done``); hand control back to
  the swarm driver.

The loop body becomes a simple match statement over these variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.handoffs import Handoff, HandoffTarget
    from troopai.adk.run.state import RunState
    from troopai.adk.swarms.yield_signal import SwarmYieldSignal
    from troopai.adk.tools.deferred_tool import DeferredToolRequests
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.output import FunctionToolCallResult


@dataclass(frozen=True)
class NextStepFinalOutput:
    """Agent produced final output — loop should terminate.

    Attributes:
        output: The agent's final output (str or structured).
    """

    output: Any
    """The agent's final output (str or structured)."""


@dataclass(frozen=True)
class NextStepHandoff:
    """Agent requested handoff — switch agents and continue.

    Attributes:
        new_agent: The agent to hand off to.
        new_messages: The prepared message list for the new agent.
        context_end: New context boundary after handoff preparation.
        target: The handoff target (for hooks and observability).
        is_deterministic: Whether this was a code-orchestrated
            (deterministic) handoff rather than an LLM-chosen one.
    """

    new_agent: Agent
    """The agent to hand off to."""

    new_messages: list[LLMInputContentItem]
    """The prepared message list for the new agent."""

    context_end: int
    """New context boundary after handoff preparation."""

    target: HandoffTarget | Handoff
    """The handoff target (for hooks and observability)."""

    is_deterministic: bool = False
    """Whether this was a code-orchestrated (deterministic) handoff."""


@dataclass(frozen=True)
class NextStepRunAgain:
    """Tools executed — send results back to the LLM.

    Attributes:
        tool_choice_override: If set, overrides the agent's
            ``tool_choice`` for the next LLM call. ``None`` means no
            override (the agent's configured choice applies).
    """

    tool_choice_override: str | None = None
    """If set, override the agent's tool_choice for the next LLM call."""


@dataclass(frozen=True)
class NextStepInterruption:
    """HITL interruption — pause for human approval.

    Attributes:
        deferred: The deferred tool requests awaiting approval or
            external execution.
        state: Serializable :class:`~troopai.adk.run.state.RunState`
            for resumption.
        completed_tool_results: Tool results that completed before the
            interruption (same batch as the deferred calls).
    """

    deferred: DeferredToolRequests
    """The deferred tool requests awaiting approval."""

    state: RunState
    """Serializable state for resumption."""

    completed_tool_results: list[FunctionToolCallResult] = field(default_factory=list)
    """Tool results that completed before the interruption (same batch)."""


@dataclass(frozen=True)
class NextStepSwarmYield:
    """Agent called a swarm-injected tool — yield control to the swarm driver.

    Emitted by ``turn_resolution`` when the active agent's turn produced
    a ``transfer_to_<member>`` call (a :class:`SwarmHandoff` signal) or
    a ``swarm_done`` call (a :class:`SwarmDone` signal) from the tool
    list the swarm policy injected for this turn.

    The single-agent runner loop surfaces this variant up to the swarm
    driver in ``run/swarm_loop.py`` (Wave 3) rather than re-entering
    itself — the driver is responsible for advancing ``SwarmState``,
    running termination checks, preparing the next turn's input, and
    invoking the runner for the next member.

    Distinct from :class:`NextStepHandoff`: ``NextStepHandoff`` applies
    to the single-agent ``Handoff`` / ``HandoffRoute`` mechanism where
    the target is known to the runner. ``NextStepSwarmYield`` delegates
    target resolution and member-graph traversal to the swarm driver.

    Attributes:
        signal: The signal that terminated the turn (``SwarmHandoff``
            or ``SwarmDone``).
        context_end: New context boundary after this turn's output was
            appended. Used by the swarm driver to slice shared history
            for the next turn's input preparation — mirrors
            :attr:`NextStepHandoff.context_end`.
    """

    signal: SwarmYieldSignal
    """The signal that terminated the turn (``SwarmHandoff`` or
    ``SwarmDone``)."""

    context_end: int
    """New context boundary after this turn's output was appended.
    Used by the swarm driver to slice shared history for the next
    turn's input preparation — mirrors :attr:`NextStepHandoff.context_end`."""


NextStep = Union[
    NextStepFinalOutput,
    NextStepHandoff,
    NextStepRunAgain,
    NextStepInterruption,
    NextStepSwarmYield,
]
"""Discriminated union for agent loop control flow."""

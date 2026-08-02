"""Swarm routing policies — who speaks next, and what tools do they see?

Three production-ready strategies plus an escape hatch, mirroring the
existing two handoff strategies (code-orchestrated ``HandoffRoute`` vs
LLM-orchestrated ``Handoff`` list):

- :class:`LLMHandoffPolicy` — AutoGen/Strands parity. The LLM picks the
  next agent by calling an injected ``transfer_to_<member>`` tool.
- :class:`RoundRobinPolicy` — deterministic rotation. Zero LLM routing
  tokens. Ideal for debates and fixed pipelines.
- :class:`StructuredRoutingPolicy` — the TroopAI differentiator. Wraps a
  :class:`~troopai.adk.handoffs.HandoffRoute`; the active agent's
  structured ``Intent`` output dispatches via ``.when(X).to(agent)``.
  Zero LLM routing tokens, full type safety.
- :class:`CustomPolicy` — bring your own ``selector`` callable.

Tools injected by a policy are merged with ``agent.tools`` at turn
dispatch time. **Agent config is never mutated.** That is load-bearing:
a swarm member can be unit-tested in isolation with no swarm awareness
because the swarm behaviour lives on the policy, not the agent.

Every policy also injects a ``swarm_done`` tool so any agent can
terminate the swarm explicitly. Never by absence — absence is not a
stop signal (see :class:`~troopai.adk.swarms.termination.TerminationCondition`).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, override

from pydantic import BaseModel, Field

from troopai.adk.handoffs.handoff import HANDOFF_TOOL_PREFIX
from troopai.adk.swarms.yield_signal import SWARM_DONE_TOOL_NAME, SwarmYieldSignal
from troopai.adk.tools.function_tool import FunctionTool

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.handoffs.handoff_route import HandoffRoute
    from troopai.adk.run.context import RunContext
    from troopai.adk.swarms.state import SwarmState


logger = logging.getLogger(__name__)


TContext = TypeVar("TContext")


# ---------------------------------------------------------------------------
# Tool argument schemas for the policy-injected tools
# ---------------------------------------------------------------------------


class _SwarmTransferArgs(BaseModel):
    """Payload for a ``transfer_to_<member>`` tool call.

    Single ``message`` field carries any explicit instruction the emitting
    agent wants the target to see. Interpreted by
    :class:`SharedContextStrategy.SCOPED` as the *only* cross-agent
    content forwarded — the target does not see the emitter's history by
    default. Developers who want broader context pick a different
    :class:`~troopai.adk.swarms.shared_context_strategy.SharedContextStrategy`.
    """

    message: str = Field(
        ...,
        description=(
            "Explicit content for the target agent. In SCOPED shared-context "
            "mode (default) this is the only message the target receives; "
            "nothing else from the emitter's turn is forwarded."
        ),
    )


class _SwarmDoneArgs(BaseModel):
    """Payload for the ``swarm_done`` termination tool.

    The ``reason`` field lands on
    :class:`~troopai.adk.swarms.stop_reason.StopReason.detail` and on the
    ``SwarmDoneEvent`` surfaced to stream consumers. This is the canonical
    way a swarm terminates — the driver does NOT guess termination from
    absence of a transfer.
    """

    reason: str = Field(
        ...,
        max_length=512,
        description=(
            "Human-readable explanation for why the swarm is done. Surfaces "
            "on SwarmRunResult.stop_reason.detail and SwarmDoneEvent. Must "
            "not exceed 512 characters — the tool call will be rejected with "
            "a validation error and the LLM will be asked to retry with a "
            "shorter reason. The cap bounds PII/token blast radius on "
            "accidental dumps, it does NOT silently truncate."
        ),
    )


def _build_swarm_done_tool() -> FunctionTool:
    """Construct the ``swarm_done`` tool injected by every policy.

    Mirrors ``Handoff.to_tool()``: no ``on_invoke`` is set because the
    swarm driver detects the tool by name in the turn-resolution path
    (``turn_resolution.py``) and converts it to a
    :class:`~troopai.adk.swarms.yield_signal.SwarmDone` yield — exactly
    analogous to how handoff tools short-circuit into a
    :class:`~troopai.adk.run.next_step.NextStepHandoff`.

    Returns:
        A :class:`~troopai.adk.tools.function_tool.FunctionTool` named
        ``swarm_done`` with the :class:`_SwarmDoneArgs` schema.
    """
    return FunctionTool(
        name=SWARM_DONE_TOOL_NAME,
        description=(
            "Call this to terminate the swarm explicitly when your "
            "contribution is complete and no further peer work is needed. "
            "Provide a concise `reason`. The swarm will stop immediately — "
            "no further agents are given a turn."
        ),
        schema=_SwarmDoneArgs,
    )


def _build_transfer_tool(target_name: str, description: str | None = None) -> FunctionTool:
    """Construct a ``transfer_to_<target_name>`` tool for LLM-handoff routing.

    Uses the same name prefix as :mod:`troopai.adk.handoffs` —
    ``HANDOFF_TOOL_PREFIX = "transfer_to_"`` — so consumers who already
    pattern-match on the prefix (e.g. verbose rendering, tracing) light
    up automatically without a new branch.

    Note: detection in the swarm driver MUST key on the tool list the
    *policy* injected for this turn — not on the name prefix alone —
    because a non-swarm agent could also have a real handoff tool of the
    same name. The driver carries the injected list forward for
    unambiguous dispatch.

    ``target_name`` MUST already be slug-safe — ``Swarm.__post_init__``
    validates member names against ``^[a-z0-9_]+$`` precisely so no
    mangling is needed here. An exact equality compare against
    ``m.name`` in the driver then round-trips without drift.

    Args:
        target_name: Name of the target swarm member. MUST be slug-safe
            (``[a-z0-9_]+``).
        description: Optional routing hint from
            :attr:`Swarm.handoff_descriptions` telling the LLM *when* to
            pick this target. Falls back to a generic description.

    Returns:
        A :class:`~troopai.adk.tools.function_tool.FunctionTool` named
        ``transfer_to_<target_name>`` with the
        :class:`_SwarmTransferArgs` schema.
    """
    if description is None:
        description = (
            f"Hand the swarm conversation to the '{target_name}' agent. "
            "Provide a `message` field with explicit instruction or context "
            "for that agent. The transfer is handled in the background — "
            "do not mention it to the user."
        )
    return FunctionTool(
        name=f"{HANDOFF_TOOL_PREFIX}{target_name}",
        description=description,
        schema=_SwarmTransferArgs,
    )


# ---------------------------------------------------------------------------
# Policy ABC
# ---------------------------------------------------------------------------


class SwarmPolicy[TContext](ABC):
    """Swarm routing policy — decides who speaks next and what tools they see.

    Policies are **pure config**: they do not execute turns, do not call
    the LLM, do not mutate agents. The swarm driver calls:

    1. ``build_step_tools(state)`` at the start of each turn to gather
       the extra ``FunctionTool`` instances for this turn. These are
       merged with ``state.current_agent.tools`` at dispatch time.
    2. ``record_yield(state, signal)`` when a turn ends with a
       :class:`~troopai.adk.swarms.yield_signal.SwarmYieldSignal`, so the
       policy can update its internal routing state if any.
    3. ``select_next(state, context)`` after the yield is recorded and
       any termination check has passed, to pick the agent for the next
       turn.

    The driver — not the policy — is responsible for:

    - Running each turn via the existing runner loop.
    - Converting the injected tool calls into
      :class:`~troopai.adk.swarms.yield_signal.SwarmYieldSignal` values.
    - Incrementing ``state.handoff_count`` / ``state.total_turns``.
    - Preparing per-turn input via the
      :class:`~troopai.adk.swarms.shared_context_strategy.SharedContextStrategy`.

    Subclasses MUST be importable and constructible without live runner
    state — they are config objects that round-trip through
    ``SwarmConfig``-style validation.

    Type Parameters:
        TContext: The user-provided context type. Matches the Swarm's
            ``TContext`` and flows through ``RunContext[TContext]`` on
            ``select_next``.
    """

    @abstractmethod
    async def select_next(
        self,
        state: SwarmState[TContext],
        context: RunContext[TContext],
    ) -> Agent[TContext]:
        """Return the agent that takes the *next* turn.

        Called after a turn completes and any ``record_yield`` /
        termination checks have run. A policy that routes via an LLM
        handoff tool (``LLMHandoffPolicy``) typically just returns
        ``state.current_agent`` because the driver has already called
        ``state.advance_to(target)`` before invoking ``select_next``.
        Deterministic policies (``RoundRobinPolicy``) compute the next
        agent from ``state.total_turns`` and the roster.

        Async so :class:`StructuredRoutingPolicy` can call
        :meth:`~troopai.adk.handoffs.handoff_route.HandoffRoute.resolve`,
        which is async.

        Args:
            state: Current swarm state after the turn completed and
                ``record_yield`` was called.
            context: The run context wrapper flowing through this run.

        Returns:
            The agent whose turn is next.
        """

    @abstractmethod
    def build_step_tools(
        self,
        state: SwarmState[TContext],
    ) -> list[FunctionTool]:
        """Return the tools to inject for the *current* turn.

        Merged with ``state.current_agent.tools`` at dispatch time. The
        agent is never mutated — the merged list is scoped to this one
        turn. Typically includes the policy-specific routing tools (e.g.
        ``transfer_to_<name>``) and the universal ``swarm_done`` tool.

        Args:
            state: Current swarm state (``state.current_agent`` is the
                agent about to take this turn).

        Returns:
            List of :class:`~troopai.adk.tools.function_tool.FunctionTool`
            instances to merge into the current agent's tool list for
            this turn only.
        """

    def record_yield(
        self,
        state: SwarmState[TContext],
        signal: SwarmYieldSignal,
    ) -> None:
        """Optional hook called when a turn emits a yield signal.

        Default: no-op. Override if the policy has internal routing
        state (e.g. an index counter) that needs updating on yields.

        Args:
            state: Current swarm state at the moment the yield is
                recorded.
            signal: The :class:`~troopai.adk.swarms.yield_signal.SwarmYieldSignal`
                emitted by the turn (``SwarmHandoff`` or ``SwarmDone``).
        """
        del state, signal


# ---------------------------------------------------------------------------
# LLMHandoffPolicy — AutoGen/Strands parity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMHandoffPolicy(SwarmPolicy[TContext]):
    """LLM picks the next agent by calling a ``transfer_to_<name>`` tool.

    For each turn, injects:

    - One ``transfer_to_<member>`` tool per *other* member of the swarm
      (the active agent cannot transfer to itself).
    - The universal ``swarm_done`` tool.

    When the LLM calls a transfer tool, the swarm driver emits a
    :class:`~troopai.adk.swarms.yield_signal.SwarmHandoff`, advances
    ``state.current_agent`` via ``state.advance_to(target)``, and then
    calls ``record_yield`` (a no-op by default). Because the driver has
    already moved the pointer, ``select_next`` simply returns the current
    agent.

    Matches AutoGen's ``Swarm`` and Strands' ``Swarm`` semantics, with
    three production improvements:

    1. **No tool injection at agent construction time.** The merge
       happens in the driver for a single turn. The ``Agent`` is
       immutable.
    2. **Explicit done signal.** ``swarm_done`` is the only way the
       swarm terminates via an LLM decision. Absence of a tool call is
       not a stop signal — see
       :class:`~troopai.adk.swarms.termination.ExplicitDoneTermination`.
    3. **No hidden system prompt.** Developers who want the LLM to know
       about transfer semantics opt in via
       :func:`~troopai.adk.swarms.swarm_prompt.prompt_with_swarm_instructions`.
    """

    @override
    def build_step_tools(
        self,
        state: SwarmState[TContext],
    ) -> list[FunctionTool]:
        tools: list[FunctionTool] = []
        current_name = state.current_agent_name
        descriptions = state.swarm.handoff_descriptions
        for member in state.swarm.members:
            if member.name == current_name:
                continue
            tools.append(_build_transfer_tool(member.name, descriptions.get(member.name)))
        tools.append(_build_swarm_done_tool())
        return tools

    @override
    async def select_next(
        self,
        state: SwarmState[TContext],
        context: RunContext[TContext],
    ) -> Agent[TContext]:
        # The driver advances state.current_agent on SwarmHandoff resolution
        # (before select_next is called). If the last yield was a SwarmDone
        # the driver would have terminated via the TerminationCondition
        # *before* calling us. If no yield occurred this turn (rare — the
        # LLM neither transferred nor called swarm_done), the same agent
        # takes another turn until the termination condition fires.
        del context
        return state.current_agent


# ---------------------------------------------------------------------------
# RoundRobinPolicy — deterministic, zero routing tokens
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundRobinPolicy(SwarmPolicy[TContext]):
    """Deterministic rotation through the member roster.

    Consumes zero LLM routing tokens — the next agent is computed from
    ``state.total_turns`` modulo the rotation length. Still injects the
    ``swarm_done`` tool so any agent can terminate explicitly (the only
    way this policy stops other than via a :class:`TerminationCondition`
    like :class:`MaxTurnsTermination`).

    Useful for debates, fixed pipelines, and deterministic test
    fixtures. Often paired with ``MaxTurnsTermination`` to cap the
    conversation length.

    Attributes:
        order: Optional rotation order by agent name. ``None`` (default)
            uses the ``Swarm.members`` order. If set, it MUST be non-empty
            and every name MUST be in ``Swarm.members`` (an unknown name
            surfaces a clear ``ValueError`` from ``select_next``).
    """

    order: tuple[str, ...] | None = None
    """Rotation order by agent name. ``None`` uses ``Swarm.members`` order."""

    def __post_init__(self) -> None:
        """Reject an empty explicit rotation order at construction time.

        An empty ``order`` is a degenerate configuration: the rotation
        length is zero, so the modulo in ``select_next`` would raise an
        opaque ``ZeroDivisionError`` mid-run. Fail fast here with a clear
        message instead. Pass ``order=None`` to fall back to the
        ``Swarm.members`` order.
        """
        if self.order is not None and len(self.order) == 0:
            raise ValueError(
                "RoundRobinPolicy.order must be non-empty when set. Pass order=None to use the Swarm.members order."
            )

    @override
    def build_step_tools(
        self,
        state: SwarmState[TContext],
    ) -> list[FunctionTool]:
        del state
        return [_build_swarm_done_tool()]

    @override
    async def select_next(
        self,
        state: SwarmState[TContext],
        context: RunContext[TContext],
    ) -> Agent[TContext]:
        del context
        rotation = self._rotation_names(state)
        # total_turns is 1-indexed (incremented at the top of each turn).
        # The entry agent already served turn 1, so the *next* agent
        # must be offset by where the entry sits in the rotation:
        #   next_idx = (total_turns + start_idx) % len(rotation)
        # This ensures that when entry == members[0] (start_idx=0) the
        # formula reduces to the original total_turns % len(rotation),
        # and when entry == members[k] the rotation starts at k so no
        # member is repeated or skipped on the first handoff.
        entry_name = state.swarm.entry.name
        try:
            start_idx = rotation.index(entry_name)
        except ValueError:
            # entry not in rotation — fall back to 0 (no offset). A valid
            # config keeps the entry inside the rotation; this branch only
            # absorbs an entry that was deliberately excluded from `order`.
            start_idx = 0
        next_idx = (state.total_turns + start_idx) % len(rotation)
        next_name = rotation[next_idx]
        for member in state.swarm.members:
            if member.name == next_name:
                return member
        raise ValueError(
            f"RoundRobinPolicy.order contains '{next_name}' but no member "
            f"with that name exists in the swarm (members: "
            f"{[m.name for m in state.swarm.members]})."
        )

    def _rotation_names(self, state: SwarmState[TContext]) -> tuple[str, ...]:
        """Return the rotation order, falling back to member order.

        Args:
            state: Current swarm state providing ``swarm.members`` when
                ``order`` is ``None``.

        Returns:
            The effective rotation order as a tuple of agent names.
        """
        if self.order is not None:
            return self.order
        return tuple(m.name for m in state.swarm.members)


# ---------------------------------------------------------------------------
# StructuredRoutingPolicy — the TroopAI differentiator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredRoutingPolicy(SwarmPolicy[TContext]):
    """Intent-based routing via the existing ``HandoffRoute`` DSL.

    Zero LLM routing tokens. The current agent's structured output must
    be an :class:`~troopai.adk.types.intents.Intent`; the policy reads
    the most recent message-output item, resolves the intent against
    the wrapped route, and dispatches to ``.when(IntentType).to(agent)``.

    This is the TroopAI differentiator: full type safety, no routing
    prompt bloat, and reuse of the ``.when(Refund).to(refunds_agent)``
    DSL that already exists for handoff routing. Most frameworks (and
    AutoGen/Strands) require an LLM tool call even for deterministic
    routing; TroopAI's Swarm lets structured output do the dispatch.

    Only the ``swarm_done`` tool is injected — structured-output
    routing doesn't need transfer tools because the routing signal is
    the output schema itself.

    Attributes:
        route: The :class:`HandoffRoute` to consult on each turn. MUST
            cover (via ``.when`` + ``.otherwise``) every intent the
            active agents can produce, or
            :class:`~troopai.adk.handoffs.handoff_route.UnhandledIntentError`
            will be raised at runtime.
    """

    route: HandoffRoute[Any, TContext]
    """The route to consult for intent → agent dispatch."""

    @override
    def build_step_tools(
        self,
        state: SwarmState[TContext],
    ) -> list[FunctionTool]:
        del state
        return [_build_swarm_done_tool()]

    @override
    async def select_next(
        self,
        state: SwarmState[TContext],
        context: RunContext[TContext],
    ) -> Agent[TContext]:
        # The swarm driver captures the parsed structured output from the
        # just-completed turn onto state.last_structured_output. This is
        # analogous to how `resolve_structured_output_step` reads
        # `response.content` in the single-agent loop — same source of
        # truth, exposed to the policy layer.
        from troopai.adk.types.intents import Intent, Respond

        intent = state.last_structured_output
        if intent is None:
            # The active agent produced no structured output this turn.
            # Typical when the triage agent has routed to a specialist
            # (no `output_schema`) — the specialist is expected to call
            # `swarm_done` or be stopped by a `TerminationCondition`.
            # Returning the current agent avoids a crash loop and keeps
            # termination in charge of ending the run.
            return state.current_agent

        # Respond is a sibling of Intent in the union — it signals "no
        # handoff, agent responded directly." Mirrors HandoffRoute.resolve,
        # which short-circuits on isinstance(intent, Respond).
        if isinstance(intent, Respond):
            return state.current_agent

        if not isinstance(intent, Intent):
            raise TypeError(
                f"StructuredRoutingPolicy expects the active agent's "
                f"structured output to be an Intent or Respond, got "
                f"{type(intent).__name__}. Update the agent's "
                "`output_schema` to an Intent subclass (or include "
                "`Respond` in the union for direct replies)."
            )

        handoff_target = await self.route.resolve(intent, context=context)
        if handoff_target is None:
            # .otherwise fallback returned None (possible when callable
            # enablement gates rejected the otherwise target). Keep the
            # current agent.
            return state.current_agent

        return handoff_target.target


# ---------------------------------------------------------------------------
# CustomPolicy — escape hatch
# ---------------------------------------------------------------------------


SwarmSelector = Callable[["SwarmState[Any]"], str]
"""Selector signature: takes the current state, returns the next agent's name."""

SwarmExtraToolsFn = Callable[["SwarmState[Any]"], list[FunctionTool]]
"""Extra-tools signature: takes the current state, returns tools for this turn.

The returned tools are merged with the universal ``swarm_done`` tool by
:class:`CustomPolicy`. Developers do not need to include ``swarm_done``
themselves.
"""


@dataclass(frozen=True)
class CustomPolicy(SwarmPolicy[TContext]):
    """Bring-your-own routing logic.

    Escape hatch for advanced patterns not covered by the three built-in
    policies (e.g. weighted rotation, probabilistic routing, external
    orchestration via context state). No magic — the developer owns the
    full selection logic.

    ``swarm_done`` is always injected; additional turn-scoped tools can
    be produced via ``extra_tools_fn``.

    Attributes:
        selector: Returns the name of the agent for the next turn.
            Synchronous — for async routing, wrap with asyncio primitives
            in a subclass.
        extra_tools_fn: Optional. Returns per-turn tools on top of
            ``swarm_done``. When ``None``, only ``swarm_done`` is
            injected.
    """

    selector: SwarmSelector
    """Returns the next agent's name from the current state."""

    extra_tools_fn: SwarmExtraToolsFn | None = field(default=None)
    """Optional extra tools per turn (beyond ``swarm_done``)."""

    @override
    def build_step_tools(
        self,
        state: SwarmState[TContext],
    ) -> list[FunctionTool]:
        tools: list[FunctionTool] = []
        if self.extra_tools_fn is not None:
            tools.extend(self.extra_tools_fn(state))
        tools.append(_build_swarm_done_tool())
        return tools

    @override
    async def select_next(
        self,
        state: SwarmState[TContext],
        context: RunContext[TContext],
    ) -> Agent[TContext]:
        del context
        next_name = self.selector(state)
        for member in state.swarm.members:
            if member.name == next_name:
                return member
        raise ValueError(
            f"CustomPolicy.selector returned '{next_name}' but no member "
            f"with that name exists in the swarm (members: "
            f"{[m.name for m in state.swarm.members]})."
        )

"""Termination conditions — composable predicates that decide when a
swarm run stops.

Steals the elegant primitive from Microsoft AutoGen: an ABC with
``__and__`` / ``__or__`` operator overloads so conditions compose
algebraically::

    termination = (
        MaxTurnsTermination(20)
        | TokenBudgetTermination(100_000)
        | ExplicitDoneTermination()
        | HandoffToTermination("user")
    )

The swarm driver calls ``termination.should_stop(state)`` at the top
of every turn before selecting the next agent. A non-None return
signals "stop here" and carries the ``StopReason`` surfaced on
``SwarmRunResult.stop_reason``.

**Never terminate by absence.** The driver never decides "the agent
forgot to hand off, so the swarm is done." It keeps running until
this predicate says stop or a ``SwarmConfig`` hard guard trips.
That rule is what prevents Strands' silent-false-complete failure
mode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from troopai.adk.swarms.stop_reason import StopReason
from troopai.adk.swarms.yield_signal import SwarmDone, SwarmHandoff
from troopai.adk.types.items.items import ItemHelpers, MessageOutputItem

if TYPE_CHECKING:
    from troopai.adk.swarms.state import SwarmState


class TerminationCondition(ABC):
    """ABC for swarm termination predicates.

    Subclasses implement ``should_stop(state)`` returning either a
    ``StopReason`` (stop) or ``None`` (keep running). Operator
    overloads compose conditions with short-circuit semantics:

    - ``a | b`` — stop when either a or b says stop.
    - ``a & b`` — stop when both a and b say stop (rare but useful
      for debounce-style conditions).

    The driver always queries ``should_stop`` before selecting the
    next agent. A non-None return ends the run immediately.
    """

    @abstractmethod
    def should_stop(self, state: SwarmState) -> StopReason | None:
        """Return a ``StopReason`` to stop the swarm, or ``None`` to keep going.

        Args:
            state: The current swarm state, inspected for progress
                counters, yield signals, and usage figures.

        Returns:
            A :class:`~troopai.adk.swarms.stop_reason.StopReason`
            when this condition fires, ``None`` to continue the run.
        """

    def validate_roster(self, member_names: tuple[str, ...]) -> None:
        """Fail-fast check against the swarm roster. Default: no-op.

        Called by ``Swarm.__post_init__`` so conditions that reference
        member names (e.g. :class:`TextMentionTermination`) reject
        typos at construction time — fail at compile time, not at run
        time. Composites recurse into their children.

        Args:
            member_names: Names of the swarm's members.

        Raises:
            ValueError: If the condition references a name not in
                ``member_names``.
        """
        del member_names

    def __or__(self, other: TerminationCondition) -> TerminationCondition:
        return OrTermination(self, other)

    def __and__(self, other: TerminationCondition) -> TerminationCondition:
        return AndTermination(self, other)


@dataclass(frozen=True)
class OrTermination(TerminationCondition):
    """Stops when either child condition fires. First match wins.

    Returned by ``TerminationCondition.__or__``. Public because it is
    part of the observable result of composing conditions — callers
    that introspect ``termination`` (e.g. for logging, or to walk a
    composite tree) need a stable, non-underscored name.

    Attributes:
        left: The first condition to check.
        right: The second condition to check.
    """

    left: TerminationCondition
    right: TerminationCondition

    @override
    def should_stop(self, state: SwarmState) -> StopReason | None:
        result = self.left.should_stop(state)
        if result is not None:
            return result
        return self.right.should_stop(state)

    @override
    def validate_roster(self, member_names: tuple[str, ...]) -> None:
        self.left.validate_roster(member_names)
        self.right.validate_roster(member_names)


@dataclass(frozen=True)
class AndTermination(TerminationCondition):
    """Stops only when both child conditions fire in the same call.

    Returned by ``TerminationCondition.__and__``. Public for the same
    reason as :class:`OrTermination` — introspection of a composed
    termination tree needs a stable name.

    Attributes:
        left: The first condition to check.
        right: The second condition to check.
    """

    left: TerminationCondition
    right: TerminationCondition

    @override
    def should_stop(self, state: SwarmState) -> StopReason | None:
        left = self.left.should_stop(state)
        if left is None:
            return None
        right = self.right.should_stop(state)
        if right is None:
            return None
        # Prefer the more specific (left) reason when both fire.
        return left

    @override
    def validate_roster(self, member_names: tuple[str, ...]) -> None:
        self.left.validate_roster(member_names)
        self.right.validate_roster(member_names)


@dataclass(frozen=True)
class MaxTurnsTermination(TerminationCondition):
    """Stop after ``n`` member turns have completed.

    Distinct from ``RunConfig.max_total_turns`` (which counts LLM
    calls across all agents and raises ``MaxTurnsExceeded`` as a
    hard safety net). This condition emits a clean ``StopReason`` and
    produces a proper ``SwarmRunResult``; ``max_total_turns`` raises.

    Attributes:
        limit: Maximum number of member turns allowed before the swarm
            stops.
    """

    limit: int

    @override
    def should_stop(self, state: SwarmState) -> StopReason | None:
        if state.total_turns >= self.limit:
            return StopReason(
                kind="max_turns",
                detail=f"Completed {state.total_turns}/{self.limit} swarm turns.",
            )
        return None


@dataclass(frozen=True)
class TokenBudgetTermination(TerminationCondition):
    """Stop when cumulative tokens meet or exceed ``limit``.

    Compares against ``state.cumulative_usage.total_tokens`` which is
    updated at the end of each turn with the per-turn delta. Same
    source of truth as ``SwarmConfig.max_total_tokens``, but emits a
    clean ``StopReason`` rather than a hard-guard trip.

    Attributes:
        limit: Cumulative token cap (input + output across all member
            turns). The swarm stops when this threshold is met or
            exceeded.
    """

    limit: int

    @override
    def should_stop(self, state: SwarmState) -> StopReason | None:
        consumed = state.cumulative_usage.total_tokens
        if consumed >= self.limit:
            return StopReason(
                kind="token_budget",
                detail=f"Consumed {consumed}/{self.limit} tokens.",
            )
        return None


@dataclass(frozen=True)
class HandoffToTermination(TerminationCondition):
    """Stop when the most recent handoff targets ``target_name``.

    Classic use: ``HandoffToTermination("user")`` to pause a swarm
    for HITL when any member emits ``SwarmHandoff(target="user", ...)``.
    The target need not be a member of the swarm — in fact this
    condition is usually paired with a pseudo-target that is not in
    the roster, so the driver cannot resolve it as a real transfer.

    Attributes:
        target_name: The handoff target name that triggers termination.
            Matched against ``state.last_yield.target`` (exact string
            equality).
    """

    target_name: str

    @override
    def should_stop(self, state: SwarmState) -> StopReason | None:
        y = state.last_yield
        if isinstance(y, SwarmHandoff) and y.target == self.target_name:
            # Intentionally omit ``y.message`` from the detail string:
            # the handoff payload is LLM-produced and may carry PII,
            # secrets, or prompt-injection. ``StopReason.detail``
            # propagates into logs, hooks, and any serialized result.
            # The message body is available on ``state.last_yield`` for
            # consumers that need it.
            return StopReason(
                kind="handoff_to",
                detail=f"Swarm handed off to {self.target_name!r}.",
            )
        return None


@dataclass(frozen=True)
class ExplicitDoneTermination(TerminationCondition):
    """Stop when the most recent yield is a ``SwarmDone`` signal.

    This is the canonical way a swarm terminates: an agent calls the
    ``swarm_done(reason)`` tool, ``turn_resolution`` produces a
    ``SwarmDone`` yield, and this condition fires. No silent-
    termination: absence of a tool call is not a stop signal.
    """

    @override
    def should_stop(self, state: SwarmState) -> StopReason | None:
        y = state.last_yield
        if isinstance(y, SwarmDone):
            return StopReason(kind="explicit_done", detail=y.reason)
        return None


@dataclass(frozen=True)
class TextMentionTermination(TerminationCondition):
    """Stop when an agent-produced message contains ``phrase``.

    Mirrors AutoGen's ``TextMentionTermination`` — a pragmatic escape
    hatch for flows where asking the LLM to call ``swarm_done`` is
    overkill (e.g. a debate that ends when a judge writes "VERDICT:").
    Only :class:`MessageOutputItem` text is scanned, so user input and
    tool payloads never trigger it.

    Explicit ``swarm_done`` remains the recommended primary stop
    signal; prefer this only for phrase-based protocols, and always
    compose it with a turn/token safety net::

        TextMentionTermination("VERDICT:") | MaxTurnsTermination(20)

    Attributes:
        phrase: The substring to look for (non-empty). Case-insensitive
            by default.
        member: Optional member name restricting whose messages can
            trigger the stop. ``None`` (default) lets any member
            trigger it. Validated against the roster at
            :class:`Swarm` construction time.
        case_sensitive: When ``True``, match ``phrase`` exactly instead
            of lowercasing both sides.
    """

    phrase: str
    member: str | None = None
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if len(self.phrase) == 0:
            raise ValueError(
                "TextMentionTermination.phrase must be non-empty — an "
                "empty phrase would stop the swarm on the first message."
            )

    @override
    def validate_roster(self, member_names: tuple[str, ...]) -> None:
        if self.member is not None and self.member not in member_names:
            raise ValueError(
                f"TextMentionTermination.member {self.member!r} is not a "
                f"swarm member. Valid members: {list(member_names)}."
            )

    @override
    def should_stop(self, state: SwarmState) -> StopReason | None:
        needle = self.phrase if self.case_sensitive else self.phrase.lower()
        for item in state.shared_history:
            if not isinstance(item, MessageOutputItem):
                continue
            if self.member is not None and item.agent_name != self.member:
                continue
            text = ItemHelpers.text_message_output(item)
            haystack = text if self.case_sensitive else text.lower()
            if needle in haystack:
                # The phrase is developer-supplied (not LLM-produced)
                # and agent_name is a roster name, so both are safe to
                # surface in logs — unlike the HandoffToTermination
                # message payload.
                return StopReason(
                    kind="text_mention",
                    detail=f"Member {item.agent_name!r} mentioned {self.phrase!r}.",
                )
        return None

"""``Swarm`` — the top-level config object that binds a roster, a policy,
a termination condition, and a shared-context strategy.

A ``Swarm`` is pure configuration. It has no ``run()`` method; it does
not hold mutable state. Execution lives in ``Runner.arun_swarm()``
which delegates to the swarm driver in ``troopai.adk.run.swarm_loop`` —
agents are configuration, the Runner is execution.

Construct once, reuse across many runs. Like ``Agent``, a ``Swarm`` is
pure configuration — it holds no mutable state and can be shared across
threads without locking.

Example — fluent builder (preferred, mirrors ``Graph.new``)::

    swarm = (
        Swarm.new("code-review", description="author → reviewer → security")
        .members(author, reviewer, security_auditor)
        .entry("author")
        .llm_handoff()
        .terminate_on(ExplicitDoneTermination() | MaxTurnsTermination(20))
        .compile()
    )
    result = await Runner.arun_swarm(swarm, "Refactor this module.")

Example — direct construction (defaults fill the rest)::

    swarm = Swarm(members=(author, reviewer), entry="author")
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, TypeVar, override

from troopai.adk.agents.agent import Agent
from troopai.adk.handoffs.handoff import HANDOFF_TOOL_PREFIX
from troopai.adk.swarms.config import SwarmConfig
from troopai.adk.swarms.hooks import SwarmHooks
from troopai.adk.swarms.policy import LLMHandoffPolicy, SwarmPolicy
from troopai.adk.swarms.termination import (
    ExplicitDoneTermination,
    MaxTurnsTermination,
    TerminationCondition,
)
from troopai.adk.swarms.yield_signal import SWARM_DONE_TOOL_NAME

if TYPE_CHECKING:
    from troopai.adk.swarms.builder import SwarmBuilder

# Member names are injected into `transfer_to_<name>` tools, so they
# MUST be safe as JSON-schema property names and MUST be slug-stable
# (no case folding, no space→underscore mangling) so detection in the
# driver (m.name == target) is exact.
_MEMBER_NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")
_RESERVED_NAMES = frozenset({SWARM_DONE_TOOL_NAME})


class _Unset:
    """Sentinel type for "argument omitted" (distinct from an explicit ``None``).

    Lets ``Swarm.__init__`` apply defaults when an argument is omitted
    while still rejecting an explicit ``None`` — which historically
    constructed fine and crashed the driver mid-run.
    """

    __slots__ = ()

    @override
    def __repr__(self) -> str:
        return "UNSET"


_UNSET: Final = _Unset()
"""Sentinel marking omitted constructor arguments (see :class:`_Unset`)."""

DEFAULT_MAX_TURNS: int = 25
"""Default member-turn cap used by :data:`DEFAULT_TERMINATION`.

Cost-conservative safety net: high enough that a well-behaved swarm
(the shipped examples cap theirs at 6–20 turns) never trips it, low
enough that a swarm whose agents never call ``swarm_done`` stops long
before the absolute ``RunConfig.max_total_turns`` net (default: 500
LLM calls) would.
"""

DEFAULT_TERMINATION: TerminationCondition = ExplicitDoneTermination() | MaxTurnsTermination(DEFAULT_MAX_TURNS)
"""Default termination: explicit ``swarm_done`` wins; the turn cap is the safety net.

Explicit termination remains the primary contract — the ``swarm_done``
tool must still be called for a clean stop. ``MaxTurnsTermination``
only bounds the damage when no member ever calls it.
"""


TContext = TypeVar("TContext")


@dataclass(frozen=True)
class Swarm[TContext]:
    """Configuration for a multi-agent swarm.

    Binds together the four orthogonal concerns of a swarm:

    - **Who** (``members``, ``entry``) — the roster and the first speaker.
    - **Routing** (``policy``) — how the next speaker is chosen.
    - **Stopping** (``termination``) — when the swarm is done.
    - **Budgets & behaviour** (``config``, ``hooks``) — safety rails and
      lifecycle callbacks.

    A ``Swarm`` is validated at construction (custom ``__init__`` +
    ``__post_init__``):

    - ``members`` is non-empty.
    - ``entry`` is one of ``members`` (passing the member *name* as a
      string is resolved to the member object).
    - Member names are unique (so ``transfer_to_<name>`` and
      ``HandoffRoute`` targets are unambiguous).
    - ``policy`` and ``termination`` are not ``None`` — never terminate
      by absence.

    Attributes:
        members: The tuple of agents participating in the swarm.
            Tuple (not list) for immutability and stable iteration.
            Order determines the default :class:`RoundRobinPolicy`
            rotation.
        entry: The agent that takes the first turn. The constructor
            also accepts the member *name* (resolved against the
            roster); this attribute is always an
            :class:`~troopai.adk.agents.agent.Agent`.
        policy: The routing policy. Default: :class:`LLMHandoffPolicy`
            (LLM picks the next speaker via ``transfer_to_<name>``
            tools). MUST NOT be ``None``.
        termination: The termination condition. Default:
            :data:`DEFAULT_TERMINATION` — explicit ``swarm_done`` or a
            25-turn safety net. MUST NOT be ``None`` (an explicit
            ``None`` raises); explicit termination remains the contract
            — the default merely ships that contract pre-wired. See
            :class:`~troopai.adk.swarms.termination.TerminationCondition`.
        config: Swarm-level budgets and limits. Default:
            :class:`SwarmConfig` with cost-conservative settings.
        hooks: Optional swarm-level lifecycle hooks. Fires alongside
            ``RunHooks`` and ``AgentHooks``.
        name: Optional human-readable swarm name (parity with
            ``Graph.id``). Pure metadata — shown in ``repr()`` and
            available for observability; not used by the driver.
        description: Optional human-readable blurb. Pure metadata.
        handoff_descriptions: Optional per-member descriptions used by
            :class:`LLMHandoffPolicy` when it builds the
            ``transfer_to_<member>`` tools — they tell the routing LLM
            *when* to route to each member (mirrors the OpenAI Agents
            SDK ``handoff_description``). Keys must be member names.

    Type Parameters:
        TContext: The user-provided context type. Flows through
            ``RunContext[TContext]`` the same way it does in
            single-agent runs.
    """

    members: tuple[Agent[TContext], ...]
    """The agent roster. Tuple for immutability and stable iteration."""

    entry: Agent[TContext]
    """The agent that takes the first turn. Construction also accepts the
    member *name* (see ``__init__``); the attribute is always an
    :class:`~troopai.adk.agents.agent.Agent`."""

    policy: SwarmPolicy[TContext] = field(default_factory=LLMHandoffPolicy)
    """Routing policy — who speaks next, what tools do they see?"""

    termination: TerminationCondition = DEFAULT_TERMINATION
    """When to stop. Never ``None``; the default ships explicit
    ``swarm_done`` termination plus a 25-turn safety net."""

    config: SwarmConfig = field(default_factory=SwarmConfig)
    """Swarm-level budgets, timeouts, shared-context strategy."""

    hooks: SwarmHooks[TContext] | None = None
    """Optional swarm-level lifecycle callbacks."""

    name: str | None = None
    """Optional human-readable swarm name. Pure metadata."""

    description: str | None = None
    """Optional human-readable description. Pure metadata."""

    handoff_descriptions: Mapping[str, str] = field(default_factory=dict)
    """Per-member ``transfer_to_<name>`` tool descriptions (keys: member names)."""

    def __init__(
        self,
        *,
        members: Iterable[Agent[TContext]],
        entry: Agent[TContext] | str,
        policy: SwarmPolicy[TContext] | None | _Unset = _UNSET,
        termination: TerminationCondition | None | _Unset = _UNSET,
        config: SwarmConfig | None | _Unset = _UNSET,
        hooks: SwarmHooks[TContext] | None = None,
        name: str | None = None,
        description: str | None = None,
        handoff_descriptions: Mapping[str, str] | None = None,
    ) -> None:
        """Build a swarm. Keyword-only, tolerant inputs, exact attributes.

        Tolerant at the boundary, exact on the instance: ``members``
        accepts any iterable (stored as a tuple), ``entry`` accepts a
        member name or the member object (stored as the object), so
        ``swarm.entry`` is always an ``Agent`` for every consumer.

        Args:
            members: The agent roster (any iterable; stored as a tuple).
            entry: First speaker — the member object or its name. A name
                is resolved against the roster here; unknown names raise
                ``ValueError`` listing the valid names.
            policy: Routing policy. Omit for the default
                :class:`LLMHandoffPolicy`. Passing ``None`` explicitly
                is an error — it used to construct fine and crash the
                driver with ``AttributeError`` mid-run.
            termination: Termination condition. Omit for
                :data:`DEFAULT_TERMINATION`. ``None`` is an error —
                never terminate by absence.
            config: Swarm-level budgets. Omit for the default
                :class:`SwarmConfig`. ``None`` is an error.
            hooks: Optional swarm-level lifecycle hooks. ``None``
                (the default) means no hooks.
            name: Optional human-readable swarm name (metadata only).
            description: Optional human-readable blurb (metadata only).
            handoff_descriptions: Optional per-member routing hints for
                :class:`LLMHandoffPolicy` transfer tools.

        Raises:
            ValueError: On an explicit ``None`` for ``policy`` /
                ``termination`` / ``config``, an unknown ``entry`` name,
                or any roster validation failure (see ``__post_init__``).
        """
        roster = tuple(members)
        if policy is None:
            raise ValueError(
                "Swarm.policy must not be None. Omit the argument to get "
                "the default LLMHandoffPolicy, or pass an explicit policy."
            )
        if termination is None:
            raise ValueError(
                "Swarm.termination must not be None — never terminate by "
                "absence. Omit the argument to get DEFAULT_TERMINATION "
                "(swarm_done or a 25-turn safety net)."
            )
        if config is None:
            raise ValueError("Swarm.config must not be None. Omit the argument to get the default SwarmConfig.")

        # Entry-by-name: resolve a string entry against the roster so
        # definitions read `entry="author"` instead of repeating the
        # agent object. Duplicate names are rejected in __post_init__
        # with the clearer error; resolution takes the first match.
        resolved_entry: Agent[TContext]
        if isinstance(entry, str):
            match = next((m for m in roster if m.name == entry), None)
            if match is None:
                raise ValueError(
                    f"Swarm.entry {entry!r} does not match any member name. Valid members: {[m.name for m in roster]}."
                )
            resolved_entry = match
        else:
            resolved_entry = entry

        object.__setattr__(self, "members", roster)
        object.__setattr__(self, "entry", resolved_entry)
        object.__setattr__(self, "policy", LLMHandoffPolicy() if isinstance(policy, _Unset) else policy)
        object.__setattr__(self, "termination", DEFAULT_TERMINATION if isinstance(termination, _Unset) else termination)
        object.__setattr__(self, "config", SwarmConfig() if isinstance(config, _Unset) else config)
        object.__setattr__(self, "hooks", hooks)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "handoff_descriptions", {} if handoff_descriptions is None else handoff_descriptions)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate the roster, the entry agent, and mandatory fields."""
        if len(self.members) == 0:
            raise ValueError("Swarm.members must be non-empty. A swarm with zero agents has nothing to run.")

        # Name uniqueness: transfer_to_<name> and HandoffRoute targets
        # both key on agent.name, so collisions would route
        # non-deterministically.
        seen: set[str] = set()
        duplicates: list[str] = []
        for m in self.members:
            if m.name in seen:
                duplicates.append(m.name)
            seen.add(m.name)
        if len(duplicates) > 0:
            raise ValueError(
                f"Swarm.members contains duplicate agent names: {duplicates}. "
                "Every member must have a unique name so transfer_to_<name> "
                "and HandoffRoute targets are unambiguous."
            )

        # Entry must be one of the members. Membership is by VALUE
        # equality (Agent is an eq=True dataclass): a distinct-but-equal
        # Agent instance also passes — acceptable, since routing keys on
        # agent.name anyway.
        if self.entry not in self.members:
            raise ValueError(
                f"Swarm.entry ({self.entry.name!r}) must be one of "
                f"Swarm.members (names: {[m.name for m in self.members]})."
            )

        # Slug-safety: member names feed `transfer_to_<name>` injected
        # tools. Enforce a strict pattern so (a) names survive as
        # JSON-schema property keys across providers, (b) no mangling
        # (lowercase / space→underscore) is needed at injection time,
        # (c) exact equality comparison (m.name == target) resolves
        # targets without silent drops, (d) the tool name cannot
        # collide with SWARM_DONE_TOOL_NAME or the transfer_to_ prefix.
        for m in self.members:
            if _MEMBER_NAME_PATTERN.match(m.name) is None:
                raise ValueError(
                    f"Swarm member name {m.name!r} is invalid. Member "
                    "names must match [a-z0-9_]+ so they can be injected "
                    "safely as `transfer_to_<name>` tool identifiers."
                )
            if m.name in _RESERVED_NAMES:
                raise ValueError(
                    f"Swarm member name {m.name!r} is reserved by the "
                    "swarm runtime (collides with SWARM_DONE_TOOL_NAME). "
                    "Pick a different name."
                )
            if m.name.startswith(HANDOFF_TOOL_PREFIX):
                raise ValueError(
                    f"Swarm member name {m.name!r} starts with the "
                    f"handoff tool prefix {HANDOFF_TOOL_PREFIX!r}. Member "
                    "names must not share the injection namespace."
                )
            # Member-owned tools must not shadow swarm-injected tool
            # names. A tool named 'swarm_done' or 'transfer_to_<peer>'
            # on an Agent would let the agent silently intercept its
            # own termination / routing.
            for t in m.tools:
                tool_name = getattr(t, "name", None)
                if tool_name is None:
                    continue
                if tool_name == SWARM_DONE_TOOL_NAME:
                    raise ValueError(
                        f"Member {m.name!r} defines a tool named "
                        f"{SWARM_DONE_TOOL_NAME!r}, which is reserved by the "
                        "swarm runtime for explicit termination."
                    )
                if tool_name.startswith(HANDOFF_TOOL_PREFIX):
                    raise ValueError(
                        f"Member {m.name!r} defines a tool named "
                        f"{tool_name!r} which starts with the handoff "
                        f"prefix {HANDOFF_TOOL_PREFIX!r}. Swarm members "
                        "must not share the injection namespace."
                    )

        # Normalize handoff descriptions into an immutable mapping and
        # reject keys that do not name a member — a typo here would
        # otherwise be silently ignored by LLMHandoffPolicy.
        descriptions = MappingProxyType(dict(self.handoff_descriptions))
        unknown = [k for k in descriptions if k not in seen]
        if len(unknown) > 0:
            raise ValueError(
                f"Swarm.handoff_descriptions keys must be member names; "
                f"unknown: {unknown}. Valid members: {[m.name for m in self.members]}."
            )
        object.__setattr__(self, "handoff_descriptions", descriptions)

        # Roster-aware termination conditions (e.g.
        # TextMentionTermination(member=...)) reject unknown names here
        # instead of silently never firing at run time.
        self.termination.validate_roster(tuple(m.name for m in self.members))

    @override
    def __repr__(self) -> str:
        """Compact, human-readable repr — the full dataclass repr dumps
        every member's system prompt and tools, which is unreadable."""
        parts: list[str] = []
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        parts.append(f"members={len(self.members)}")
        parts.append(f"entry={self.entry.name!r}")
        return f"Swarm({', '.join(parts)})"

    @override
    def __getstate__(self) -> dict[str, Any]:
        """Pickle support: ``MappingProxyType`` is not picklable.

        Stores ``handoff_descriptions`` as a plain dict;
        :meth:`__setstate__` re-wraps it on restore. Without this,
        ``pickle.dumps`` / ``copy.deepcopy`` on a Swarm raise
        ``TypeError: cannot pickle 'mappingproxy' object``.
        """
        state = dict(self.__dict__)
        state["handoff_descriptions"] = dict(self.handoff_descriptions)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`__getstate__`, re-wrapping the immutable mapping."""
        for key, value in state.items():
            object.__setattr__(self, key, value)
        object.__setattr__(
            self,
            "handoff_descriptions",
            MappingProxyType(dict(self.handoff_descriptions)),
        )

    @staticmethod
    def new(name: str | None = None, *, description: str | None = None) -> SwarmBuilder[Any]:
        """Start a fluent swarm definition (mirrors ``Graph.new``).

        Args:
            name: Optional human-readable swarm name (metadata only).
            description: Optional human-readable blurb (metadata only).

        Returns:
            A :class:`~troopai.adk.swarms.builder.SwarmBuilder`;
            call ``.compile()`` to produce the frozen :class:`Swarm`.
        """
        from troopai.adk.swarms.builder import SwarmBuilder

        return SwarmBuilder(name=name, description=description)

    def get_member(self, name: str) -> Agent[TContext]:
        """Look up a member by name. Raises if not found.

        Used by the swarm driver when resolving
        :class:`~troopai.adk.swarms.yield_signal.SwarmHandoff` targets.

        Args:
            name: The agent name to look up (matched by exact equality
                against ``Agent.name``).

        Returns:
            The matching :class:`~troopai.adk.agents.agent.Agent`
            instance.

        Raises:
            KeyError: If no member with ``name`` exists in the swarm.
        """
        for m in self.members:
            if m.name == name:
                return m
        raise KeyError(f"No member named {name!r} in swarm (members: {[x.name for x in self.members]}).")

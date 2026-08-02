"""``SwarmBuilder`` — fluent, opinionated API for constructing a :class:`Swarm`.

Design goals (mirrors :mod:`troopai.adk.graphs.builder`):

1. **Readability first.** A swarm definition should read top-to-bottom
   like a spec. Compare:

   - AutoGen: ``Swarm([travel_agent, refunder], termination_condition=HandoffTermination("user") | TextMentionTermination("TERMINATE"))``
     with routing declared as strings scattered across per-agent
     ``handoffs=[...]`` lists.
   - TroopAI (this file): ``Swarm.new("support").members(triage, refunds).entry("triage").llm_handoff().terminate_on(ExplicitDoneTermination() | MaxTurnsTermination(12)).compile()``.
     The roster, the entry point, the routing strategy, and the stop
     rule each get exactly one line.

2. **Fail at compile time, not at run time.** :meth:`compile`
   constructs the :class:`Swarm`, so every ``__post_init__`` check
   (non-empty roster, unique names, entry ∈ members, reserved names,
   tool shadowing, roster-aware termination validation) fires before
   the first LLM call is billed.

3. **Defaults for the common case.** No policy call means
   :class:`LLMHandoffPolicy`; no termination call means
   :data:`~troopai.adk.swarms.swarm.DEFAULT_TERMINATION` (explicit
   ``swarm_done`` or a 25-turn safety net). A single-member swarm needs
   no ``.entry()`` call.

The builder is mutable during construction; :meth:`compile` returns the
frozen :class:`Swarm`. Mutating the builder afterwards has no effect on
the compiled swarm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from troopai.adk.swarms.config import SwarmConfig
from troopai.adk.swarms.policy import (
    CustomPolicy,
    LLMHandoffPolicy,
    RoundRobinPolicy,
    StructuredRoutingPolicy,
    SwarmPolicy,
)
from troopai.adk.swarms.swarm import Swarm

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.handoffs.handoff_route import HandoffRoute
    from troopai.adk.swarms.hooks import SwarmHooks
    from troopai.adk.swarms.policy import SwarmExtraToolsFn, SwarmSelector
    from troopai.adk.swarms.termination import TerminationCondition


TContext = TypeVar("TContext")


@dataclass
class SwarmBuilder[TContext]:
    """Mutable, fluent builder producing a frozen :class:`Swarm`.

    Every method except :meth:`compile` returns ``self`` so callers can
    chain. Nothing is validated until :meth:`compile` — at which point
    the full :class:`Swarm` validation fires.

    Attributes:
        name: Optional human-readable swarm name (metadata only).
        description: Optional human-readable blurb (metadata only).
        _members: Working roster, in insertion order.
        _handoff_descriptions: Working per-member routing hints for
            :class:`LLMHandoffPolicy` transfer tools.
        _entry: Working entry (agent object or member name). ``None``
            until :meth:`entry` is called.
        _policy: Working routing policy. ``None`` means "use the
            default :class:`LLMHandoffPolicy`".
        _termination: Working termination condition. ``None`` means
            "use :data:`DEFAULT_TERMINATION`".
        _config: Working :class:`SwarmConfig`.
        _hooks: Working :class:`SwarmHooks`.
    """

    name: str | None = None
    """Optional human-readable swarm name (metadata only)."""

    description: str | None = None
    """Optional human-readable description (metadata only)."""

    _members: list[Agent[TContext]] = field(default_factory=list)
    """Working roster, in insertion order."""

    _handoff_descriptions: dict[str, str] = field(default_factory=dict)
    """Working per-member transfer-tool descriptions."""

    _entry: Agent[TContext] | str | None = None
    """Working entry (agent or member name)."""

    _policy: SwarmPolicy[TContext] | None = None
    """Working routing policy; ``None`` → default LLM handoff."""

    _termination: TerminationCondition | None = None
    """Working termination; ``None`` → ``DEFAULT_TERMINATION``."""

    _config: SwarmConfig = field(default_factory=SwarmConfig)
    """Working :class:`SwarmConfig`."""

    _hooks: SwarmHooks[TContext] | None = None
    """Working swarm hooks."""

    # -- Roster ------------------------------------------------------

    def member(
        self,
        agent: Agent[TContext],
        *,
        handoff_description: str | None = None,
    ) -> SwarmBuilder[TContext]:
        """Add one agent to the roster.

        Args:
            agent: The agent to add. Duplicate names are rejected at
                :meth:`compile` time by :class:`Swarm` validation.
            handoff_description: Optional routing hint used as the
                ``transfer_to_<name>`` tool description by
                :class:`LLMHandoffPolicy` — tells the routing LLM
                *when* to pick this member (mirrors the OpenAI Agents
                SDK ``handoff_description``).

        Returns:
            ``self``, for chaining.
        """
        self._members.append(agent)
        if handoff_description is not None:
            self._handoff_descriptions[agent.name] = handoff_description
        return self

    def members(self, *agents: Agent[TContext]) -> SwarmBuilder[TContext]:
        """Add several agents to the roster, in order.

        Args:
            *agents: The agents to add.

        Returns:
            ``self``, for chaining.
        """
        for agent in agents:
            self.member(agent)
        return self

    # -- Topology ----------------------------------------------------

    def entry(self, agent_or_name: Agent[TContext] | str) -> SwarmBuilder[TContext]:
        """Declare which member takes the first turn.

        Args:
            agent_or_name: The entry agent, or its name. Names read
                better in a chain (``entry="author"``) and are resolved
                against the roster at :meth:`compile` time.

        Returns:
            ``self``, for chaining.
        """
        self._entry = agent_or_name
        return self

    # -- Routing -----------------------------------------------------

    def policy(self, policy: SwarmPolicy[TContext]) -> SwarmBuilder[TContext]:
        """Set an explicit routing policy (escape hatch).

        Prefer the named shortcuts — :meth:`llm_handoff`,
        :meth:`round_robin`, :meth:`routed`, :meth:`custom_policy` —
        which read better; use this when the policy needs constructor
        arguments the shortcuts do not expose.

        Args:
            policy: The :class:`SwarmPolicy` to use.

        Returns:
            ``self``, for chaining.
        """
        self._policy = policy
        return self

    def llm_handoff(self) -> SwarmBuilder[TContext]:
        """Route via LLM-called ``transfer_to_<member>`` tools.

        Returns:
            ``self``, for chaining.
        """
        self._policy = LLMHandoffPolicy()
        return self

    def round_robin(self, order: tuple[str, ...] | None = None) -> SwarmBuilder[TContext]:
        """Route via deterministic rotation (zero LLM routing tokens).

        Args:
            order: Optional explicit rotation of member names.
                Defaults to roster order.

        Returns:
            ``self``, for chaining.
        """
        self._policy = RoundRobinPolicy(order=order)
        return self

    def routed(self, route: HandoffRoute[Any, TContext]) -> SwarmBuilder[TContext]:
        """Route via structured intent output (``HandoffRoute``).

        Args:
            route: The :class:`HandoffRoute` mapping intent types to
                members, e.g. ``HandoffRoute("s").when(X).to(agent)``.

        Returns:
            ``self``, for chaining.
        """
        self._policy = StructuredRoutingPolicy(route=route)
        return self

    def custom_policy(
        self,
        selector: SwarmSelector,
        *,
        extra_tools: SwarmExtraToolsFn | None = None,
    ) -> SwarmBuilder[TContext]:
        """Route via a custom ``(state) -> member_name`` callable.

        Args:
            selector: Callable returning the next member's name.
            extra_tools: Optional callable returning extra
                ``FunctionTool`` instances to inject per turn.

        Returns:
            ``self``, for chaining.
        """
        self._policy = CustomPolicy(selector=selector, extra_tools_fn=extra_tools)
        return self

    # -- Stopping ----------------------------------------------------

    def terminate_on(self, condition: TerminationCondition) -> SwarmBuilder[TContext]:
        """Set the termination condition (composables with ``|`` / ``&``).

        Args:
            condition: The :class:`TerminationCondition` tree, e.g.
                ``ExplicitDoneTermination() | MaxTurnsTermination(12)``.

        Returns:
            ``self``, for chaining.
        """
        self._termination = condition
        return self

    # -- Budgets & hooks ---------------------------------------------

    def with_config(self, config: SwarmConfig) -> SwarmBuilder[TContext]:
        """Attach swarm-level budgets and shared-context strategy.

        Args:
            config: The :class:`SwarmConfig` to use.

        Returns:
            ``self``, for chaining.
        """
        self._config = config
        return self

    def with_hooks(self, hooks: SwarmHooks[TContext]) -> SwarmBuilder[TContext]:
        """Attach swarm-level lifecycle hooks.

        Args:
            hooks: The :class:`SwarmHooks` observer to attach.

        Returns:
            ``self``, for chaining.
        """
        self._hooks = hooks
        return self

    # -- Terminal ----------------------------------------------------

    def compile(self) -> Swarm[TContext]:
        """Validate and freeze into a :class:`Swarm`.

        Returns:
            The frozen, validated :class:`Swarm`.

        Raises:
            ValueError: If no members were added, if the entry was
                never set for a multi-member roster, or if any
                :class:`Swarm` validation fails (duplicate names,
                unknown entry, reserved names, tool shadowing,
                roster-aware termination errors).
        """
        if len(self._members) == 0:
            raise ValueError(
                "SwarmBuilder.compile: no members added. Add at least one "
                "agent via .member(agent) or .members(a, b, ...) first."
            )
        entry = self._entry
        if entry is None:
            if len(self._members) == 1:
                entry = self._members[0]
            else:
                raise ValueError(
                    "SwarmBuilder.compile: no entry set. Call "
                    ".entry(name_or_agent) — only a single-member swarm "
                    "can default its entry."
                )
        kwargs: dict[str, Any] = {}
        if self._policy is not None:
            kwargs["policy"] = self._policy
        if self._termination is not None:
            kwargs["termination"] = self._termination
        return Swarm(
            members=tuple(self._members),
            entry=entry,
            config=self._config,
            hooks=self._hooks,
            name=self.name,
            description=self.description,
            handoff_descriptions=self._handoff_descriptions,
            **kwargs,
        )

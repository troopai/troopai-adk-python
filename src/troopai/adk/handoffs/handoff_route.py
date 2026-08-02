from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Self,
)

from troopai.adk.handoffs.handoff_config import HandoffConfig
from troopai.adk.handoffs.handoff_helpers import evaluate_enabled
from troopai.adk.handoffs.handoff_target import (
    HandoffEnabledCallback,
    HandoffInputFilter,
    HandoffTarget,
    OnHandoffCallback,
    TAgent,
)
from troopai.adk.run.context import RunContext, TContext
from troopai.adk.types.intents import Intent, Respond

if TYPE_CHECKING:
    from troopai.adk.agents import Agent


def handoff_route(
    *routes: tuple[type[Intent], Agent[Any]],
    otherwise: Agent[Any] | None = None,
    name: str | None = None,
) -> HandoffRoute[Any, Any]:
    """Build a HandoffRoute from intent-agent tuples.

    Convenience factory for code-orchestrated handoffs. For LLM-orchestrated
    handoffs, use :func:`~troopai.adk.handoffs.handoff.handoff` or pass
    a list of agents/Handoff objects directly to ``Agent.handoffs``.

    Args:
        *routes: Tuples of (IntentType, Agent) for code-orchestrated routing.
        otherwise: Default agent if no intents match.
        name: Optional name for the route.

    Returns:
        HandoffRoute configured with the routing rules.

    Raises:
        ValueError: If no routes are provided.

    Example:
        agent.handoffs = handoff_route(
            (RefundIntent, refunds_agent),
            (BillingIntent, billing_agent),
            otherwise=general_agent,
        )
    """
    if len(routes) == 0 and otherwise is None:
        raise ValueError("Must provide at least one route or an otherwise agent.")

    built_route: HandoffRoute[Any, Any] = HandoffRoute(name)
    for intent_type, target_agent in routes:
        built_route.when(intent_type).to(target_agent)

    if otherwise is not None:
        built_route.otherwise(otherwise)

    return built_route


class UnhandledIntentError(Exception):
    """Raised when an intent falls through the router with no matching rule or fallback."""

    pass


class RouteSealedError(RuntimeError):
    """Raised when attempting to modify a route after it has started resolving intents."""

    pass


class HandoffPendingRoute(Generic[TAgent, TContext]):
    """Incomplete routing rule waiting for ``.to()`` to specify the target agent.

    Created by :meth:`HandoffRoute.when` and holds the matched intent types
    until ``.to()`` is called to complete the rule with a target agent.
    """

    def __init__(
        self,
        route: HandoffRoute[TAgent, TContext],
        intent_types: tuple[type[Intent], ...],
    ) -> None:
        """Store the parent route and the matched intent types.

        Args:
            route: The parent HandoffRoute that created this pending rule.
            intent_types: One or more Intent subclass types to match.
        """
        self._route = route
        self._intent_types = intent_types

    def to(
        self,
        target: Agent[TContext],
        on_handoff: OnHandoffCallback | None = None,
        input_filter: HandoffInputFilter | None = None,
        enabled: HandoffEnabledCallback = True,
        config: HandoffConfig | None = None,
    ) -> HandoffRoute[TAgent, TContext]:
        """Complete the pending routing rule by specifying the target agent.

        Registers the intent-to-agent mapping on the parent HandoffRoute and
        returns the route so further ``.when()`` calls can be chained.

        Args:
            target: The agent to hand off to when a matched intent is resolved.
            on_handoff: Optional callback invoked when this rule fires.
            input_filter: Optional function to transform handoff data before
                passing to the target agent.
            enabled: Whether this rule is active (bool or callable).
            config: Handoff configuration for this rule. Defaults to
                ``HandoffConfig()`` if not provided.

        Returns:
            The parent HandoffRoute for further chaining.

        Raises:
            RouteSealedError: If the route has already started resolving.
            ValueError: If the new intent type is a subclass of an already
                registered intent type (shadowing guard).
        """

        if self._route._sealed:
            raise RouteSealedError("Cannot add rules to a sealed HandoffRoute.")

        # GUARD: Prevent subclass shadowing
        for existing_types, _ in self._route._rules:
            for existing_type in existing_types:
                for new_type in self._intent_types:
                    if issubclass(new_type, existing_type):
                        raise ValueError(
                            f"Shadowing Error: '{new_type.__name__}' is a subclass of "
                            f"already registered '{existing_type.__name__}'. "
                            f"Register the more specific intent first."
                        )

        handoff_target: HandoffTarget[Any, TContext] = HandoffTarget(
            target=target,
            on_handoff=on_handoff,
            input_filter=input_filter,
            enabled=enabled,
            config=config if config is not None else HandoffConfig(),
        )

        self._route._rules.append((self._intent_types, handoff_target))
        return self._route


class HandoffRoute(Generic[TAgent, TContext]):
    """Fluent DSL for defining intent-based routing rules with full type safety.

    Routes map Intent types to HandoffTargets deterministically.
    Build rules with ``.when(*intents).to(agent)`` and register a fallback
    with ``.otherwise(agent)``. Call ``handoff_route()`` for the convenience
    factory.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialise an empty route.

        Args:
            name: Optional human-readable name used in error messages and
                tracing. Defaults to ``route_<id(self)>`` if not provided.
        """
        self.name: str = name if name is not None else f"route_{id(self)}"
        # Rules now store a tuple of Intent Types to allow multi-matching
        self._rules: list[tuple[tuple[type[Intent], ...], HandoffTarget[Any, TContext]]] = []
        self._otherwise: HandoffTarget[Any, TContext] | None = None
        self._sealed: bool = False

    def all_targets(self) -> list[HandoffTarget[Any, TContext]]:
        """Return every routing target registered on this route.

        Lists each rule target in registration order followed by the
        ``otherwise`` fallback when one is set. Lets external introspection
        (graph building, visualization, audit) enumerate handoff targets
        without reaching into the route's internal rule storage.

        Returns:
            A list of :class:`HandoffTarget` instances; each
            ``target.target`` is the destination :class:`Agent`.
        """
        targets: list[HandoffTarget[Any, TContext]] = [target for _, target in self._rules]
        if self._otherwise is not None:
            targets.append(self._otherwise)
        return targets

    def when(self, *intent_types: type[Intent]) -> HandoffPendingRoute[TAgent, TContext]:
        """Start defining a routing rule for one or more intent types.

        Accepts one or more Intent types as positional arguments.

        Example:
            .when(Refund).to(refunds_agent)
            .when(Billing, CancelSubscription).to(billing_agent)

        Args:
            *intent_types: One or more Intent subclass types to match.

        Returns:
            HandoffPendingRoute waiting for ``.to()`` to complete the rule.

        Raises:
            RouteSealedError: If the route has already started resolving.
            ValueError: If no intent types are provided.
        """
        if self._sealed:
            raise RouteSealedError("Cannot add rules to a sealed HandoffRoute.")

        if len(intent_types) == 0:
            raise ValueError("when() requires at least one Intent type.")

        return HandoffPendingRoute(self, intent_types)

    def otherwise(
        self,
        target: Agent[TContext],
        on_handoff: OnHandoffCallback | None = None,
        input_filter: HandoffInputFilter | None = None,
        enabled: HandoffEnabledCallback = True,
        config: HandoffConfig | None = None,
    ) -> Self:
        """Define the default routing rule used when no intent rule matches.

        Args:
            target: The fallback agent to hand off to.
            on_handoff: Optional callback invoked when the fallback fires.
            input_filter: Optional function to transform handoff data before
                passing to the target agent.
            enabled: Whether the fallback is active (bool or callable).
            config: Handoff configuration for the fallback rule. Defaults to
                ``HandoffConfig()`` if not provided.

        Returns:
            This route, for method chaining.

        Raises:
            RouteSealedError: If the route has already started resolving.
        """
        if self._sealed:
            raise RouteSealedError("Cannot set fallback on a sealed HandoffRoute.")

        self._otherwise = HandoffTarget(
            target=target,
            on_handoff=on_handoff,
            input_filter=input_filter,
            enabled=enabled,
            config=config if config is not None else HandoffConfig(),
        )
        return self

    async def resolve(
        self,
        intent: Intent | Respond,
        /,
        context: RunContext[TContext] | None = None,
    ) -> HandoffTarget[Any, TContext] | None:
        """Determine the appropriate HandoffTarget based on the output intent.

        Accepts either an ``Intent`` (triggers routing) or a ``Respond``
        (LLM chose to answer directly — no handoff).

        Seals the route on first call to prevent concurrent mutations.

        Args:
            intent: The resolved intent or direct-response sentinel.
            context: Run context passed to ``enabled`` callables.

        Returns:
            The matching HandoffTarget, or ``None`` when ``intent`` is a
            ``Respond`` (no handoff required).

        Raises:
            UnhandledIntentError: If no rule matches and no valid
                ``otherwise`` fallback is enabled.
        """
        # Explicitly handle direct responses (no handoff required).
        # A Respond means no handoff was taken; the route must stay usable
        # for future turns, so we seal only when an actual routing decision
        # is made (below).
        if isinstance(intent, Respond):
            return None

        # Seal the router after the first real routing decision to prevent
        # unexpected rule mutations once intents are actively being resolved.
        self._sealed = True

        # 1. Check explicitly defined rules in order
        for intent_types, handoff_target in self._rules:
            # isinstance naturally supports checking against a tuple of types!
            if isinstance(intent, intent_types) and await self._is_enabled(handoff_target, intent, context):
                return handoff_target

        # 2. Fallback to the 'otherwise' rule if no specific rule matched/was enabled
        if self._otherwise is not None and await self._is_enabled(self._otherwise, intent, context):
            return self._otherwise

        # 3. If we get here, the intent fell through the cracks. Fail fast and loud.
        raise UnhandledIntentError(
            f"Intent '{type(intent).__name__}' was not handled by route '{self.name}', "
            f"and no valid fallback (.otherwise) was executed."
        )

    @staticmethod
    async def _is_enabled(
        target: HandoffTarget[Any, Any],
        intent: Intent,
        context: RunContext[Any] | None = None,
    ) -> bool:
        """Check if a code-orchestrated route target is enabled.

        Delegates to :func:`evaluate_enabled` so LLM-orch (Handoff) and
        code-orch (HandoffRoute) share one dispatch contract — same
        arity rules, same async + async-generator + non-bool guards,
        same missing-context error.

        ``intent`` is supplied as the second positional argument to
        2-arg callables (mirroring how LLM-orch supplies the target
        Agent).
        """
        agent_name = getattr(target.target, "name", "<unknown>")
        return await evaluate_enabled(
            target.enabled,
            context,
            intent,
            handoff_name=f"route→{agent_name}",
        )

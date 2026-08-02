"""Swarm lifecycle hooks — complement ``RunHooks`` and ``AgentHooks``.

Three hook scopes compose during a swarm run:

1. ``RunHooks`` (run-level) — fires on every agent; unchanged.
2. ``AgentHooks`` (per-agent) — fires on its own agent; unchanged.
3. ``SwarmHooks`` (new, this file) — swarm-level lifecycle callbacks
   that fire around turn boundaries, handoffs, and final termination.

The base-class methods follow the existing project pattern: no-op
implementations that use ``del`` on parameters to signal "intentionally
unused" to linters. Override only the hooks you need.

Example::

    class SwarmMetrics(SwarmHooks):
        async def on_swarm_turn_end(self, context, state, items):
            metric("swarm.turn.tokens", state.cumulative_usage.total_tokens)

        async def on_swarm_handoff(self, context, from_agent, to_agent, message):
            metric("swarm.handoff", 1, tags={"from": from_agent, "to": to_agent})
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from troopai.adk.graphs.interrupt import Interrupt
    from troopai.adk.run.context import RunContext
    from troopai.adk.swarms.state import SwarmState
    from troopai.adk.swarms.stop_reason import StopReason
    from troopai.adk.types.items.items import RunItem


logger = logging.getLogger(__name__)


TContext = TypeVar("TContext")


class SwarmHooks[TContext]:
    """Base class for swarm-level lifecycle hooks.

    All methods are async and optional — override only what you need.
    Fires alongside ``RunHooks`` and ``AgentHooks``; does not replace
    them. The swarm driver calls these at well-defined boundaries:

    - ``on_swarm_start`` — once, before the first turn's LLM call.
    - ``on_swarm_turn_start`` — before each member turn.
    - ``on_swarm_handoff`` — when a ``SwarmHandoff`` is resolved,
      between the emitting agent's turn-end and the target agent's
      turn-start.
    - ``on_swarm_turn_end`` — after each member turn, after
      ``on_agent_end`` on ``RunHooks``.
    - ``on_swarm_done`` — once, after the last turn, before
      ``SwarmRunResult`` is returned to the caller.

    Firing order on the terminal turn (important for implementors):

    1. The agent calls the ``swarm_done`` tool.
    2. ``turn_resolution`` builds a ``SwarmDone`` yield and appends
       it to ``state.last_yield``.
    3. ``on_swarm_turn_end`` fires — **at this point
       ``state.last_yield`` is already a ``SwarmDone`` instance**.
       Hook implementors can detect termination in this callback by
       branching on ``isinstance(state.last_yield, SwarmDone)``.
    4. The driver's termination predicate returns a ``StopReason`` and
       the loop exits.
    5. ``on_swarm_done`` fires with the terminal ``StopReason`` and
       ``final_output``.

    A turn that ends in a normal handoff fires ``on_swarm_turn_end``
    with ``state.last_yield`` as the corresponding ``SwarmHandoff``.
    The shape of ``last_yield`` is the canonical way to tell one kind
    of turn-end from another inside this hook — the driver does not
    fire a separate "terminal turn end" callback.

    Type Parameters:
        TContext: The user-provided context type (same ``TContext``
            as ``RunHooks`` — hooks share the same context object
            through ``RunContext``).
    """

    propagate_errors: bool = False
    """When ``True``, errors raised by this hook's callbacks propagate out of
    the fan-out instead of being logged and swallowed.  The default ``False``
    keeps observer hooks best-effort.  Subclasses that provide a persistence
    guarantee (e.g. checkpointers) set this to ``True`` so a failed save
    surfaces to the caller rather than being silently dropped.
    """

    async def on_swarm_start(
        self,
        context: RunContext[TContext],
        state: SwarmState[TContext],
    ) -> None:
        """Called once at the beginning of the swarm run.

        Args:
            context: The run context wrapper (usage tracking, user context).
            state: Initial swarm state (entry agent set, no turns yet).
        """
        del context, state

    async def on_swarm_turn_start(
        self,
        context: RunContext[TContext],
        state: SwarmState[TContext],
        member_name: str,
    ) -> None:
        """Called before each member turn's LLM call.

        Args:
            context: The run context wrapper.
            state: Current swarm state (``state.current_agent`` is the
                agent about to take this turn; ``state.total_turns``
                is incremented before this hook fires).
            member_name: Name of the swarm member about to take this
                turn. Mirrors ``state.current_agent_name``; provided as
                an explicit parameter so overriding subclasses do not
                need to inspect the state to get the name.
        """
        del context, state, member_name

    async def on_swarm_handoff(
        self,
        context: RunContext[TContext],
        state: SwarmState[TContext],
        from_agent: str,
        to_agent: str,
        message: str,
    ) -> None:
        """Called when a ``SwarmHandoff`` signal is resolved.

        Fires after the emitter's ``on_swarm_turn_end`` and before
        the target's ``on_swarm_turn_start``.

        Args:
            context: The run context wrapper.
            state: Current swarm state (``handoff_count`` is already
                incremented).
            from_agent: Name of the emitting agent.
            to_agent: Name of the target agent.
            message: Explicit handoff payload (``SwarmHandoff.message``).
        """
        del context, state, from_agent, to_agent, message

    async def on_swarm_turn_end(
        self,
        context: RunContext[TContext],
        state: SwarmState[TContext],
        items: list[RunItem],
    ) -> None:
        """Called after each member turn, after the agent's
        ``on_agent_end``.

        Args:
            context: The run context wrapper.
            state: Current swarm state (items already appended to
                ``shared_history`` and ``per_agent_scratch``).
            items: Layer 3 items produced by this turn.
        """
        del context, state, items

    async def on_swarm_done(
        self,
        context: RunContext[TContext],
        state: SwarmState[TContext],
        reason: StopReason,
        final_output: Any,
    ) -> None:
        """Called once at the end of the swarm run, before the
        ``SwarmRunResult`` is returned.

        Args:
            context: The run context wrapper.
            state: Final swarm state.
            reason: Why the swarm stopped.
            final_output: The terminal agent's final output
                (``None`` if termination was not via ``swarm_done``).
        """
        del context, state, reason, final_output

    async def on_swarm_turn_interrupt(
        self,
        context: RunContext[TContext],
        state: SwarmState[TContext],
        member_name: str,
        interrupt: Interrupt,
    ) -> None:
        """Called when a member turn suspends on a cooperative interrupt.

        Fires once per parked turn, at the moment the swarm loop catches
        the inner ``InterruptException`` (HITL via
        ``request_human_input``) or lifts an :class:`AgentToolDeferral`
        to a :class:`NestedAgentInterrupt`. The concrete ``Interrupt``
        subclass distinguishes the cause.

        Args:
            context: The run context wrapper.
            state: Current swarm state (interrupt already parked on
                ``state.pending_interrupts[member_name]``).
            member_name: The name of the swarm member whose turn parked.
            interrupt: The :class:`Interrupt` payload (subclassed for
                nested-agent suspends).
        """
        del context, state, member_name, interrupt


class HookRegistry:
    """Aggregates multiple :class:`SwarmHooks` and fans calls out to all.

    Mirrors the graphs :class:`troopai.adk.graphs.hooks.HookRegistry`
    pattern. The swarm loop fires each ``on_*`` once per event; the
    registry forwards to every attached hook.

    Errors from observer hooks (``propagate_errors=False``) are logged
    as warnings and do not halt the swarm run.  Hooks with
    ``propagate_errors=True`` — such as checkpointers that provide a
    persistence guarantee — re-raise their errors so a failed save
    surfaces to the caller instead of being silently dropped.
    """

    def __init__(self) -> None:
        self._hooks: list[SwarmHooks[Any]] = []

    def add(self, hooks: SwarmHooks[Any]) -> None:
        """Attach a :class:`SwarmHooks` instance.

        Args:
            hooks: The :class:`SwarmHooks` to attach. Its callbacks
                will fire on every subsequent event.
        """
        self._hooks.append(hooks)

    async def _fan_out(
        self,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Invoke ``method_name`` on every attached hook.

        Hooks with ``propagate_errors=True`` re-raise on failure so
        persistence-critical callbacks (e.g. checkpointer saves) surface
        errors to the caller.  All other hooks are best-effort: errors are
        logged as warnings and do not halt the swarm run.

        Args:
            method_name: Name of the ``SwarmHooks`` method to call.
            *args: Positional arguments forwarded to the method.
            **kwargs: Keyword arguments forwarded to the method.
        """
        for h in self._hooks:
            try:
                await getattr(h, method_name)(*args, **kwargs)
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                # Task cancellation and process signals must propagate
                # immediately — they are BaseException subclasses and must
                # not be swallowed by the per-hook error-handling logic.
                raise
            except Exception as exc:
                if h.propagate_errors:
                    logger.exception(
                        "SwarmHooks.%s raised on %s; propagating (persistence-critical).",
                        method_name,
                        type(h).__name__,
                    )
                    raise
                logger.warning(
                    "SwarmHooks.%s raised on %s; continuing: %s",
                    method_name,
                    type(h).__name__,
                    exc,
                )

    async def on_swarm_start(
        self,
        context: RunContext[Any],
        state: SwarmState[Any],
    ) -> None:
        await self._fan_out("on_swarm_start", context, state)

    async def on_swarm_turn_start(
        self,
        context: RunContext[Any],
        state: SwarmState[Any],
        member_name: str,
    ) -> None:
        await self._fan_out("on_swarm_turn_start", context, state, member_name)

    async def on_swarm_handoff(
        self,
        context: RunContext[Any],
        state: SwarmState[Any],
        from_agent: str,
        to_agent: str,
        message: str,
    ) -> None:
        await self._fan_out("on_swarm_handoff", context, state, from_agent, to_agent, message)

    async def on_swarm_turn_end(
        self,
        context: RunContext[Any],
        state: SwarmState[Any],
        items: list[RunItem],
    ) -> None:
        await self._fan_out("on_swarm_turn_end", context, state, items)

    async def on_swarm_done(
        self,
        context: RunContext[Any],
        state: SwarmState[Any],
        reason: StopReason,
        final_output: Any,
    ) -> None:
        await self._fan_out("on_swarm_done", context, state, reason, final_output)

    async def on_swarm_turn_interrupt(
        self,
        context: RunContext[Any],
        state: SwarmState[Any],
        member_name: str,
        interrupt: Interrupt,
    ) -> None:
        await self._fan_out("on_swarm_turn_interrupt", context, state, member_name, interrupt)

"""Graph lifecycle hooks — complement ``RunHooks`` / ``AgentHooks`` / ``SwarmHooks``.

A graph run layers hook scopes the same way a swarm run does:

1. ``RunHooks`` (run-level) — fires on every nested agent, unchanged.
2. ``AgentHooks`` (per-agent) — fires on its own agent, unchanged.
3. ``SwarmHooks`` (when a node hosts a Swarm) — fires on that nested
   swarm, unchanged.
4. ``GraphHooks`` (new, this file) — graph-wide lifecycle callbacks.

The key architectural insight is that the :class:`Checkpointer`
protocol extends :class:`HookProvider`. The graph loop never calls
``checkpointer.save()`` directly — it just fires hooks, and the
checkpointer implements :meth:`GraphHooks.on_node_end` /
:meth:`GraphHooks.on_graph_end` to persist state. This keeps the
loop code pure of persistence concerns and makes checkpointers
swappable without touching the loop. Strands pioneered this
pattern; LangGraph bakes checkpointing into the engine.

:class:`HookRegistry` aggregates multiple hook providers and
collects interrupt exceptions (for human-in-the-loop pauses).
Multiple hooks on the same event can each request interrupts; the
registry gathers them all rather than short-circuiting on the first
raise.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from troopai.adk.graphs.interrupt import InterruptException

if TYPE_CHECKING:
    from troopai.adk.graphs.interrupt import Interrupt
    from troopai.adk.graphs.result import GraphRunStatus
    from troopai.adk.graphs.state import GraphState
    from troopai.adk.orchestration.executable import ExecutableInput, NodeResult
    from troopai.adk.run.context import RunContext
    from troopai.adk.types.items.items import RunItem


logger = logging.getLogger(__name__)


TContext = TypeVar("TContext")


class GraphHooks[TContext]:
    """Base class for graph-level lifecycle hooks.

    All methods are async and optional — override only what you need.
    The graph driver calls these at well-defined boundaries:

    - :meth:`on_graph_start` — once, before the first superstep.
    - :meth:`on_superstep_start` — before each superstep.
    - :meth:`on_node_start` — before each node invocation.
    - :meth:`on_node_end` — after each node invocation.
    - :meth:`on_node_error` — on node exception (when fail_fast=False).
    - :meth:`on_node_interrupt` — when a node raises
      ``InterruptException`` to suspend (HITL or nested-agent defer).
    - :meth:`on_superstep_end` — after each superstep, including the
      list of node ids that fired.
    - :meth:`on_graph_end` — once, after the last superstep, before
      :class:`GraphRunResult` is returned.

    The base-class methods are no-ops that ``del`` their parameters
    to silence linters — same pattern as ``SwarmHooks``.

    Type Parameters:
        TContext: The user-provided context type.
    """

    propagate_errors: bool = False
    """When ``True``, errors raised by this hook's callbacks propagate out of
    the fan-out instead of being logged and swallowed.  The default ``False``
    keeps observer hooks best-effort.  Subclasses that provide a persistence
    guarantee (e.g. checkpointers) set this to ``True`` so a failed save
    surfaces to the caller rather than being silently dropped.
    """

    async def on_graph_start(
        self,
        context: RunContext[TContext],
        state: GraphState[TContext],
    ) -> None:
        """Called once at graph-run start, before the first superstep.

        Args:
            context: Run context for the graph run.
            state: Initial :class:`GraphState` (``superstep == 0``).
        """
        del context, state

    async def on_superstep_start(
        self,
        context: RunContext[TContext],
        state: GraphState[TContext],
        ready_nodes: tuple[str, ...],
    ) -> None:
        """Called at the top of each superstep.

        Args:
            context: Run context.
            state: Current graph state (``state.superstep`` is
                incremented before this hook fires).
            ready_nodes: Node ids about to fire in this superstep.
        """
        del context, state, ready_nodes

    async def on_node_start(
        self,
        context: RunContext[TContext],
        state: GraphState[TContext],
        node_id: str,
        input: ExecutableInput,
    ) -> None:
        """Called before a node's :meth:`Executable.invoke`.

        Args:
            context: Run context.
            state: Current graph state.
            node_id: Id of the node about to fire.
            input: The :class:`ExecutableInput` being passed in.
        """
        del context, state, node_id, input

    async def on_node_end(
        self,
        context: RunContext[TContext],
        state: GraphState[TContext],
        node_id: str,
        result: NodeResult,
    ) -> None:
        """Called after a node's :meth:`Executable.invoke` returns cleanly.

        This is where :class:`Checkpointer` persists state — it
        overrides this method to call ``self.save(state)``.

        Args:
            context: Run context.
            state: Current graph state (``result`` is already
                recorded on the state).
            node_id: Id of the node that just fired.
            result: Its :class:`NodeResult`.
        """
        del context, state, node_id, result

    async def on_node_error(
        self,
        context: RunContext[TContext],
        state: GraphState[TContext],
        node_id: str,
        error: BaseException,
    ) -> None:
        """Called when a node raises.

        Fires even when :attr:`GraphConfig.fail_fast` is ``True`` —
        ``fail_fast`` controls sibling cancellation, not hook
        emission. An observability hook may still want to record
        the error.

        ``InterruptException`` is NOT delivered here — it is a
        cooperative pause, not a failure. See
        :meth:`on_node_interrupt`.

        Args:
            context: Run context.
            state: Current graph state.
            node_id: Id of the node that raised.
            error: The exception instance.
        """
        del context, state, node_id, error

    async def on_node_interrupt(
        self,
        context: RunContext[TContext],
        state: GraphState[TContext],
        node_id: str,
        interrupt: Interrupt,
    ) -> None:
        """Called when a node suspends by raising ``InterruptException``.

        Fires once per suspending node, at the moment the BSP loop
        captures the interrupt and before the superstep ends. The
        concrete ``Interrupt`` subclass distinguishes the cause:

        - :class:`Interrupt` — HITL ``request_human_input``.
        - :class:`NestedAgentInterrupt` — an inner agent deferred a
          tool call; the outer graph node is parking for the reply.

        Siblings in the same superstep still run; the run as a whole
        transitions to ``INTERRUPTED`` only at the superstep boundary.

        Args:
            context: Run context.
            state: Current graph state.
            node_id: Id of the node that suspended.
            interrupt: The :class:`Interrupt` payload (subclassed for
                nested-agent suspends).
        """
        del context, state, node_id, interrupt

    async def on_superstep_end(
        self,
        context: RunContext[TContext],
        state: GraphState[TContext],
        fired_nodes: tuple[str, ...],
        items: list[RunItem],
    ) -> None:
        """Called after a superstep completes.

        Args:
            context: Run context.
            state: Current graph state.
            fired_nodes: Node ids that fired this superstep (same
                order as :meth:`on_superstep_start.ready_nodes`
                except for nodes that errored).
            items: Layer 3 items produced during this superstep.
        """
        del context, state, fired_nodes, items

    async def on_graph_end(
        self,
        context: RunContext[TContext],
        state: GraphState[TContext],
        status: GraphRunStatus,
        final_output: Any,
    ) -> None:
        """Called once at graph-run end, before the result is returned.

        Checkpointers typically override this to mark the run
        finalised or to clean up transient in-flight state.

        Args:
            context: Run context.
            state: Final graph state.
            status: Terminal :class:`GraphRunStatus`.
            final_output: Aggregate graph output.
        """
        del context, state, status, final_output


@runtime_checkable
class HookProvider(Protocol):
    """Objects that can attach themselves to a :class:`HookRegistry`.

    A :class:`Checkpointer` implements this Protocol and uses it to
    subscribe to the hooks it cares about (``on_node_end``,
    ``on_graph_end``). User code implements it to attach arbitrary
    observers to a graph run.

    The Protocol is ``@runtime_checkable`` so user code can pass raw
    objects in ``Runner.arun_graph(..., hooks=[metrics, checkpointer])``
    without inheriting from a concrete base.
    """

    def register(self, registry: HookRegistry) -> None:
        """Register this provider's callbacks on the supplied registry.

        Called once at the start of :func:`run_graph_loop` for every
        attached provider.
        """
        ...


class HookRegistry:
    """Aggregates multiple :class:`GraphHooks` and :class:`HookProvider`\\ s.

    The graph loop calls ``registry.on_*`` exactly once per event;
    the registry fans out to every attached hook. This lets a graph
    run attach both a :class:`GraphHooks` (the plain lifecycle
    observer) and one or more :class:`Checkpointer` instances
    without the loop needing to know about either.

    Errors from observer hooks (``propagate_errors=False``) are logged
    as warnings and do not halt the graph run.  Hooks with
    ``propagate_errors=True`` — such as checkpointers that provide a
    persistence guarantee — re-raise their errors so a failed save
    surfaces to the caller instead of being silently dropped.
    """

    def __init__(self) -> None:
        self._hooks: list[GraphHooks[Any]] = []

    def add(self, hooks: GraphHooks[Any]) -> None:
        """Attach a :class:`GraphHooks` instance."""
        self._hooks.append(hooks)

    def add_provider(self, provider: HookProvider) -> None:
        """Attach a :class:`HookProvider`.

        The provider is asked to register itself on this registry —
        most providers will call :meth:`add` with their own
        :class:`GraphHooks` implementation.
        """
        provider.register(self)

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
        logged as warnings and do not halt the graph run.
        """
        for h in self._hooks:
            if not hasattr(h, method_name):
                # All items in _hooks are typed as GraphHooks, which defines
                # every method as a no-op. A missing method is a programmer
                # error (e.g. a hook that forgot to inherit properly); raise
                # so it is caught at development time rather than silently
                # swallowed.
                raise TypeError(
                    f"GraphHooks._fan_out: hook {type(h).__name__!r} has no "
                    f"method {method_name!r}. Ensure it inherits from GraphHooks."
                )
            try:
                await getattr(h, method_name)(*args, **kwargs)
            except InterruptException:
                # A hook requested a cooperative pause (e.g. a custom HITL
                # checkpointer). This is NOT a failure — never swallow it or
                # downgrade it to a logged warning regardless of
                # ``propagate_errors``. Re-raise so the graph driver can
                # surface the run as INTERRUPTED, the same terminal state a
                # node-level ``request_human_input`` produces.
                raise
            except Exception as exc:
                if h.propagate_errors:
                    logger.exception(
                        "GraphHooks.%s raised on %s; propagating (persistence-critical).",
                        method_name,
                        type(h).__name__,
                    )
                    raise
                logger.warning(
                    "GraphHooks.%s raised on %s; continuing: %s",
                    method_name,
                    type(h).__name__,
                    exc,
                )

    # -- Convenience pass-throughs ---------------------------------

    async def on_graph_start(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
    ) -> None:
        await self._fan_out("on_graph_start", context, state)

    async def on_superstep_start(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        ready_nodes: tuple[str, ...],
    ) -> None:
        await self._fan_out("on_superstep_start", context, state, ready_nodes)

    async def on_node_start(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        node_id: str,
        input: ExecutableInput,
    ) -> None:
        await self._fan_out("on_node_start", context, state, node_id, input)

    async def on_node_end(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        node_id: str,
        result: NodeResult,
    ) -> None:
        await self._fan_out("on_node_end", context, state, node_id, result)

    async def on_node_error(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        node_id: str,
        error: BaseException,
    ) -> None:
        await self._fan_out("on_node_error", context, state, node_id, error)

    async def on_node_interrupt(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        node_id: str,
        interrupt: Interrupt,
    ) -> None:
        await self._fan_out("on_node_interrupt", context, state, node_id, interrupt)

    async def on_superstep_end(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        fired_nodes: tuple[str, ...],
        items: list[RunItem],
    ) -> None:
        await self._fan_out("on_superstep_end", context, state, fired_nodes, items)

    async def on_graph_end(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        status: GraphRunStatus,
        final_output: Any,
    ) -> None:
        await self._fan_out("on_graph_end", context, state, status, final_output)


__all__ = [
    "GraphHooks",
    "HookProvider",
    "HookRegistry",
]

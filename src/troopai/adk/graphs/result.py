"""Result types for graph execution.

Mirrors :class:`~troopai.adk.swarms.result.SwarmRunResult` so consumers
who already read ``.final_output`` / ``.new_items`` / ``.context`` on
single-agent runs get the same ergonomics for graphs. Adds
graph-specific fields that do not make sense on a single-agent or
swarm result:

- ``per_node_usage`` — cost attribution broken down by node id. The
  feature neither LangGraph nor Strands surfaces ergonomically on
  their Graph result.
- ``node_results`` — the full :class:`NodeResult` per node, available
  without unpacking :attr:`state`.
- ``status`` — :class:`GraphRunStatus` lifecycle tag (``completed``,
  ``failed``, ``max_supersteps``, ``max_tokens``, ``no_ready_nodes``, or
  ``interrupted``).

:class:`GraphRunResultStreaming` is the streaming twin of
:class:`GraphRunResult`, produced by
:meth:`~troopai.adk.run.runner.Runner.arun_graph_streamed` and
``Runner.configure().graph(graph).arun(stream=True)``. Events are consumed via
:meth:`GraphRunResultStreaming.stream_events`; terminal fields
(``final_output``, ``status``, ``state``, usage, etc.) are populated
when the run completes.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from troopai.adk.graphs.interrupt import Interrupt, NestedAgentInterrupt, NestedGraphInterrupt
from troopai.adk.run.stream import CancelMode, QueueCompleteSentinel
from troopai.adk.types.tokens.llm_usage import LLMUsage

if TYPE_CHECKING:
    from troopai.adk.graphs.state import GraphState
    from troopai.adk.orchestration.executable import NodeResult
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.types import UserPrompt
    from troopai.adk.types.items.items import RunItem


TContext = TypeVar("TContext")


@dataclass(frozen=True)
class StructuredInterrupts:
    """A structured view of pending interrupts on a graph run result.

    Groups the flat ``interrupts`` tuple into named categories so
    consumers can distinguish pending human decisions from each other
    without inspecting raw dict keys.  This is additive — the existing
    ``interrupts`` field on :class:`GraphRunResult` and
    :class:`GraphRunResultStreaming` is unchanged.

    Attributes:
        generic: Plain :class:`Interrupt` instances (``kind="generic"``
            or any non-typed kind).
        nested_agent: :class:`NestedAgentInterrupt` instances — a nested
            agent deferred a tool call awaiting human approval.
        nested_graph: :class:`NestedGraphInterrupt` instances — an inner
            graph suspended on a plain interrupt that was lifted to the
            outer graph.
        by_node: All pending interrupts keyed by node id for direct
            lookup without iterating any of the category tuples.
    """

    generic: tuple[Interrupt, ...] = ()
    """Plain interrupts (kind not matched by a more specific subtype)."""

    nested_agent: tuple[NestedAgentInterrupt, ...] = ()
    """Nested-agent tool-approval interrupts."""

    nested_graph: tuple[NestedGraphInterrupt, ...] = ()
    """Lifted inner-graph interrupts."""

    by_node: Mapping[str, Interrupt] = field(default_factory=dict)
    """All pending interrupts keyed by node id (read-only mapping)."""

    def __post_init__(self) -> None:
        # Wrap any plain dict so the frozen contract extends to the
        # mapping's contents, matching the tuple-typed sibling fields.
        if isinstance(self.by_node, dict):
            object.__setattr__(self, "by_node", MappingProxyType(self.by_node))

    @classmethod
    def from_interrupts(cls, interrupts: tuple[Interrupt, ...]) -> StructuredInterrupts:
        """Build a :class:`StructuredInterrupts` from a flat interrupt tuple.

        Classifies each interrupt into the appropriate category based on
        its concrete type.  ``by_node`` is built from all entries;
        later entries with the same node id overwrite earlier ones (a
        node can only have one pending interrupt at a time).

        Args:
            interrupts: The ``GraphRunResult.interrupts`` tuple, usually
                sourced from ``GraphState.pending_interrupts.values()``.

        Returns:
            A fully populated :class:`StructuredInterrupts`.
        """
        generic: list[Interrupt] = []
        nested_agent: list[NestedAgentInterrupt] = []
        nested_graph_list: list[NestedGraphInterrupt] = []
        by_node: dict[str, Interrupt] = {}

        for iv in interrupts:
            by_node[iv.node_id] = iv
            if isinstance(iv, NestedAgentInterrupt):
                nested_agent.append(iv)
            elif isinstance(iv, NestedGraphInterrupt):
                nested_graph_list.append(iv)
            else:
                generic.append(iv)

        return cls(
            generic=tuple(generic),
            nested_agent=tuple(nested_agent),
            nested_graph=tuple(nested_graph_list),
            by_node=MappingProxyType(by_node),
        )


class GraphRunStatus(StrEnum):
    """Terminal lifecycle tag for a graph run."""

    COMPLETED = "completed"
    """Every terminal node fired at least once; loop exited cleanly."""

    FAILED = "failed"
    """A node raised an exception that was surfaced.  With
    :attr:`GraphConfig.fail_fast` set (the default), the run stops
    immediately.  With ``fail_fast=False``, node errors are accumulated and
    the run continues; if errored nodes cause all downstream paths to
    deadlock (no ready nodes and no terminal fired), the status is set to
    ``FAILED`` and :attr:`GraphRunResult.error` names which nodes
    failed."""

    MAX_SUPERSTEPS = "max_supersteps"
    """Loop exited because :attr:`GraphConfig.max_supersteps` was hit."""

    MAX_TOKENS = "max_tokens"
    """Loop exited because :attr:`GraphConfig.max_total_tokens` was hit."""

    NO_READY_NODES = "no_ready_nodes"
    """Loop exited because no more nodes were schedulable and no
    terminal had fired — typically a conditional-edge setup where
    every branch's predicate returned ``False``."""

    INTERRUPTED = "interrupted"
    """Loop paused because a node raised :class:`InterruptException`;
    the caller must supply a :class:`~troopai.adk.graphs.interrupt.GraphResume`
    to continue."""


@dataclass
class GraphRunResult[TContext]:
    """Result of a completed graph run.

    A :class:`GraphRunResult` is produced for every terminal outcome
    — completed, failed, or budget-exceeded. The ``status`` field
    tells the caller which one. For hard-crash exits (e.g.
    ``MaxTurnsExceeded`` from a nested agent) the graph loop still
    raises rather than returning a result.

    Attributes:
        final_output: Aggregate graph output. When the graph has one
            terminal, this is that terminal's
            :attr:`NodeResult.output`. When it has multiple
            terminals, this is a dict ``{terminal_id: output}``.
        status: :class:`GraphRunStatus` lifecycle tag.
        user_prompt: The original input passed to
            :meth:`Runner.arun_graph`.
        new_items: Layer 3 items produced across the whole run, in
            completion order. Equivalent to
            :attr:`GraphState.all_items` but surfaced at the top level
            for API parity with :class:`RunResult`.
        state: Final :class:`GraphState`. Serialisable via
            :meth:`GraphState.to_json`.
        node_results: Latest :class:`NodeResult` per node id, at loop
            exit. Mirrors :attr:`GraphState.node_results`.
        context: The :class:`RunContext` shared across the run.
        per_node_usage: Per-node cost attribution keyed by node id.
        cumulative_usage: Graph-wide cumulative usage. Equal to the
            sum of :attr:`per_node_usage` values.
        total_supersteps: Mirror of :attr:`GraphState.superstep`.
        error: Populated only when ``status == GraphRunStatus.FAILED``.
        interrupts: Pending interrupts when ``status == INTERRUPTED``.
            Empty for all other statuses.
    """

    final_output: Any
    """Aggregate graph output."""

    status: GraphRunStatus
    """Lifecycle tag."""

    user_prompt: UserPrompt
    """Original input."""

    new_items: list[RunItem] = field(default_factory=list)
    """Layer 3 items produced across the whole run."""

    state: GraphState[TContext] | None = None
    """Final graph state."""

    node_results: dict[str, NodeResult] = field(default_factory=dict)
    """Latest :class:`NodeResult` per node id."""

    context: RunContext[TContext] | None = None
    """The run context."""

    per_node_usage: dict[str, LLMUsage] = field(default_factory=dict)
    """Per-node cost attribution."""

    cumulative_usage: LLMUsage = field(default_factory=LLMUsage)
    """Graph-wide cumulative usage."""

    total_supersteps: int = 0
    """Mirror of :attr:`GraphState.superstep`."""

    error: str | None = None
    """Serialised error when ``status == FAILED``."""

    interrupts: tuple[Interrupt, ...] = ()
    """Pending interrupts when ``status == INTERRUPTED``.  Empty for all
    other statuses."""

    structured_interrupts: StructuredInterrupts = field(default_factory=StructuredInterrupts)
    """Structured view of :attr:`interrupts`, grouped by subtype.

    Populated when ``status == INTERRUPTED``; all category tuples are
    empty for all other statuses.  Additive — does not replace
    :attr:`interrupts`.
    """

    def release_agents(self) -> None:
        """Drop strong references to heavy fields.

        Parity with :meth:`RunResult.release_agents` /
        :meth:`SwarmRunResult.release_agents`. Caches holding many
        completed :class:`GraphRunResult` instances can pin whole
        agent+swarm+sub-graph structures via the node executables;
        call this after you're done with ``new_items`` and
        ``node_results`` to free them for GC.
        """
        self.new_items = []
        self.node_results = {}
        if self.state is not None:
            # ``GraphState.graph`` is required during a run; making it Optional
            # cascades None-checks across every consumer. The ignore is scoped
            # to this single post-run GC-release assignment.
            self.state.graph = None  # type: ignore[assignment]
            self.state.node_results = {}
            self.state.all_items = []


@dataclass
class GraphRunResultStreaming[TContext]:
    """Streaming twin of :class:`GraphRunResult`.

    Produced by :meth:`~troopai.adk.run.runner.Runner.arun_graph_streamed`
    and ``Runner.configure().graph(graph).arun(stream=True)``. Iterate events
    in real time via :meth:`stream_events`, which yields :class:`GraphStreamEvent`
    instances until the run completes or is cancelled. Terminal fields
    (``final_output``, ``status``, ``state``, ``per_node_usage``,
    ``cumulative_usage``, etc.) are populated once the run completes.
    Cancellation is available via :meth:`cancel`.

    Attributes:
        final_output: Aggregate graph output. ``None`` while streaming;
            populated on run completion.
        status: Terminal :class:`GraphRunStatus`. ``None`` while
            streaming.
        user_prompt: Original input passed to the run entry point.
            Set at start.
        new_items: Layer 3 items accumulated so far.
        state: Live :class:`GraphState` reference.
        node_results: Latest per-node results accumulated so far.
        context: The :class:`RunContext` shared across the run.
        per_node_usage: Per-node usage accumulated so far.
        cumulative_usage: Graph-wide cumulative usage accumulated so far.
        total_supersteps: Current superstep count.
        error: Serialised error string when ``status == FAILED``;
            ``None`` otherwise.
        interrupts: Pending interrupts when ``status == INTERRUPTED``.
            Empty while streaming and for all non-interrupted terminal
            statuses.
        _event_queue: Producer-to-consumer FIFO queue. Internal; do not
            access directly.
        _run_task: Background driver task. Internal.
        _node_tasks: In-flight concurrent superstep node tasks (for
            immediate cancel). Internal.
        _cancel_mode: Current cancellation state. Internal.
        _stored_exception: Exception propagated from the driver to the
            consumer. Internal.
        _deferred_run_impl: Lazily-created driver coroutine factory used
            when the run is started outside an active event loop.
            Internal.
    """

    final_output: Any = None
    """Populated on completion; ``None`` while streaming."""

    status: GraphRunStatus | None = None
    """``None`` while streaming."""

    user_prompt: UserPrompt | None = None
    """Original input. Set on start."""

    new_items: list[RunItem] = field(default_factory=list)
    """Layer 3 items accumulated so far."""

    state: GraphState[TContext] | None = None
    """Live graph state."""

    node_results: dict[str, NodeResult] = field(default_factory=dict)
    """Latest per-node results so far."""

    context: RunContext[TContext] | None = None
    """The run context."""

    per_node_usage: dict[str, LLMUsage] = field(default_factory=dict)
    """Per-node usage accumulated so far."""

    cumulative_usage: LLMUsage = field(default_factory=LLMUsage)
    """Graph-wide cumulative usage accumulated so far."""

    total_supersteps: int = 0
    """Current superstep count."""

    error: str | None = None
    """Serialised error when ``status == FAILED``; ``None`` otherwise."""

    interrupts: tuple[Interrupt, ...] = ()
    """Pending interrupts when ``status == INTERRUPTED``.  Empty while
    streaming and for all non-interrupted terminal statuses."""

    structured_interrupts: StructuredInterrupts = field(default_factory=StructuredInterrupts)
    """Structured view of :attr:`interrupts`, grouped by subtype.

    Populated when ``status == INTERRUPTED``; all category tuples are
    empty while streaming and for all non-interrupted terminal statuses.
    Additive — does not replace :attr:`interrupts`.
    """

    _event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    """Producer (driver) -> consumer (stream_events) FIFO."""

    _run_task: asyncio.Task | None = None
    """Background run_graph_loop_streamed task."""

    _node_tasks: set[asyncio.Task] = field(default_factory=set)
    """In-flight concurrent superstep node tasks (for immediate cancel)."""

    _cancel_mode: CancelMode = CancelMode.NONE
    """Cancellation state."""

    _stored_exception: Exception | None = None
    """Exception propagated from the driver to the consumer."""

    _deferred_run_impl: Any | None = None
    """Lazily-created driver coroutine when started outside a running loop."""

    async def stream_events(self) -> AsyncIterator[Any]:
        """Yield graph events in real time; re-raise driver exceptions.

        Starts the background driver task if it has been deferred.
        Iterates until the driver enqueues the sentinel, an immediate
        cancel is requested, or the driver raises. On completion all
        terminal fields are populated.

        Raises:
            Exception: Re-raises any exception that the driver task
                stored via :meth:`set_exception`.
        """
        if self._run_task is None and self._deferred_run_impl is not None:
            self._run_task = asyncio.get_running_loop().create_task(self._deferred_run_impl())
            self._deferred_run_impl = None
        try:
            while True:
                if self._stored_exception is not None:
                    break
                if self._cancel_mode == CancelMode.IMMEDIATE:
                    break
                item = await self._event_queue.get()
                if isinstance(item, QueueCompleteSentinel):
                    self._event_queue.task_done()
                    break
                yield item
                self._event_queue.task_done()
        finally:
            # Retrieve the driver task's outcome so a driver-side exception
            # is not reported by asyncio as "never retrieved" (the consumer-
            # facing error is already in _stored_exception, re-raised below).
            # When the consumer stops iterating before the run finished — an
            # early ``break`` / ``aclose()`` on the generator, or an
            # IMMEDIATE cancel — the driver task is still running: cancel it
            # (and any in-flight node tasks it spawned) rather than awaiting
            # the whole run to completion, which would defeat early
            # termination and keep the graph executing detached. A run that
            # already finished leaves the task done, so the cancel is skipped
            # and we simply drain it.
            if self._run_task is not None:
                if not self._run_task.done():
                    self._run_task.cancel()
                    for node_task in self._node_tasks:
                        if not node_task.done():
                            node_task.cancel()
                with contextlib.suppress(BaseException):
                    await self._run_task
            if self._stored_exception is not None:
                raise self._stored_exception

    def cancel(self, mode: Literal["immediate", "after_superstep", "drain"] = "immediate") -> None:
        """Cancel the streamed run.

        ``immediate``: drop pending events, cancel the driver task and
        every in-flight node task, wake the consumer with the sentinel.
        ``after_superstep``: cooperative — the driver stops at the next
        superstep boundary.
        ``drain``: let in-flight nodes in the current superstep complete,
        schedule NO new nodes, checkpoint, then exit cleanly.

        Args:
            mode: ``"immediate"`` (default) cancels everything now;
                ``"after_superstep"`` waits for the current superstep
                to finish before stopping; ``"drain"`` waits for in-flight
                nodes without scheduling new ones.
        """
        if mode == "immediate":
            self._cancel_mode = CancelMode.IMMEDIATE
            while not self._event_queue.empty():
                try:
                    self._event_queue.get_nowait()
                    self._event_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            if self._run_task is not None and not self._run_task.done():
                self._run_task.cancel()
            for t in self._node_tasks:
                if not t.done():
                    t.cancel()
            with contextlib.suppress(asyncio.QueueFull):
                self._event_queue.put_nowait(QueueCompleteSentinel())
        elif mode == "drain":
            self._cancel_mode = CancelMode.DRAIN
        else:
            self._cancel_mode = CancelMode.AFTER_SUPERSTEP

    async def put_event(self, event: Any) -> None:
        """Enqueue an event unless an immediate cancel is in flight.

        Args:
            event: The :class:`GraphStreamEvent` (or any object) to enqueue.
        """
        if self._cancel_mode != CancelMode.IMMEDIATE:
            await self._event_queue.put(event)

    async def complete(self) -> None:
        """Signal end of stream (no-op once an immediate cancel drained the queue)."""
        if self._cancel_mode != CancelMode.IMMEDIATE:
            await self._event_queue.put(QueueCompleteSentinel())

    def set_exception(self, exc: Exception) -> None:
        """Store an exception to re-raise out of stream_events().

        Args:
            exc: The exception the driver encountered.
        """
        self._stored_exception = exc

    def set_run_task(self, task: asyncio.Task) -> None:
        """Record the background driver task.

        Args:
            task: The ``asyncio.Task`` running the graph loop driver.
        """
        self._run_task = task

    def set_deferred_run_impl(self, impl: Any) -> None:
        """Store the driver coroutine factory for lazy task creation.

        Called when :meth:`Runner.arun_graph_streamed` is invoked outside
        an active event loop. The task is created on the first call to
        :meth:`stream_events`.

        Args:
            impl: A zero-argument callable that returns the driver
                coroutine when called.
        """
        self._deferred_run_impl = impl

    def register_node_task(self, task: asyncio.Task) -> None:
        """Track an in-flight node task for immediate cancel.

        Args:
            task: The ``asyncio.Task`` running the node's executable.
        """
        self._node_tasks.add(task)

    def discard_node_task(self, task: asyncio.Task) -> None:
        """Stop tracking a finished node task.

        Args:
            task: The completed ``asyncio.Task`` to remove from the
                tracking set.
        """
        self._node_tasks.discard(task)

    @property
    def cancel_mode(self) -> CancelMode:
        """Current cancellation mode."""
        return self._cancel_mode


__all__ = [
    "GraphRunResult",
    "GraphRunResultStreaming",
    "GraphRunStatus",
    "StructuredInterrupts",
]

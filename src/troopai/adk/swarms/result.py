"""Result types for swarm execution.

Mirrors the shape of :class:`~troopai.adk.types.run.run_result.RunResult`
so consumers who already read ``.final_output`` / ``.new_items`` /
``.context`` on single-agent runs get the same ergonomics for swarm runs.
Adds swarm-specific fields that do not make sense on a single-agent
result:

- ``stop_reason`` — the :class:`~troopai.adk.swarms.stop_reason.StopReason`
  that ended the run (never ``None`` on a completed swarm — absence of
  a stop reason is an invalid state because the driver only builds a
  result *because* a termination fired).
- ``state`` — the final :class:`~troopai.adk.swarms.state.SwarmState`,
  serializable for inspection / persistence.
- ``last_agent`` — the agent that emitted the terminal output (the one
  that called ``swarm_done`` or produced the last turn when a
  non-explicit termination fired).
- ``per_member_usage`` — token usage broken down by agent name, for
  cost attribution.

``SwarmRunResultStreaming`` is the streaming twin returned by
:meth:`Runner.arun_swarm_streamed`; lives in this file so the public
API shape is discoverable from one import.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, override

from troopai.adk.graphs.interrupt import Interrupt
from troopai.adk.run.stream import CancelMode, QueueCompleteSentinel
from troopai.adk.swarms.stop_reason import StopReason
from troopai.adk.types.tokens.llm_usage import LLMUsage

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.types import UserPrompt
    from troopai.adk.swarms.events import SwarmEvent
    from troopai.adk.swarms.state import SwarmState
    from troopai.adk.types.items.items import RunItem


TContext = TypeVar("TContext")

_OUTPUT_PREVIEW_CHARS = 60
"""Max characters of ``final_output`` shown in ``SwarmRunResult.__repr__``."""


def _output_preview(output: Any) -> str:
    """Render a one-line, length-capped preview of a final output for reprs."""
    if output is None:
        return "None"
    text = output if isinstance(output, str) else repr(output)
    text = text.replace("\n", " ")
    if len(text) > _OUTPUT_PREVIEW_CHARS:
        text = text[: _OUTPUT_PREVIEW_CHARS - 1] + "…"
    return repr(text) if isinstance(output, str) else text


@dataclass
class SwarmRunResult[TContext]:
    """Result of a completed swarm run.

    A ``SwarmRunResult`` is only produced when a
    :class:`~troopai.adk.swarms.termination.TerminationCondition` fires
    or a hard guard trips with a clean
    :class:`~troopai.adk.swarms.stop_reason.StopReason`. Hard-crash exits
    (e.g. ``MaxTurnsExceeded`` from the underlying runner loop) still
    raise rather than returning a result — same rule as single-agent
    runs.

    Attributes:
        final_output: Terminal output from the swarm. When stopped via
            ``swarm_done`` the driver resolves this to the terminal
            agent's last text (``ItemHelpers.extract_last_text`` over
            that turn's items) if the inner run did not already
            populate ``final_output``. For other terminations it is the
            last turn's ``final_output`` when present.
        stop_reason: Why the swarm stopped. Never ``None``.
        user_prompt: The original input passed to ``arun_swarm``.
        new_items: Layer 3 RunItems produced across the entire swarm
            run, in order. Equivalent to ``state.shared_history``
            but surfaced at the top level for API parity with
            :class:`RunResult`.
        state: Final swarm state — serializable via ``to_json()`` for
            inspection, post-hoc analysis, or warm-start of a follow-up
            run (useful for HITL pause/resume patterns built on
            :class:`~troopai.adk.swarms.termination.HandoffToTermination`).
        last_agent: The agent that produced the terminal output. When
            stopped via ``swarm_done`` this is the emitter; otherwise
            the agent active on the final turn.
        context: The :class:`RunContext` shared across the whole run
            (usage tracking, user context).
        per_member_usage: Per-agent token usage breakdown, keyed by
            agent name. Sum across agents equals
            ``state.cumulative_usage``. Useful for cost attribution in
            multi-agent workflows.
        total_turns: Mirror of ``state.total_turns`` for quick access
            without unpacking state.
        handoff_count: Mirror of ``state.handoff_count``.
    """

    final_output: Any
    """The terminal output. ``None`` when stopped via ``swarm_done(reason=...)``
    with no ``final_output`` payload."""

    stop_reason: StopReason
    """Why the swarm stopped. Never ``None`` on a valid ``SwarmRunResult``."""

    user_prompt: UserPrompt
    """The original user prompt passed to ``Runner.arun_swarm``."""

    new_items: list[RunItem] = field(default_factory=list)
    """Layer 3 items produced across the whole run, in chronological order."""

    state: SwarmState[TContext] | None = None
    """Final swarm state. Serializable via ``to_json()``."""

    last_agent: Agent[TContext] | None = None
    """The agent that produced the terminal output."""

    context: RunContext[TContext] | None = None
    """The run context (usage, user context). ``None`` on a result that
    was built without a run context, which should not happen in practice."""

    per_member_usage: dict[str, LLMUsage] = field(default_factory=dict)
    """Per-agent usage breakdown, keyed by agent name."""

    total_turns: int = 0
    """Mirror of ``state.total_turns``."""

    handoff_count: int = 0
    """Mirror of ``state.handoff_count``."""

    interrupts: tuple[Interrupt, ...] = ()
    """Interrupts parked when the swarm suspends mid-turn.

    Empty tuple on a clean run. When the swarm exits with
    ``stop_reason.kind == "interrupted"`` this carries the
    :class:`Interrupt` objects keyed by member name in
    :attr:`SwarmState.pending_interrupts`, in lexicographic order.
    Mirrors :attr:`GraphRunResult.interrupts`."""

    def release_agents(self) -> None:
        """Drop strong references to agents and item history.

        Parity with :meth:`RunResult.release_agents`. Long-lived caches
        holding many completed ``SwarmRunResult`` instances can pin
        whole agent graphs (system prompts, tool closures, policy
        references) in memory; calling this after you're done with the
        heavyweight fields keeps the cheap metadata available while
        freeing the agents for GC.
        """
        self.last_agent = None
        self.new_items = []
        if self.state is not None:
            # ``SwarmState.swarm`` and ``current_agent`` are required during a
            # run; making them Optional cascades None-checks across every
            # consumer. The ignores are scoped to this GC-release block.
            self.state.swarm = None  # type: ignore[assignment]
            self.state.current_agent = None  # type: ignore[assignment]

    @override
    def __repr__(self) -> str:
        """One-line run summary for humans.

        The full dataclass repr dumps every RunItem and the whole
        ``SwarmState`` — unreadable in a REPL or log line. This shows
        what a human checks first: which swarm, why it stopped, how
        much it cost, and a preview of the output. Tolerates
        :meth:`release_agents` (``state.swarm`` may be ``None``).
        """
        parts: list[str] = []
        swarm = self.state.swarm if self.state is not None else None
        if swarm is not None and swarm.name is not None:
            parts.append(f"swarm={swarm.name!r}")
        parts.append(f"stop={self.stop_reason.kind!r}")
        parts.append(f"turns={self.total_turns}")
        parts.append(f"handoffs={self.handoff_count}")
        if self.state is not None:
            parts.append(f"tokens={self.state.cumulative_usage.total_tokens}")
        parts.append(f"final_output={_output_preview(self.final_output)}")
        return f"SwarmRunResult({', '.join(parts)})"


@dataclass
class SwarmRunResultStreaming[TContext]:
    """Streaming twin of :class:`SwarmRunResult`.

    Produced by :meth:`Runner.arun_swarm_streamed`. Iterate events
    in real time via :meth:`stream_events`, which yields events
    from the same union as ``Runner.arun_streamed`` plus the
    swarm-scoped variants (``SwarmStartEvent``,
    ``SwarmTurnStartEvent``, ``SwarmHandoffEvent``,
    ``SwarmTurnEndEvent`` / ``SwarmTurnInterruptEvent``,
    ``SwarmDoneEvent``). Cancellation is available via
    :meth:`cancel`. Terminal fields are populated once the run
    completes.

    Attributes:
        user_prompt: The original user prompt passed to
            ``Runner.arun_swarm_streamed``.
        final_output: Populated when the run completes; ``None`` while
            still streaming.
        stop_reason: ``None`` while streaming, set on completion.
        state: Live swarm state — the same instance the driver mutates.
            Read-only from the consumer's perspective.
        last_agent: Most recently active agent.
        context: The run context.
        per_member_usage: Per-agent usage breakdown accumulated so far,
            keyed by agent name.
        new_items: Layer 3 items accumulated so far. Grows as the
            stream progresses.
        interrupts: Pending interrupts when the swarm suspends
            mid-turn. Empty while streaming; populated on terminal
            interrupt-style stop reasons.
        total_turns: Mirror of ``state.total_turns``.
        handoff_count: Mirror of ``state.handoff_count``.
    """

    user_prompt: UserPrompt
    """The original user prompt passed to ``Runner.arun_swarm_streamed``."""

    final_output: Any = None
    """Populated when the run completes; ``None`` while still streaming."""

    stop_reason: StopReason | None = None
    """``None`` while streaming, set on completion."""

    state: SwarmState[TContext] | None = None
    """Live swarm state — the same instance the driver mutates. Read-only
    from the consumer's perspective."""

    last_agent: Agent[TContext] | None = None
    """Most recently active agent."""

    context: RunContext[TContext] | None = None
    """The run context."""

    per_member_usage: dict[str, LLMUsage] = field(default_factory=dict)
    """Per-agent usage breakdown accumulated so far."""

    new_items: list[RunItem] = field(default_factory=list)
    """Layer 3 items accumulated so far. Grows as the stream progresses."""

    interrupts: tuple[Interrupt, ...] = ()
    """Pending interrupts when the swarm suspends mid-turn.  Empty while
    streaming; populated on terminal interrupt-style stop reasons."""

    total_turns: int = 0
    """Mirror of ``state.total_turns``."""

    handoff_count: int = 0
    """Mirror of ``state.handoff_count``."""

    _event_queue: asyncio.Queue[SwarmEvent | QueueCompleteSentinel] = field(default_factory=asyncio.Queue)
    """Producer (driver) -> consumer (stream_events) FIFO."""

    _run_task: asyncio.Task[None] | None = None
    """Background swarm-driver task."""

    _cancel_mode: CancelMode = CancelMode.NONE
    """Cancellation state."""

    _stored_exception: BaseException | None = None
    """Exception propagated from the driver to the consumer."""

    _deferred_run_impl: Callable[[], Coroutine[Any, Any, None]] | None = None
    """Lazily-created driver coroutine when started outside a running loop."""

    _complete: bool = False
    """True once a completion sentinel has been posted; guards idempotency."""

    async def stream_events(self) -> AsyncIterator[SwarmEvent]:
        """Yield swarm events in real time; re-raise driver exceptions.

        Raises:
            RuntimeError: When called on a result with no scheduled
                driver (neither ``set_run_task`` nor
                ``set_deferred_run_impl`` was called). Without a
                producer, the queue is empty and the consumer would
                otherwise block forever; the explicit error names
                the actionable misuse instead of silently hanging.
        """
        if self._run_task is None and self._deferred_run_impl is not None:
            if self._cancel_mode == CancelMode.IMMEDIATE:
                # Already cancelled before the driver ever launched (the
                # consumer called cancel() before the first stream_events()).
                # Discard the deferred impl so the driver never runs and bills
                # no LLM tokens for a run the developer already cancelled.
                self._deferred_run_impl = None
            else:
                self._run_task = asyncio.get_running_loop().create_task(self._deferred_run_impl())
                self._deferred_run_impl = None
        if self._run_task is None and self._cancel_mode != CancelMode.IMMEDIATE:
            raise RuntimeError(
                "SwarmRunResultStreaming.stream_events: no driver scheduled. "
                "The runner must call set_run_task() (or set_deferred_run_impl() "
                "when no event loop is running) before the consumer iterates "
                "the stream."
            )
        try:
            while True:
                # Check stored exception first so a driver that called
                # set_exception() WITHOUT also calling complete() still
                # wakes the consumer instead of blocking on get() forever.
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
            # Cancel a still-running driver before draining it. Breaking out
            # of the loop on an immediate cancel (or a stored exception) leaves
            # the driver task suspended at an await; cancelling it here stops
            # further member turns / LLM calls instead of letting the loop run
            # to completion and bill tokens for a stream the consumer has
            # already abandoned. Then retrieve the task's outcome so a
            # driver-side exception is not reported by asyncio as "never
            # retrieved"; the consumer-facing error is already in
            # _stored_exception and is re-raised below.
            if self._run_task is not None:
                if not self._run_task.done():
                    self._run_task.cancel()
                with contextlib.suppress(BaseException):
                    await self._run_task
            if self._stored_exception is not None:
                raise self._stored_exception

    def cancel(self) -> None:
        """Cancel the streamed run.

        Drops pending events, cancels the driver task, and wakes the
        consumer with the completion sentinel.
        """
        self._cancel_mode = CancelMode.IMMEDIATE
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
                self._event_queue.task_done()
            except asyncio.QueueEmpty:
                break
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()
        self._complete = True
        with contextlib.suppress(asyncio.QueueFull):
            self._event_queue.put_nowait(QueueCompleteSentinel())

    async def put_event(self, event: Any) -> None:
        """Enqueue an event unless an immediate cancel is in flight.

        Accepts ``Any`` because the swarm queue also carries inner-agent
        passthrough events (raw-response / run-item / agent-updated), not
        only ``SwarmEvent`` variants.
        """
        if self._cancel_mode != CancelMode.IMMEDIATE:
            await self._event_queue.put(event)

    async def complete(self) -> None:
        """Signal end of stream (idempotent)."""
        if self._cancel_mode == CancelMode.IMMEDIATE:
            return
        if self._complete:
            return
        self._complete = True
        await self._event_queue.put(QueueCompleteSentinel())

    def set_exception(self, exc: BaseException) -> None:
        """Store an exception and wake the consumer (idempotent).

        Posts the completion sentinel so a driver that crashed and
        cannot reach ``complete()`` still terminates the consumer's
        ``stream_events()`` loop. The stored exception is re-raised
        from ``stream_events()``'s ``finally`` block after the queue
        drains.
        """
        self._stored_exception = exc
        if not self._complete:
            self._complete = True
            with contextlib.suppress(asyncio.QueueFull):
                self._event_queue.put_nowait(QueueCompleteSentinel())

    def set_run_task(self, task: asyncio.Task[None]) -> None:
        """Record the background driver task."""
        self._run_task = task

    def set_deferred_run_impl(self, impl: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Store the driver coroutine factory for lazy task creation.

        Called when :meth:`Runner.arun_swarm_streamed` is invoked outside
        an active event loop. The task is created on the first call to
        :meth:`stream_events`.
        """
        self._deferred_run_impl = impl

    @property
    def cancel_mode(self) -> CancelMode:
        """Current cancellation mode."""
        return self._cancel_mode

"""Result types for Flow execution.

:class:`FlowRunResult` is the eager, non-streaming result returned by
:meth:`Runner.arun_flow`. :class:`FlowRunResultStreaming` is the
streaming variant returned by :meth:`Runner.arun_flow_streamed`.

The non-streaming result captures the final state, the cumulative LLM
usage across every step, and per-step usage attribution (useful for
cost analysis — analogous to ``GraphRunResult.per_node_usage``).

HITL surface mirrors the tool layer exactly
(:class:`troopai.adk.types.run.run_result.RunResult`):

- Both :class:`FlowRunResult` and :class:`FlowRunResultStreaming`
  expose ``deferred_steps`` plus ``checkpoint`` and a
  ``requires_action`` property.
- The streaming variant does NOT expose a live-inject
  ``send_approval`` channel — same contract as
  :class:`RunResultStreaming`. Decisions are recorded on the
  :class:`FlowCheckpoint` between stream end and the next
  :meth:`Runner.arun_flow_from_checkpoint` call.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, override

from troopai.adk.types.tokens.llm_usage import LLMUsage

if TYPE_CHECKING:
    from troopai.adk.flows.checkpoint import FlowCheckpoint
    from troopai.adk.flows.deferred import FlowDeferredStep
    from troopai.adk.flows.events import FlowEvent
    from troopai.adk.types.items.items import RunItem
    from troopai.adk.types.run.guardrail_audit import GuardrailAuditRecord


_STATE_PREVIEW_CHARS = 60
"""Max characters of ``final_state`` shown in ``FlowRunResult.__repr__``."""


def _state_preview(state: Any) -> str:
    """Render a one-line, length-capped preview of a final state for reprs."""
    if state is None:
        return "None"
    text = state if isinstance(state, str) else repr(state)
    text = text.replace("\n", " ")
    if len(text) > _STATE_PREVIEW_CHARS:
        text = text[: _STATE_PREVIEW_CHARS - 1] + "…"
    return repr(text) if isinstance(state, str) else text


FlowRunStatus = Literal[
    "completed",
    "failed",
    "deferred",
    "halted_max_steps",
    "halted_max_tokens",
]
"""Final status of a Flow run.

- ``"completed"`` — every reachable step ran and the flow terminated
  cleanly with no pending listeners.
- ``"failed"`` — a step raised under ``error_policy="halt"`` and the
  flow stopped immediately. The :attr:`FlowRunResult.error` field
  carries the message.
- ``"deferred"`` — one or more steps with ``requires_approval`` gates
  fired and the run halted pending human decisions. The
  :attr:`FlowRunResult.deferred_steps` field lists them and
  :attr:`FlowRunResult.checkpoint` carries the resume payload.
- ``"halted_max_steps"`` — the :attr:`FlowConfig.max_steps` cap was hit
  before the flow could terminate naturally.
- ``"halted_max_tokens"`` — the :attr:`FlowConfig.max_total_tokens` cap
  was hit before the flow could terminate naturally.
"""


@dataclass(frozen=True, kw_only=True)
class FlowRunResult[StateT]:
    """Result of a completed Flow run.

    Attributes:
        final_state: The :class:`Flow` instance's ``state`` after every
            step has run. Read-only from the consumer's perspective.
        flow_id: Stable identifier of the Flow instance.
        status: Final status — see :data:`FlowRunStatus`.
        completed_steps: Tuple of step method names that ran, in
            completion order. Skip slots from listener gates that never
            fired are NOT included.
        cumulative_usage: Sum of LLM usage across every step's inner
            :meth:`Runner.arun` calls. Equals
            ``sum(per_step_usage.values())``.
        per_step_usage: Mapping from step name to LLM usage delta for
            that step's body. Useful for per-step cost attribution —
            analogous to :attr:`troopai.adk.graphs.result.GraphRunResult.per_node_usage`.
        new_items: Tuple of :class:`RunItem` instances produced by inner
            :meth:`Runner.arun` calls inside step bodies. Order is the
            order in which steps completed.
        error: Human-readable error description when ``status="failed"``;
            ``None`` otherwise.
        deferred_steps: Steps paused by a ``requires_approval`` gate.
            Populated when ``status="deferred"``; empty tuple otherwise.
        checkpoint: Resume payload when ``status="deferred"``. Pass to
            :meth:`Runner.arun_flow_from_checkpoint` to continue.
            ``None`` when not deferred.
        metadata: Open-ended developer metadata dict. The framework
            never writes to it; useful for tracing tags carried alongside
            the result.
    """

    final_state: StateT
    """The Flow's typed state after every step has run."""

    flow_id: str
    """Stable identifier of the Flow instance."""

    status: FlowRunStatus
    """Final flow status."""

    completed_steps: tuple[str, ...]
    """Step method names that ran, in completion order."""

    cumulative_usage: LLMUsage
    """Sum of all step LLM usage."""

    per_step_usage: dict[str, LLMUsage] = field(default_factory=dict)
    """Per-step LLM usage attribution."""

    new_items: tuple[RunItem, ...] = ()
    """RunItems produced by inner Runner.arun calls inside step bodies."""

    error: str | None = None
    """Human-readable error description when status is ``"failed"``."""

    deferred_steps: tuple[FlowDeferredStep, ...] = ()
    """Steps paused by a ``requires_approval`` gate. Populated when ``status="deferred"``."""

    checkpoint: FlowCheckpoint | None = None
    """Resume payload when ``status="deferred"``. Pass to ``Runner.arun_flow_from_checkpoint``."""

    guardrail_audit: tuple[GuardrailAuditRecord, ...] = ()
    """Per-action guardrail audit trail (hashes, never raw payloads) for this run.

    Populated when the flow shares a run context (the opt-in ``flow.run_context``);
    empty otherwise."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Open-ended developer metadata; framework never writes here."""

    @override
    def __repr__(self) -> str:
        """One-line run summary for humans.

        The full dataclass repr dumps the entire ``final_state`` —
        unreadable in a REPL or log line. This shows what a human
        checks first: which flow, how it ended, how many steps ran,
        and a capped preview of the final state.
        """
        parts: list[str] = []
        parts.append(f"flow_id={self.flow_id!r}")
        parts.append(f"status={self.status!r}")
        parts.append(f"steps={len(self.completed_steps)}")
        parts.append(f"final_state={_state_preview(self.final_state)}")
        return f"FlowRunResult({', '.join(parts)})"

    @property
    def requires_action(self) -> bool:
        """``True`` when at least one step is awaiting human approval.

        Exact mirror of
        :attr:`troopai.adk.types.run.run_result.RunResult.requires_action`
        at the Flow layer. Equivalent to
        ``len(deferred_steps) > 0`` — true exactly when
        ``status == "deferred"``.
        """
        return len(self.deferred_steps) > 0


@dataclass
class FlowRunResultStreaming[StateT]:
    """Streaming variant of :class:`FlowRunResult`.

    Wraps an :class:`asyncio.Queue` of :class:`FlowEvent` items produced
    by a background producer task (the streaming flow loop). Consumers
    iterate via :meth:`stream_events` and inspect the final state /
    status after the stream completes.

    The result instance is created by :meth:`Runner.arun_flow_streamed`
    BEFORE the producer task starts running, so callers can register
    cancel handlers / inspect ``flow_id`` immediately. The producer is
    scheduled lazily on the first :meth:`stream_events` call (same
    pattern as :class:`troopai.adk.run.stream.RunResultStreaming`).

    Attributes:
        flow_id: Stable identifier of the Flow instance being run.
        final_state: Populated when the stream completes; ``None`` while
            still streaming.
        status: Populated when the stream completes; ``None`` while
            still streaming.
        completed_steps: Accumulated as steps complete during the run.
        cumulative_usage: Accumulated LLM usage delta as steps complete.
        per_step_usage: Per-step LLM usage attribution, accumulated as
            each step body returns.
        new_items: :class:`RunItem` instances accumulated as inner
            :meth:`Runner.arun` calls produce results.
        error: Set when a step raises and ``error_policy="halt"``;
            ``None`` otherwise.
        deferred_steps: Set when ``status="deferred"``. Mirrors
            :attr:`FlowRunResult.deferred_steps`.
        checkpoint: Resume payload when ``status="deferred"``. The
            developer records decisions on this checkpoint off-stream
            via :meth:`FlowCheckpoint.approve` / :meth:`FlowCheckpoint.reject`
            and resumes via :meth:`Runner.arun_flow_from_checkpoint`.
            There is no live-inject channel on this class.
    """

    flow_id: str
    """Stable identifier of the Flow instance being run."""

    final_state: StateT | None = None
    """Populated when the stream completes; ``None`` while streaming."""

    status: FlowRunStatus | None = None
    """Populated when the stream completes; ``None`` while streaming."""

    completed_steps: tuple[str, ...] = ()
    """Accumulated as steps complete."""

    cumulative_usage: LLMUsage = field(default_factory=LLMUsage)
    """Accumulated as steps complete."""

    per_step_usage: dict[str, LLMUsage] = field(default_factory=dict)
    """Accumulated as steps complete."""

    new_items: tuple[RunItem, ...] = ()
    """Accumulated as inner Runner.arun calls produce RunItems."""

    error: str | None = None
    """Set when a step raises and ``error_policy="halt"``."""

    deferred_steps: tuple[FlowDeferredStep, ...] = ()
    """Set when ``status="deferred"``. Mirrors :attr:`FlowRunResult.deferred_steps`."""

    checkpoint: FlowCheckpoint | None = None
    """Resume payload when ``status="deferred"``.

    The developer records :meth:`FlowCheckpoint.approve` /
    :meth:`FlowCheckpoint.reject` decisions on this checkpoint off-stream,
    then resumes via :meth:`Runner.arun_flow_from_checkpoint(flow, checkpoint)`.
    There is intentionally no live-inject channel on this class — same
    contract as :class:`RunResultStreaming`.
    """

    guardrail_audit: tuple[GuardrailAuditRecord, ...] = ()
    """Per-action guardrail audit trail (hashes, never raw payloads). Mirrors
    :attr:`FlowRunResult.guardrail_audit`; populated when the stream completes."""

    _event_queue: asyncio.Queue[FlowEvent | None] = field(default_factory=asyncio.Queue)
    """Internal — producer pushes events; ``None`` is the end-of-stream sentinel."""

    _producer_task: asyncio.Task[None] | None = None
    """Internal — background task driving the executor; ``None`` until first stream_events()."""

    _deferred_run_impl: Callable[[], Coroutine[Any, Any, None]] | None = field(default=None, repr=False)
    """Internal — producer coroutine factory scheduled lazily on the first
    ``stream_events()`` call when ``arun_flow_streamed`` ran outside a loop."""

    def push_event(self, event: FlowEvent) -> None:
        """Enqueue a :class:`FlowEvent` for consumers iterating :meth:`stream_events`.

        Public accessor used by :class:`troopai.adk.run.runner.Runner` to
        forward executor events into the queue without reaching into
        the private ``_event_queue`` attribute. Same encapsulation
        pattern as :meth:`troopai.adk.run.stream.RunResultStreaming.put_event`.

        Args:
            event: The event to enqueue.
        """
        self._event_queue.put_nowait(event)

    def complete(self) -> None:
        """Signal end-of-stream by pushing the sentinel ``None``.

        Called by the producer task in its ``finally`` block after the
        executor returns, regardless of whether the run succeeded or
        failed. Subsequent calls are idempotent at the queue level — a
        second sentinel is harmless because :meth:`stream_events` exits
        on the first one it sees.
        """
        self._event_queue.put_nowait(None)

    def set_producer_task(self, task: asyncio.Task[None]) -> None:
        """Set the background producer task driving the executor.

        Public accessor used by :class:`troopai.adk.run.runner.Runner` to
        attach the task without reaching into the private
        ``_producer_task`` attribute. Same encapsulation pattern as
        :meth:`troopai.adk.run.stream.RunResultStreaming.set_run_task`.

        Args:
            task: The producer task scheduled on the event loop.
        """
        self._producer_task = task

    def set_deferred_run_impl(self, impl: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Store the producer coroutine factory for lazy scheduling.

        Used by :meth:`troopai.adk.run.runner.Runner.arun_flow_streamed`
        when it is called outside a running event loop: no producer task
        can be created yet, so the factory is held here and scheduled on
        the first :meth:`stream_events` call (which runs inside a loop).
        Same encapsulation pattern as
        :meth:`troopai.adk.run.stream.RunResultStreaming.set_deferred_run_impl`.

        Args:
            impl: A zero-argument callable returning the producer coroutine.
        """
        self._deferred_run_impl = impl

    @property
    def requires_action(self) -> bool:
        """``True`` when the stream terminated awaiting human approval.

        Mirrors :attr:`FlowRunResult.requires_action` and
        :attr:`troopai.adk.types.run.run_result.RunResult.requires_action`.
        Equivalent to ``len(deferred_steps) > 0``.
        """
        return len(self.deferred_steps) > 0

    async def stream_events(self) -> AsyncIterator[FlowEvent]:
        """Async iterator over the events produced during the Flow run.

        Drains :attr:`_event_queue` until the producer sends the
        end-of-stream sentinel (``None``). Consumers MAY call
        ``stream_events()`` at most once per result instance — the queue
        is single-consumer.

        Yields:
            :class:`FlowEvent` instances in the order they were produced.
        """
        # Lazy producer scheduling for when arun_flow_streamed was called
        # outside a running event loop: no producer task could be created
        # then, so it is scheduled here on first iteration (inside a loop).
        if self._producer_task is None and self._deferred_run_impl is not None:
            self._producer_task = asyncio.get_running_loop().create_task(self._deferred_run_impl())
            self._deferred_run_impl = None
        try:
            while True:
                event = await self._event_queue.get()
                if event is None:
                    return
                yield event
        finally:
            # Early close (consumer breaks out and calls aclose, or the
            # generator is GC'd) must not leak the background producer driving
            # the executor. Cancel it and await the cancellation so it unwinds
            # cleanly instead of running to completion against an abandoned
            # queue. Mirrors RunResultStreaming._cleanup.
            task = self._producer_task
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

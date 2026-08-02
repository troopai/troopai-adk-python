"""Streaming support for agent execution.

This module provides streaming capabilities for real-time agent responses:
- StreamEvent types for different event categories
- RunResultStreaming for streaming execution results
- Cancellation support for graceful termination

Uses ``response_format`` (JSON schema mode) for structured output.

Example:
    from troopai.adk import Agent, Runner

    agent = Agent(name="Assistant", system_prompt="You are helpful.")

    # Stream execution
    result = Runner.run(agent, "Write a story", stream=True)

    async for event in result.stream_events():
        if event.type == "raw_response_event":
            logger.info(event.data)
        elif event.type == "run_item_stream_event":
            logger.info(f"\\n[{event.name}]: {event.item}")

    logger.info(f"Final: {result.final_output}")
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from troopai.adk.agents.agent_guardrails import AgentGuardrailResults
from troopai.adk.run.config import DEFAULT_MAX_TURNS

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.llms.llm_usage import LLMUsage
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.state import RunState
    from troopai.adk.swarms.yield_signal import SwarmYieldSignal
    from troopai.adk.tools.deferred_tool import DeferredToolRequests
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.items.items import RunItem
    from troopai.adk.types.run.guardrail_audit import GuardrailAuditRecord


class RunItemType(StrEnum):
    """Types of run items that can be streamed."""

    MESSAGE_OUTPUT_CREATED = "message_output_created"
    """A message response from the agent."""

    TOOL_CALLED = "tool_called"
    """A tool was invoked by the agent."""

    TOOL_OUTPUT = "tool_output"
    """A tool returned its result."""

    HANDOFF_REQUESTED = "handoff_requested"
    """Agent requested handoff to another agent."""

    HANDOFF_OCCURRED = "handoff_occurred"
    """Handoff to another agent completed."""

    GUARDRAIL_TRIGGERED = "guardrail_triggered"
    """A guardrail was triggered."""

    PARTIAL_OUTPUT = "partial_output"
    """Partial structured output (for output_type with streaming)."""

    TOOL_PARTIAL_OUTPUT = "tool_partial_output"
    """A streaming tool yielded a partial event.

    Emitted by the tool executor while draining a streaming function
    tool's ``AsyncIterator[ToolStreamEvent]``. The LLM still receives
    exactly one ``TOOL_OUTPUT`` for the tool call (carrying the final
    accumulated value); these partial events are surfaced to consumers
    of ``Runner.arun(stream=True)`` only.
    """

    TOOL_APPROVAL_REQUESTED = "tool_approval_requested"
    """A tool requires human approval before execution (HITL)."""

    # Routing events
    SIGNAL_EXTRACTED = "signal_extracted"
    """Signals were extracted from input."""

    ROUTING_EVALUATED = "routing_evaluated"
    """Routing decision was made."""

    ROUTE_MATCHED = "route_matched"
    """A route matched and handoff will occur."""


@dataclass
class RawResponseStreamEvent:
    """Raw streaming event directly from the LLM.

    Contains token-level data as it streams from the model.
    Useful for real-time display of agent responses.

    Attributes:
        data: Raw streaming data (token, delta, or chunk).
        type: Discriminator literal — always ``"raw_response_event"``.
    """

    data: Any
    """Raw streaming data (token, delta, or chunk)."""

    type: Literal["raw_response_event"] = "raw_response_event"


@dataclass
class RunItemStreamEvent:
    """Semantic event wrapping a run item.

    Provides higher-level events like tool calls, outputs,
    and handoffs during agent execution.

    Attributes:
        name: The :class:`RunItemType` discriminating which kind of item this is.
        item: The actual item data (message, tool call, etc.).
        type: Discriminator literal — always ``"run_item_stream_event"``.
    """

    name: RunItemType
    """The type of run item."""

    item: Any
    """The actual item data (message, tool call, etc.)."""

    type: Literal["run_item_stream_event"] = "run_item_stream_event"


@dataclass
class AgentUpdatedStreamEvent:
    """Event signaling agent transition during handoffs.

    Emitted when execution switches from one agent to another.

    Attributes:
        new_agent: The agent that is now executing.
        type: Discriminator literal — always ``"agent_updated_stream_event"``.
    """

    new_agent: Agent
    """The agent that is now executing."""

    type: Literal["agent_updated_stream_event"] = "agent_updated_stream_event"


class HookEventKind(StrEnum):
    """Discriminator values for :class:`HookLifecycleEvent`.

    Each member corresponds to a :class:`~troopai.adk.hooks.RunHooks`
    method that fires at a tool or guardrail lifecycle boundary.
    """

    TOOL_START = "tool_start"
    """Fired just before a tool is executed."""

    TOOL_END = "tool_end"
    """Fired just after a tool returns its result."""

    GUARDRAIL_INPUT_START = "guardrail_input_start"
    """Fired before an input guardrail runs."""

    GUARDRAIL_INPUT_END = "guardrail_input_end"
    """Fired after an input guardrail completes."""

    GUARDRAIL_OUTPUT_START = "guardrail_output_start"
    """Fired before an output guardrail runs."""

    GUARDRAIL_OUTPUT_END = "guardrail_output_end"
    """Fired after an output guardrail completes."""


@dataclass
class HookLifecycleEvent:
    """Stream event carrying a hook lifecycle moment.

    Emitted only when :attr:`~troopai.adk.run.config.RunConfig.include_hook_events`
    is ``True``.  Carries the hook kind and any available payload data.

    Attributes:
        kind: Which lifecycle moment this event represents.
        agent_name: Name of the agent that owns this lifecycle point.
        payload: Kind-specific data dict (``tool_name``, ``tool_input``,
            ``tool_output``, ``guardrail_name``, or ``guardrail_result`` —
            whatever is available at the call site).
        type: Discriminator literal — always ``"hook_lifecycle_event"``.
    """

    kind: HookEventKind
    """Which lifecycle moment this event represents."""

    agent_name: str
    """Name of the agent that owns this lifecycle point."""

    payload: dict[str, Any]
    """Kind-specific data dict."""

    type: Literal["hook_lifecycle_event"] = "hook_lifecycle_event"


# All single-agent stream events.
#
# Swarm runs expose an iterator that yields ``StreamEvent | SwarmEvent``
# — the :class:`~troopai.adk.swarms.events.SwarmEvent` variants bracket
# each member turn while the per-agent events below continue to flow
# unchanged. The multiplexed union lives on the swarm streaming result
# (``SwarmRunResultStreaming.stream_events``) rather than here to avoid
# a module-level import cycle between ``run.stream`` and ``swarms``
# (the ``swarms`` package transitively imports ``run.context`` via
# ``handoffs.handoff_target``).
StreamEvent = RawResponseStreamEvent | RunItemStreamEvent | AgentUpdatedStreamEvent | HookLifecycleEvent


class CancelMode(StrEnum):
    """Cancellation modes for streaming execution."""

    NONE = "none"
    """No cancellation requested."""

    IMMEDIATE = "immediate"
    """Stop immediately, clear pending events."""

    AFTER_TURN = "after_turn"
    """Complete current LLM response and tools, then stop."""

    AFTER_SUPERSTEP = "after_superstep"
    """Graph: complete the current superstep, then stop."""

    DRAIN = "drain"
    """Graph: let all in-flight nodes in the current superstep complete,
    schedule NO new nodes, checkpoint, then exit cleanly.

    Distinct from ``AFTER_SUPERSTEP`` in that new supersteps are never
    started — as soon as the last in-flight task lands the loop exits
    without evaluating barriers or scheduling the next ready set.  All
    completed node results are still recorded and the checkpointer fires
    its normal hooks before the run returns.
    """


@dataclass
class QueueCompleteSentinel:
    """Sentinel value to signal queue completion."""

    pass


@dataclass
class RunResultStreaming:
    """Result of a streaming agent run.

    Provides an async iterator over stream events and access
    to the final result after streaming completes.

    Attributes:
        current_agent: The currently executing agent.
        current_turn: The current turn number.
        max_turns: Maximum turns allowed.
        final_output: The final output (populated after streaming completes).
        is_complete: Whether streaming has completed.
        user_prompt: The original user prompt provided to the run.
        new_items: Layer 3 :class:`~troopai.adk.types.items.items.RunItem`
            objects generated during execution.
        context: The run context carrying cumulative usage metrics.
        deferred_requests: Tools captured for approval or external execution.
            ``None`` if the run completed without interruption.
        state: Serializable :class:`~troopai.adk.run.state.RunState` for
            resuming an interrupted run (HITL). ``None`` when not interrupted.
        swarm_yield: Set only by the streamed swarm driver when an agent
            turn yielded control. ``None`` for every non-swarm run.
        has_emitted_tokens: ``True`` once the first content delta has been
            forwarded to the consumer. Read by the streaming routing layer
            to decide whether escalation to a next candidate is still safe.
        guardrail_results: Per-phase agent-level guardrail audit trail for
            the streamed run. Accumulated during streaming; treat as
            read-only after ``stream_events()`` completes.

    Example:
        result = Runner.run(agent, "Hello!", stream=True)

        async for event in result.stream_events():
            match event.type:
                case "raw_response_event":
                    logger.info(event.data)
                case "run_item_stream_event":
                    if event.name == RunItemType.TOOL_CALLED:
                        logger.info(f"Calling tool: {event.item['name']}")

        logger.info(f"Final: {result.final_output}")
    """

    current_agent: Agent
    """The currently executing agent."""

    current_turn: int = 0
    """The current turn number."""

    max_turns: int = DEFAULT_MAX_TURNS
    """Maximum turns allowed."""

    final_output: Any = None
    """The final output (populated after streaming completes)."""

    is_complete: bool = False
    """Whether streaming has completed."""

    user_prompt: str | list[LLMInputContentItem] = ""

    recovered: bool = False
    """``True`` when an error handler produced ``final_output`` after the
    streamed run raised; session/memory persistence is skipped for the run."""
    """The original user prompt provided."""

    new_items: list[RunItem] = field(default_factory=list)
    """Layer 3 RunItems generated during execution."""

    context: RunContext[Any] | None = None
    """The run context with usage metrics."""

    deferred_requests: DeferredToolRequests | None = None
    """Tools captured for approval/external execution. None if run completed."""

    state: RunState | None = None
    """Serializable state for resuming interrupted runs (HITL)."""

    swarm_yield: SwarmYieldSignal | None = None
    """Set only by the streamed swarm driver when an agent turn yielded control.

    Mirrors ``RunResult.swarm_yield`` on the non-streamed path. ``None``
    for every non-swarm run. Populated by ``run_agent_loop_streamed``
    when the turn was dispatched with ``swarm_tool_names`` and the LLM
    called ``transfer_to_<member>`` or ``swarm_done``.
    """

    has_emitted_tokens: bool = False
    """True once the first content delta has been forwarded to the consumer.

    Set to ``True`` at the exact point the first ``RawResponseStreamEvent``
    is enqueued. The streaming routing layer reads this flag to decide
    whether escalation to a next candidate is still safe: escalation is
    only valid before any token reaches the consumer.
    """

    guardrail_results: AgentGuardrailResults = field(default_factory=AgentGuardrailResults)
    """Per-phase agent-level guardrail audit trail for the streamed run.

    Mirrors ``RunResult.guardrail_results``. Accumulated during
    streaming; treat as read-only after ``stream_events()`` completes.
    """

    guardrail_audit: tuple[GuardrailAuditRecord, ...] = field(default_factory=tuple)
    """Per-action guardrail audit trail across every level (agent, tool, flow).

    Mirrors ``RunResult.guardrail_audit``: hashes, never raw payloads. Empty
    when no guardrail ran; drained from the run context once the run completes.
    """

    # Internal state
    _event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    """Queue for streaming events."""

    _run_task: asyncio.Task | None = None
    """The main execution task."""

    _cancel_mode: CancelMode = CancelMode.NONE
    """Current cancellation mode."""

    _stored_exception: BaseException | None = None
    """Exception to raise after cleanup."""

    _input_guardrails_task: asyncio.Task | None = None
    """Task for running input guardrails."""

    _deferred_run_impl: Callable[[], Coroutine[Any, Any, None]] | None = None
    """Deferred run implementation for lazy task creation.

    Stored as a coroutine function when ``_run_streamed()`` is called
    outside an active event loop. The task is created lazily on the
    first call to ``stream_events()``.
    """

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        """Stream events as they are generated.

        Yields events in real-time during agent execution.
        After the iterator completes, final_output is available.

        Yields:
            StreamEvent: Events as they occur (raw tokens, items, agent changes).

        Raises:
            MaxTurnsExceeded: If agent exceeds max_turns limit.
            AgentInputGuardrailTripwireTriggered: If input guardrail fails.
            AgentOutputGuardrailTripwireTriggered: If output guardrail fails.
            Exception: Any other execution error.

        Example:
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    logger.info(event.data)
        """
        # Lazy task creation for when _run_streamed was called outside event loop
        if self._run_task is None and self._deferred_run_impl is not None:
            self._run_task = asyncio.get_running_loop().create_task(self._deferred_run_impl())
            self._deferred_run_impl = None

        try:
            while True:
                if self._stored_exception is not None:
                    self.is_complete = True
                    break

                if self._cancel_mode == CancelMode.IMMEDIATE:
                    self.is_complete = True
                    break

                # Block until the producer enqueues the next event or the
                # completion sentinel. `cancel()` enqueues a sentinel
                # synchronously, so a cancelled run wakes this up on the
                # next receive without any polling.
                item = await self._event_queue.get()

                if isinstance(item, QueueCompleteSentinel):
                    self._event_queue.task_done()
                    break

                yield item
                self._event_queue.task_done()

        finally:
            # Cleanup tasks
            await self._cleanup()

            # Raise stored exception if any
            if self._stored_exception is not None:
                raise self._stored_exception

    def cancel(self, mode: Literal["immediate", "after_turn"] = "immediate") -> None:
        """Cancel the streaming execution.

        Args:
            mode: Cancellation mode:
                - ``"immediate"``: Stop instantly. Drains pending events,
                  cancels the producer task synchronously, and enqueues a
                  sentinel so the consumer returns on its next receive.
                - ``"after_turn"``: Let the current LLM response and its
                  tool batch finish, then stop. The producer observes the
                  flag at turn-level checkpoints.

        Safe to call from any async context. The event loop does not
        need to be ticking for the flag to take effect — the consumer
        sees it on its next awake.

        Example:
            result = Runner.run(agent, "Hello!", stream=True)

            async for event in result.stream_events():
                if should_stop(event):
                    result.cancel(mode="immediate")
                    break
        """
        if mode == "immediate":
            self._cancel_mode = CancelMode.IMMEDIATE
            # Drain pending events so the consumer sees the sentinel
            # (and nothing else) on its next receive.
            while not self._event_queue.empty():
                try:
                    self._event_queue.get_nowait()
                    self._event_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            # Cancel the producer task synchronously. The task is
            # suspended at some await point; `task.cancel()` schedules a
            # CancelledError at that point. The cancellation propagates
            # out of run_impl's try/except (CancelledError is a
            # BaseException, not caught) and `finally` still runs
            # `result.complete()` — but we enqueue our own sentinel here
            # so the consumer never waits on that.
            if self._run_task is not None and not self._run_task.done():
                self._run_task.cancel()
            with contextlib.suppress(asyncio.QueueFull):
                self._event_queue.put_nowait(QueueCompleteSentinel())
        else:
            self._cancel_mode = CancelMode.AFTER_TURN

    async def _cleanup(self) -> None:
        """Clean up tasks and resources."""
        # Cancel running tasks
        for task in [
            self._run_task,
            self._input_guardrails_task,
        ]:
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def put_event(self, event: StreamEvent) -> None:
        """Add an event to the streaming queue."""
        if self._cancel_mode != CancelMode.IMMEDIATE:
            await self._event_queue.put(event)

    async def complete(self) -> None:
        """Signal that streaming execution has finished."""
        self.is_complete = True
        await self._event_queue.put(QueueCompleteSentinel())

    def set_exception(self, exc: BaseException) -> None:
        """Store an exception for later raising during stream_events()."""
        self._stored_exception = exc

    def set_run_task(self, task: asyncio.Task) -> None:
        """Set the background execution task."""
        self._run_task = task

    def set_deferred_run_impl(self, impl: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Set deferred run implementation for lazy task creation."""
        self._deferred_run_impl = impl

    def set_input_guardrails_task(self, task: asyncio.Task) -> None:
        """Set the input guardrails task."""
        self._input_guardrails_task = task

    def get_input_guardrails_task(self) -> asyncio.Task | None:
        """Get the input guardrails task."""
        return self._input_guardrails_task

    def clear_input_guardrails_task(self) -> None:
        """Clear the input guardrails task."""
        self._input_guardrails_task = None

    @property
    def cancel_mode(self) -> CancelMode:
        """Current cancellation mode (internal use)."""
        return self._cancel_mode

    @property
    def requires_action(self) -> bool:
        """True if human approval or external action is needed.

        Available after stream_events() completes.
        """
        return self.deferred_requests is not None and (
            len(self.deferred_requests.approvals) > 0 or len(self.deferred_requests.calls) > 0
        )

    @property
    def usage(self) -> LLMUsage | None:
        """Convenience property to access usage from context."""
        if self.context is not None:
            return self.context.usage
        return None

    def to_input_list(self) -> list[Any]:
        """Convert result to input list for continued conversation.

        Converts Layer 3 RunItems to Layer 1 params for passing
        as input to a subsequent Runner call. Prepends the original
        ``user_prompt`` so the returned list carries the full turn —
        ``new_items`` holds only LLM-generated items, never the user
        message.
        """
        from troopai.adk.types.items.items import ItemHelpers

        user_items: list[Any]
        if isinstance(self.user_prompt, str):
            user_items = [{"role": "user", "content": self.user_prompt}]
        else:
            user_items = list(self.user_prompt)
        return user_items + ItemHelpers.run_items_to_params(self.new_items)

"""``A2AExecutor`` — bridges a2a-sdk's ``AgentExecutor`` ABC to ``Runner.arun``.

The executor is the server-side mirror of :class:`A2AClient`. Where the
client translates outbound framework prompts into A2A wire-format
requests, the executor translates inbound A2A tasks into local
:func:`Runner.arun` calls, then projects the resulting
:class:`RunResult` back into A2A artifacts and status updates.

Implements two abstract methods from ``a2a.server.agent_execution.AgentExecutor``:

* :meth:`execute` — run a task to completion. Pulls the prompt from
  the inbound :class:`Message`, drives :func:`Runner.arun` against the
  local :class:`Agent`, publishes the result as a single text artifact,
  and marks the task ``TASK_STATE_COMPLETED``. Maps every framework
  exception class to the matching terminal :class:`TaskState`.
* :meth:`cancel` — cancel a running task by ``task_id``. Routes to the
  asyncio task tracked in :attr:`_running_tasks` so the inner
  ``Runner.arun`` coroutine receives a :class:`asyncio.CancelledError``
  and unwinds cleanly.

A single ``A2AExecutor`` instance handles many concurrent tasks. Each
``execute()`` call records its asyncio task in ``_running_tasks`` for
the duration so concurrent ``cancel()`` calls can reach in. The dict
is small (one entry per in-flight task) and protected by Python's GIL
on the dict-mutation methods — no extra lock is needed.

Task persistence is optional.  Pass a :class:`~troopai.adk.a2a.task_store.TaskStore`
to ``task_store`` to persist tasks across process restarts.  The default
(``task_store=None``) uses an :class:`~troopai.adk.a2a.task_store.InMemoryTaskStore`
which replicates the original in-memory behavior with zero new dependencies.

Tracing: every ``execute()`` call opens a :func:`function_span` with
``a2a_data={"task_id", "context_id"}`` so the inner ``Runner.arun``
spans nest as children. Same span pattern :class:`A2AClient` uses on
the outbound side; both sides share the ``a2a.<task_id>`` naming
convention.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import TYPE_CHECKING, Any, override

try:
    from a2a.helpers.proto_helpers import new_text_part
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.events import EventQueue
    from a2a.server.tasks import TaskUpdater
    from a2a.types import Message, Role, Task, TaskState, TaskStatus
except ImportError as ie:
    if ie.name is not None and ie.name != "a2a" and not ie.name.startswith("a2a."):
        # A transitive dependency of an installed a2a-sdk failed — surface
        # the real error instead of mislabeling it "extra not installed".
        raise
    raise ImportError(
        "Please install the 'a2a' extra to use A2A protocol support. Run: pip install 'troopai-adk-python[a2a]'",
        # name="a2a" lets optional-extra guards (e.g. the graphs adapter's
        # A2A fallthrough) recognize the missing extra and degrade gracefully.
        name="a2a",
    ) from ie

from troopai.adk.a2a import converters
from troopai.adk.a2a.task_store import InMemoryTaskStore, TaskStore
from troopai.adk.exceptions import (
    AgentInputGuardrailTripwireTriggered,
    AgentOutputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    UsageLimitExceeded,
)
from troopai.adk.run.runner import Runner
from troopai.adk.tracing import function_span

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.config import RunConfig

logger = logging.getLogger(__name__)


def _wrap_text_as_message(text: str) -> Message:
    """Build an A2A agent-role ``Message`` carrying a single text Part.

    Used when the executor needs to attach a human-readable failure
    reason to a non-completed terminal status (``failed`` / ``rejected``).
    The protobuf ``TaskUpdater.failed/reject/cancel`` methods accept
    ``Message | None`` — wrapping a string in a one-part agent message
    is the canonical way to surface a string reason without
    constructing a full Task.

    Args:
        text: The human-readable reason string to embed in the message.

    Returns:
        A :class:`a2a.types.Message` with ``ROLE_AGENT`` carrying a
        single text :class:`a2a.types.Part`.
    """
    return Message(role=Role.ROLE_AGENT, parts=[new_text_part(text)])


def _exception_to_state(exc: BaseException) -> TaskState:
    """Map a run exception to the appropriate terminal ``TaskState``.

    Mirrors the state-mapping table in :meth:`A2AExecutor._handle_run_exception`
    so the task-store snapshot records the same verdict that gets published
    to the A2A event queue.

    Args:
        exc: The exception that terminated the run.

    Returns:
        A :class:`~a2a.types.TaskState` value (one of COMPLETED / FAILED /
        CANCELED / REJECTED).
    """
    if isinstance(exc, AgentInputGuardrailTripwireTriggered):
        return TaskState.TASK_STATE_REJECTED
    if isinstance(exc, asyncio.CancelledError):
        return TaskState.TASK_STATE_CANCELED
    return TaskState.TASK_STATE_FAILED


class A2AExecutor(AgentExecutor):
    """Bridges A2A task execution to a local Agent's ``Runner.arun``.

    A single instance is shared across all in-flight tasks for the
    server it backs. The instance is stateful only for cancellation
    routing (``_running_tasks: dict[str, asyncio.Task]``); it holds
    no per-task buffer or replay state — those live in the
    :class:`a2a.server.tasks.TaskStore` configured on the
    :class:`A2AServer`.
    """

    def __init__(
        self,
        *,
        agent: Agent[Any],
        max_turns: int = 10,
        run_config: RunConfig | None = None,
        task_store: TaskStore | None = None,
    ) -> None:
        """Construct an :class:`A2AExecutor`.

        Args:
            agent: The local :class:`Agent` to execute when a task
                arrives.
            max_turns: Maximum local-agent loop turns per task. Mapped
                directly to ``Runner.arun(max_turns=...)``.
            run_config: Optional :class:`RunConfig` override forwarded
                to ``Runner.arun(run_config=...)``. ``None`` means the
                framework defaults apply.
            task_store: Optional :class:`~troopai.adk.a2a.task_store.TaskStore`
                for persisting tasks across process restarts.  ``None``
                (the default) uses an
                :class:`~troopai.adk.a2a.task_store.InMemoryTaskStore`
                with identical behavior to the original dict-based
                implementation.
        """
        self._agent = agent
        self._max_turns = max_turns
        self._run_config = run_config
        self._task_store: TaskStore = task_store if task_store is not None else InMemoryTaskStore()
        # Per-task asyncio.Task references for cancellation routing.
        # Keyed by A2A task_id; populated for the duration of execute()
        # and removed in the finally block.
        self._running_tasks: dict[str, asyncio.Task[None]] = {}

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Drive a single A2A task to completion.

        Method body kept under the 60-line function limit by delegating
        to :meth:`_resolve_ids`, :meth:`_enqueue_initial_task`,
        :meth:`_extract_prompt`, :meth:`_run_under_span`, and
        :meth:`_handle_run_exception`.

        Args:
            context: The a2a-sdk :class:`a2a.server.agent_execution.RequestContext`
                carrying the inbound task identifiers and message.
            event_queue: The :class:`a2a.server.events.EventQueue` to
                publish task status and artifact events onto.
        """
        try:
            task_id, context_id = self._resolve_ids(context)
        except RuntimeError as exc:
            # _resolve_ids raises when the a2a-sdk did not populate task_id /
            # context_id. Before the fix, this escaped the method without
            # publishing any terminal event, leaving the client stuck in
            # SUBMITTED forever. Catch it: synthesise fallback ids from
            # whatever partial context we have, enqueue a Task, then publish
            # FAILED so the client can unblock.
            fallback_task_id = context.task_id or ""
            fallback_context_id = context.context_id or ""
            logger.exception(
                "A2AExecutor.execute could not resolve ids (task_id=%r, context_id=%r); "
                "publishing synthetic FAILED event to unblock client.",
                fallback_task_id,
                fallback_context_id,
            )
            await self._enqueue_initial_task(event_queue, context, fallback_task_id, fallback_context_id)
            fallback_updater = TaskUpdater(event_queue, fallback_task_id, fallback_context_id)
            await fallback_updater.failed(message=_wrap_text_as_message(f"Internal server error: {exc}"))
            return
        initial_task = await self._enqueue_initial_task(event_queue, context, task_id, context_id)
        try:
            await self._task_store.save(initial_task)
        except Exception:
            logger.exception(
                "A2AExecutor: failed to persist initial task %s to task_store; wire behavior is unaffected.",
                task_id,
            )
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.update_status(TaskState.TASK_STATE_WORKING)
        prompt = self._extract_prompt(context)
        if prompt is None:
            await updater.failed(message=_wrap_text_as_message("Empty input message."))
            await self._save_task_state(task_id, context_id, TaskState.TASK_STATE_FAILED, context)
            return
        await self._run_under_span(prompt, task_id, context_id, updater, context)

    async def _run_under_span(
        self,
        prompt: str,
        task_id: str,
        context_id: str,
        updater: TaskUpdater,
        context: RequestContext,
    ) -> None:
        """Run the local agent under a tracing span; publish result + state.

        Records the asyncio task for cancellation routing, runs
        :func:`Runner.arun`, maps the result to A2A artifact +
        complete, or routes exceptions via
        :meth:`_handle_run_exception`. Persists the final terminal state
        to ``_task_store``. Cleans up :attr:`_running_tasks` in
        ``finally``.

        Args:
            prompt: The extracted prompt text to pass to
                ``Runner.arun``.
            task_id: The A2A task identifier for span naming and
                cancellation tracking.
            context_id: The conversation context identifier for span
                metadata.
            updater: The :class:`a2a.server.tasks.TaskUpdater` for
                publishing artifact and terminal status events.
            context: The inbound request context; used when persisting
                the terminal task snapshot to the store.
        """
        with function_span(
            name=f"task.{task_id}",
            input=prompt,
            a2a_data={
                "task_id": task_id,
                "context_id": context_id,
                "agent_name": self._agent.name,
            },
        ) as span:
            current_task = asyncio.current_task()
            if current_task is not None:
                self._running_tasks[task_id] = current_task
            else:
                logger.warning(
                    "A2AExecutor.execute is not running inside an asyncio.Task "
                    "(asyncio.current_task() returned None); cancel() will be a "
                    "no-op for task_id=%s. Ensure execute() is called from an "
                    "async handler running under an event loop (e.g. Starlette).",
                    task_id,
                )
            terminal_state: TaskState = TaskState.TASK_STATE_FAILED
            try:
                result = await Runner.arun(
                    self._agent,
                    user_prompt=prompt,
                    max_turns=self._max_turns,
                    run_config=self._run_config,
                )
                text = str(result.final_output)
                span.data = dataclasses.replace(span.data, output=text)
                await updater.add_artifact([new_text_part(text)])
                await updater.complete()
                terminal_state = TaskState.TASK_STATE_COMPLETED
            except (asyncio.CancelledError, Exception) as exc:
                # Route EVERY error through _handle_run_exception: the
                # well-known framework types map to their specific
                # terminal states; anything else surfaces as FAILED and
                # re-raises. Catching only the framework types would let
                # a provider error or tool bug escape, leaving the task
                # stuck WORKING forever. CancelledError is BaseException,
                # not Exception, so it stays listed explicitly; process
                # signals (KeyboardInterrupt/SystemExit) still propagate.
                terminal_state = _exception_to_state(exc)
                await self._handle_run_exception(updater, task_id, exc)
            finally:
                self._running_tasks.pop(task_id, None)
                await self._save_task_state(task_id, context_id, terminal_state, context)

    async def _save_task_state(
        self,
        task_id: str,
        context_id: str,
        state: TaskState,
        context: RequestContext,
    ) -> None:
        """Persist a terminal (or failed-early) task snapshot to the store.

        Silently logs on store errors so a persistence failure does not
        propagate into the A2A event stream.

        Args:
            task_id: The A2A task identifier.
            context_id: The conversation context identifier.
            state: The :class:`~a2a.types.TaskState` to record.
            context: The inbound request context; the message history is
                copied into the stored snapshot.
        """
        try:
            snapshot = Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=state),
                history=[context.message] if context.message is not None else [],
            )
            await self._task_store.save(snapshot)
        except Exception:
            logger.exception(
                "A2AExecutor: failed to persist task %s to task_store; wire behavior is unaffected.",
                task_id,
            )

    @staticmethod
    def _resolve_ids(context: RequestContext) -> tuple[str, str]:
        """Materialise non-null ``task_id`` / ``context_id`` from a request.

        Upstream types both fields as ``str | None`` because the a2a SDK
        auto-mints them for some flows; by the time ``execute()`` runs
        they are populated. An explicit boundary check raises if not.

        Args:
            context: The inbound :class:`a2a.server.agent_execution.RequestContext`.

        Returns:
            A ``(task_id, context_id)`` tuple, both non-empty.

        Raises:
            RuntimeError: If either identifier is empty.
        """
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        if len(task_id) == 0 or len(context_id) == 0:
            raise RuntimeError(
                "A2AExecutor.execute received a RequestContext with "
                "empty task_id or context_id; the a2a-sdk's request "
                "handler should have populated these."
            )
        return task_id, context_id

    @staticmethod
    async def _enqueue_initial_task(
        event_queue: EventQueue,
        context: RequestContext,
        task_id: str,
        context_id: str,
    ) -> Task:
        """Enqueue the initial ``Task`` event before any status update.

        The a2a-sdk's task consumer requires a ``Task`` event to be
        enqueued first; ``TaskUpdater`` only emits status / artifact
        events. Skipping this step trips
        ``InvalidAgentResponseError("Agent should enqueue Task before
        TaskStatusUpdateEvent event")`` on the first ``update_status``
        call. This is the load-bearing precondition every server-side
        request depends on.

        Args:
            event_queue: The :class:`a2a.server.events.EventQueue` to
                push the initial :class:`a2a.types.Task` event onto.
            context: The inbound request context, used to copy the
                message into the task's ``history``.
            task_id: The resolved task identifier.
            context_id: The resolved conversation context identifier.

        Returns:
            The :class:`a2a.types.Task` that was enqueued, for
            persistence to the task store.
        """
        initial_task = Task(
            id=task_id,
            context_id=context_id,
            status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            history=[context.message] if context.message is not None else [],
        )
        await event_queue.enqueue_event(initial_task)
        return initial_task

    @staticmethod
    def _extract_prompt(context: RequestContext) -> str | None:
        """Return the inbound prompt text, or ``None`` if empty.

        ``context.message`` is ``Message | None`` upstream because the
        a2a SDK can be invoked via paths that don't carry a message
        (e.g. resubscribe); ``execute`` always has one. Treats absence
        and empty-text uniformly as ``None`` so the caller emits a
        single FAILED status with a clear "empty input" reason.

        Args:
            context: The inbound :class:`a2a.server.agent_execution.RequestContext`.

        Returns:
            The extracted prompt string, or ``None`` when the message
            is absent or carries no text content.
        """
        if context.message is None:
            return None
        prompt = converters.extract_text_from_message(context.message)
        if len(prompt) == 0:
            return None
        return prompt

    @staticmethod
    async def _handle_run_exception(
        updater: TaskUpdater,
        task_id: str,
        exc: BaseException,
    ) -> None:
        """Map a framework exception to the right terminal task state.

        Splits the verdict-publication from :meth:`execute` so the
        method body stays under the 60-line function limit and the
        state-mapping table is testable in isolation. ``CancelledError``
        is published as CANCELED then re-raised so the asyncio task
        itself unwinds cleanly.

        Args:
            updater: The :class:`a2a.server.tasks.TaskUpdater` for
                publishing the terminal status event.
            task_id: The A2A task identifier, used in log messages.
            exc: The caught exception to map to a terminal state.

        Raises:
            asyncio.CancelledError: Re-raised after publishing CANCELED
                so the asyncio task unwinds normally.
            BaseException: Any exception not matching a known
                framework type is re-raised after publishing FAILED.
        """
        if isinstance(exc, AgentInputGuardrailTripwireTriggered):
            logger.info("A2A task %s rejected by input guardrail: %s", task_id, exc)
            await updater.reject(message=_wrap_text_as_message(f"Input rejected: {exc}"))
            return
        if isinstance(exc, AgentOutputGuardrailTripwireTriggered):
            logger.warning("A2A task %s failed by output guardrail: %s", task_id, exc)
            await updater.failed(message=_wrap_text_as_message(f"Output validation failed: {exc}"))
            return
        if isinstance(exc, (MaxTurnsExceeded, UsageLimitExceeded)):
            logger.warning("A2A task %s exceeded a budget limit: %s", task_id, exc)
            await updater.failed(message=_wrap_text_as_message(f"Budget exceeded: {exc}"))
            return
        if isinstance(exc, asyncio.CancelledError):
            logger.info("A2A task %s cancelled", task_id)
            await updater.cancel()
            raise exc
        # Unknown exception — surface as FAILED and re-raise.
        logger.exception("A2A task %s failed with unexpected exception", task_id)
        await updater.failed(message=_wrap_text_as_message(f"Internal error: {exc}"))
        raise exc

    @override
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Request cancellation of an in-flight task.

        Looks up the asyncio task by ``context.task_id`` and calls
        ``.cancel()`` on it. The inner ``Runner.arun`` coroutine
        receives a :class:`asyncio.CancelledError` on its next
        ``await`` point and unwinds; the ``except CancelledError``
        branch in :meth:`execute` publishes the terminal CANCELED
        status before re-raising.

        Cancelling an unknown ``task_id`` is a no-op — the task may
        have already completed before the cancel reached us, in
        which case the terminal state was already published.

        Args:
            context: The a2a-sdk :class:`a2a.server.agent_execution.RequestContext`
                carrying the ``task_id`` to cancel.
            event_queue: Unused — the :meth:`execute` coroutine
                publishes its own terminal status event when it catches
                the :class:`asyncio.CancelledError`.
        """
        del event_queue  # The execute() task publishes its own status.
        # task_id may be ``None`` on a malformed cancel; treat the
        # missing-id case as "task not tracked" — same as a stale id.
        task_id = context.task_id or ""
        task = self._running_tasks.get(task_id)
        if task is None:
            logger.debug(
                "A2A cancel request for unknown task_id %s (already completed?)",
                task_id,
            )
            return
        if task.done():
            logger.debug(
                "A2A cancel request for already-finished task %s",
                task_id,
            )
            return
        task.cancel()


__all__ = ["A2AExecutor", "TaskStore"]

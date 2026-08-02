"""Tests for ``A2AExecutor`` — the bridge between A2A and ``Runner.arun``.

The executor is the server-side counterpart to ``A2AClient``. Tests
mock ``Runner.arun`` so the test suite doesn't make real LLM calls.
Verifies:

* Happy path: ``Runner.arun`` runs to completion → status WORKING → add_artifact → complete.
* Empty input message → ``failed("Empty input")``.
* Each well-known framework exception type maps to the correct
  terminal :class:`TaskState`:
  - ``AgentInputGuardrailTripwireTriggered`` → REJECTED
  - ``AgentOutputGuardrailTripwireTriggered`` → FAILED
  - ``MaxTurnsExceeded`` / ``UsageLimitExceeded`` → FAILED
  - ``asyncio.CancelledError`` → CANCELED then re-raise
* ``cancel()`` finds the running task and calls ``.cancel()``.
* ``cancel()`` on unknown ``task_id`` is a no-op.
* The ``_running_tasks`` dict is cleaned up in the ``finally`` block.
* ``function_span`` is opened with ``a2a_data`` populated.
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip the module if optional `a2a` extra is missing.
pytest.importorskip("a2a.server.agent_execution")

from a2a.types import Message, Part, Role

from troopai.adk.a2a.executor import A2AExecutor
from troopai.adk.agents import Agent
from troopai.adk.exceptions import (
    AgentInputGuardrailTripwireTriggered,
    AgentOutputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
    UsageLimitExceeded,
)
from troopai.adk.types.run import RunResult
from troopai.adk.types.tracing.span_data import FunctionSpanData

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent() -> Agent[None]:
    return Agent(name="test_agent", system_prompt="Be helpful.")


@pytest.fixture
def executor(agent: Agent[None]) -> A2AExecutor:
    return A2AExecutor(agent=agent, max_turns=5)


def _make_message(text: str) -> Message:
    return Message(role=Role.ROLE_USER, parts=[Part(text=text)])


def _make_request_context(prompt: str = "hi", task_id: str = "t1", context_id: str = "c1") -> MagicMock:
    """Build a mock RequestContext with the fields A2AExecutor reads."""
    ctx = MagicMock()
    ctx.task_id = task_id
    ctx.context_id = context_id
    ctx.message = _make_message(prompt)
    return ctx


def _make_event_queue() -> MagicMock:
    queue = MagicMock()
    queue.enqueue_event = AsyncMock()
    return queue


def _make_run_result(text: str) -> RunResult:
    """Build a minimal RunResult mock — only fields the executor reads."""
    result = MagicMock(spec=RunResult)
    result.final_output = text
    return result


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    async def test_complete_runs_runner_and_publishes_artifact(self, executor: A2AExecutor) -> None:
        ctx = _make_request_context(prompt="What is 2+2?")
        queue = _make_event_queue()
        with patch(
            "troopai.adk.a2a.executor.Runner.arun",
            AsyncMock(return_value=_make_run_result("4")),
        ) as mock_arun:
            await executor.execute(ctx, queue)
        mock_arun.assert_awaited_once()
        # Some events should have been enqueued: at minimum the
        # WORKING status, the artifact, and the completion.
        assert queue.enqueue_event.await_count >= 3

    async def test_runner_arun_called_with_executor_config(self, executor: A2AExecutor) -> None:
        ctx = _make_request_context()
        queue = _make_event_queue()
        with patch(
            "troopai.adk.a2a.executor.Runner.arun",
            AsyncMock(return_value=_make_run_result("done")),
        ) as mock_arun:
            await executor.execute(ctx, queue)
        # The executor should pass max_turns and run_config through.
        # ``await_args`` is ``call | None`` per the unittest.mock
        # stubs; assert non-None to narrow before subscripting.
        assert mock_arun.await_args is not None
        call_kwargs = mock_arun.await_args.kwargs
        assert call_kwargs["max_turns"] == 5
        assert call_kwargs["run_config"] is None
        assert call_kwargs["user_prompt"] == "hi"


class TestEmptyInput:
    async def test_empty_message_marks_failed(self, executor: A2AExecutor) -> None:
        ctx = MagicMock()
        ctx.task_id = "t1"
        ctx.context_id = "c1"
        ctx.message = Message(role=Role.ROLE_USER, parts=[])  # empty parts list
        queue = _make_event_queue()
        # Runner should NEVER be called when the input is empty.
        with patch("troopai.adk.a2a.executor.Runner.arun", AsyncMock()) as mock_arun:
            await executor.execute(ctx, queue)
        mock_arun.assert_not_awaited()


# ---------------------------------------------------------------------------
# State-mapping table — every exception class
# ---------------------------------------------------------------------------


class TestStateMapping:
    @pytest.mark.parametrize(
        "exception_factory,expected_method_name",
        [
            # InputGuardrail trip → REJECTED (the request was unacceptable).
            (
                lambda: AgentInputGuardrailTripwireTriggered(
                    guardrail_result=MagicMock(),
                    message="injection detected",
                ),
                "reject",
            ),
            # OutputGuardrail trip → FAILED (server attempted but output bad).
            (
                lambda: AgentOutputGuardrailTripwireTriggered(
                    guardrail_result=MagicMock(),
                    message="PII leak detected",
                ),
                "failed",
            ),
            # Budget exceeded → FAILED.
            (lambda: MaxTurnsExceeded("max_turns=10 exceeded"), "failed"),
            (lambda: UsageLimitExceeded("token budget exceeded"), "failed"),
        ],
    )
    async def test_exception_maps_to_terminal_state(
        self,
        executor: A2AExecutor,
        exception_factory: object,
        expected_method_name: str,
    ) -> None:
        ctx = _make_request_context()
        queue = _make_event_queue()
        with patch("troopai.adk.a2a.executor.TaskUpdater") as mock_updater_cls:
            mock_updater = MagicMock()
            mock_updater.update_status = AsyncMock()
            mock_updater.add_artifact = AsyncMock()
            mock_updater.complete = AsyncMock()
            mock_updater.failed = AsyncMock()
            mock_updater.reject = AsyncMock()
            mock_updater.cancel = AsyncMock()
            mock_updater_cls.return_value = mock_updater
            with patch(
                "troopai.adk.a2a.executor.Runner.arun",
                AsyncMock(side_effect=exception_factory()),  # type: ignore[operator]
            ):
                await executor.execute(ctx, queue)
        # Verify the correct terminal-state method was called exactly once.
        terminal_method = getattr(mock_updater, expected_method_name)
        terminal_method.assert_awaited_once()


class TestUnexpectedException:
    async def test_unexpected_exception_marks_failed_then_reraises(self, executor: A2AExecutor) -> None:
        """A non-framework exception must surface the task as FAILED.

        The ``execute`` exception handler must route *every* error
        (a provider error, a tool bug, a plain ``RuntimeError``) through
        ``_handle_run_exception``. If it caught only the five well-known
        framework types, any other exception would escape the handler:
        ``updater.failed`` would never be called and the A2A task would
        stay WORKING forever. The handler publishes FAILED and re-raises.
        """
        ctx = _make_request_context()
        queue = _make_event_queue()
        with patch("troopai.adk.a2a.executor.TaskUpdater") as mock_updater_cls:
            mock_updater = MagicMock()
            mock_updater.update_status = AsyncMock()
            mock_updater.failed = AsyncMock()
            mock_updater_cls.return_value = mock_updater
            with (
                patch(
                    "troopai.adk.a2a.executor.Runner.arun",
                    AsyncMock(side_effect=RuntimeError("provider blew up")),
                ),
                pytest.raises(RuntimeError, match="provider blew up"),
            ):
                await executor.execute(ctx, queue)
        # The task is surfaced as FAILED, not left WORKING.
        mock_updater.failed.assert_awaited_once()
        # The running-task slot is still cleaned up.
        assert ctx.task_id not in executor._running_tasks


class TestCancellation:
    async def test_cancelled_error_publishes_cancel_then_reraises(self, executor: A2AExecutor) -> None:
        ctx = _make_request_context()
        queue = _make_event_queue()
        with patch("troopai.adk.a2a.executor.TaskUpdater") as mock_updater_cls:
            mock_updater = MagicMock()
            mock_updater.update_status = AsyncMock()
            mock_updater.cancel = AsyncMock()
            mock_updater_cls.return_value = mock_updater
            with (
                patch(
                    "troopai.adk.a2a.executor.Runner.arun",
                    AsyncMock(side_effect=asyncio.CancelledError()),
                ),
                pytest.raises(asyncio.CancelledError),
            ):
                await executor.execute(ctx, queue)
        mock_updater.cancel.assert_awaited_once()


# ---------------------------------------------------------------------------
# cancel() — task-lookup behaviour
# ---------------------------------------------------------------------------


class TestCancel:
    async def test_cancel_unknown_task_is_noop(self, executor: A2AExecutor) -> None:
        ctx = _make_request_context(task_id="unknown")
        # Should not raise.
        await executor.cancel(ctx, _make_event_queue())

    async def test_cancel_in_flight_task_calls_cancel(self, executor: A2AExecutor) -> None:
        # Insert a fake asyncio.Task into _running_tasks; verify
        # cancel() reaches it.
        async def hang() -> None:
            await asyncio.sleep(60)

        running_task = asyncio.create_task(hang())
        executor._running_tasks["t1"] = running_task

        ctx = _make_request_context(task_id="t1")
        await executor.cancel(ctx, _make_event_queue())

        # Brief yield so the cancel propagates.
        await asyncio.sleep(0)
        assert running_task.cancelled() or running_task.cancelling() > 0
        running_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await running_task

    async def test_cancel_done_task_is_noop(self, executor: A2AExecutor) -> None:
        async def quick() -> None:
            return None

        done_task = asyncio.create_task(quick())
        await done_task  # let it complete
        executor._running_tasks["t1"] = done_task
        ctx = _make_request_context(task_id="t1")
        # Should not raise — already-done tasks are skipped.
        await executor.cancel(ctx, _make_event_queue())


# ---------------------------------------------------------------------------
# _running_tasks lifecycle
# ---------------------------------------------------------------------------


class TestRunningTasksLifecycle:
    async def test_running_tasks_cleaned_up_after_success(self, executor: A2AExecutor) -> None:
        ctx = _make_request_context()
        queue = _make_event_queue()
        with patch(
            "troopai.adk.a2a.executor.Runner.arun",
            AsyncMock(return_value=_make_run_result("done")),
        ):
            await executor.execute(ctx, queue)
        assert ctx.task_id not in executor._running_tasks

    async def test_running_tasks_cleaned_up_after_exception(self, executor: A2AExecutor) -> None:
        ctx = _make_request_context()
        queue = _make_event_queue()
        with patch(
            "troopai.adk.a2a.executor.Runner.arun",
            AsyncMock(side_effect=MaxTurnsExceeded("oops")),
        ):
            await executor.execute(ctx, queue)
        assert ctx.task_id not in executor._running_tasks


# ---------------------------------------------------------------------------
# Tracing — function_span opened with a2a_data
# ---------------------------------------------------------------------------


class TestTracing:
    async def test_function_span_opened_with_a2a_data(self, executor: A2AExecutor) -> None:
        ctx = _make_request_context(task_id="t-abc", context_id="c-xyz")
        queue = _make_event_queue()
        with patch("troopai.adk.a2a.executor.function_span") as mock_span_factory:
            mock_span = MagicMock()
            mock_span.__enter__ = MagicMock(return_value=mock_span)
            mock_span.__exit__ = MagicMock(return_value=False)
            # ``span.data`` MUST be a real dataclass so the executor's
            # ``dataclasses.replace(span.data, output=text)`` call works.
            mock_span.data = FunctionSpanData(name="a2a.task.t-abc", input="hi")
            mock_span_factory.return_value = mock_span
            with patch(
                "troopai.adk.a2a.executor.Runner.arun",
                AsyncMock(return_value=_make_run_result("ok")),
            ):
                await executor.execute(ctx, queue)
        # The factory was called once with a2a_data carrying both ids.
        mock_span_factory.assert_called_once()
        call = mock_span_factory.call_args
        assert call is not None
        a2a_data = call.kwargs["a2a_data"]
        assert a2a_data["task_id"] == "t-abc"
        assert a2a_data["context_id"] == "c-xyz"
        assert a2a_data["agent_name"] == "test_agent"


class TestResolveIdsFailurePublishesTerminalEvent:
    async def test_missing_task_id_publishes_failed_not_hangs(
        self,
        executor: A2AExecutor,
    ) -> None:
        # Regression: _resolve_ids raised RuntimeError before _enqueue_initial_task,
        # leaving the client stuck in SUBMITTED with no terminal event ever published.
        # Fix: catch the RuntimeError, enqueue a Task then publish failed status.
        ctx = MagicMock()
        ctx.task_id = None  # missing — _resolve_ids will raise
        ctx.context_id = "c1"
        ctx.message = _make_message("hi")
        queue = _make_event_queue()

        # Should not raise; should instead publish a terminal event.
        await executor.execute(ctx, queue)

        # At minimum, some event must have been enqueued (the Task + failed status).
        assert queue.enqueue_event.await_count >= 1, (
            f"Expected at least one enqueue_event call when _resolve_ids fails, got {queue.enqueue_event.await_count}"
        )


class TestCurrentTaskNoneWarning:
    async def test_execute_warns_when_current_task_is_none(
        self,
        executor: A2AExecutor,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Regression: when asyncio.current_task() returns None (execute()
        # not running inside an asyncio.Task), cancel() becomes a silent
        # no-op. A logger.warning must surface this misconfiguration.
        ctx = _make_request_context()
        queue = _make_event_queue()
        with (
            patch("troopai.adk.a2a.executor.asyncio.current_task", return_value=None),
            patch(
                "troopai.adk.a2a.executor.Runner.arun",
                AsyncMock(return_value=_make_run_result("done")),
            ),
            caplog.at_level("WARNING", logger="troopai.adk.a2a.executor"),
        ):
            await executor.execute(ctx, queue)
        warning_msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("current_task" in m or "no-op" in m or "asyncio.Task" in m for m in warning_msgs), (
            f"Expected a warning about current_task=None; got: {warning_msgs}"
        )

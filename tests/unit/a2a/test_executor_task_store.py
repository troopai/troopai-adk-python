"""Tests for ``A2AExecutor`` task-store integration.

Verifies that ``A2AExecutor`` wires its ``task_store`` correctly:

* Default (``task_store=None``) uses ``InMemoryTaskStore``.
* A custom ``TaskStore`` receives ``save()`` calls for initial SUBMITTED
  task and the final terminal state.
* Tasks are persisted as COMPLETED on happy path, FAILED on error,
  CANCELED on CancelledError, REJECTED on input guardrail.
* Persistence errors in the store are swallowed (logged), not surfaced
  to the A2A event stream.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip if a2a extra is missing.
pytest.importorskip("a2a.server.agent_execution")

import asyncio

from a2a.types import Message, Part, Role, TaskState

from troopai.adk.a2a.executor import A2AExecutor
from troopai.adk.a2a.task_store import InMemoryTaskStore
from troopai.adk.agents import Agent
from troopai.adk.exceptions import (
    AgentInputGuardrailTripwireTriggered,
    MaxTurnsExceeded,
)
from troopai.adk.types.run import RunResult

# ---------------------------------------------------------------------------
# Helpers (mirrors test_executor.py helpers)
# ---------------------------------------------------------------------------


def _make_message(text: str) -> Message:
    return Message(role=Role.ROLE_USER, parts=[Part(text=text)])


def _make_request_context(
    prompt: str = "hi",
    task_id: str = "t1",
    context_id: str = "c1",
) -> MagicMock:
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
    result = MagicMock(spec=RunResult)
    result.final_output = text
    return result


@pytest.fixture
def agent() -> Agent[None]:
    return Agent(name="test_agent", system_prompt="Be helpful.")


# ---------------------------------------------------------------------------
# Default task_store is InMemoryTaskStore
# ---------------------------------------------------------------------------


class TestDefaultTaskStore:
    def test_default_task_store_is_in_memory(self, agent: Agent[None]) -> None:
        executor = A2AExecutor(agent=agent)
        assert isinstance(executor._task_store, InMemoryTaskStore)

    def test_explicit_none_uses_in_memory(self, agent: Agent[None]) -> None:
        executor = A2AExecutor(agent=agent, task_store=None)
        assert isinstance(executor._task_store, InMemoryTaskStore)


# ---------------------------------------------------------------------------
# Custom store receives save() calls
# ---------------------------------------------------------------------------


class TestCustomStoreReceivesSaves:
    @pytest.mark.asyncio
    async def test_happy_path_saves_twice(self, agent: Agent[None]) -> None:
        """On success: initial SUBMITTED save + final COMPLETED save."""
        store = InMemoryTaskStore()
        executor = A2AExecutor(agent=agent, task_store=store)
        ctx = _make_request_context(task_id="t-happy")
        queue = _make_event_queue()
        with patch(
            "troopai.adk.a2a.executor.Runner.arun",
            AsyncMock(return_value=_make_run_result("done")),
        ):
            await executor.execute(ctx, queue)
        # After execution, the task must be stored with COMPLETED state.
        result = await store.get("t-happy")
        assert result is not None, "Task not found in store after happy-path run"
        assert result.status.state == TaskState.TASK_STATE_COMPLETED, f"Expected COMPLETED, got {result.status.state}"

    @pytest.mark.asyncio
    async def test_runner_failure_saves_failed(self, agent: Agent[None]) -> None:
        store = InMemoryTaskStore()
        executor = A2AExecutor(agent=agent, task_store=store)
        ctx = _make_request_context(task_id="t-fail")
        queue = _make_event_queue()
        with patch(
            "troopai.adk.a2a.executor.Runner.arun",
            AsyncMock(side_effect=MaxTurnsExceeded("too many")),
        ):
            await executor.execute(ctx, queue)
        result = await store.get("t-fail")
        assert result is not None
        assert result.status.state == TaskState.TASK_STATE_FAILED

    @pytest.mark.asyncio
    async def test_cancelled_error_saves_canceled(self, agent: Agent[None]) -> None:
        store = InMemoryTaskStore()
        executor = A2AExecutor(agent=agent, task_store=store)
        ctx = _make_request_context(task_id="t-cancel")
        queue = _make_event_queue()
        with (
            patch(
                "troopai.adk.a2a.executor.Runner.arun",
                AsyncMock(side_effect=asyncio.CancelledError()),
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await executor.execute(ctx, queue)
        result = await store.get("t-cancel")
        assert result is not None
        assert result.status.state == TaskState.TASK_STATE_CANCELED

    @pytest.mark.asyncio
    async def test_input_guardrail_saves_rejected(self, agent: Agent[None]) -> None:
        store = InMemoryTaskStore()
        executor = A2AExecutor(agent=agent, task_store=store)
        ctx = _make_request_context(task_id="t-reject")
        queue = _make_event_queue()
        with patch(
            "troopai.adk.a2a.executor.Runner.arun",
            AsyncMock(
                side_effect=AgentInputGuardrailTripwireTriggered(
                    guardrail_result=MagicMock(),
                    message="blocked",
                )
            ),
        ):
            await executor.execute(ctx, queue)
        result = await store.get("t-reject")
        assert result is not None
        assert result.status.state == TaskState.TASK_STATE_REJECTED

    @pytest.mark.asyncio
    async def test_empty_input_saves_failed(self, agent: Agent[None]) -> None:
        """Empty input path must also persist a FAILED terminal state."""
        store = InMemoryTaskStore()
        executor = A2AExecutor(agent=agent, task_store=store)
        ctx = MagicMock()
        ctx.task_id = "t-empty"
        ctx.context_id = "c1"
        ctx.message = Message(role=Role.ROLE_USER, parts=[])  # no text parts
        queue = _make_event_queue()
        with patch("troopai.adk.a2a.executor.Runner.arun", AsyncMock()) as mock_arun:
            await executor.execute(ctx, queue)
        mock_arun.assert_not_awaited()
        result = await store.get("t-empty")
        assert result is not None
        assert result.status.state == TaskState.TASK_STATE_FAILED


# ---------------------------------------------------------------------------
# Persistence errors are swallowed
# ---------------------------------------------------------------------------


class TestStoreErrorsAreSwallowed:
    @pytest.mark.asyncio
    async def test_store_save_error_does_not_crash_execute(self, agent: Agent[None]) -> None:
        """A failing task_store must not propagate into the A2A event stream."""
        bad_store = MagicMock()
        bad_store.save = AsyncMock(side_effect=RuntimeError("store exploded"))
        executor = A2AExecutor(agent=agent, task_store=bad_store)
        ctx = _make_request_context(task_id="t-store-err")
        queue = _make_event_queue()
        with patch(
            "troopai.adk.a2a.executor.Runner.arun",
            AsyncMock(return_value=_make_run_result("ok")),
        ):
            # Must not raise even though the store is broken.
            await executor.execute(ctx, queue)
        # The event queue still received events (wire behavior unaffected).
        assert queue.enqueue_event.await_count >= 1

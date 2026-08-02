"""Tests for ``RunResultStreaming.cancel()``.

Covers both cancellation modes:

- ``mode="immediate"`` — drains the queue, cancels the producer task
  synchronously, and enqueues a completion sentinel so the consumer
  returns on its next receive (no 0.1s polling).
- ``mode="after_turn"`` — flips the flag only; the agent loop observes
  it at turn-level checkpoints.

Also covers the between-tool cancel check in
``execute_tool_calls_streamed`` that stops processing further tools in
a batch once IMMEDIATE cancel has been requested.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.config import RunConfig
from troopai.adk.run.context import RunContext
from troopai.adk.run.stream import (
    CancelMode,
    QueueCompleteSentinel,
    RawResponseStreamEvent,
    RunResultStreaming,
)
from troopai.adk.run.tools_executor import execute_tool_calls_streamed
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

# ── Helpers ──────────────────────────────────────────────────────────


@dataclass
class _FakeAgent:
    """Minimal Agent stand-in — only ``name`` is read by the code paths
    we exercise here."""

    name: str = "test-agent"
    handoffs: list[Any] = field(default_factory=list)


def _make_result() -> RunResultStreaming:
    return RunResultStreaming(
        current_agent=_FakeAgent(),  # type: ignore[arg-type]
        max_turns=3,
    )


def _make_tool_call(call_id: str, name: str, args: str = "{}") -> LLMResponseFunctionToolCall:
    return LLMResponseFunctionToolCall(call_id=call_id, name=name, arguments=args)


# ── `cancel(mode="immediate")` ───────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_immediate_sets_mode_and_enqueues_sentinel() -> None:
    result = _make_result()
    # Pre-seed a couple of real events — cancel() must drain them so
    # the consumer sees the sentinel next, not stale events.
    await result.put_event(RawResponseStreamEvent(data="chunk-1"))
    await result.put_event(RawResponseStreamEvent(data="chunk-2"))

    result.cancel(mode="immediate")

    assert result.cancel_mode == CancelMode.IMMEDIATE
    # Exactly one item remains: the sentinel.
    assert result._event_queue.qsize() == 1
    item = result._event_queue.get_nowait()
    assert isinstance(item, QueueCompleteSentinel)


@pytest.mark.asyncio
async def test_cancel_immediate_cancels_run_task() -> None:
    result = _make_result()

    async def long_running() -> None:
        # Deliberately longer than the test so cancellation can fire.
        await asyncio.sleep(10)

    task = asyncio.get_running_loop().create_task(long_running())
    result.set_run_task(task)

    result.cancel(mode="immediate")

    # The task must be flagged cancelled synchronously; the
    # CancelledError surfaces at the next tick.
    assert task.cancelled() or task.cancelling() > 0
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert task.done()


@pytest.mark.asyncio
async def test_cancel_immediate_with_no_task_is_noop() -> None:
    """Calling cancel() before ``stream_events()`` is legal — the
    producer task may not exist yet (deferred impl path)."""
    result = _make_result()
    assert result._run_task is None

    # Must not raise.
    result.cancel(mode="immediate")

    assert result.cancel_mode == CancelMode.IMMEDIATE
    assert result._event_queue.qsize() == 1
    assert isinstance(result._event_queue.get_nowait(), QueueCompleteSentinel)


@pytest.mark.asyncio
async def test_stream_events_exits_promptly_on_immediate_cancel() -> None:
    """End-to-end: consumer is blocked on ``queue.get()``; cancel()
    from inside the iterator must wake it up without the old 0.1s
    polling delay."""
    result = _make_result()

    async def producer() -> None:
        # Emit one event so the consumer yields it, then block "forever".
        await result.put_event(RawResponseStreamEvent(data="first"))
        await asyncio.sleep(10)

    task = asyncio.get_running_loop().create_task(producer())
    result.set_run_task(task)

    received: list[Any] = []

    async def consume() -> None:
        async for ev in result.stream_events():
            received.append(ev)
            if isinstance(ev, RawResponseStreamEvent) and ev.data == "first":
                result.cancel(mode="immediate")

    # If the consumer hangs, the outer timeout fails the test. The
    # budget is deliberately tight — the old polling implementation
    # needed ~100ms per wake; plain `get()` should be sub-ms.
    await asyncio.wait_for(consume(), timeout=1.0)

    assert len(received) == 1
    assert received[0].data == "first"
    assert result.is_complete is True
    assert result.cancel_mode == CancelMode.IMMEDIATE


# ── `cancel(mode="after_turn")` ──────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_after_turn_sets_flag_and_leaves_task_alone() -> None:
    result = _make_result()

    async def still_running() -> None:
        await asyncio.sleep(10)

    task = asyncio.get_running_loop().create_task(still_running())
    result.set_run_task(task)

    await result.put_event(RawResponseStreamEvent(data="mid-turn"))

    result.cancel(mode="after_turn")

    assert result.cancel_mode == CancelMode.AFTER_TURN
    # Task is NOT cancelled — after_turn lets the current batch finish.
    assert not task.cancelled()
    assert task.cancelling() == 0
    # Queue is NOT drained — the in-flight event must still reach the
    # consumer so the "finish current response" contract holds.
    assert result._event_queue.qsize() == 1

    # Cleanup so pytest doesn't warn about pending tasks.
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ── Between-tool cancel check in execute_tool_calls_streamed ─────────


def _make_executor_agent(tools: list[FunctionTool]) -> Any:
    from troopai.adk.agents.middleware import Middleware

    return SimpleNamespace(
        name="cancel-victim",
        tools=tools,
        tool_use_behavior="run_llm_again",
        handoffs=None,
        llm=None,
        hooks=None,
        middleware=Middleware(),
    )


@pytest.mark.asyncio
async def test_execute_tool_calls_streamed_stops_on_immediate_cancel() -> None:
    """When IMMEDIATE is already set before the batch starts, no tool
    runs at all — the first iteration short-circuits."""
    executed: list[str] = []

    async def recording_handler(_ctx: Any, raw_args: str) -> str:
        import json

        payload = json.loads(raw_args)
        executed.append(payload["label"])
        return f"done-{payload['label']}"

    tool = FunctionTool(
        name="slow_tool",
        description="A tool that records labels.",
        schema={
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        },
        on_invoke=recording_handler,
    )
    agent = _make_executor_agent([tool])
    tool_calls = [_make_tool_call(f"call-{i}", "slow_tool", f'{{"label": "t{i}"}}') for i in range(3)]

    result = _make_result()
    result.current_agent = agent
    result.cancel(mode="immediate")  # flipped BEFORE the batch

    ctx_wrapper: RunContext[Any] = RunContext(context=None)
    hooks: RunHooks[Any] = RunHooks()
    config = RunConfig()

    results, deferred = await execute_tool_calls_streamed(
        agent=agent,
        tool_calls=tool_calls,
        ctx_wrapper=ctx_wrapper,
        hooks=hooks,
        config=config,
        result=result,
        tool_failure_counts={},
    )

    assert executed == []
    assert results == []
    assert deferred is None


@pytest.mark.asyncio
async def test_execute_tool_calls_streamed_stops_mid_batch_on_cancel() -> None:
    """The between-tool check fires between iterations. A tool that
    flips the cancel flag as its side effect runs to completion; the
    NEXT tool must be skipped."""
    executed: list[str] = []
    result = _make_result()

    async def record_then_cancel(_ctx: Any, raw_args: str) -> str:
        import json

        payload = json.loads(raw_args)
        executed.append(payload["label"])
        # Flip IMMEDIATE mid-batch — the executor's check is at the
        # TOP of the next iteration, so this tool's result still lands.
        result.cancel(mode="immediate")
        return f"done-{payload['label']}"

    tool = FunctionTool(
        name="record_then_cancel",
        description="Record a label, then cancel the stream.",
        schema={
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        },
        on_invoke=record_then_cancel,
    )
    agent = _make_executor_agent([tool])
    tool_calls = [_make_tool_call(f"call-{i}", "record_then_cancel", f'{{"label": "t{i}"}}') for i in range(3)]

    result.current_agent = agent

    ctx_wrapper: RunContext[Any] = RunContext(context=None)
    hooks: RunHooks[Any] = RunHooks()
    config = RunConfig()

    results, deferred = await execute_tool_calls_streamed(
        agent=agent,
        tool_calls=tool_calls,
        ctx_wrapper=ctx_wrapper,
        hooks=hooks,
        config=config,
        result=result,
        tool_failure_counts={},
    )

    # First tool ran (and flipped the flag); the remaining two did not.
    assert executed == ["t0"]
    assert len(results) == 1
    assert deferred is None

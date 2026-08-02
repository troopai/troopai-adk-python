"""Tests for :mod:`troopai.adk.flows.result`.

Locks in the lazy-producer scheduling contract on
:class:`FlowRunResultStreaming`: when the streamed result is constructed
outside a running event loop (so no producer task can be created yet) the
producer factory is stored and scheduled on the first ``stream_events()``
call. Without lazy scheduling the consumer's first ``await queue.get()``
would block forever.
"""

from __future__ import annotations

import asyncio

import pytest

from troopai.adk.flows.result import FlowRunResultStreaming


@pytest.mark.asyncio
async def test_stream_events_lazy_schedules_deferred_producer() -> None:
    """A deferred producer set outside a loop runs on first stream_events().

    Mirrors the cross-loop usage: build the result with no producer task,
    register a deferred producer factory (as the runner does when called
    outside a running loop), then iterate inside a loop. The producer must
    be scheduled and the stream must terminate rather than deadlock.
    """
    result: FlowRunResultStreaming[None] = FlowRunResultStreaming(flow_id="flow-1")

    started = asyncio.Event()

    async def producer() -> None:
        started.set()
        result.push_event("first")  # type: ignore[arg-type]  # str stand-in for FlowEvent
        result.push_event("second")  # type: ignore[arg-type]
        result.complete()

    # Simulate arun_flow_streamed being called outside a running loop.
    result.set_deferred_run_impl(producer)
    assert result._producer_task is None

    collected: list[object] = []
    async for event in result.stream_events():
        collected.append(event)

    assert started.is_set()
    assert collected == ["first", "second"]
    # The deferred impl was consumed and the task was created exactly once.
    assert result._producer_task is not None
    assert result._deferred_run_impl is None


@pytest.mark.asyncio
async def test_stream_events_does_not_hang_without_deferred_impl() -> None:
    """With a producer task already attached, stream_events() drains normally.

    Guards the common in-loop path: no deferred impl, the queue is fed by an
    externally scheduled task. The lazy-start guard must not interfere.
    """
    result: FlowRunResultStreaming[None] = FlowRunResultStreaming(flow_id="flow-2")

    async def producer() -> None:
        result.push_event("only")  # type: ignore[arg-type]
        result.complete()

    result.set_producer_task(asyncio.get_running_loop().create_task(producer()))

    collected: list[object] = []
    async for event in result.stream_events():
        collected.append(event)

    assert collected == ["only"]
    # No deferred impl was ever stored, so the guard left it None.
    assert result._deferred_run_impl is None

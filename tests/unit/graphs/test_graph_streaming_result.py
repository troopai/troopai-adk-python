"""Unit tests for GraphRunResultStreaming producer/consumer plumbing."""

from __future__ import annotations

import asyncio

import pytest

from troopai.adk.graphs.result import GraphRunResultStreaming


async def test_stream_events_yields_then_completes() -> None:
    r: GraphRunResultStreaming = GraphRunResultStreaming()
    await r.put_event({"type": "a"})
    await r.put_event({"type": "b"})
    await r.complete()
    got = [ev async for ev in r.stream_events()]
    assert got == [{"type": "a"}, {"type": "b"}]


async def test_set_exception_reraised_through_iterator() -> None:
    r: GraphRunResultStreaming = GraphRunResultStreaming()
    r.set_exception(RuntimeError("boom"))
    await r.complete()
    with pytest.raises(RuntimeError, match="boom"):
        async for _ in r.stream_events():
            pass


async def test_cancel_immediate_drains_and_wakes_consumer() -> None:
    """After immediate cancel the queue is drained and stream_events exits
    cleanly (no exception) when there is no stored exception.

    The finally block uses suppress(BaseException) so the CancelledError
    from the cancelled driver task does NOT propagate — only a stored
    exception set via set_exception() would be re-raised.
    """
    r: GraphRunResultStreaming = GraphRunResultStreaming()
    await r.put_event({"type": "x"})

    async def fake_producer() -> None:
        await asyncio.sleep(10)

    task = asyncio.get_running_loop().create_task(fake_producer())
    r.set_run_task(task)
    r.register_node_task(task)
    r.cancel("immediate")
    # With suppress(BaseException) the driver task's CancelledError is
    # swallowed; no exception propagates out of stream_events when there
    # is no stored exception.
    events: list = []
    async for ev in r.stream_events():
        events.append(ev)
    # Queue was drained before iteration — no events.
    assert events == []
    await asyncio.sleep(0)
    assert task.cancelled() or task.cancelling() > 0


async def test_cancel_after_superstep_sets_flag_and_does_not_suppress() -> None:
    from troopai.adk.run.stream import CancelMode

    r: GraphRunResultStreaming = GraphRunResultStreaming()
    r.cancel("after_superstep")
    assert r.cancel_mode == CancelMode.AFTER_SUPERSTEP
    await r.put_event({"type": "k"})
    await r.complete()
    assert [ev async for ev in r.stream_events()] == [{"type": "k"}]

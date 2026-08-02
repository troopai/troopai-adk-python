"""Unit tests for SwarmRunResultStreaming."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from troopai.adk.swarms.events import SwarmDoneEvent, SwarmStartEvent
from troopai.adk.swarms.result import SwarmRunResultStreaming
from troopai.adk.swarms.stop_reason import StopReason


def _start_event() -> SwarmStartEvent:
    return SwarmStartEvent(entry_agent="m", member_names=("m",))


def _done_event() -> SwarmDoneEvent:
    return SwarmDoneEvent(
        reason=StopReason(kind="max_turns", detail=""),
        final_output=None,
    )


async def _noop_driver_task() -> asyncio.Task[None]:
    """Register a completed driver task so stream_events()'s
    no-driver-scheduled guard doesn't fire when unit-testing the
    queue/sentinel machinery in isolation."""

    async def _done() -> None:
        return None

    task = asyncio.get_running_loop().create_task(_done())
    # Drain so the task is `done()` by the time stream_events()'s
    # finally tries to await it.
    await asyncio.sleep(0)
    return task


class TestSwarmRunResultStreamingBasics:
    async def test_stream_events_drains_queue_and_exits_at_sentinel(self) -> None:
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(
            user_prompt="go",
        )
        result.set_run_task(await _noop_driver_task())
        await result.put_event(_start_event())
        await result.put_event(_done_event())
        await result.complete()

        events: list[Any] = []
        async for ev in result.stream_events():
            events.append(ev)
        assert len(events) == 2
        assert isinstance(events[0], SwarmStartEvent)
        assert isinstance(events[1], SwarmDoneEvent)

    async def test_set_exception_re_raises_after_drain(self) -> None:
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(
            user_prompt="go",
        )
        result.set_run_task(await _noop_driver_task())
        await result.put_event(_start_event())
        result.set_exception(RuntimeError("boom"))
        await result.complete()

        events: list[Any] = []
        with pytest.raises(RuntimeError, match="boom"):
            async for ev in result.stream_events():
                events.append(ev)
        # The top-of-loop _stored_exception check exits before reading
        # the queued event — set_exception is meant to wake the consumer
        # immediately, not deliver more events.
        assert events == []

    async def test_stream_events_raises_when_no_driver_scheduled(self) -> None:
        """A bare result with no driver MUST raise immediately, not
        block forever on the empty queue."""
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(
            user_prompt="go",
        )

        with pytest.raises(RuntimeError, match="no driver scheduled"):
            async for _ in result.stream_events():
                pass

    async def test_set_exception_without_complete_wakes_consumer(self) -> None:
        """A driver that errors and exits without complete() must NOT
        leave the consumer blocked on the queue."""
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(
            user_prompt="go",
        )

        async def _error_driver() -> None:
            await asyncio.sleep(0)
            result.set_exception(RuntimeError("driver crashed"))
            # Deliberately NO complete() call — the consumer must
            # still exit via the top-of-loop exception check.

        task = asyncio.get_running_loop().create_task(_error_driver())
        result.set_run_task(task)

        with pytest.raises(RuntimeError, match="driver crashed"):
            async for _ in result.stream_events():
                pass


class TestSwarmRunResultStreamingCancel:
    async def test_cancel_drains_and_exits(self) -> None:
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(
            user_prompt="go",
        )
        await result.put_event(_start_event())

        async def _hang() -> None:
            await asyncio.sleep(100)

        task = asyncio.get_running_loop().create_task(_hang())
        result.set_run_task(task)

        result.cancel()

        events: list[Any] = []
        async for ev in result.stream_events():
            events.append(ev)
        assert events == []
        # Briefly yield so the cancellation propagates.
        await asyncio.sleep(0)
        assert task.cancelled() or task.done()


class TestSwarmRunResultStreamingDeferredRun:
    async def test_deferred_run_impl_scheduled_on_first_stream_events(self) -> None:
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(
            user_prompt="go",
        )

        ran = asyncio.Event()

        async def _impl() -> None:
            ran.set()
            await result.put_event(_start_event())
            await result.complete()

        result.set_deferred_run_impl(_impl)
        assert result._run_task is None  # type: ignore[union-attr]

        events: list[Any] = []
        async for ev in result.stream_events():
            events.append(ev)
        assert ran.is_set()
        assert len(events) == 1

    async def test_cancel_before_first_stream_events_skips_deferred_driver(self) -> None:
        """cancel() before the first stream_events() must discard the
        deferred driver so it never launches and never bills LLM tokens
        for a run the consumer already cancelled."""
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(
            user_prompt="go",
        )

        ran = asyncio.Event()

        async def _impl() -> None:
            # If this ever runs, the cancelled run is still paying for LLM
            # turns — exactly the cost the guard must prevent.
            ran.set()
            await result.put_event(_start_event())
            await result.complete()

        result.set_deferred_run_impl(_impl)
        # Consumer cancels BEFORE iterating — _run_task is still None here,
        # so the old cancel() had nothing to cancel.
        result.cancel()

        events: list[Any] = []
        async for ev in result.stream_events():
            events.append(ev)

        # Driver must not have started; stream exits cleanly with no events.
        assert not ran.is_set()
        assert events == []
        assert result._run_task is None  # type: ignore[union-attr]
        assert result._deferred_run_impl is None  # type: ignore[union-attr]


class TestSwarmRunResultStreamingCancelCost:
    async def test_consumer_break_cancels_still_running_driver(self) -> None:
        """A consumer that breaks out of its loop while the driver is mid-turn
        must have the driver cancelled in stream_events()'s finally (mirroring
        the single-agent cleanup), not merely awaited — otherwise the finally
        blocks until the in-flight turn finishes, billing LLM tokens for a
        stream the consumer has already abandoned.

        The driver here finishes its in-flight turn shortly after the break.
        With the in-finally cancel the turn is cut short (it never reaches
        ``ran_to_completion``); without it the finally awaits the turn to its
        natural end, so ``ran_to_completion`` flips True.
        """
        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(
            user_prompt="go",
        )

        ran_to_completion = False

        async def _driver() -> None:
            nonlocal ran_to_completion
            try:
                await result.put_event(_start_event())
                # Stand in for an in-flight member turn / LLM call that would
                # finish shortly if left uninterrupted.
                await asyncio.sleep(0.3)
                ran_to_completion = True
            finally:
                await result.complete()

        task = asyncio.get_running_loop().create_task(_driver())
        result.set_run_task(task)

        # Consumer takes the first event then breaks; the break closes the
        # async generator, running stream_events()'s finally.
        async for ev in result.stream_events():
            assert isinstance(ev, SwarmStartEvent)
            break

        # Drive the cancellation the finally requested to completion.
        with contextlib.suppress(BaseException):
            await task

        assert task.cancelled() or task.done()
        # The in-flight turn was cut short by the finally's cancel, not run to
        # its natural end (which the un-cancelled finally would have awaited).
        assert not ran_to_completion


class TestSwarmDriverCancelSemantics:
    async def test_developer_cancel_does_not_surface_cancelled_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A naive ``result.cancel(); break`` must read as a clean stop.

        The real streamed driver's outer handler catches the
        ``CancelledError`` raised when ``cancel()`` cancels the driver
        task. It must record that exception for the consumer only when the
        cancel came from outside (cancel_mode not IMMEDIATE); a
        developer-issued immediate cancel is a clean, requested stop and
        must NOT re-raise a spurious ``CancelledError`` from
        ``stream_events()``.
        """
        from troopai.adk.run import swarm_loop_streamed as driver_mod

        result: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(
            user_prompt="go",
        )

        body_running = asyncio.Event()

        async def _blocking_body(**_kwargs: Any) -> None:
            # Emit one event so the consumer wakes, then stand in for an
            # in-flight member turn. Suspended at the sleep when cancel()
            # cancels the driver task, so CancelledError is raised here and
            # propagates into run_swarm_loop_streamed's outer except clause.
            await result.put_event(_start_event())
            body_running.set()
            await asyncio.sleep(100)

        monkeypatch.setattr(driver_mod, "_run_streamed_body", _blocking_body)

        task = asyncio.get_running_loop().create_task(
            driver_mod.run_swarm_loop_streamed(
                swarm=None,  # type: ignore[arg-type]
                user_prompt="go",
                ctx_wrapper=None,  # type: ignore[arg-type]
                hooks=None,  # type: ignore[arg-type]
                config=None,  # type: ignore[arg-type]
                result=result,
            )
        )
        result.set_run_task(task)
        await body_running.wait()

        events: list[Any] = []
        # Naive consumer pattern: cancel mid-stream and break. The break must
        # NOT raise a CancelledError out of the async-for.
        async for ev in result.stream_events():
            events.append(ev)
            result.cancel()
            break

        # Drive the cancelled driver to completion so its except/finally runs.
        with contextlib.suppress(BaseException):
            await task
        assert task.done()
        # No spurious exception leaked, and none was stored for the consumer.
        assert result._stored_exception is None  # type: ignore[union-attr]
        # The consumer received the one in-flight event before cancelling.
        assert len(events) == 1
        assert isinstance(events[0], SwarmStartEvent)

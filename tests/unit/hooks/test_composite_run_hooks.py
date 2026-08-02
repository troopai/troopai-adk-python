"""Tests for :class:`CompositeRunHooks` — fan-out composition over multiple hook instances.

Exercises two invariants:

1. **Every public ``RunHooks`` method has a matching forwarder** on
   :class:`CompositeRunHooks`. If a new public hook is added to
   :class:`RunHooks` without a matching override on the composite,
   fan-out silently becomes a no-op and verbose/user hooks lose
   visibility. The reflection test below fails in that scenario.
2. **Exception handling**: a failing member does not prevent the
   others from firing, and the first exception is re-raised after
   all members have run.
"""

from __future__ import annotations

import asyncio
import inspect
import logging

import pytest

from troopai.adk.hooks.hooks import (
    CompositeRunHooks,
    RunHooks,
    compose_run_hooks,
)


class _RecordingHooks(RunHooks):
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[str] = []

    async def on_agent_start(self, context, agent) -> None:
        del context, agent
        self.calls.append("on_agent_start")

    async def on_tool_start(self, context, agent, tool_name, tool_input) -> None:
        del context, agent, tool_name, tool_input
        self.calls.append("on_tool_start")


class _RaisingHooks(RunHooks):
    async def on_agent_start(self, context, agent) -> None:
        del context, agent
        raise RuntimeError("boom")


def _public_async_hook_methods() -> list[str]:
    """Return the names of every public async method on :class:`RunHooks`."""
    names: list[str] = []
    for name, value in inspect.getmembers(RunHooks):
        if name.startswith("_"):
            continue
        if not inspect.iscoroutinefunction(value):
            continue
        names.append(name)
    return names


def test_every_public_runhook_method_is_forwarded() -> None:
    """Guard rail — a future addition to RunHooks must also update CompositeRunHooks."""
    method_names = _public_async_hook_methods()
    assert len(method_names) > 0
    for name in method_names:
        forwarded = getattr(CompositeRunHooks, name, None)
        assert forwarded is not None, f"CompositeRunHooks missing forwarder for {name}"
        # Must be an async function (coroutine function).
        assert inspect.iscoroutinefunction(forwarded), f"{name} on CompositeRunHooks is not a coroutine function"
        # Must NOT inherit from RunHooks (i.e. must be overridden, not
        # the no-op base).
        assert forwarded is not getattr(RunHooks, name), (
            f"CompositeRunHooks.{name} is not overridden (inherits RunHooks no-op)"
        )


@pytest.mark.asyncio
async def test_fanout_fires_all_members_in_order() -> None:
    a = _RecordingHooks("a")
    b = _RecordingHooks("b")
    composite = CompositeRunHooks([a, b])

    await composite.on_agent_start(None, None)  # type: ignore[arg-type]

    assert a.calls == ["on_agent_start"]
    assert b.calls == ["on_agent_start"]


@pytest.mark.asyncio
async def test_fanout_continues_past_failure_and_reraises() -> None:
    a = _RaisingHooks()
    b = _RecordingHooks("b")
    composite = CompositeRunHooks([a, b])

    with pytest.raises(RuntimeError, match="boom"):
        await composite.on_agent_start(None, None)  # type: ignore[arg-type]

    # b must still have fired despite a's exception.
    assert b.calls == ["on_agent_start"]


@pytest.mark.asyncio
async def test_fanout_logs_every_member_error_not_just_first(caplog) -> None:
    """Every member error is logged; the first is still re-raised.

    Regression: _fanout kept only the FIRST exception and silently
    discarded later members' errors — a broken metrics/observability hook's
    failure vanished with no log. All member errors are now logged at ERROR
    while the first is re-raised (the re-raise contract is preserved).
    """

    class _RaiseA(RunHooks):
        async def on_agent_start(self, context, agent) -> None:
            del context, agent
            raise RuntimeError("error-from-A")

    class _RaiseB(RunHooks):
        async def on_agent_start(self, context, agent) -> None:
            del context, agent
            raise ValueError("error-from-B")

    composite = CompositeRunHooks([_RaiseA(), _RaiseB()])

    with (
        caplog.at_level(logging.ERROR, logger="troopai.adk.hooks.hooks"),
        pytest.raises(RuntimeError, match="error-from-A"),
    ):
        await composite.on_agent_start(None, None)  # type: ignore[arg-type]

    # Both errors must be logged — the second must not vanish.
    assert "error-from-A" in caplog.text
    assert "error-from-B" in caplog.text


@pytest.mark.asyncio
async def test_fanout_propagates_cancellation_without_running_later_members() -> None:
    """asyncio.CancelledError from a member must propagate immediately.

    Regression: _fanout caught BaseException, so a CancelledError (and
    KeyboardInterrupt / SystemExit) was collected like an ordinary hook
    error and the remaining members still ran — delaying/eating
    cancellation and Ctrl-C. Only Exception is now caught, so control-flow
    exceptions short-circuit the fan-out.
    """

    class _CancelHook(RunHooks):
        async def on_agent_start(self, context, agent) -> None:
            del context, agent
            raise asyncio.CancelledError

    later = _RecordingHooks("later")
    composite = CompositeRunHooks([_CancelHook(), later])

    with pytest.raises(asyncio.CancelledError):
        await composite.on_agent_start(None, None)  # type: ignore[arg-type]

    # The later member must NOT have run — cancellation short-circuits.
    assert later.calls == []


def test_compose_run_hooks_returns_noop_when_all_none() -> None:
    result = compose_run_hooks(None, None)
    assert type(result) is RunHooks


def test_compose_run_hooks_returns_sole_member_directly() -> None:
    only = _RecordingHooks("only")
    result = compose_run_hooks(None, only)
    assert result is only


def test_compose_run_hooks_wraps_multiple() -> None:
    a = _RecordingHooks("a")
    b = _RecordingHooks("b")
    result = compose_run_hooks(a, b)
    assert isinstance(result, CompositeRunHooks)

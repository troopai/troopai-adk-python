"""Unit tests for ``propagate_errors`` on :class:`SwarmHooks`.

Covers:
- A hook with ``propagate_errors=True`` causes the registry fan-out to
  re-raise, so the error reaches the caller.
- A hook with ``propagate_errors=False`` (the default) has its error
  logged and swallowed — the registry call completes normally.
- :class:`SwarmCheckpointerHooks` declares ``propagate_errors = True``
  so a failed checkpointer save propagates to the caller.
- End-to-end path: a fake checkpointer whose ``save`` raises is
  registered on a :class:`HookRegistry`; firing ``on_swarm_turn_end``
  propagates the error.
"""

from __future__ import annotations

from typing import Any, override

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.context import RunContext
from troopai.adk.swarms.checkpointers.hooks import SwarmCheckpointerHooks
from troopai.adk.swarms.hooks import HookRegistry, SwarmHooks
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.state import SwarmState
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_swarm() -> Swarm:
    member = Agent(name="m1", system_prompt="x")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(3),
    )


def _make_state(swarm: Swarm) -> SwarmState[Any]:
    state: SwarmState[Any] = SwarmState(
        swarm=swarm,
        current_agent=swarm.members[0],
        current_agent_name="m1",
    )
    state.total_turns = 1
    return state


def _make_ctx() -> RunContext[None]:
    return RunContext(context=None)  # type: ignore[arg-type]  # test scaffolding: hooks ignore the context payload


class _CriticalHook(SwarmHooks[Any]):
    """Hook with ``propagate_errors=True`` that always raises on turn end."""

    propagate_errors = True

    @override
    async def on_swarm_turn_end(
        self,
        context: RunContext[Any],
        state: SwarmState[Any],
        items: list[Any],
    ) -> None:
        del context, state, items
        raise RuntimeError("swarm-boom")


class _ObserverHook(SwarmHooks[Any]):
    """Hook with ``propagate_errors=False`` (default) that always raises."""

    @override
    async def on_swarm_turn_end(
        self,
        context: RunContext[Any],
        state: SwarmState[Any],
        items: list[Any],
    ) -> None:
        del context, state, items
        raise RuntimeError("observer-swarm-boom")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_critical_hook_propagates_error() -> None:
    """A hook with ``propagate_errors=True`` causes the fan-out to re-raise."""
    registry = HookRegistry()
    registry.add(_CriticalHook())

    sw = _make_swarm()
    state = _make_state(sw)

    with pytest.raises(RuntimeError, match="swarm-boom"):
        await registry.on_swarm_turn_end(
            context=_make_ctx(),
            state=state,
            items=[],
        )


async def test_observer_hook_error_is_swallowed() -> None:
    """A hook with ``propagate_errors=False`` has its error swallowed."""
    registry = HookRegistry()
    registry.add(_ObserverHook())

    sw = _make_swarm()
    state = _make_state(sw)

    # Must NOT raise — the observer error is logged and discarded.
    await registry.on_swarm_turn_end(
        context=_make_ctx(),
        state=state,
        items=[],
    )


def test_swarm_checkpointer_hooks_is_critical() -> None:
    """``SwarmCheckpointerHooks`` must have ``propagate_errors = True``."""
    assert SwarmCheckpointerHooks.propagate_errors is True


async def test_fake_checkpointer_save_error_propagates_via_registry() -> None:
    """End-to-end: a fake checkpointer whose ``save`` raises is registered
    on a HookRegistry; firing on_swarm_turn_end propagates the error.
    """
    from troopai.adk.swarms.checkpointer import SwarmCheckpoint, SwarmCheckpointer

    class _FailingCheckpointer:
        async def save(self, checkpoint: SwarmCheckpoint) -> None:
            del checkpoint
            raise OSError("disk full")

        async def load(self, thread_id: str, swarm: Swarm) -> SwarmCheckpoint | None:
            del thread_id, swarm
            return None

        async def list_checkpoints(self) -> list[str]:
            return []

        async def delete(self, thread_id: str) -> None:
            del thread_id

        def register(self, registry: HookRegistry) -> None:
            registry.add(SwarmCheckpointerHooks(self, "fail-thr"))  # type: ignore[arg-type]  # fake checkpointer structurally satisfies the SwarmCheckpointer Protocol

    assert isinstance(_FailingCheckpointer(), SwarmCheckpointer)

    cp = _FailingCheckpointer()
    registry = HookRegistry()
    cp.register(registry)

    sw = _make_swarm()
    state = _make_state(sw)

    with pytest.raises(OSError, match="disk full"):
        await registry.on_swarm_turn_end(
            context=_make_ctx(),
            state=state,
            items=[],
        )


async def test_cancelled_error_propagates_through_fan_out() -> None:
    """asyncio.CancelledError (a BaseException) must escape the fan-out
    immediately even when sandwiched between non-propagating hooks."""
    import asyncio

    class _CancellingHook(SwarmHooks[Any]):
        """Raises CancelledError — simulates task cancellation mid-hook."""

        @override
        async def on_swarm_turn_end(
            self,
            context: RunContext[Any],
            state: SwarmState[Any],
            items: list[Any],
        ) -> None:
            del context, state, items
            raise asyncio.CancelledError("task cancelled")

    # Register: observer (non-propagating), then the hook that raises CancelledError.
    # The CancelledError must NOT be swallowed by the except Exception block.
    registry = HookRegistry()
    registry.add(_ObserverHook())
    registry.add(_CancellingHook())

    sw = _make_swarm()
    state = _make_state(sw)

    with pytest.raises(asyncio.CancelledError):
        await registry.on_swarm_turn_end(
            context=_make_ctx(),
            state=state,
            items=[],
        )


async def test_observer_before_critical_does_not_block_propagation() -> None:
    """Observer hook fires first; its error is swallowed.  The subsequent
    critical hook still propagates its error.
    """
    registry = HookRegistry()
    registry.add(_ObserverHook())
    registry.add(_CriticalHook())

    sw = _make_swarm()
    state = _make_state(sw)

    with pytest.raises(RuntimeError, match="swarm-boom"):
        await registry.on_swarm_turn_end(
            context=_make_ctx(),
            state=state,
            items=[],
        )


# ---------------------------------------------------------------------------
# Regression: on_swarm_turn_start must forward member_name to subclasses (#MED)
# ---------------------------------------------------------------------------


async def test_on_swarm_turn_start_forwards_member_name() -> None:
    """Regression: HookRegistry.on_swarm_turn_start must forward member_name
    to every attached SwarmHooks instance — previously the registry discarded
    it and always called the base with only (context, state)."""

    received: list[str] = []

    class _CaptureMemberName(SwarmHooks[Any]):
        @override
        async def on_swarm_turn_start(
            self,
            context: RunContext[Any],
            state: SwarmState[Any],
            member_name: str,
        ) -> None:
            del context, state
            received.append(member_name)

    registry = HookRegistry()
    registry.add(_CaptureMemberName())

    sw = _make_swarm()
    state = _make_state(sw)

    await registry.on_swarm_turn_start(
        context=_make_ctx(),
        state=state,
        member_name="m1",
    )

    assert received == ["m1"], f"on_swarm_turn_start must forward member_name; got {received!r}"

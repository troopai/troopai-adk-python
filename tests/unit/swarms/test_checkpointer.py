"""Tests for SwarmCheckpoint + SwarmCheckpointer protocol + reference impls."""

from __future__ import annotations

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.swarms.checkpointer import (
    SwarmCheckpoint,
    SwarmCheckpointer,
)
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_swarm() -> Swarm:
    """Single-member swarm fixture for checkpointer tests."""
    member = Agent(name="m1", system_prompt="x")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(3),
    )


class TestSwarmCheckpointConstruction:
    def test_construct_with_required_fields(self) -> None:
        cp = SwarmCheckpoint(
            thread_id="thr-1",
            state={"current_agent_name": "m1"},
            turn=0,
        )
        assert cp.thread_id == "thr-1"
        assert cp.turn == 0
        assert cp.state == {"current_agent_name": "m1"}

    def test_checkpoint_is_frozen(self) -> None:
        cp = SwarmCheckpoint(thread_id="t", state={}, turn=0)
        with pytest.raises(Exception):  # FrozenInstanceError
            cp.thread_id = "other"  # type: ignore[misc]


class TestSwarmCheckpointerProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        """Any object with save/load/list_checkpoints/delete/register satisfies the Protocol."""

        class _Recorder:
            async def save(self, checkpoint: SwarmCheckpoint) -> None:
                del checkpoint

            async def load(self, thread_id: str, swarm: Swarm) -> SwarmCheckpoint | None:
                del thread_id, swarm
                return None

            async def list_checkpoints(self) -> list[str]:
                return []

            async def delete(self, thread_id: str) -> None:
                del thread_id

            def register(self, registry: object) -> None:
                del registry

        assert isinstance(_Recorder(), SwarmCheckpointer)


class TestInMemorySwarmCheckpointerSaveLoad:
    async def test_save_and_load_roundtrip(self) -> None:
        from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer

        cp = InMemorySwarmCheckpointer()
        sw = _make_swarm()
        checkpoint = SwarmCheckpoint(
            thread_id="thr-rt",
            state={"current_agent_name": "m1", "total_turns": 1},
            turn=1,
        )
        await cp.save(checkpoint)
        restored = await cp.load("thr-rt", sw)
        assert restored is not None
        assert restored.turn == 1
        assert restored.state["current_agent_name"] == "m1"

    async def test_load_unknown_thread_id_returns_none(self) -> None:
        from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer

        cp = InMemorySwarmCheckpointer()
        sw = _make_swarm()
        result = await cp.load("does-not-exist", sw)
        assert result is None

    async def test_save_overwrites_previous_checkpoint_for_same_thread(self) -> None:
        from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer

        cp = InMemorySwarmCheckpointer()
        sw = _make_swarm()
        first = SwarmCheckpoint(thread_id="thr-ov", state={}, turn=1)
        second = SwarmCheckpoint(thread_id="thr-ov", state={}, turn=3)
        await cp.save(first)
        await cp.save(second)
        restored = await cp.load("thr-ov", sw)
        assert restored is not None
        assert restored.turn == 3

    async def test_register_attaches_auto_save_hooks_to_registry(self) -> None:
        """register() should attach a SwarmHooks that auto-saves on turn end."""
        from troopai.adk.run.context import RunContext
        from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer
        from troopai.adk.swarms.hooks import HookRegistry
        from troopai.adk.swarms.state import SwarmState

        cp = InMemorySwarmCheckpointer(thread_id="auto-thr")
        registry = HookRegistry()
        cp.register(registry)

        sw = _make_swarm()
        state = SwarmState(
            swarm=sw,
            current_agent=sw.members[0],
            current_agent_name="m1",
        )
        state.total_turns = 2

        # Firing the registry's on_swarm_turn_end should trigger
        # the auto-save hook the checkpointer registered.
        ctx: RunContext = RunContext(context=None)
        await registry.on_swarm_turn_end(ctx, state, [])

        restored = await cp.load("auto-thr", sw)
        assert restored is not None
        assert restored.turn == 2
        assert restored.state["current_agent_name"] == "m1"


class TestAutoSaveOnInterrupt:
    """Verify the auto-save hook snapshots state on a parked interrupt.

    The swarm loop returns from a member turn before ``on_swarm_turn_end``
    fires when the turn suspends on a cooperative interrupt. Without the
    ``on_swarm_turn_interrupt`` auto-save override the parked
    ``pending_interrupts`` and ``nested_agent_snapshots`` would never
    reach the checkpoint store and the resume path would deadlock.
    """

    async def test_on_swarm_turn_interrupt_saves_checkpoint(self) -> None:
        from troopai.adk.graphs.interrupt import Interrupt
        from troopai.adk.run.context import RunContext
        from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer
        from troopai.adk.swarms.hooks import HookRegistry
        from troopai.adk.swarms.state import SwarmState

        cp = InMemorySwarmCheckpointer(thread_id="parked-thr")
        registry = HookRegistry()
        cp.register(registry)

        sw = _make_swarm()
        state = SwarmState(
            swarm=sw,
            current_agent=sw.members[0],
            current_agent_name="m1",
        )
        state.total_turns = 1
        interrupt = Interrupt(
            node_id="m1",
            question="Approve action?",
            kind="tool_approval",
            metadata={"tool_call_id": "call-1"},
        )
        state.pending_interrupts["m1"] = interrupt

        ctx: RunContext = RunContext(context=None)
        await registry.on_swarm_turn_interrupt(ctx, state, "m1", interrupt)

        restored = await cp.load("parked-thr", sw)
        assert restored is not None
        assert restored.turn == 1
        assert restored.state["current_agent_name"] == "m1"
        parked = restored.state["pending_interrupts"]
        assert "m1" in parked
        assert parked["m1"]["question"] == "Approve action?"
        assert parked["m1"]["kind"] == "tool_approval"

    async def test_save_idempotent_across_end_and_interrupt(self) -> None:
        from troopai.adk.graphs.interrupt import Interrupt
        from troopai.adk.run.context import RunContext
        from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer
        from troopai.adk.swarms.hooks import HookRegistry
        from troopai.adk.swarms.state import SwarmState

        cp = InMemorySwarmCheckpointer(thread_id="dual-thr")
        registry = HookRegistry()
        cp.register(registry)

        sw = _make_swarm()
        state = SwarmState(
            swarm=sw,
            current_agent=sw.members[0],
            current_agent_name="m1",
        )
        state.total_turns = 2

        ctx: RunContext = RunContext(context=None)
        # First: a normal turn-end save.
        await registry.on_swarm_turn_end(ctx, state, [])
        after_end = await cp.load("dual-thr", sw)
        assert after_end is not None
        assert after_end.turn == 2
        assert len(after_end.state["pending_interrupts"]) == 0

        # Then: the next turn parks on an interrupt — the interrupt
        # auto-save MUST overwrite under the same thread_id (single-slot
        # dict semantic) and carry the parked interrupt forward.
        state.total_turns = 3
        interrupt = Interrupt(
            node_id="m1",
            question="Confirm?",
            kind="generic",
        )
        state.pending_interrupts["m1"] = interrupt
        await registry.on_swarm_turn_interrupt(ctx, state, "m1", interrupt)

        after_interrupt = await cp.load("dual-thr", sw)
        assert after_interrupt is not None
        # Single-slot semantic: load returns the latest write for the
        # thread_id — the interrupt-save (turn=3) overwrites the prior
        # turn-end save (turn=2).
        assert after_interrupt.turn == 3
        assert "m1" in after_interrupt.state["pending_interrupts"]
        # The post-end load already happened at turn=2; re-loading now
        # MUST return the parked snapshot, not the earlier one.
        assert after_interrupt.turn != after_end.turn


# ---------------------------------------------------------------------------
# (b) list_checkpoints / delete
# ---------------------------------------------------------------------------


class TestListAndDelete:
    async def test_list_checkpoints_sorted(self) -> None:
        from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer

        cp = InMemorySwarmCheckpointer()
        await cp.save(SwarmCheckpoint(thread_id="t2", state={}, turn=1))
        await cp.save(SwarmCheckpoint(thread_id="t1", state={}, turn=1))
        assert await cp.list_checkpoints() == ["t1", "t2"]

    async def test_list_checkpoints_empty(self) -> None:
        from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer

        cp = InMemorySwarmCheckpointer()
        assert await cp.list_checkpoints() == []

    async def test_delete_removes_thread(self) -> None:
        from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer

        cp = InMemorySwarmCheckpointer()
        await cp.save(SwarmCheckpoint(thread_id="t1", state={}, turn=1))
        await cp.save(SwarmCheckpoint(thread_id="t2", state={}, turn=1))
        await cp.delete("t1")
        assert await cp.list_checkpoints() == ["t2"]

    async def test_delete_missing_is_noop(self) -> None:
        from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer

        cp = InMemorySwarmCheckpointer()
        # Must not raise.
        await cp.delete("does-not-exist")
        assert await cp.list_checkpoints() == []

    def test_protocol_includes_list_and_delete(self) -> None:
        """Any object that adds list_checkpoints + delete satisfies the Protocol."""

        class _Full:
            async def save(self, checkpoint: SwarmCheckpoint) -> None:
                del checkpoint

            async def load(self, thread_id: str, swarm: Swarm) -> SwarmCheckpoint | None:
                del thread_id, swarm
                return None

            def register(self, registry: object) -> None:
                del registry

            async def list_checkpoints(self) -> list[str]:
                return []

            async def delete(self, thread_id: str) -> None:
                del thread_id

        assert isinstance(_Full(), SwarmCheckpointer)


# ---------------------------------------------------------------------------
# (c) SwarmCheckpointerHooks extracted class
# ---------------------------------------------------------------------------


class TestSwarmCheckpointerHooks:
    async def test_on_swarm_turn_end_saves_under_thread_id(self) -> None:
        from troopai.adk.run.context import RunContext
        from troopai.adk.swarms.checkpointers.hooks import SwarmCheckpointerHooks
        from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer
        from troopai.adk.swarms.state import SwarmState

        cp = InMemorySwarmCheckpointer(thread_id="hooks-thr")
        sw = _make_swarm()
        hooks = SwarmCheckpointerHooks(cp, "hooks-thr")
        state = SwarmState(
            swarm=sw,
            current_agent=sw.members[0],
            current_agent_name="m1",
        )
        state.total_turns = 5

        ctx: RunContext = RunContext(context=None)
        await hooks.on_swarm_turn_end(ctx, state, [])

        restored = await cp.load("hooks-thr", sw)
        assert restored is not None
        assert restored.thread_id == "hooks-thr"
        assert restored.turn == 5

    async def test_on_swarm_turn_interrupt_saves_under_thread_id(self) -> None:
        from troopai.adk.graphs.interrupt import Interrupt
        from troopai.adk.run.context import RunContext
        from troopai.adk.swarms.checkpointers.hooks import SwarmCheckpointerHooks
        from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer
        from troopai.adk.swarms.state import SwarmState

        cp = InMemorySwarmCheckpointer(thread_id="ihooks-thr")
        sw = _make_swarm()
        hooks = SwarmCheckpointerHooks(cp, "ihooks-thr")
        state = SwarmState(
            swarm=sw,
            current_agent=sw.members[0],
            current_agent_name="m1",
        )
        state.total_turns = 3
        interrupt = Interrupt(node_id="m1", question="?", kind="generic")

        ctx: RunContext = RunContext(context=None)
        await hooks.on_swarm_turn_interrupt(ctx, state, "m1", interrupt)

        restored = await cp.load("ihooks-thr", sw)
        assert restored is not None
        assert restored.turn == 3

    def test_in_memory_register_uses_swarm_checkpointer_hooks(self) -> None:
        """register() must install a SwarmCheckpointerHooks, not _AutoSaveHooks."""
        from troopai.adk.swarms.checkpointers.hooks import SwarmCheckpointerHooks
        from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer

        attached: list[object] = []

        class _FakeRegistry:
            def add(self, hooks: object) -> None:
                attached.append(hooks)

        cp = InMemorySwarmCheckpointer(thread_id="reg-thr")
        cp.register(_FakeRegistry())  # type: ignore[arg-type]
        assert len(attached) == 1
        assert isinstance(attached[0], SwarmCheckpointerHooks)

    def test_no_private_auto_save_hooks_in_in_memory_module(self) -> None:
        """_AutoSaveHooks must be removed from in_memory after extraction."""
        import troopai.adk.swarms.checkpointers.in_memory as mod

        assert not hasattr(mod, "_AutoSaveHooks")

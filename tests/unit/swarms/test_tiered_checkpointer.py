"""Tests for ``TieredSwarmCheckpointer`` — hot/cold composite swarm checkpointer.

Covers:
- Load falls back to cold and re-warms hot.
- archive() with archive_after_seconds=0.0 moves a saved entry hot→cold.
- list_checkpoints() is the union of both tiers.
- delete() removes from both tiers.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.swarms.checkpointer import SwarmCheckpoint
from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer
from troopai.adk.swarms.checkpointers.tiered import TieredSwarmCheckpointer
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.state import SwarmState, SwarmStateDict
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination

# ---------------------------------------------------------------------------
# Helpers
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


def _make_state(swarm: Swarm, turns: int = 1) -> SwarmState:
    state = SwarmState(
        swarm=swarm,
        current_agent=swarm.members[0],
        current_agent_name=swarm.members[0].name,
    )
    state.total_turns = turns
    return state


def _ckpt(thread_id: str, state: SwarmState, turn: int) -> SwarmCheckpoint:
    return SwarmCheckpoint(
        thread_id=thread_id,
        state=cast(dict[str, Any], state.to_dict()),
        turn=turn,
    )


# ---------------------------------------------------------------------------
# Cold fallback + re-warm
# ---------------------------------------------------------------------------


async def test_load_falls_back_to_cold_and_rewarms_hot() -> None:
    """When hot is empty and cold has a checkpoint, load returns it and
    re-warms the hot tier so a subsequent list_checkpoints includes the thread."""
    swarm = _make_swarm()
    hot = InMemorySwarmCheckpointer()
    cold = InMemorySwarmCheckpointer()
    tiered = TieredSwarmCheckpointer(hot=hot, cold=cold, archive_after_seconds=3600.0)

    # Seed cold only (bypass the tiered composite to simulate a prior archive).
    state = _make_state(swarm, turns=5)
    await cold.save(_ckpt("t1", state, 5))

    # Hot must be empty before the load.
    assert await hot.list_checkpoints() == []

    # load() must find the checkpoint through cold.
    cp = await tiered.load("t1", swarm)
    assert cp is not None
    assert cp.turn == 5

    # After load, hot must have been re-warmed.
    assert await hot.list_checkpoints() == ["t1"]


async def test_load_returns_none_when_both_empty() -> None:
    """load() returns None when neither tier has the thread_id."""
    swarm = _make_swarm()
    hot = InMemorySwarmCheckpointer()
    cold = InMemorySwarmCheckpointer()
    tiered = TieredSwarmCheckpointer(hot=hot, cold=cold, archive_after_seconds=3600.0)

    assert await tiered.load("does-not-exist", swarm) is None


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


async def test_archive_moves_hot_to_cold() -> None:
    """archive() with archive_after_seconds=0.0 immediately moves the entry
    hot→cold; hot becomes empty, cold holds the thread, return value == moved count."""
    swarm = _make_swarm()
    hot = InMemorySwarmCheckpointer()
    cold = InMemorySwarmCheckpointer()
    tiered = TieredSwarmCheckpointer(hot=hot, cold=cold, archive_after_seconds=0.0)

    state = _make_state(swarm, turns=2)
    await tiered.save(_ckpt("t1", state, 2))

    # Sanity: hot has it, cold does not.
    assert await hot.list_checkpoints() == ["t1"]
    assert await cold.list_checkpoints() == []

    moved = await tiered.archive(swarm)

    assert moved == 1
    assert await hot.list_checkpoints() == []
    assert await cold.list_checkpoints() == ["t1"]


async def test_archive_cold_copy_round_trips() -> None:
    """After archive(), the cold copy re-hydrates to the original state."""
    swarm = _make_swarm()
    hot = InMemorySwarmCheckpointer()
    cold = InMemorySwarmCheckpointer()
    tiered = TieredSwarmCheckpointer(hot=hot, cold=cold, archive_after_seconds=0.0)

    state = _make_state(swarm, turns=7)
    await tiered.save(_ckpt("t1", state, 7))
    await tiered.archive(swarm)

    cp = await cold.load("t1", swarm)
    assert cp is not None
    assert cp.turn == 7
    rehydrated = SwarmState.from_dict(cast(SwarmStateDict, cp.state), swarm)
    assert rehydrated.total_turns == 7
    assert rehydrated.current_agent_name == "m1"


async def test_archive_skips_fresh_entries() -> None:
    """archive() with a far-future cutoff leaves recent entries in hot untouched."""
    swarm = _make_swarm()
    hot = InMemorySwarmCheckpointer()
    cold = InMemorySwarmCheckpointer()
    tiered = TieredSwarmCheckpointer(hot=hot, cold=cold, archive_after_seconds=9999.0)

    state = _make_state(swarm, turns=1)
    await tiered.save(_ckpt("t1", state, 1))

    moved = await tiered.archive(swarm)

    assert moved == 0
    assert await hot.list_checkpoints() == ["t1"]
    assert await cold.list_checkpoints() == []


# ---------------------------------------------------------------------------
# list_checkpoints — union
# ---------------------------------------------------------------------------


async def test_list_checkpoints_returns_union() -> None:
    """list_checkpoints() merges thread ids from both tiers without duplicates."""
    swarm = _make_swarm()
    hot = InMemorySwarmCheckpointer()
    cold = InMemorySwarmCheckpointer()
    tiered = TieredSwarmCheckpointer(hot=hot, cold=cold, archive_after_seconds=3600.0)

    state = _make_state(swarm, turns=1)
    await tiered.save(_ckpt("hot-only", state, 1))
    # Seed cold directly to simulate an already-archived entry.
    await cold.save(_ckpt("cold-only", state, 1))

    ids = await tiered.list_checkpoints()
    assert ids == ["cold-only", "hot-only"]


async def test_list_checkpoints_deduplicates() -> None:
    """When the same thread_id is in both tiers it appears only once."""
    swarm = _make_swarm()
    hot = InMemorySwarmCheckpointer()
    cold = InMemorySwarmCheckpointer()
    tiered = TieredSwarmCheckpointer(hot=hot, cold=cold, archive_after_seconds=3600.0)

    state = _make_state(swarm, turns=1)
    await hot.save(_ckpt("shared", state, 1))
    await cold.save(_ckpt("shared", state, 1))

    ids = await tiered.list_checkpoints()
    assert ids == ["shared"]


# ---------------------------------------------------------------------------
# delete — removes from both tiers
# ---------------------------------------------------------------------------


async def test_delete_removes_from_both_tiers() -> None:
    """delete() removes a thread_id from hot and cold; list returns empty afterwards."""
    swarm = _make_swarm()
    hot = InMemorySwarmCheckpointer()
    cold = InMemorySwarmCheckpointer()
    tiered = TieredSwarmCheckpointer(hot=hot, cold=cold, archive_after_seconds=3600.0)

    state = _make_state(swarm, turns=1)
    await hot.save(_ckpt("t1", state, 1))
    await cold.save(_ckpt("t1", state, 1))

    await tiered.delete("t1")

    assert await hot.list_checkpoints() == []
    assert await cold.list_checkpoints() == []
    assert await tiered.list_checkpoints() == []


async def test_delete_noop_on_missing() -> None:
    """delete() on an unknown thread_id is a no-op (not an error)."""
    hot = InMemorySwarmCheckpointer()
    cold = InMemorySwarmCheckpointer()
    tiered = TieredSwarmCheckpointer(hot=hot, cold=cold, archive_after_seconds=3600.0)

    # Must not raise.
    await tiered.delete("does-not-exist")
    assert await tiered.list_checkpoints() == []


# ---------------------------------------------------------------------------
# Negative archive_after_seconds
# ---------------------------------------------------------------------------


def test_negative_archive_after_rejected() -> None:
    """Constructing TieredSwarmCheckpointer with a negative archive_after_seconds raises ValueError."""
    hot = InMemorySwarmCheckpointer()
    cold = InMemorySwarmCheckpointer()
    with pytest.raises(ValueError, match="archive_after_seconds"):
        TieredSwarmCheckpointer(hot=hot, cold=cold, archive_after_seconds=-1.0)


# ---------------------------------------------------------------------------
# Hook-path archive (FIX 1)
# ---------------------------------------------------------------------------


async def test_archive_sees_hook_driven_saves() -> None:
    """register() routes hook-saves through the composite, so _saved_at is
    populated and archive() can migrate them."""
    from troopai.adk.run.context import RunContext
    from troopai.adk.swarms.hooks import HookRegistry

    swarm = _make_swarm()
    hot = InMemorySwarmCheckpointer()
    cold = InMemorySwarmCheckpointer()
    tiered = TieredSwarmCheckpointer(hot=hot, cold=cold, archive_after_seconds=0.0, thread_id="hooked")
    registry = HookRegistry()
    tiered.register(registry)

    # Fire the auto-save hook with a real state (mirrors how test_checkpointer.py
    # drives on_swarm_turn_end).
    state = _make_state(swarm, turns=2)
    ctx: RunContext = RunContext(context=None)
    await registry.on_swarm_turn_end(ctx, state, [])

    assert await hot.list_checkpoints() == ["hooked"]

    moved = await tiered.archive(swarm)

    assert moved == 1
    assert await cold.list_checkpoints() == ["hooked"]
    assert await hot.list_checkpoints() == []


# ---------------------------------------------------------------------------
# Regression: delete() must NOT log "removed from both tiers" on partial failure
# and must propagate the error (#LOW)
# ---------------------------------------------------------------------------


async def test_delete_partial_failure_raises_and_does_not_log_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: when one tier raises during delete(), the success log
    must NOT fire and the error must propagate to the caller."""
    import logging

    from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer

    class _FailOnDelete(InMemorySwarmCheckpointer):
        """Always raises on delete regardless of whether the key exists."""

        async def delete(self, thread_id: str) -> None:
            raise OSError("storage unavailable")

    hot = InMemorySwarmCheckpointer()
    cold = _FailOnDelete()
    tiered = TieredSwarmCheckpointer(hot=hot, cold=cold, archive_after_seconds=3600.0)

    swarm = _make_swarm()
    state = _make_state(swarm, turns=1)
    await hot.save(_ckpt("t1", state, 1))

    with (
        pytest.raises(OSError, match="storage unavailable"),
        caplog.at_level(logging.DEBUG, logger="troopai.adk.swarms.checkpointers.tiered"),
    ):
        await tiered.delete("t1")

    # The "removed from both tiers" success message must NOT appear
    success_messages = [r.message for r in caplog.records if "removed from both tiers" in r.message]
    assert len(success_messages) == 0, f"delete() must not log success when a tier failed; got: {success_messages}"

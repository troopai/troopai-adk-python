"""Tests for ``TieredCheckpointer`` — hot/cold composite graph checkpointer.

Covers:
- Load falls back to cold and re-warms hot.
- archive() with archive_after_seconds=0.0 moves a saved entry hot→cold.
- list_checkpoints() is the union of both tiers.
- delete() removes from both tiers.
- register() routes hook-driven saves through the composite (FIX 1).
"""

from __future__ import annotations

import asyncio

import pytest

from troopai.adk.graphs.checkpointer import GraphCheckpoint
from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
from troopai.adk.graphs.checkpointers.tiered import TieredCheckpointer
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.hooks import HookRegistry
from troopai.adk.graphs.state import GraphState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _g() -> Graph:
    """Minimal two-node graph for checkpointer tests."""
    return (
        Graph.new("tiered-cp-test")
        .node("a", lambda: "a")
        .node("b", lambda: "b")
        .edge("a", "b")
        .entry("a")
        .terminal("b")
        .compile()
    )


def _checkpoint(
    graph: Graph,
    thread_id: str,
    superstep: int = 0,
) -> GraphCheckpoint:
    """Build a minimal :class:`GraphCheckpoint` for ``graph`` and ``thread_id``."""
    state = GraphState(graph=graph, thread_id=thread_id)
    state.superstep = superstep
    return GraphCheckpoint(
        thread_id=thread_id,
        graph_id=graph.id,
        state=state.to_dict(),
        superstep=superstep,
    )


# ---------------------------------------------------------------------------
# Cold fallback + re-warm
# ---------------------------------------------------------------------------


async def test_load_falls_back_to_cold_and_rewarms_hot() -> None:
    """When hot is empty and cold has a checkpoint, load returns the state and
    re-warms the hot tier so a subsequent list_checkpoints includes the thread."""
    g = _g()
    hot = InMemoryCheckpointer()
    cold = InMemoryCheckpointer()
    tiered = TieredCheckpointer(hot=hot, cold=cold, archive_after_seconds=3600.0)

    # Seed cold only (bypass the tiered composite to simulate a prior archive).
    await cold.save(_checkpoint(g, "t1", superstep=2))

    # Hot must be empty before the load.
    assert await hot.list_checkpoints() == []

    # load() must find the state through cold.
    state = await tiered.load("t1", g)
    assert state is not None
    assert state.superstep == 2

    # After load, hot must have been re-warmed.
    assert await hot.list_checkpoints() == ["t1"]


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


async def test_archive_moves_hot_to_cold() -> None:
    """archive() with archive_after_seconds=0.0 immediately moves the entry
    hot→cold; hot becomes empty, cold holds the thread, and the return value
    equals the number moved."""
    g = _g()
    hot = InMemoryCheckpointer()
    cold = InMemoryCheckpointer()
    tiered = TieredCheckpointer(hot=hot, cold=cold, archive_after_seconds=0.0)

    # Save through the composite so _saved_at is populated.
    await tiered.save(_checkpoint(g, "t1", superstep=1))

    # Sanity: hot has it, cold does not.
    assert await hot.list_checkpoints() == ["t1"]
    assert await cold.list_checkpoints() == []

    moved = await tiered.archive(g)

    assert moved == 1
    assert await hot.list_checkpoints() == []
    assert await cold.list_checkpoints() == ["t1"]


async def test_archive_skips_fresh_entries() -> None:
    """archive() with a future cutoff leaves recent entries in hot untouched."""
    g = _g()
    hot = InMemoryCheckpointer()
    cold = InMemoryCheckpointer()
    tiered = TieredCheckpointer(hot=hot, cold=cold, archive_after_seconds=9999.0)

    await tiered.save(_checkpoint(g, "t1", superstep=1))

    moved = await tiered.archive(g)

    assert moved == 0
    assert await hot.list_checkpoints() == ["t1"]
    assert await cold.list_checkpoints() == []


# ---------------------------------------------------------------------------
# list_checkpoints — union
# ---------------------------------------------------------------------------


async def test_list_checkpoints_returns_union() -> None:
    """list_checkpoints() merges thread ids from both tiers without duplicates."""
    g = _g()
    hot = InMemoryCheckpointer()
    cold = InMemoryCheckpointer()
    tiered = TieredCheckpointer(hot=hot, cold=cold, archive_after_seconds=3600.0)

    await tiered.save(_checkpoint(g, "hot-only", superstep=1))
    # Seed cold directly to simulate an already-archived entry.
    await cold.save(_checkpoint(g, "cold-only", superstep=2))

    ids = await tiered.list_checkpoints()
    assert ids == ["cold-only", "hot-only"]


async def test_list_checkpoints_deduplicates() -> None:
    """When the same thread_id is in both tiers it appears only once."""
    g = _g()
    hot = InMemoryCheckpointer()
    cold = InMemoryCheckpointer()
    tiered = TieredCheckpointer(hot=hot, cold=cold, archive_after_seconds=3600.0)

    await hot.save(_checkpoint(g, "shared", superstep=1))
    await cold.save(_checkpoint(g, "shared", superstep=1))

    ids = await tiered.list_checkpoints()
    assert ids == ["shared"]


# ---------------------------------------------------------------------------
# delete — removes from both tiers
# ---------------------------------------------------------------------------


async def test_delete_removes_from_both_tiers() -> None:
    """delete() removes a thread_id from hot and cold; list returns empty afterwards."""
    g = _g()
    hot = InMemoryCheckpointer()
    cold = InMemoryCheckpointer()
    tiered = TieredCheckpointer(hot=hot, cold=cold, archive_after_seconds=3600.0)

    await hot.save(_checkpoint(g, "t1", superstep=0))
    await cold.save(_checkpoint(g, "t1", superstep=0))

    await tiered.delete("t1")

    assert await hot.list_checkpoints() == []
    assert await cold.list_checkpoints() == []
    assert await tiered.list_checkpoints() == []


async def test_delete_noop_on_missing() -> None:
    """delete() on an unknown thread_id is a no-op (not an error)."""
    hot = InMemoryCheckpointer()
    cold = InMemoryCheckpointer()
    tiered = TieredCheckpointer(hot=hot, cold=cold, archive_after_seconds=3600.0)

    # Must not raise.
    await tiered.delete("does-not-exist")
    assert await tiered.list_checkpoints() == []


# ---------------------------------------------------------------------------
# Negative archive_after_seconds
# ---------------------------------------------------------------------------


def test_negative_archive_after_rejected() -> None:
    """Constructing TieredCheckpointer with a negative archive_after_seconds raises ValueError."""
    hot = InMemoryCheckpointer()
    cold = InMemoryCheckpointer()
    with pytest.raises(ValueError, match="archive_after_seconds"):
        TieredCheckpointer(hot=hot, cold=cold, archive_after_seconds=-1.0)


# ---------------------------------------------------------------------------
# Hook-path archive (FIX 1)
# ---------------------------------------------------------------------------


async def test_archive_sees_hook_driven_saves() -> None:
    """register() routes hook-saves through the composite, so _saved_at is
    populated and archive() can migrate them."""
    from troopai.adk.orchestration.executable import NodeResult

    g = _g()
    hot = InMemoryCheckpointer()
    cold = InMemoryCheckpointer()
    tiered = TieredCheckpointer(hot=hot, cold=cold, archive_after_seconds=0.0)
    registry = HookRegistry()
    tiered.register(registry)

    # Fire the auto-save hook with a real state (mirrors how test_checkpointer.py
    # drives on_node_end).
    state = GraphState(graph=g, thread_id="hooked")
    state.superstep = 1

    await registry.on_node_end(
        context=None,  # type: ignore[arg-type]
        state=state,
        node_id="a",
        result=NodeResult(output="x"),
    )

    assert await hot.list_checkpoints() == ["hooked"]

    moved = await tiered.archive(g)

    assert moved == 1
    assert await cold.list_checkpoints() == ["hooked"]
    assert await hot.list_checkpoints() == []


# ---------------------------------------------------------------------------
# Archive vs concurrent save — no checkpoint loss
# ---------------------------------------------------------------------------


class _SlowSaveCold(InMemoryCheckpointer):
    """Cold tier whose save() parks on an event, exposing the migrate window.

    The data-loss window in ``_migrate_one`` lies between ``hot.load``
    returning and ``hot.delete`` running; parking the cold save holds the
    migration inside that window.
    """

    def __init__(self) -> None:
        super().__init__()
        self.save_started = asyncio.Event()
        self.release_save = asyncio.Event()

    async def save(self, checkpoint: GraphCheckpoint) -> None:
        self.save_started.set()
        await self.release_save.wait()
        await super().save(checkpoint)


@pytest.mark.asyncio
async def test_archive_does_not_discard_concurrent_save() -> None:
    """A save landing mid-migration must survive, not be hot-deleted.

    Sequence without the per-thread guard: archive loads superstep 0 from
    hot and starts writing it to cold; a concurrent save writes superstep 1
    to hot; the migration then deletes hot — silently discarding
    superstep 1. With the guard, the save waits until migration completes
    and re-creates the hot entry.
    """
    graph = _g()
    hot = InMemoryCheckpointer()
    cold = _SlowSaveCold()
    tiered = TieredCheckpointer(hot=hot, cold=cold, archive_after_seconds=0.0)

    await tiered.save(_checkpoint(graph, "t1", superstep=0))

    archive_task = asyncio.create_task(tiered.archive(graph))
    await cold.save_started.wait()  # migration parked between hot.load and hot.delete

    # Concurrent newer save while the migration is mid-window.
    save_task = asyncio.create_task(tiered.save(_checkpoint(graph, "t1", superstep=1)))
    await asyncio.sleep(0)  # let the save run (or block on the guard)
    cold.release_save.set()

    moved = await archive_task
    await save_task
    assert moved == 1

    # The newer superstep must still be loadable: hot has 1, cold kept 0.
    state = await tiered.load("t1", graph)
    assert state is not None
    assert state.superstep == 1, "superstep-1 save was discarded by the archive migration"


# ---------------------------------------------------------------------------
# Per-thread guard is never orphaned (refcounted lock map)
# ---------------------------------------------------------------------------


class _OverlapDetectingHot(InMemoryCheckpointer):
    """Hot tier that records the peak number of overlapping save() bodies.

    While ``arm`` is set, each save() announces entry, parks on
    ``release_saves`` holding its critical section, then completes. If two
    callers are *not* serialised by the per-thread guard, both reach the
    park at once and ``max_concurrent_saves`` rises above one — a
    deterministic witness of the orphaned-lock race. Before arming, save()
    behaves like the plain in-memory store so setup does not deadlock.
    """

    def __init__(self) -> None:
        super().__init__()
        self.arm = False
        self.in_save = 0
        self.max_concurrent_saves = 0
        self.entered = asyncio.Event()
        self.release_saves = asyncio.Event()

    async def save(self, checkpoint: GraphCheckpoint) -> None:
        if not self.arm:
            await super().save(checkpoint)
            return
        self.in_save += 1
        self.max_concurrent_saves = max(self.max_concurrent_saves, self.in_save)
        self.entered.set()
        try:
            await self.release_saves.wait()
            await super().save(checkpoint)
        finally:
            self.in_save -= 1


@pytest.mark.asyncio
async def test_guard_not_orphaned_by_migration_with_pending_waiter() -> None:
    """A migration that finishes while a save waits on the same lock must not
    drop that lock — otherwise the waiting save and a fresh third caller would
    serialise on two different lock objects and run the per-thread critical
    section concurrently.

    Trigger: migration M holds lock L1 (parked in cold.save); save B awaits the
    SAME L1; M finishes and (with the bug) deletes L1 right after release while
    B's waiter is scheduled-but-not-resumed; B then holds an orphaned L1 while a
    third save C, finding no map entry, creates a fresh L2 and overlaps B.
    """
    graph = _g()
    hot = _OverlapDetectingHot()
    cold = _SlowSaveCold()
    tiered = TieredCheckpointer(hot=hot, cold=cold, archive_after_seconds=0.0)

    # Seed through the composite so hot has the entry AND _saved_at is recorded
    # for archive(); cold.save runs immediately for this initial save.
    cold.release_save.set()
    await tiered.save(_checkpoint(graph, "t1", superstep=0))
    cold.release_save.clear()
    cold.save_started.clear()
    hot.arm = True  # from here, hot.save parks so overlap is observable

    # M starts and parks between hot.load and hot.delete (holding L1).
    archive_task = asyncio.create_task(tiered.archive(graph))
    await cold.save_started.wait()

    # B awaits the SAME L1 (setdefault returns the existing lock).
    save_b = asyncio.create_task(tiered.save(_checkpoint(graph, "t1", superstep=1)))
    await asyncio.sleep(0)  # let B reach the lock acquisition

    # Release M: it deletes hot and exits its guard. With the bug, the lock
    # entry is dropped here while B's waiter is scheduled but not yet resumed.
    cold.release_save.set()
    await hot.entered.wait()  # B (the next save) reaches the hot critical section

    # C arrives while B is parked in its critical section. With the bug it
    # builds a fresh lock and enters concurrently; with the fix it waits on L1.
    save_c = asyncio.create_task(tiered.save(_checkpoint(graph, "t1", superstep=2)))
    await asyncio.sleep(0)
    await asyncio.sleep(0)  # give C every chance to (wrongly) enter

    overlap = hot.max_concurrent_saves
    hot.release_saves.set()  # drain B and C

    await archive_task
    await save_b
    await save_c

    assert overlap == 1, (
        "two coroutines entered the per-thread critical section at once — "
        "the per-thread guard was orphaned by the migration"
    )

    # The lock map must not leak entries once every caller is done.
    assert tiered._thread_locks == {}
    assert tiered._lock_refs == {}


@pytest.mark.asyncio
async def test_lock_map_is_cleaned_after_plain_save() -> None:
    """A save with no concurrent waiter leaves no lock-map entry behind, so the
    guard table is bounded by concurrently in-flight threads, not total runs."""
    graph = _g()
    hot = InMemoryCheckpointer()
    cold = InMemoryCheckpointer()
    tiered = TieredCheckpointer(hot=hot, cold=cold, archive_after_seconds=3600.0)

    await tiered.save(_checkpoint(graph, "t1", superstep=0))

    assert tiered._thread_locks == {}
    assert tiered._lock_refs == {}

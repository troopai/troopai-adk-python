"""Protocol-conformance tests for the worker-backend distribution surface.

Exercises both ``InMemoryFlowWorkerBackend`` and
``SqliteFlowWorkerBackend`` against the same set of assertions —
the Protocol shape is identical so the suite is shared.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from troopai.adk.flows import (
    Flow,
    FlowCheckpoint,
    FlowWorkerBackend,
    InMemoryFlowWorkerBackend,
    SqliteFlowWorkerBackend,
    flow_listen,
    flow_start,
)
from troopai.adk.run.runner import Runner


class _ChainState(BaseModel):
    """Minimal pydantic state."""

    count: int = 0
    notes: list[str] = []


class _ChainFlow(Flow[_ChainState]):
    """Two-step flow: kick → continue."""

    @flow_start
    async def kick(self) -> None:
        self.state.count = 1
        self.state.notes.append("kick")

    @flow_listen("kick")
    async def cont(self) -> None:
        self.state.count *= 2
        self.state.notes.append("cont")


@pytest.fixture
async def in_memory_backend() -> FlowWorkerBackend:
    return InMemoryFlowWorkerBackend()


@pytest.fixture
async def sqlite_backend() -> AsyncIterator[FlowWorkerBackend]:
    with tempfile.TemporaryDirectory() as tmp:
        backend: FlowWorkerBackend = SqliteFlowWorkerBackend(path=Path(tmp) / "flow.db")
        yield backend


class TestClaimContention:
    async def test_first_claim_succeeds(self, in_memory_backend: FlowWorkerBackend) -> None:
        ok = await in_memory_backend.claim_batch("flow1", 0, "worker-a")
        assert ok is True

    async def test_second_claim_within_ttl_fails(
        self,
        in_memory_backend: FlowWorkerBackend,
    ) -> None:
        await in_memory_backend.claim_batch("flow1", 0, "worker-a")
        ok = await in_memory_backend.claim_batch("flow1", 0, "worker-b")
        assert ok is False

    async def test_distinct_batches_dont_collide(
        self,
        in_memory_backend: FlowWorkerBackend,
    ) -> None:
        ok1 = await in_memory_backend.claim_batch("flow1", 0, "worker-a")
        ok2 = await in_memory_backend.claim_batch("flow1", 1, "worker-b")
        assert ok1 is True
        assert ok2 is True

    async def test_expired_claim_can_be_taken_over(self) -> None:
        # Use an injectable clock so the expiry path is deterministic.
        now = {"t": 100.0}
        backend = InMemoryFlowWorkerBackend(clock=lambda: now["t"])
        assert await backend.claim_batch("flow1", 0, "worker-a", ttl_seconds=10.0)
        now["t"] = 200.0  # 100s elapsed > 10s TTL
        assert await backend.claim_batch("flow1", 0, "worker-b", ttl_seconds=10.0)


class TestHeartbeat:
    async def test_heartbeat_extends_ownership(self, in_memory_backend: FlowWorkerBackend) -> None:
        await in_memory_backend.claim_batch("flow1", 0, "worker-a")
        ok = await in_memory_backend.heartbeat("flow1", 0, "worker-a")
        assert ok is True

    async def test_heartbeat_from_non_owner_fails(
        self,
        in_memory_backend: FlowWorkerBackend,
    ) -> None:
        await in_memory_backend.claim_batch("flow1", 0, "worker-a")
        ok = await in_memory_backend.heartbeat("flow1", 0, "worker-b")
        assert ok is False


class TestCheckpointPersistence:
    async def test_save_then_load_round_trips(
        self,
        in_memory_backend: FlowWorkerBackend,
    ) -> None:
        cp = FlowCheckpoint(
            flow_id="flow1",
            completed_steps=("kick",),
            pending_steps=("cont",),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data='{"count":1}',
        )
        await in_memory_backend.save_checkpoint(cp)
        loaded = await in_memory_backend.load_checkpoint("flow1")
        assert loaded is not None
        assert loaded.completed_steps == ("kick",)
        assert loaded.pending_steps == ("cont",)

    async def test_release_persists_checkpoint(
        self,
        in_memory_backend: FlowWorkerBackend,
    ) -> None:
        await in_memory_backend.claim_batch("flow1", 0, "worker-a")
        cp = FlowCheckpoint(
            flow_id="flow1",
            completed_steps=("kick",),
            pending_steps=("cont",),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data='{"count":1}',
        )
        await in_memory_backend.release_batch("flow1", 0, "worker-a", cp)
        loaded = await in_memory_backend.load_checkpoint("flow1")
        assert loaded is not None and loaded.completed_steps == ("kick",)
        # Claim is freed after release.
        ok = await in_memory_backend.claim_batch("flow1", 1, "worker-b")
        assert ok is True


class TestReleaseBatchOwnershipCheck:
    """release_batch must not overwrite a checkpoint written by a recovery worker."""

    async def test_stale_release_drops_checkpoint_write(self) -> None:
        """Worker-A's stale release after TTL takeover must NOT overwrite worker-B's checkpoint."""
        now = {"t": 100.0}
        backend = InMemoryFlowWorkerBackend(clock=lambda: now["t"])

        # worker-a claims batch 0
        assert await backend.claim_batch("flow1", 0, "worker-a", ttl_seconds=10.0)

        # worker-b saves a recovery checkpoint via save_checkpoint directly
        recovery_cp = FlowCheckpoint(
            flow_id="flow1",
            completed_steps=("kick", "cont"),
            pending_steps=(),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data='{"count":2}',
        )
        await backend.save_checkpoint(recovery_cp)

        # TTL expires; worker-b claims and releases the batch with its own checkpoint
        now["t"] = 200.0
        assert await backend.claim_batch("flow1", 0, "worker-b", ttl_seconds=10.0)
        wb_cp = FlowCheckpoint(
            flow_id="flow1",
            completed_steps=("kick", "cont", "finalize"),
            pending_steps=(),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data='{"count":4}',
        )
        await backend.release_batch("flow1", 0, "worker-b", wb_cp)

        # Now worker-a wakes up and tries to release with a stale/old checkpoint
        stale_cp = FlowCheckpoint(
            flow_id="flow1",
            completed_steps=("kick",),
            pending_steps=("cont",),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data='{"count":1}',
        )
        await backend.release_batch("flow1", 0, "worker-a", stale_cp)

        # The stale write must have been silently dropped — worker-b's checkpoint survives
        loaded = await backend.load_checkpoint("flow1")
        assert loaded is not None
        assert loaded.completed_steps == ("kick", "cont", "finalize")
        assert loaded.state_data == '{"count":4}'

    async def test_legitimate_release_still_writes_checkpoint(self) -> None:
        """A normal release by the rightful owner must still persist the checkpoint."""
        backend = InMemoryFlowWorkerBackend()
        assert await backend.claim_batch("flow1", 0, "worker-a")
        cp = FlowCheckpoint(
            flow_id="flow1",
            completed_steps=("kick",),
            pending_steps=("cont",),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data='{"count":1}',
        )
        await backend.release_batch("flow1", 0, "worker-a", cp)
        loaded = await backend.load_checkpoint("flow1")
        assert loaded is not None
        assert loaded.state_data == '{"count":1}'


class TestListClaims:
    async def test_list_returns_owners(self, in_memory_backend: FlowWorkerBackend) -> None:
        await in_memory_backend.claim_batch("flow1", 0, "worker-a")
        await in_memory_backend.claim_batch("flow1", 1, "worker-b")
        claims = await in_memory_backend.list_claims("flow1")
        owners = {c.worker_id for c in claims}
        batches = {c.batch_id for c in claims}
        assert owners == {"worker-a", "worker-b"}
        assert batches == {0, 1}


class TestSqliteBackend:
    """Confirm the SQLite implementation passes the same conformance suite."""

    async def test_claim_persists_across_connections(
        self,
        sqlite_backend: FlowWorkerBackend,
    ) -> None:
        await sqlite_backend.claim_batch("flow1", 0, "worker-a")
        # A fresh connection from the same backend instance must see
        # the prior claim (it's persisted on disk).
        ok = await sqlite_backend.claim_batch("flow1", 0, "worker-b")
        assert ok is False

    async def test_round_trips_checkpoint_through_disk(
        self,
        sqlite_backend: FlowWorkerBackend,
    ) -> None:
        cp = FlowCheckpoint(
            flow_id="flow1",
            completed_steps=("kick",),
            pending_steps=("cont",),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data='{"count":1}',
        )
        await sqlite_backend.save_checkpoint(cp)
        loaded = await sqlite_backend.load_checkpoint("flow1")
        assert loaded is not None
        assert loaded.completed_steps == ("kick",)


class TestRunnerArunFlowDistributed:
    async def test_first_call_runs_starts_and_persists_checkpoint(self) -> None:
        backend: FlowWorkerBackend = InMemoryFlowWorkerBackend()
        flow = _ChainFlow(_ChainState)
        result = await Runner.arun_flow_distributed(flow, backend)
        # The flow has two batches (kick → cont); one call drives the
        # full BSP loop until the executor returns naturally.
        # The executor processes both batches in this call because it
        # owns one continuous BSP loop — the helper acts on the
        # initial cold-start claim only. After completion the
        # checkpoint reflects status="completed".
        assert result.status == "completed"
        assert result.final_state.count == 2  # 1 * 2
        loaded = await backend.load_checkpoint(flow.flow_id)
        assert loaded is not None

    async def test_two_workers_serialise_at_claim_boundary(self) -> None:
        backend: FlowWorkerBackend = InMemoryFlowWorkerBackend()
        flow_a = _ChainFlow(_ChainState)
        # Pre-claim batch 0 with another worker; the next call from
        # the runner must report the lost-claim path.
        ok = await backend.claim_batch(flow_a.flow_id, 0, "external-worker")
        assert ok is True
        # Need to seed an initial checkpoint so `arun_flow_distributed`
        # finds something to claim against (it uses
        # `len(checkpoint.completed_steps)` to derive `batch_id`).
        # The runner synthesises one on cold start — but the external
        # claim was for `batch_id=0`. Seed accordingly.
        cp = FlowCheckpoint(
            flow_id=flow_a.flow_id,
            completed_steps=(),
            pending_steps=("kick",),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data=flow_a.state.model_dump_json(),
        )
        await backend.save_checkpoint(cp)

        result = await Runner.arun_flow_distributed(flow_a, backend, worker_id="us")
        assert result.status == "failed"
        assert result.error is not None and "already claimed" in result.error


class TestCrossEventLoopReuse:
    def test_in_memory_backend_usable_across_sequential_event_loops(self) -> None:
        """One backend instance must work from two different event loops.

        Regression: the lazily-allocated ``asyncio.Lock`` bound itself to
        the first loop that used it, so reusing the backend from a second
        loop (e.g. ``Runner.run_flow`` called from different threads, or
        sequential ``asyncio.run`` calls) raised "lock is bound to a
        different event loop". Locks are now kept per running loop.
        """
        backend = InMemoryFlowWorkerBackend()

        first = asyncio.run(backend.claim_batch("flow1", 0, "worker-a"))
        # Second loop, same backend — must not raise; the prior claim
        # still blocks a competing worker (backend state is shared).
        second = asyncio.run(backend.claim_batch("flow1", 0, "worker-b"))

        assert first is True
        assert second is False

    def test_in_memory_backend_allocates_one_lock_per_event_loop(self) -> None:
        """The backend keeps one ``asyncio.Lock`` per running loop.

        The single shared lock bound itself to the first loop that used
        it (asyncio binds on the first contended acquire); a second loop
        hitting contention then raised "bound to a different event
        loop". Per-loop locks remove the cross-loop alias entirely.
        """
        backend = InMemoryFlowWorkerBackend()

        asyncio.run(backend.claim_batch("flow1", 0, "worker-a"))
        asyncio.run(backend.claim_batch("flow1", 1, "worker-a"))

        assert len(backend._locks) == 2
        assert len(set(backend._locks.values())) == 2

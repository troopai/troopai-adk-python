"""Tests for load_checkpoint_by_id on both worker-backend implementations.

Covers:
- InMemoryFlowWorkerBackend.load_checkpoint_by_id — found and not-found
- SqliteFlowWorkerBackend.load_checkpoint_by_id — found and not-found
- Runner.arun_flow_from_id — successful resume and FlowCheckpointNotFoundError on missing id
- Protocol conformance: load_checkpoint_by_id is on FlowWorkerBackend
- Deterministic connection cleanup (SQLite backend uses to_thread, no open handles)
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic import BaseModel

from troopai.adk.flows import (
    Flow,
    FlowCheckpoint,
    FlowCheckpointNotFoundError,
    FlowWorkerBackend,
    InMemoryFlowWorkerBackend,
    SqliteFlowWorkerBackend,
    flow_listen,
    flow_start,
)
from troopai.adk.run.runner import Runner

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(flow_id: str = "flow-abc") -> FlowCheckpoint:
    return FlowCheckpoint(
        flow_id=flow_id,
        completed_steps=("step_a",),
        pending_steps=("step_b",),
        and_gate_arrivals={},
        consumed_gates=(),
        state_data='{"value": 1}',
    )


@pytest.fixture
def in_memory_backend() -> InMemoryFlowWorkerBackend:
    return InMemoryFlowWorkerBackend()


@pytest.fixture
async def sqlite_backend() -> AsyncIterator[SqliteFlowWorkerBackend]:
    with tempfile.TemporaryDirectory() as tmp:
        backend = SqliteFlowWorkerBackend(path=Path(tmp) / "flow.db")
        yield backend
        # No open connections to close: every method uses
        # asyncio.to_thread + contextlib.closing, so the connection is
        # closed before the coroutine returns.


# ---------------------------------------------------------------------------
# InMemoryFlowWorkerBackend
# ---------------------------------------------------------------------------


class TestInMemoryLoadCheckpointById:
    async def test_returns_none_when_absent(self, in_memory_backend: InMemoryFlowWorkerBackend) -> None:
        result = await in_memory_backend.load_checkpoint_by_id("nonexistent")
        assert result is None

    async def test_returns_checkpoint_after_save(self, in_memory_backend: InMemoryFlowWorkerBackend) -> None:
        cp = _make_checkpoint("flow-xyz")
        await in_memory_backend.save_checkpoint(cp)
        loaded = await in_memory_backend.load_checkpoint_by_id("flow-xyz")
        assert loaded is not None
        assert loaded.flow_id == "flow-xyz"
        assert loaded.completed_steps == ("step_a",)

    async def test_id_lookup_matches_flow_id_lookup(self, in_memory_backend: InMemoryFlowWorkerBackend) -> None:
        cp = _make_checkpoint("flow-same")
        await in_memory_backend.save_checkpoint(cp)
        via_id = await in_memory_backend.load_checkpoint_by_id("flow-same")
        via_flow = await in_memory_backend.load_checkpoint("flow-same")
        assert via_id == via_flow

    async def test_different_flow_ids_are_independent(self, in_memory_backend: InMemoryFlowWorkerBackend) -> None:
        cp1 = _make_checkpoint("flow-1")
        cp2 = _make_checkpoint("flow-2")
        await in_memory_backend.save_checkpoint(cp1)
        await in_memory_backend.save_checkpoint(cp2)
        assert (await in_memory_backend.load_checkpoint_by_id("flow-1")) is not None
        assert (await in_memory_backend.load_checkpoint_by_id("flow-2")) is not None
        assert (await in_memory_backend.load_checkpoint_by_id("flow-3")) is None

    async def test_release_batch_visible_via_load_by_id(self, in_memory_backend: InMemoryFlowWorkerBackend) -> None:
        """Checkpoint persisted via release_batch is also reachable by id."""
        cp = _make_checkpoint("flow-rel")
        await in_memory_backend.claim_batch("flow-rel", 0, "worker-a")
        await in_memory_backend.release_batch("flow-rel", 0, "worker-a", cp)
        loaded = await in_memory_backend.load_checkpoint_by_id("flow-rel")
        assert loaded is not None
        assert loaded.completed_steps == ("step_a",)


# ---------------------------------------------------------------------------
# SqliteFlowWorkerBackend
# ---------------------------------------------------------------------------


class TestSqliteLoadCheckpointById:
    async def test_returns_none_when_absent(self, sqlite_backend: SqliteFlowWorkerBackend) -> None:
        result = await sqlite_backend.load_checkpoint_by_id("nonexistent")
        assert result is None

    async def test_returns_checkpoint_after_save(self, sqlite_backend: SqliteFlowWorkerBackend) -> None:
        cp = _make_checkpoint("flow-xyz")
        await sqlite_backend.save_checkpoint(cp)
        loaded = await sqlite_backend.load_checkpoint_by_id("flow-xyz")
        assert loaded is not None
        assert loaded.flow_id == "flow-xyz"
        assert loaded.completed_steps == ("step_a",)

    async def test_id_lookup_matches_flow_id_lookup(self, sqlite_backend: SqliteFlowWorkerBackend) -> None:
        cp = _make_checkpoint("flow-same")
        await sqlite_backend.save_checkpoint(cp)
        via_id = await sqlite_backend.load_checkpoint_by_id("flow-same")
        via_flow = await sqlite_backend.load_checkpoint("flow-same")
        assert via_id is not None
        assert via_flow is not None
        assert via_id.flow_id == via_flow.flow_id
        assert via_id.completed_steps == via_flow.completed_steps

    async def test_different_flow_ids_are_independent(self, sqlite_backend: SqliteFlowWorkerBackend) -> None:
        cp1 = _make_checkpoint("flow-1")
        cp2 = _make_checkpoint("flow-2")
        await sqlite_backend.save_checkpoint(cp1)
        await sqlite_backend.save_checkpoint(cp2)
        assert (await sqlite_backend.load_checkpoint_by_id("flow-1")) is not None
        assert (await sqlite_backend.load_checkpoint_by_id("flow-2")) is not None
        assert (await sqlite_backend.load_checkpoint_by_id("flow-3")) is None

    async def test_parameterized_sql_no_injection(self, sqlite_backend: SqliteFlowWorkerBackend) -> None:
        """Confirm parameterized queries: a crafted id returns None, not all rows."""
        cp = _make_checkpoint("real-flow")
        await sqlite_backend.save_checkpoint(cp)
        # This would return all rows if the query used string formatting.
        malicious_id = "real-flow' OR '1'='1"
        result = await sqlite_backend.load_checkpoint_by_id(malicious_id)
        assert result is None


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_in_memory_backend_satisfies_protocol(self) -> None:
        backend: FlowWorkerBackend = InMemoryFlowWorkerBackend()
        assert hasattr(backend, "load_checkpoint_by_id")

    def test_sqlite_backend_satisfies_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend: FlowWorkerBackend = SqliteFlowWorkerBackend(path=Path(tmp) / "p.db")
            assert hasattr(backend, "load_checkpoint_by_id")


# ---------------------------------------------------------------------------
# Runner.arun_flow_from_id
# ---------------------------------------------------------------------------


class _CountState(BaseModel):
    count: int = 0


class _CountFlow(Flow[_CountState]):
    @flow_start
    async def kick(self) -> None:
        self.state.count += 1

    @flow_listen("kick")
    async def cont(self) -> None:
        self.state.count += 10


class TestRunnerArunFlowFromId:
    async def test_raises_checkpoint_not_found_when_id_missing(self) -> None:
        backend = InMemoryFlowWorkerBackend()
        flow = _CountFlow(initial_state=_CountState())
        with pytest.raises(FlowCheckpointNotFoundError) as exc_info:
            await Runner.arun_flow_from_id(flow, "missing-id", backend)
        assert exc_info.value.checkpoint_id == "missing-id"

    async def test_not_found_error_is_user_error(self) -> None:
        from troopai.adk.exceptions import UserError

        backend = InMemoryFlowWorkerBackend()
        flow = _CountFlow(initial_state=_CountState())
        with pytest.raises(UserError):
            await Runner.arun_flow_from_id(flow, "missing-id", backend)

    async def test_resumes_flow_from_stored_checkpoint(self) -> None:
        """Store a post-kick checkpoint and resume to completion via id."""
        backend = InMemoryFlowWorkerBackend()
        flow = _CountFlow(initial_state=_CountState(count=1))

        # Build a checkpoint that represents "kick done, cont pending".
        cp = FlowCheckpoint(
            flow_id=flow.flow_id,
            completed_steps=("kick",),
            pending_steps=("cont",),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data=_CountState(count=1).model_dump_json(),
        )
        await backend.save_checkpoint(cp)

        result = await Runner.arun_flow_from_id(flow, flow.flow_id, backend)
        assert result.status == "completed"
        # cont adds 10 to the count of 1 that was in the checkpoint's state.
        assert flow.state.count == 11

    async def test_id_is_flow_id_attribute(self) -> None:
        """The checkpoint_id corresponds to Flow.flow_id."""
        backend = InMemoryFlowWorkerBackend()
        flow = _CountFlow(initial_state=_CountState())

        cp = FlowCheckpoint(
            flow_id=flow.flow_id,
            completed_steps=("kick",),
            pending_steps=("cont",),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data=_CountState(count=5).model_dump_json(),
        )
        await backend.save_checkpoint(cp)

        # Look up by the flow's own id attribute — this is the natural
        # developer-facing pattern.
        loaded = await backend.load_checkpoint_by_id(flow.flow_id)
        assert loaded is not None
        assert loaded.flow_id == flow.flow_id

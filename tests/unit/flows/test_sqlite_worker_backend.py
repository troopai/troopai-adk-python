"""Regression tests for :class:`SqliteFlowWorkerBackend` defect fixes.

Covers three confirmed defects:

1. ``path=":memory:"`` is rejected at construction (each connection
   would open an independent empty database, so the eagerly-created
   schema would not survive).
2. Writes use column-named ``INSERT OR REPLACE`` so additive schema
   evolution (``ALTER TABLE ... ADD COLUMN ... DEFAULT``) keeps working.
3. ``_save_checkpoint_sync`` opens ``BEGIN IMMEDIATE`` outside the
   ``try`` so a failed BEGIN under write contention surfaces the real
   lock error instead of a masking "no transaction is active".
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from troopai.adk.flows import FlowCheckpoint, SqliteFlowWorkerBackend


def _make_checkpoint(flow_id: str = "flow1") -> FlowCheckpoint:
    return FlowCheckpoint(
        flow_id=flow_id,
        completed_steps=("kick",),
        pending_steps=("cont",),
        and_gate_arrivals={},
        consumed_gates=(),
        state_data='{"count":1}',
    )


class TestMemoryPathRejected:
    """Finding 1: a ``:memory:`` path must fail loudly at construction."""

    def test_memory_path_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match=":memory:"):
            SqliteFlowWorkerBackend(path=":memory:")

    async def test_file_path_constructs_and_operates(self) -> None:
        # A real file path must continue to work end-to-end.
        with tempfile.TemporaryDirectory() as tmp:
            backend = SqliteFlowWorkerBackend(path=Path(tmp) / "flow.db")
            await backend.save_checkpoint(_make_checkpoint())
            loaded = await backend.load_checkpoint("flow1")
            assert loaded is not None
            assert loaded.completed_steps == ("kick",)


class TestAdditiveSchemaEvolution:
    """Finding 2: writes must survive an additive ``ADD COLUMN``."""

    async def test_save_checkpoint_after_add_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "flow.db"
            backend = SqliteFlowWorkerBackend(path=db_path)

            # Apply the documented forward-compatibility evolution path:
            # add a new column with a safe default.
            conn = sqlite3.connect(str(db_path), isolation_level=None)
            try:
                conn.execute("ALTER TABLE flow_checkpoints ADD COLUMN extra TEXT DEFAULT NULL")
            finally:
                conn.close()

            # The positional-INSERT bug would raise here:
            # "table flow_checkpoints has 4 columns but 3 values were supplied".
            await backend.save_checkpoint(_make_checkpoint())
            loaded = await backend.load_checkpoint("flow1")
            assert loaded is not None
            assert loaded.completed_steps == ("kick",)

    async def test_claim_after_add_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "flow.db"
            backend = SqliteFlowWorkerBackend(path=db_path)

            conn = sqlite3.connect(str(db_path), isolation_level=None)
            try:
                conn.execute("ALTER TABLE flow_claims ADD COLUMN extra TEXT DEFAULT NULL")
            finally:
                conn.close()

            # Positional INSERT into flow_claims would raise on the extra column.
            ok = await backend.claim_batch("flow1", 0, "worker-a")
            assert ok is True


class TestSaveCheckpointBeginOutsideTry:
    """Finding 3: a failed BEGIN must surface the real lock error."""

    async def test_save_under_write_lock_surfaces_lock_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "flow.db"
            backend = SqliteFlowWorkerBackend(path=db_path)

            # Hold a RESERVED lock from a separate connection so the
            # backend's BEGIN IMMEDIATE cannot acquire it.
            holder = sqlite3.connect(str(db_path), isolation_level=None, timeout=5.0)
            holder.execute("PRAGMA journal_mode=WAL")
            holder.execute("BEGIN IMMEDIATE")
            # Force the lock to materialise with an actual write.
            holder.execute(
                "INSERT OR REPLACE INTO flow_checkpoints (flow_id, payload, updated_at) VALUES (?, ?, ?)",
                ("other", "{}", 0.0),
            )

            # Shorten the backend's busy-timeout so the test does not
            # block on the default 30s; run the sync path directly.
            def _short_timeout_connect() -> sqlite3.Connection:
                conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=0.2)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
                return conn

            backend._connect = _short_timeout_connect  # type: ignore[method-assign]

            try:
                with pytest.raises(sqlite3.OperationalError) as excinfo:
                    backend._save_checkpoint_sync(_make_checkpoint())
                # The genuine lock error must propagate, NOT the masking
                # "cannot rollback - no transaction is active".
                msg = str(excinfo.value)
                assert "no transaction is active" not in msg
                assert "locked" in msg
            finally:
                holder.execute("ROLLBACK")
                holder.close()

"""Unit + wiring tests for the shared snapshot-discard guards.

`reject_unsupported_snapshot_store` / `warn_discarded_snapshot`
(clients/base.py) are the single source of the no-snapshot-
persistence contract every backend's create() routes through.
RECEPTION of the reject path per-backend is covered by
test_unsupported_snapshot_store.py; this pins the helpers
themselves plus the warn-path wiring on the no-network backend.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from troopai.adk.exceptions.exceptions import UnsupportedSnapshotFeatureError
from troopai.adk.sandbox.clients.base import (
    reject_unsupported_snapshot_store,
    warn_discarded_snapshot,
)
from troopai.adk.types.sandbox.snapshot import LocalSnapshotSpec


class TestRejectUnsupportedSnapshotStore:
    def test_none_is_noop(self) -> None:
        reject_unsupported_snapshot_store(None, "k8s_pod")  # must not raise

    def test_non_none_raises_with_feature_and_backend(self) -> None:
        with pytest.raises(UnsupportedSnapshotFeatureError) as excinfo:
            reject_unsupported_snapshot_store(object(), "k8s_pod")
        assert excinfo.value.feature == "snapshot_store"
        assert excinfo.value.backend_id == "k8s_pod"


class TestWarnDiscardedSnapshot:
    def test_none_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        log = logging.getLogger("t.snapshot.none")
        with caplog.at_level(logging.WARNING, logger="t.snapshot.none"):
            warn_discarded_snapshot(None, "docker", log)
        assert len(caplog.records) == 0

    def test_non_none_warns_once_naming_backend(self, caplog: pytest.LogCaptureFixture) -> None:
        log = logging.getLogger("t.snapshot.warn")
        with caplog.at_level(logging.WARNING, logger="t.snapshot.warn"):
            warn_discarded_snapshot(object(), "docker", log)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "docker" in message
        assert "does not implement snapshot restore" in message


class TestLocalBackendWarnWiring:
    async def test_local_create_with_snapshot_logs_discard_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        # The local backend's create() does real (no-network) tempdir
        # setup AFTER the guards, so this exercises the warn wiring
        # end-to-end (snapshot is discarded, session still created).
        from troopai.adk.sandbox.clients.local.subprocess_client import (
            LocalSandboxClientOptions,
            LocalSubprocessSandboxClient,
        )

        client = LocalSubprocessSandboxClient(warn_banner=False)
        with caplog.at_level(logging.WARNING):
            session = await client.create(
                # Real spec; the local backend warns then discards it.
                snapshot=LocalSnapshotSpec(base_path=Path("/tmp/troopai-test-snap")),
                options=LocalSandboxClientOptions(),
            )
        try:
            messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
            assert any("does not implement snapshot restore" in m and "unix_local" in m for m in messages)
        finally:
            await session.aclose()

    async def test_snapshot_store_reject_precedes_snapshot_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        # Both set: reject(snapshot_store) MUST fire FIRST — it raises
        # before warn(snapshot) is reached, so NO discard warning is
        # emitted. Fences the reject→warn call order in create().
        from troopai.adk.sandbox.clients.local.subprocess_client import (
            LocalSandboxClientOptions,
            LocalSubprocessSandboxClient,
        )

        client = LocalSubprocessSandboxClient(warn_banner=False)
        with caplog.at_level(logging.WARNING), pytest.raises(UnsupportedSnapshotFeatureError):
            await client.create(
                snapshot=LocalSnapshotSpec(base_path=Path("/tmp/troopai-test-snap")),
                snapshot_store=object(),
                options=LocalSandboxClientOptions(),
            )
        assert not any("does not implement snapshot restore" in r.getMessage() for r in caplog.records)

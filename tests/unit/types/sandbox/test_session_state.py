"""Tests for ``troopai.adk.types.sandbox.session_state``."""

from __future__ import annotations

from troopai.adk.types.sandbox.session_state import SandboxSessionState
from troopai.adk.types.sandbox.snapshot import SnapshotRef


class TestSandboxSessionState:
    def test_minimal_construction(self) -> None:
        s = SandboxSessionState(backend_id="docker")
        assert s.backend_id == "docker"
        assert s.snapshot is None
        assert s.provider_payload == {}

    def test_with_snapshot(self) -> None:
        ref = SnapshotRef(snapshot_id="x", store_uri="s3://b")
        s = SandboxSessionState(backend_id="docker", snapshot=ref)
        assert s.snapshot == ref

    def test_provider_payload_preserved(self) -> None:
        s = SandboxSessionState(
            backend_id="e2b",
            provider_payload={"sandbox_id": "abc123", "region": "us-west"},
        )
        assert s.provider_payload["sandbox_id"] == "abc123"

    def test_round_trip_via_model_dump(self) -> None:
        ref = SnapshotRef(snapshot_id="x", store_uri="s3://b")
        s = SandboxSessionState(
            backend_id="k8s_pod",
            snapshot=ref,
            provider_payload={"pod_name": "sandbox-7d4"},
        )
        dumped = s.model_dump()
        restored = SandboxSessionState.model_validate(dumped)
        assert restored.backend_id == s.backend_id
        assert restored.snapshot == s.snapshot
        assert restored.provider_payload == s.provider_payload

"""Tests for ``SandboxSpanData`` in ``troopai.adk.types.tracing.span_data``."""

from __future__ import annotations

from troopai.adk.types.tracing import SandboxSpanData


class TestSandboxSpanData:
    def test_minimal_construction(self) -> None:
        s = SandboxSpanData(backend_id="docker")
        assert s.backend_id == "docker"
        assert s.command is None
        assert s.exit_code is None
        assert s.duration_ms is None
        assert s.type == "sandbox"

    def test_full_construction(self) -> None:
        s = SandboxSpanData(
            backend_id="k8s_pod",
            command="ls /tmp",
            exit_code=0,
            duration_ms=42,
            manifest_hash="sha256:abc",
            resource_usage={
                "cpu_ms": 10,
                "memory_peak_mb": 64,
                "bytes_read": 1024,
                "bytes_written": 256,
            },
            snapshot_id="snap-7d4",
        )
        assert s.backend_id == "k8s_pod"
        assert s.command == "ls /tmp"
        assert s.exit_code == 0

    def test_export_shape(self) -> None:
        s = SandboxSpanData(
            backend_id="unix_local",
            command="echo hi",
            exit_code=0,
            duration_ms=10,
        )
        exported = s.export()
        assert exported["type"] == "sandbox"
        assert exported["backend_id"] == "unix_local"
        assert exported["command"] == "echo hi"
        assert exported["exit_code"] == 0
        # None fields are preserved as-is so downstream exporters can
        # decide whether to drop them.
        assert exported["manifest_hash"] is None
        assert exported["snapshot_id"] is None

    def test_export_is_json_safe(self) -> None:
        import json

        s = SandboxSpanData(
            backend_id="docker",
            command="cat /etc/hostname",
            exit_code=0,
            resource_usage={"cpu_ms": 5},
        )
        # Round-trip via json.dumps to confirm every value is serializable.
        rendered = json.dumps(s.export())
        assert "docker" in rendered
        assert "sandbox" in rendered

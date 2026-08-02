"""Tests for ``troopai.adk.types.sandbox.snapshot``."""

from __future__ import annotations

from pathlib import Path

import pytest

from troopai.adk.types.sandbox.snapshot import (
    LocalSnapshotSpec,
    NoopSnapshotSpec,
    RemoteSnapshotSpec,
    SnapshotMetadata,
    SnapshotRef,
)


class TestSnapshotRef:
    def test_construction(self) -> None:
        r = SnapshotRef(snapshot_id="abc", store_uri="file:///tmp/snaps")
        assert r.snapshot_id == "abc"
        assert r.store_uri == "file:///tmp/snaps"


class TestSnapshotMetadata:
    def test_with_manifest_hash(self) -> None:
        ref = SnapshotRef(snapshot_id="x", store_uri="s3://b")
        m = SnapshotMetadata(
            ref=ref,
            created_at_iso="2025-01-01T00:00:00Z",
            size_bytes=1024,
            manifest_hash="sha256:deadbeef",
        )
        assert m.size_bytes == 1024
        assert m.manifest_hash == "sha256:deadbeef"


class TestLocalSnapshotSpec:
    def test_construction(self) -> None:
        s = LocalSnapshotSpec(base_path=Path("/tmp/snaps"))
        assert s.type == "local"
        assert s.base_path == Path("/tmp/snaps")

    def test_str_coerced_to_path(self) -> None:
        s = LocalSnapshotSpec(base_path="/tmp/snaps")  # type: ignore[arg-type]
        assert s.base_path == Path("/tmp/snaps")

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            LocalSnapshotSpec(base_path="")  # type: ignore[arg-type]


class TestRemoteSnapshotSpec:
    def test_s3_uri(self) -> None:
        s = RemoteSnapshotSpec(store_uri="s3://bucket/prefix")
        assert s.type == "remote"
        assert s.client_options == {}

    def test_uri_without_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="must include a scheme"):
            RemoteSnapshotSpec(store_uri="bucket/prefix")

    def test_empty_uri_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            RemoteSnapshotSpec(store_uri="")

    def test_client_options(self) -> None:
        s = RemoteSnapshotSpec(
            store_uri="s3://b/p",
            client_options={"region": "us-east-1"},
        )
        assert s.client_options == {"region": "us-east-1"}


class TestNoopSnapshotSpec:
    def test_construction(self) -> None:
        s = NoopSnapshotSpec()
        assert s.type == "noop"

"""Tests for S3 + GCS snapshot store stubs (P41 + P42)."""

from __future__ import annotations


class TestS3StoreImports:
    def test_class_importable(self) -> None:
        from troopai.adk.sandbox.snapshot.s3_store import S3SnapshotStore

        assert S3SnapshotStore is not None


class TestGCSStoreImports:
    def test_class_importable(self) -> None:
        from troopai.adk.sandbox.snapshot.gcs_store import GCSSnapshotStore

        assert GCSSnapshotStore is not None

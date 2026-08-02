"""Regression tests for GCSSnapshotStore.delete error handling.

Uses unittest.mock so the suite runs without google-cloud-storage
installed — the store accepts an injected client in the ctor. The real
``google.api_core.exceptions.NotFound`` is used so the production lazy
import resolves the same class the store checks against.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from google.api_core.exceptions import Forbidden, NotFound

from troopai.adk.exceptions.exceptions import SnapshotError
from troopai.adk.sandbox.snapshot.gcs_store import GCSSnapshotStore
from troopai.adk.types.sandbox.snapshot import SnapshotRef


def _mock_gcs_client() -> tuple[MagicMock, MagicMock]:
    blob = MagicMock()
    blob.delete = MagicMock()
    bucket = MagicMock()
    bucket.blob = MagicMock(return_value=blob)
    client = MagicMock()
    client.bucket = MagicMock(return_value=bucket)
    return client, blob


class TestGCSDeleteErrorHandling:
    async def test_delete_raises_on_real_failure(self) -> None:
        # A permission/network failure must NOT be swallowed: a retention
        # caller would otherwise assume the objects are gone when they remain.
        gcs, blob = _mock_gcs_client()
        blob.delete.side_effect = Forbidden("permission denied")
        store = GCSSnapshotStore(bucket="b", client=gcs)
        with pytest.raises(SnapshotError, match="delete failed"):
            await store.delete(SnapshotRef(snapshot_id="x", store_uri="gs://b/"))

    async def test_delete_treats_not_found_as_idempotent_success(self) -> None:
        # A missing object is the idempotent case — already gone — and must
        # return normally without raising.
        gcs, blob = _mock_gcs_client()
        blob.delete.side_effect = NotFound("already gone")
        store = GCSSnapshotStore(bucket="b", client=gcs)
        await store.delete(SnapshotRef(snapshot_id="x", store_uri="gs://b/"))
        # Both keys (object + metadata) were attempted.
        assert blob.delete.call_count == 2

    async def test_delete_raises_when_only_second_key_fails(self) -> None:
        # First key deletes fine; the metadata key hits a real failure. The
        # method must still attempt both and then surface the failure.
        gcs, blob = _mock_gcs_client()
        blob.delete.side_effect = [None, Forbidden("denied")]
        store = GCSSnapshotStore(bucket="b", client=gcs)
        with pytest.raises(SnapshotError, match="delete failed"):
            await store.delete(SnapshotRef(snapshot_id="x", store_uri="gs://b/"))
        assert blob.delete.call_count == 2

    async def test_delete_succeeds_when_no_errors(self) -> None:
        gcs, blob = _mock_gcs_client()
        store = GCSSnapshotStore(bucket="b", client=gcs)
        await store.delete(SnapshotRef(snapshot_id="x", store_uri="gs://b/"))
        assert blob.delete.call_count == 2

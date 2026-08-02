"""Tests for S3SnapshotStore + GCSSnapshotStore.

Uses unittest.mock so the suite runs without boto3 / google-cloud-storage
installed — the stores accept injected clients in the ctor.
"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import pytest

from troopai.adk.exceptions.exceptions import (
    SnapshotError,
    SnapshotPersistError,
    SnapshotRestoreError,
)
from troopai.adk.types.sandbox.snapshot import SnapshotRef


class _FakeClientError(Exception):
    """Stand-in for botocore ClientError carrying response.Error.Code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------


def _mock_s3_client() -> MagicMock:
    client = MagicMock()
    client.put_object = MagicMock()
    client.delete_object = MagicMock()
    head_obj = MagicMock(return_value={})
    client.head_object = head_obj
    body_mock = MagicMock()
    body_mock.read.return_value = b"hello-tar-payload"
    client.get_object = MagicMock(return_value={"Body": body_mock})
    paginator = MagicMock()
    paginator.paginate.return_value = iter([])
    client.get_paginator = MagicMock(return_value=paginator)
    return client


class TestS3Store:
    @pytest.mark.asyncio
    async def test_save_uploads_object_and_metadata(self) -> None:
        from troopai.adk.sandbox.snapshot.s3_store import S3SnapshotStore

        s3 = _mock_s3_client()
        store = S3SnapshotStore(bucket="b", prefix="snaps/", client=s3)
        meta = await store.save(
            snapshot_id="snap-1",
            data=BytesIO(b"payload"),
            manifest_hash="hash-1",
        )
        assert meta.ref.snapshot_id == "snap-1"
        assert meta.size_bytes == len(b"payload")
        assert meta.manifest_hash == "hash-1"
        # put_object called twice: object + metadata.
        assert s3.put_object.call_count == 2

    @pytest.mark.asyncio
    async def test_save_failure_raises_persist_error(self) -> None:
        from troopai.adk.sandbox.snapshot.s3_store import S3SnapshotStore

        s3 = _mock_s3_client()
        s3.put_object.side_effect = RuntimeError("S3 down")
        store = S3SnapshotStore(bucket="b", client=s3)
        with pytest.raises(SnapshotPersistError):
            await store.save(snapshot_id="snap-2", data=BytesIO(b"x"))

    @pytest.mark.asyncio
    async def test_load_returns_payload(self) -> None:
        from troopai.adk.sandbox.snapshot.s3_store import S3SnapshotStore

        s3 = _mock_s3_client()
        store = S3SnapshotStore(bucket="b", client=s3)
        ref = SnapshotRef(snapshot_id="snap-1", store_uri="s3://b/")
        stream = await store.load(ref)
        assert stream.read() == b"hello-tar-payload"

    @pytest.mark.asyncio
    async def test_load_failure_raises_restore_error(self) -> None:
        from troopai.adk.sandbox.snapshot.s3_store import S3SnapshotStore

        s3 = _mock_s3_client()
        s3.get_object.side_effect = RuntimeError("not found")
        store = S3SnapshotStore(bucket="b", client=s3)
        with pytest.raises(SnapshotRestoreError):
            await store.load(SnapshotRef(snapshot_id="x", store_uri="s3://b/"))

    @pytest.mark.asyncio
    async def test_exists_returns_true_when_head_ok(self) -> None:
        from troopai.adk.sandbox.snapshot.s3_store import S3SnapshotStore

        s3 = _mock_s3_client()
        store = S3SnapshotStore(bucket="b", client=s3)
        assert await store.exists(SnapshotRef(snapshot_id="snap-1", store_uri="s3://b/")) is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_on_404(self) -> None:
        from troopai.adk.sandbox.snapshot.s3_store import S3SnapshotStore

        s3 = _mock_s3_client()
        s3.head_object.side_effect = _FakeClientError("404")
        store = S3SnapshotStore(bucket="b", client=s3)
        assert await store.exists(SnapshotRef(snapshot_id="missing", store_uri="s3://b/")) is False

    @pytest.mark.asyncio
    async def test_exists_raises_on_non_404(self) -> None:
        # 403 / expired creds / 5xx must NOT masquerade as "snapshot absent".
        from troopai.adk.sandbox.snapshot.s3_store import S3SnapshotStore

        s3 = _mock_s3_client()
        s3.head_object.side_effect = _FakeClientError("403")
        store = S3SnapshotStore(bucket="b", client=s3)
        with pytest.raises(SnapshotError, match="exists failed"):
            await store.exists(SnapshotRef(snapshot_id="denied", store_uri="s3://b/"))

    @pytest.mark.asyncio
    async def test_save_metadata_failure_raises_and_cleans_orphan(self) -> None:
        # Object write succeeds, metadata write fails: must raise (not lie that
        # the snapshot is durable) AND delete the orphaned object.
        from troopai.adk.sandbox.snapshot.s3_store import S3SnapshotStore

        s3 = _mock_s3_client()
        # First put_object (object) succeeds; second (metadata) fails.
        s3.put_object.side_effect = [None, RuntimeError("metadata 500")]
        store = S3SnapshotStore(bucket="b", client=s3)
        with pytest.raises(SnapshotPersistError, match="metadata write failed"):
            await store.save(snapshot_id="snap-3", data=BytesIO(b"x"), manifest_hash="h")
        # The orphaned object blob was cleaned up.
        s3.delete_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_calls_remove_for_both_objects(self) -> None:
        from troopai.adk.sandbox.snapshot.s3_store import S3SnapshotStore

        s3 = _mock_s3_client()
        store = S3SnapshotStore(bucket="b", client=s3)
        await store.delete(SnapshotRef(snapshot_id="x", store_uri="s3://b/"))
        # delete_object called twice — object + metadata.
        assert s3.delete_object.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_treats_not_found_as_success(self) -> None:
        # A genuine "not found" is the idempotent case (already gone): delete()
        # must return cleanly, NOT raise, even though delete_object errored.
        from troopai.adk.sandbox.snapshot.s3_store import S3SnapshotStore

        s3 = _mock_s3_client()
        s3.delete_object.side_effect = _FakeClientError("404")
        store = S3SnapshotStore(bucket="b", client=s3)
        # No exception — both keys reported absent.
        await store.delete(SnapshotRef(snapshot_id="x", store_uri="s3://b/"))
        assert s3.delete_object.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_raises_on_access_denied(self) -> None:
        # 403 / expired creds / throttling / 5xx are real failures and MUST
        # surface — silently swallowing them would record the delete as done
        # while the (possibly sensitive) payload remains in the bucket.
        from troopai.adk.sandbox.snapshot.s3_store import S3SnapshotStore

        s3 = _mock_s3_client()
        s3.delete_object.side_effect = _FakeClientError("403")
        store = S3SnapshotStore(bucket="b", client=s3)
        with pytest.raises(SnapshotError, match="failed to remove"):
            await store.delete(SnapshotRef(snapshot_id="denied", store_uri="s3://b/"))

    @pytest.mark.asyncio
    async def test_delete_raises_on_non_client_error(self) -> None:
        # A non-ClientError exception (no response.Error.Code) is unambiguously
        # a failure, not a not-found — it must propagate as a SnapshotError.
        from troopai.adk.sandbox.snapshot.s3_store import S3SnapshotStore

        s3 = _mock_s3_client()
        s3.delete_object.side_effect = RuntimeError("network reset")
        store = S3SnapshotStore(bucket="b", client=s3)
        with pytest.raises(SnapshotError, match="failed to remove"):
            await store.delete(SnapshotRef(snapshot_id="x", store_uri="s3://b/"))

    @pytest.mark.asyncio
    async def test_sse_kms_forwarded(self) -> None:
        from troopai.adk.sandbox.snapshot.s3_store import S3SnapshotStore

        s3 = _mock_s3_client()
        store = S3SnapshotStore(
            bucket="b",
            server_side_encryption="aws:kms",
            kms_key_id="alias/k1",
            client=s3,
        )
        await store.save(snapshot_id="s", data=BytesIO(b""))
        call_kwargs = s3.put_object.call_args_list[0].kwargs
        assert call_kwargs["ServerSideEncryption"] == "aws:kms"
        assert call_kwargs["SSEKMSKeyId"] == "alias/k1"


# ---------------------------------------------------------------------------
# GCS
# ---------------------------------------------------------------------------


def _mock_gcs_client() -> tuple[MagicMock, MagicMock]:
    blob = MagicMock()
    blob.upload_from_string = MagicMock()
    blob.delete = MagicMock()
    blob.exists = MagicMock(return_value=True)
    blob.download_as_bytes = MagicMock(return_value=b"hello-tar-payload")
    bucket = MagicMock()
    bucket.blob = MagicMock(return_value=blob)
    client = MagicMock()
    client.bucket = MagicMock(return_value=bucket)
    client.list_blobs = MagicMock(return_value=[])
    return client, blob


class TestGCSStore:
    @pytest.mark.asyncio
    async def test_save_uploads_object_and_metadata(self) -> None:
        from troopai.adk.sandbox.snapshot.gcs_store import GCSSnapshotStore

        gcs, blob = _mock_gcs_client()
        store = GCSSnapshotStore(bucket="b", prefix="snaps/", client=gcs)
        meta = await store.save(
            snapshot_id="snap-1",
            data=BytesIO(b"payload"),
            manifest_hash="hash-1",
        )
        assert meta.ref.snapshot_id == "snap-1"
        assert meta.size_bytes == len(b"payload")
        # upload_from_string called twice: object + metadata.
        assert blob.upload_from_string.call_count == 2

    @pytest.mark.asyncio
    async def test_save_failure_raises_persist_error(self) -> None:
        from troopai.adk.sandbox.snapshot.gcs_store import GCSSnapshotStore

        gcs, blob = _mock_gcs_client()
        blob.upload_from_string.side_effect = RuntimeError("GCS down")
        store = GCSSnapshotStore(bucket="b", client=gcs)
        with pytest.raises(SnapshotPersistError):
            await store.save(snapshot_id="snap-2", data=BytesIO(b"x"))

    @pytest.mark.asyncio
    async def test_load_returns_payload(self) -> None:
        from troopai.adk.sandbox.snapshot.gcs_store import GCSSnapshotStore

        gcs, _blob = _mock_gcs_client()
        store = GCSSnapshotStore(bucket="b", client=gcs)
        stream = await store.load(SnapshotRef(snapshot_id="snap-1", store_uri="gs://b/"))
        assert stream.read() == b"hello-tar-payload"

    @pytest.mark.asyncio
    async def test_load_failure_raises_restore_error(self) -> None:
        from troopai.adk.sandbox.snapshot.gcs_store import GCSSnapshotStore

        gcs, blob = _mock_gcs_client()
        blob.download_as_bytes.side_effect = RuntimeError("not found")
        store = GCSSnapshotStore(bucket="b", client=gcs)
        with pytest.raises(SnapshotRestoreError):
            await store.load(SnapshotRef(snapshot_id="x", store_uri="gs://b/"))

    @pytest.mark.asyncio
    async def test_exists_returns_true(self) -> None:
        from troopai.adk.sandbox.snapshot.gcs_store import GCSSnapshotStore

        gcs, _blob = _mock_gcs_client()
        store = GCSSnapshotStore(bucket="b", client=gcs)
        assert await store.exists(SnapshotRef(snapshot_id="s", store_uri="gs://b/")) is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_when_blob_absent(self) -> None:
        # blob.exists() returning False is the not-found signal (no exception).
        from troopai.adk.sandbox.snapshot.gcs_store import GCSSnapshotStore

        gcs, blob = _mock_gcs_client()
        blob.exists = MagicMock(return_value=False)
        store = GCSSnapshotStore(bucket="b", client=gcs)
        assert await store.exists(SnapshotRef(snapshot_id="missing", store_uri="gs://b/")) is False

    @pytest.mark.asyncio
    async def test_exists_raises_on_error(self) -> None:
        # An exception from blob.exists() is a real failure (auth, network),
        # NOT a not-found — it must propagate, not silently report absent.
        from troopai.adk.sandbox.snapshot.gcs_store import GCSSnapshotStore

        gcs, blob = _mock_gcs_client()
        blob.exists.side_effect = RuntimeError("permission denied")
        store = GCSSnapshotStore(bucket="b", client=gcs)
        with pytest.raises(SnapshotError, match="exists failed"):
            await store.exists(SnapshotRef(snapshot_id="m", store_uri="gs://b/"))

    @pytest.mark.asyncio
    async def test_save_metadata_failure_raises_and_cleans_orphan(self) -> None:
        # Object write succeeds, metadata write fails: raise + delete the orphan.
        from troopai.adk.sandbox.snapshot.gcs_store import GCSSnapshotStore

        gcs, blob = _mock_gcs_client()
        blob.upload_from_string.side_effect = [None, RuntimeError("metadata 500")]
        store = GCSSnapshotStore(bucket="b", client=gcs)
        with pytest.raises(SnapshotPersistError, match="metadata write failed"):
            await store.save(snapshot_id="snap-3", data=BytesIO(b"x"), manifest_hash="h")
        blob.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_calls_remove_for_both_objects(self) -> None:
        from troopai.adk.sandbox.snapshot.gcs_store import GCSSnapshotStore

        gcs, blob = _mock_gcs_client()
        store = GCSSnapshotStore(bucket="b", client=gcs)
        await store.delete(SnapshotRef(snapshot_id="x", store_uri="gs://b/"))
        assert blob.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_cmek_key_forwarded(self) -> None:
        from troopai.adk.sandbox.snapshot.gcs_store import GCSSnapshotStore

        gcs, blob = _mock_gcs_client()
        store = GCSSnapshotStore(bucket="b", kms_key_name="projects/p/keys/k", client=gcs)
        await store.save(snapshot_id="s", data=BytesIO(b""))
        # The blob.kms_key_name attribute is set before upload.
        assert blob.kms_key_name == "projects/p/keys/k"

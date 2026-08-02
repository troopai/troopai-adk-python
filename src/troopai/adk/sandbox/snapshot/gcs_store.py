"""GCSSnapshotStore — Google Cloud Storage-backed snapshot store.

Uses google-cloud-storage. The SDK is synchronous; every call is
wrapped in ``asyncio.to_thread``. CMEK (customer-managed encryption
key) supported via ``kms_key_name``.

Stored objects live at ``gs://{bucket}/{prefix}{snapshot_id}.tar``
with a sibling ``gs://{bucket}/{prefix}{snapshot_id}.json`` metadata
object holding manifest hash + created_at + size.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from io import BytesIO, IOBase
from typing import Any, override

from troopai.adk.exceptions.exceptions import (
    SnapshotError,
    SnapshotPersistError,
    SnapshotRestoreError,
    UnsupportedSandboxClientError,
)
from troopai.adk.sandbox.snapshot.store import SnapshotStore
from troopai.adk.types.sandbox.snapshot import SnapshotMetadata, SnapshotRef

logger = logging.getLogger(__name__)

__all__ = ["GCSSnapshotStore"]


def _require_gcs() -> None:
    try:
        # google-cloud-storage ships as google.cloud.storage; live under
        # the [sandbox-gcs] extra. Bare import for the ImportError signal only.
        import google.cloud.storage  # type: ignore[import-untyped]  # noqa: F401  # pyright: ignore[reportMissingImports, reportUnusedImport]
    except ImportError as exc:
        raise UnsupportedSandboxClientError(
            "GCSSnapshotStore requires the optional 'google-cloud-storage' package: "
            "pip install 'troopai-adk-python[sandbox-gcs]'",
        ) from exc


def _is_not_found(exc: Exception) -> bool:
    """True iff ``exc`` is the SDK's missing-object signal (NotFound)."""
    try:
        from google.api_core.exceptions import (
            NotFound,  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
        )
    except ImportError:
        # The SDK is not importable, so a real NotFound cannot have been
        # raised; treat the exception as a genuine failure.
        return False
    return isinstance(exc, NotFound)


class GCSSnapshotStore(SnapshotStore):
    """Google Cloud Storage-backed snapshot store."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        project: str | None = None,
        kms_key_name: str | None = None,
        client: Any = None,
    ) -> None:
        """Construct the store.

        ``client`` injects a pre-built google.cloud.storage.Client
        (primary use case: tests). When ``None`` the store lazy-loads
        google-cloud-storage and builds its own client.
        """
        self._bucket_name = bucket
        self._prefix = prefix.rstrip("/") + "/" if len(prefix) > 0 and not prefix.endswith("/") else prefix
        self._project = project
        self._kms_key_name = kms_key_name
        if client is not None:
            self._client = client
            self._bucket = client.bucket(bucket)
            return
        _require_gcs()
        import google.cloud.storage as gcs_storage  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]

        self._client = gcs_storage.Client(project=project)
        self._bucket = self._client.bucket(bucket)

    def _object_key(self, snapshot_id: str) -> str:
        return f"{self._prefix}{snapshot_id}.tar"

    def _metadata_key(self, snapshot_id: str) -> str:
        return f"{self._prefix}{snapshot_id}.json"

    def _store_uri(self) -> str:
        return f"gs://{self._bucket_name}/{self._prefix}"

    def _upload_blob(self, key: str, payload: bytes) -> None:
        blob = self._bucket.blob(key)
        if self._kms_key_name is not None:
            blob.kms_key_name = self._kms_key_name
        blob.upload_from_string(payload)

    @override
    async def save(
        self,
        *,
        snapshot_id: str,
        data: IOBase,
        manifest_hash: str | None = None,
    ) -> SnapshotMetadata:
        payload = data.read()
        try:
            await asyncio.to_thread(
                self._upload_blob,
                self._object_key(snapshot_id),
                payload,
            )
        except Exception as exc:
            raise SnapshotPersistError(
                f"GCSSnapshotStore.save failed for {snapshot_id}: {exc}",
            ) from exc
        created_at = datetime.now(UTC).isoformat()
        metadata_blob = json.dumps(
            {
                "snapshot_id": snapshot_id,
                "created_at_iso": created_at,
                "size_bytes": len(payload),
                "manifest_hash": manifest_hash,
            },
        ).encode("utf-8")
        try:
            await asyncio.to_thread(
                self._upload_blob,
                self._metadata_key(snapshot_id),
                metadata_blob,
            )
        except Exception as exc:
            # The blob is written but its metadata sidecar (manifest_hash +
            # size) is not. Reporting success would be a durability lie:
            # list() would surface the orphan with no manifest_hash. Best-
            # effort delete the blob, then fail loudly.
            try:
                orphan = self._bucket.blob(self._object_key(snapshot_id))
                await asyncio.to_thread(orphan.delete)
            except Exception:
                logger.warning(
                    "GCSSnapshotStore.save: orphaned-blob cleanup failed for %s",
                    snapshot_id,
                    exc_info=True,
                )
            raise SnapshotPersistError(
                f"GCSSnapshotStore.save: metadata write failed for {snapshot_id}: {exc}",
            ) from exc
        ref = SnapshotRef(snapshot_id=snapshot_id, store_uri=self._store_uri())
        return SnapshotMetadata(
            ref=ref,
            created_at_iso=created_at,
            size_bytes=len(payload),
            manifest_hash=manifest_hash,
        )

    @override
    async def load(self, ref: SnapshotRef) -> IOBase:
        try:
            blob = self._bucket.blob(self._object_key(ref.snapshot_id))
            payload = await asyncio.to_thread(blob.download_as_bytes)
        except Exception as exc:
            raise SnapshotRestoreError(
                f"GCSSnapshotStore.load failed for {ref.snapshot_id}: {exc}",
            ) from exc
        return BytesIO(payload)

    @override
    async def delete(self, ref: SnapshotRef) -> None:
        errors: list[Exception] = []
        for key in (self._object_key(ref.snapshot_id), self._metadata_key(ref.snapshot_id)):
            try:
                blob = self._bucket.blob(key)
                await asyncio.to_thread(blob.delete)
            except Exception as exc:
                # A missing object is the idempotent case (already gone) and is
                # NOT an error; the SDK signals it with NotFound. Anything else
                # (permission, network, 5xx) is a real failure that MUST surface
                # rather than masquerade as a successful delete.
                if _is_not_found(exc):
                    continue
                logger.warning("GCSSnapshotStore.delete: %s failed: %s", key, exc)
                errors.append(exc)
        if len(errors) > 0:
            raise SnapshotError(
                f"GCSSnapshotStore.delete failed for {ref.snapshot_id}: {errors[0]}",
            ) from errors[0]

    @override
    async def list(self, prefix: str | None = None) -> list[SnapshotMetadata]:
        search_prefix = f"{self._prefix}{prefix}" if prefix is not None else self._prefix

        def _list_blobs() -> list[Any]:
            return list(self._client.list_blobs(self._bucket_name, prefix=search_prefix))

        try:
            blobs = await asyncio.to_thread(_list_blobs)
        except Exception as exc:
            raise SnapshotRestoreError(
                f"GCSSnapshotStore.list failed: {exc}",
            ) from exc
        results: list[SnapshotMetadata] = []
        for blob in blobs:
            name = getattr(blob, "name", "")
            if not name.endswith(".tar"):
                continue
            snapshot_id = name[len(self._prefix) :].removesuffix(".tar")
            metadata_blob = self._bucket.blob(self._metadata_key(snapshot_id))
            try:
                meta_raw = await asyncio.to_thread(metadata_blob.download_as_bytes)
                meta_dict = json.loads(meta_raw.decode("utf-8"))
                results.append(
                    SnapshotMetadata(
                        ref=SnapshotRef(snapshot_id=snapshot_id, store_uri=self._store_uri()),
                        created_at_iso=meta_dict.get("created_at_iso", ""),
                        size_bytes=int(meta_dict.get("size_bytes", 0)),
                        manifest_hash=meta_dict.get("manifest_hash"),
                    ),
                )
            except Exception as exc:
                logger.debug(
                    "GCSSnapshotStore.list: metadata for %s missing (%s); deriving from blob",
                    snapshot_id,
                    exc,
                )
                size_bytes = int(getattr(blob, "size", 0) or 0)
                time_created = getattr(blob, "time_created", None)
                iso = time_created.isoformat() if time_created else ""
                results.append(
                    SnapshotMetadata(
                        ref=SnapshotRef(snapshot_id=snapshot_id, store_uri=self._store_uri()),
                        created_at_iso=iso,
                        size_bytes=size_bytes,
                        manifest_hash=None,
                    ),
                )
        return results

    @override
    async def exists(self, ref: SnapshotRef) -> bool:
        try:
            blob = self._bucket.blob(self._object_key(ref.snapshot_id))
            return bool(await asyncio.to_thread(blob.exists))
        except Exception as exc:
            # blob.exists() already returns False for a missing object, so any
            # exception here is a real failure (auth, network, 5xx) -- not a
            # "not found". Raise rather than silently report the snapshot absent.
            raise SnapshotError(
                f"GCSSnapshotStore.exists failed for {ref.snapshot_id}: {exc}",
            ) from exc

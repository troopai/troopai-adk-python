"""S3SnapshotStore — AWS S3-backed snapshot store.

The boto3 SDK is synchronous; this store wraps every call in
``asyncio.to_thread``. Server-side encryption is enabled by default
(AES256); pass ``server_side_encryption="aws:kms"`` for KMS-backed
buckets and supply ``kms_key_id`` to pin the key.

Stored objects live at ``s3://{bucket}/{prefix}{snapshot_id}.tar``
with a sibling ``s3://{bucket}/{prefix}{snapshot_id}.json`` metadata
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

__all__ = ["S3SnapshotStore"]


def _read_stream_bytes(stream: Any) -> bytes:
    """Drain a boto3 streaming body to bytes (used inside to_thread)."""
    payload: bytes = stream.read()
    return payload


def _require_boto3() -> None:
    try:
        import boto3  # type: ignore[import-not-found]  # noqa: F401  # pyright: ignore[reportMissingImports, reportUnusedImport]
    except ImportError as exc:
        raise UnsupportedSandboxClientError(
            "S3SnapshotStore requires the optional 'boto3' package: pip install 'troopai-adk-python[sandbox-s3]'",
        ) from exc


class S3SnapshotStore(SnapshotStore):
    """S3-backed snapshot store."""

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
        server_side_encryption: str = "AES256",
        kms_key_id: str | None = None,
        client: Any = None,
    ) -> None:
        """Construct the store.

        ``client`` injects a pre-built boto3 S3 client (primary use
        case: tests). When ``None`` the store lazy-loads boto3 and
        builds its own client.
        """
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/" if len(prefix) > 0 and not prefix.endswith("/") else prefix
        self._region = region
        self._sse = server_side_encryption
        self._kms_key_id = kms_key_id
        if client is not None:
            self._s3 = client
            return
        _require_boto3()
        import boto3  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        self._s3 = boto3.client("s3", region_name=region)

    def _object_key(self, snapshot_id: str) -> str:
        return f"{self._prefix}{snapshot_id}.tar"

    def _metadata_key(self, snapshot_id: str) -> str:
        return f"{self._prefix}{snapshot_id}.json"

    def _store_uri(self) -> str:
        return f"s3://{self._bucket}/{self._prefix}"

    def _put_kwargs(self, *, key: str, body: bytes) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "Bucket": self._bucket,
            "Key": key,
            "Body": body,
            "ServerSideEncryption": self._sse,
        }
        if self._sse == "aws:kms" and self._kms_key_id is not None:
            kwargs["SSEKMSKeyId"] = self._kms_key_id
        return kwargs

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
                self._s3.put_object,
                **self._put_kwargs(key=self._object_key(snapshot_id), body=payload),
            )
        except Exception as exc:
            raise SnapshotPersistError(
                f"S3SnapshotStore.save failed for {snapshot_id}: {exc}",
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
                self._s3.put_object,
                **self._put_kwargs(key=self._metadata_key(snapshot_id), body=metadata_blob),
            )
        except Exception as exc:
            # The blob is written but its metadata sidecar (carrying
            # manifest_hash + size) is not. Reporting success would be a
            # durability lie: list() would surface the orphan with no
            # manifest_hash. Best-effort delete the blob, then fail loudly.
            try:
                await asyncio.to_thread(
                    self._s3.delete_object,
                    Bucket=self._bucket,
                    Key=self._object_key(snapshot_id),
                )
            except Exception:
                logger.warning(
                    "S3SnapshotStore.save: orphaned-blob cleanup failed for %s",
                    snapshot_id,
                    exc_info=True,
                )
            raise SnapshotPersistError(
                f"S3SnapshotStore.save: metadata write failed for {snapshot_id}: {exc}",
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
            response = await asyncio.to_thread(
                self._s3.get_object,
                Bucket=self._bucket,
                Key=self._object_key(ref.snapshot_id),
            )
        except Exception as exc:
            raise SnapshotRestoreError(
                f"S3SnapshotStore.load failed for {ref.snapshot_id}: {exc}",
            ) from exc
        body = response.get("Body")
        if body is None:
            raise SnapshotRestoreError(
                f"S3SnapshotStore.load: empty Body for {ref.snapshot_id}",
            )

        payload = await asyncio.to_thread(_read_stream_bytes, body)
        return BytesIO(payload)

    @override
    async def delete(self, ref: SnapshotRef) -> None:
        failures: list[tuple[str, Exception]] = []
        for key in (self._object_key(ref.snapshot_id), self._metadata_key(ref.snapshot_id)):
            try:
                await asyncio.to_thread(
                    self._s3.delete_object,
                    Bucket=self._bucket,
                    Key=key,
                )
            except Exception as exc:
                # Idempotent — a genuine "not found" is success. But 403 /
                # expired creds / throttling / 5xx MUST surface: hiding them
                # would record the delete as done while the payload remains.
                response = getattr(exc, "response", None)
                error_code = ""
                if isinstance(response, dict):
                    error_code = str(response.get("Error", {}).get("Code", ""))
                if error_code in ("404", "NoSuchKey", "NotFound"):
                    continue
                logger.warning("S3SnapshotStore.delete: %s failed: %s", key, exc)
                failures.append((key, exc))
        if len(failures) > 0:
            keys_str = ", ".join(k for k, _ in failures)
            raise SnapshotError(
                f"S3SnapshotStore.delete: failed to remove {keys_str} for {ref.snapshot_id}",
            ) from failures[0][1]

    @override
    async def list(self, prefix: str | None = None) -> list[SnapshotMetadata]:
        search_prefix = f"{self._prefix}{prefix}" if prefix is not None else self._prefix

        def _list_objects() -> list[dict[str, Any]]:
            paginator = self._s3.get_paginator("list_objects_v2")
            collected: list[dict[str, Any]] = []
            for page in paginator.paginate(Bucket=self._bucket, Prefix=search_prefix):
                contents = page.get("Contents", [])
                collected.extend(contents)
            return collected

        try:
            objects = await asyncio.to_thread(_list_objects)
        except Exception as exc:
            raise SnapshotRestoreError(
                f"S3SnapshotStore.list failed: {exc}",
            ) from exc
        results: list[SnapshotMetadata] = []
        for obj in objects:
            key = obj.get("Key", "")
            if not key.endswith(".tar"):
                continue
            snapshot_id = key[len(self._prefix) :].removesuffix(".tar")
            metadata_key = self._metadata_key(snapshot_id)
            try:
                meta_resp = await asyncio.to_thread(
                    self._s3.get_object,
                    Bucket=self._bucket,
                    Key=metadata_key,
                )
                meta_body = meta_resp.get("Body")
                if meta_body is None:
                    raise ValueError("metadata body missing")

                meta_raw = await asyncio.to_thread(_read_stream_bytes, meta_body)
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
                    "S3SnapshotStore.list: metadata for %s missing (%s); deriving from object",
                    snapshot_id,
                    exc,
                )
                last_modified = obj.get("LastModified")
                iso = last_modified.isoformat() if last_modified else ""
                results.append(
                    SnapshotMetadata(
                        ref=SnapshotRef(snapshot_id=snapshot_id, store_uri=self._store_uri()),
                        created_at_iso=iso,
                        size_bytes=int(obj.get("Size", 0)),
                        manifest_hash=None,
                    ),
                )
        return results

    @override
    async def exists(self, ref: SnapshotRef) -> bool:
        try:
            await asyncio.to_thread(
                self._s3.head_object,
                Bucket=self._bucket,
                Key=self._object_key(ref.snapshot_id),
            )
        except Exception as exc:
            # Only a genuine "not found" is a False; 403 / expired creds / 5xx
            # must NOT masquerade as "snapshot absent" (that would silently
            # treat a restorable snapshot as missing). boto3 ClientError carries
            # response.Error.Code; head_object on a missing key is "404".
            response = getattr(exc, "response", None)
            error_code = ""
            if isinstance(response, dict):
                error_code = str(response.get("Error", {}).get("Code", ""))
            if error_code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise SnapshotError(
                f"S3SnapshotStore.exists failed for {ref.snapshot_id}: {exc}",
            ) from exc
        return True

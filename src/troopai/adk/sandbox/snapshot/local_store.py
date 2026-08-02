"""``LocalSnapshotStore`` — filesystem-backed snapshot persistence.

Tar streams are written to ``{base_path}/{snapshot_id}.tar`` via an
atomic temp-file write. Listing scans the base directory for ``.tar``
files. Production deployments typically want
``S3SnapshotStore`` or ``GCSSnapshotStore``; the local
store covers dev iteration + forensic replay.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from io import IOBase
from pathlib import Path
from typing import override

from troopai.adk.exceptions.exceptions import (
    SnapshotError,
    SnapshotPersistError,
    SnapshotRestoreError,
)
from troopai.adk.sandbox.snapshot.store import SnapshotStore
from troopai.adk.types.sandbox.snapshot import SnapshotMetadata, SnapshotRef

logger = logging.getLogger(__name__)

__all__ = ["LocalSnapshotStore"]


_TAR_SUFFIX = ".tar"


class LocalSnapshotStore(SnapshotStore):
    """Filesystem-rooted snapshot store.

    Attributes:
        base_path: Directory snapshots are written into. Created on
            first ``save`` if missing.
    """

    def __init__(self, base_path: Path | str) -> None:
        self._base_path = Path(base_path)
        self._lock = asyncio.Lock()

    @property
    def base_path(self) -> Path:
        return self._base_path

    @property
    def store_uri(self) -> str:
        return f"file://{self._base_path.resolve()}"

    def _ref_path(self, snapshot_id: str) -> Path:
        # ``snapshot_id`` becomes a filename under ``base_path``; a value
        # carrying path separators or parent references (e.g.
        # ``"../../etc/cron.d/x"``) would escape the store root. Require a
        # single plain path component so a crafted id cannot traverse outside
        # ``base_path`` (fail closed).
        candidate = Path(f"{snapshot_id}{_TAR_SUFFIX}")
        if len(snapshot_id) == 0 or candidate.is_absolute() or len(candidate.parts) != 1:
            raise SnapshotError(
                f"LocalSnapshotStore: invalid snapshot_id {snapshot_id!r}; must be a single "
                "path component without separators or parent references",
            )
        return self._base_path / candidate

    @override
    async def save(
        self,
        *,
        snapshot_id: str,
        data: IOBase,
        manifest_hash: str | None = None,
    ) -> SnapshotMetadata:
        async with self._lock:
            self._base_path.mkdir(parents=True, exist_ok=True)
            destination = self._ref_path(snapshot_id)
            payload = data.read()
            # Atomic temp-file write: write to <base>/<id>.tmp.<pid>
            # then os.replace into <base>/<id>.tar so a crashed save
            # never leaves a half-written file at the canonical path.
            tmp_path = self._base_path / f"{snapshot_id}.tmp.{os.getpid()}"
            try:
                tmp_path.write_bytes(payload)
                os.replace(tmp_path, destination)
            except OSError as exc:
                # Best-effort temp-file cleanup. A cleanup OSError MUST NOT
                # mask the original write failure, so swallow + log it and let
                # the original exception surface via ``raise ... from exc``.
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "LocalSnapshotStore.save: temp-file cleanup failed for %s",
                        tmp_path,
                        exc_info=True,
                    )
                raise SnapshotPersistError(f"LocalSnapshotStore.save failed for {snapshot_id}: {exc}") from exc
            size = destination.stat().st_size
            return SnapshotMetadata(
                ref=SnapshotRef(snapshot_id=snapshot_id, store_uri=self.store_uri),
                created_at_iso=datetime.now(UTC).isoformat(),
                size_bytes=size,
                manifest_hash=manifest_hash,
            )

    @override
    async def load(self, ref: SnapshotRef) -> IOBase:
        path = self._ref_path(ref.snapshot_id)
        if not path.exists():
            raise SnapshotRestoreError(f"LocalSnapshotStore.load: snapshot {ref.snapshot_id!r} not found at {path}")
        try:
            return open(path, "rb")
        except OSError as exc:
            raise SnapshotRestoreError(f"LocalSnapshotStore.load failed for {ref.snapshot_id}: {exc}") from exc

    @override
    async def delete(self, ref: SnapshotRef) -> None:
        path = self._ref_path(ref.snapshot_id)
        # Idempotent — missing is fine.
        path.unlink(missing_ok=True)

    @override
    async def list(self, prefix: str | None = None) -> list[SnapshotMetadata]:
        if not self._base_path.exists():
            return []
        result: list[SnapshotMetadata] = []
        for entry in sorted(self._base_path.iterdir()):
            if not entry.is_file() or entry.suffix != _TAR_SUFFIX:
                continue
            snapshot_id = entry.stem
            if prefix is not None and not snapshot_id.startswith(prefix):
                continue
            stat = entry.stat()
            result.append(
                SnapshotMetadata(
                    ref=SnapshotRef(
                        snapshot_id=snapshot_id,
                        store_uri=self.store_uri,
                    ),
                    created_at_iso=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                    size_bytes=stat.st_size,
                    manifest_hash=None,
                ),
            )
        return result

    @override
    async def exists(self, ref: SnapshotRef) -> bool:
        return self._ref_path(ref.snapshot_id).exists()

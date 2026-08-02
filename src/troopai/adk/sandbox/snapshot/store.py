"""``SnapshotStore`` — abstract backend for workspace snapshot persistence.

Concrete implementations (``LocalSnapshotStore``, ``S3SnapshotStore``,
``GCSSnapshotStore``) extend this ABC. Current sandbox backends expose
these stores for direct use; run-level automatic restore/persist wiring
rejects ``SandboxRunConfig.snapshot_store`` instead of silently
discarding durability settings.
"""

from __future__ import annotations

import abc
from io import IOBase
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.types.sandbox.snapshot import SnapshotMetadata, SnapshotRef

__all__ = ["SnapshotStore"]


class SnapshotStore(abc.ABC):
    """Abstract backend for persisted sandbox workspace snapshots."""

    @abc.abstractmethod
    async def save(
        self,
        *,
        snapshot_id: str,
        data: IOBase,
        manifest_hash: str | None = None,
    ) -> SnapshotMetadata:
        """Persist ``data`` under ``snapshot_id``; return metadata."""

    @abc.abstractmethod
    async def load(self, ref: SnapshotRef) -> IOBase:
        """Open the snapshot referenced by ``ref`` for reading."""

    @abc.abstractmethod
    async def delete(self, ref: SnapshotRef) -> None:
        """Remove the snapshot referenced by ``ref``; idempotent."""

    @abc.abstractmethod
    async def list(self, prefix: str | None = None) -> list[SnapshotMetadata]:
        """List snapshots matching ``prefix`` (or all when ``None``)."""

    @abc.abstractmethod
    async def exists(self, ref: SnapshotRef) -> bool:
        """True iff the snapshot referenced by ``ref`` is present."""

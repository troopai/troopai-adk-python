"""Snapshot persistence backends for sandbox workspaces.

A ``SnapshotStore`` reads and writes serialized workspace snapshots
(typically tar streams produced by ``BaseSandboxSession.persist_workspace``).
``LocalSnapshotStore`` is filesystem-backed; ``S3SnapshotStore`` and
``GCSSnapshotStore`` extend the same ABC behind optional extras
(``[sandbox-s3]`` / ``[sandbox-gcs]``).

Current backends reject ``SandboxRunConfig.snapshot_store`` during
run-level session creation. Use stores directly until automatic
session-start restore and session-stop persistence are wired.
"""

from __future__ import annotations

from troopai.adk.sandbox.snapshot.gcs_store import GCSSnapshotStore
from troopai.adk.sandbox.snapshot.local_store import LocalSnapshotStore
from troopai.adk.sandbox.snapshot.s3_store import S3SnapshotStore
from troopai.adk.sandbox.snapshot.store import SnapshotStore

__all__ = [
    "GCSSnapshotStore",
    "LocalSnapshotStore",
    "S3SnapshotStore",
    "SnapshotStore",
]

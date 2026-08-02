"""Workspace snapshot types for sandbox persistence.

A ``SnapshotSpec`` describes WHERE saved workspace contents live;
the backend uses it both to restore on session start and to persist
on session stop. ``SnapshotRef`` and ``SnapshotMetadata`` are the
records produced and returned by a snapshot store.

Concrete subclasses (``LocalSnapshotSpec``, ``RemoteSnapshotSpec``,
``NoopSnapshotSpec``) live here; the runtime store implementations
that consume them live under ``troopai.adk.sandbox.snapshot``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "LocalSnapshotSpec",
    "NoopSnapshotSpec",
    "RemoteSnapshotSpec",
    "SnapshotMetadata",
    "SnapshotRef",
    "SnapshotSpec",
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SnapshotRef:
    """Address of a single snapshot in a store.

    Attributes:
        snapshot_id: Opaque identifier scoped to the store. Backends
            generate this when they persist; callers pass it back to
            restore.
        store_uri: URI describing the store the snapshot lives in
            (e.g. ``"file:///tmp/snaps"``, ``"s3://bucket/prefix"``,
            ``"gs://bucket/prefix"``). The store implementation
            interprets the scheme.
    """

    snapshot_id: str
    """Opaque identifier scoped to the store."""

    store_uri: str
    """URI describing the store the snapshot lives in."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class SnapshotMetadata:
    """Descriptive record about a persisted snapshot.

    Returned by ``SnapshotStore.list()`` and ``SnapshotStore.save()``
    so callers can audit + cull stored snapshots without having to
    re-download the payload.

    Attributes:
        ref: Address of the snapshot.
        created_at_iso: ISO-8601 timestamp the snapshot was persisted.
        size_bytes: Serialized snapshot size in bytes.
        manifest_hash: Optional content hash of the manifest used to
            create the originating session. Helps detect divergence
            between a saved snapshot and a fresh manifest.
    """

    ref: SnapshotRef
    """Address of the snapshot."""

    created_at_iso: str
    """ISO-8601 timestamp the snapshot was persisted."""

    size_bytes: int
    """Serialized snapshot size in bytes."""

    manifest_hash: str | None = None
    """Optional content hash of the manifest used to create the session."""


class SnapshotSpec(BaseModel):
    """Abstract spec describing where snapshots live.

    Concrete subclasses select a store implementation at runtime via
    the ``type`` discriminator.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    type: str
    """Discriminator string for subclass dispatch."""


class LocalSnapshotSpec(SnapshotSpec):
    """Snapshot store rooted at a local filesystem directory.

    Attributes:
        base_path: Directory containing the snapshot files. The
            directory is created on first persist if missing.
    """

    type: Literal["local"] = "local"
    """Discriminator. Always ``"local"``."""

    base_path: Path
    """Directory containing the snapshot files."""

    @field_validator("base_path", mode="before")
    @classmethod
    def _coerce_base_path(cls, value: object) -> Path:
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            if len(value) == 0:
                raise ValueError("LocalSnapshotSpec.base_path must be non-empty")
            return Path(value)
        raise TypeError(f"LocalSnapshotSpec.base_path must be str or Path, got {type(value).__name__}")


class RemoteSnapshotSpec(SnapshotSpec):
    """Snapshot store backed by a remote object store.

    The ``store_uri`` scheme picks the concrete client (``s3://``,
    ``gs://``, ``azure://``, …); the store implementation injects
    credentials from the host environment or an explicit secret-store
    reference.

    Attributes:
        store_uri: Store URI (e.g. ``"s3://bucket/prefix"``).
        client_options: Provider-specific extra options (region,
            endpoint URL, SSE config, …) the store implementation
            forwards to its SDK. Opaque to the framework.
    """

    type: Literal["remote"] = "remote"
    """Discriminator. Always ``"remote"``."""

    store_uri: str
    """Store URI (e.g. ``"s3://bucket/prefix"``)."""

    client_options: dict[str, str] = Field(default_factory=dict)
    """Provider-specific extra options forwarded to the store SDK."""

    @field_validator("store_uri")
    @classmethod
    def _validate_store_uri(cls, value: str) -> str:
        if len(value) == 0:
            raise ValueError("RemoteSnapshotSpec.store_uri must be non-empty")
        if "://" not in value:
            raise ValueError(f"RemoteSnapshotSpec.store_uri must include a scheme (e.g. 's3://'), got {value!r}")
        return value


class NoopSnapshotSpec(SnapshotSpec):
    """Fallback spec that persists nothing.

    Used by the runtime when no snapshot spec is provided AND no
    default local store can be set up. Calls to ``save`` return a
    placeholder ref; calls to ``load`` raise.
    """

    type: Literal["noop"] = "noop"
    """Discriminator. Always ``"noop"``."""

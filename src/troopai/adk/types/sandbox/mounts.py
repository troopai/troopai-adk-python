"""Remote storage mounts for sandbox workspaces.

Mounts attach external storage (S3, GCS, R2, Azure Blob, Box, S3 Files)
into the sandbox so the agent can read or write objects as if they
were local files. The mount *entry* describes WHAT to expose; the
mount *strategy* describes HOW the backend attaches it (rclone,
mountpoint, fuse, Docker volume drivers, etc.).

Snapshot + persistence flows skip mounted paths instead of copying
remote storage into saved workspace state.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal, TypedDict, override

from pydantic import BaseModel, ConfigDict, Field, field_validator

from troopai.adk.types.sandbox.entries import BaseEntry

__all__ = [
    "AzureBlobMount",
    "BoxMount",
    "DockerVolumeMountSpec",
    "DockerVolumeMountStrategy",
    "FuseMountPattern",
    "GCSMount",
    "InContainerMountSpec",
    "InContainerMountStrategy",
    "Mount",
    "MountPattern",
    "MountSpec",
    "MountStrategy",
    "MountpointMountPattern",
    "R2Mount",
    "RcloneMountPattern",
    "S3FilesMount",
    "S3FilesMountPattern",
    "S3Mount",
]


class MountPattern(BaseModel):
    """Abstract base for in-container mount patterns.

    A pattern names a specific tool + flags the backend should invoke
    once inside the container to bind the remote storage. Concrete
    patterns: ``RcloneMountPattern``, ``MountpointMountPattern``,
    ``FuseMountPattern``, ``S3FilesMountPattern``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    type: str
    """Discriminator string for pattern dispatch."""


class RcloneMountPattern(MountPattern):
    """rclone-driven mount inside the sandbox container.

    The widest-supported pattern; rclone speaks S3, GCS, R2, Azure
    Blob, Box, and more. ``mode="fuse"`` mounts via FUSE; ``"nfs"``
    runs rclone serve nfs and the sandbox mounts the resulting NFS
    export.

    Attributes:
        mode: Mount mode. ``"fuse"`` (most portable) or ``"nfs"``.
        remote_name: Name of the rclone remote to bind.
        extra_args: Additional CLI arguments passed to rclone.
        nfs_addr: Address rclone-nfs listens on (mode="nfs" only).
        nfs_mount_options: Mount-side options for the NFS client.
        config_file_path: Path to the rclone config file inside the sandbox.
    """

    # Pydantic discriminator: narrows the parent's `type: str` to a
    # concrete `Literal` subtype. mypy + pyright both accept this
    # (Pydantic's plugin treats the field as the union discriminator);
    # no type-checker suppression is needed.
    type: Literal["rclone"] = "rclone"
    """Discriminator. Always ``"rclone"``."""

    mode: Literal["fuse", "nfs"] = "fuse"
    """Mount mode."""

    remote_name: str
    """Name of the rclone remote to bind."""

    extra_args: tuple[str, ...] = ()
    """Additional CLI arguments passed to rclone."""

    nfs_addr: str | None = None
    """Address rclone-nfs listens on (mode="nfs" only)."""

    nfs_mount_options: tuple[str, ...] = ()
    """Mount-side options for the NFS client."""

    config_file_path: str | None = None
    """Path to the rclone config file inside the sandbox."""


class MountpointMountPattern(MountPattern):
    """AWS mount-s3 / mountpoint-style S3 mount.

    Use when the sandbox image ships ``mount-s3``. Requires S3 or
    S3-compatible storage.

    Attributes:
        bucket: S3 bucket name.
        prefix: Optional bucket prefix to scope the mount to.
        endpoint_url: Optional custom endpoint (S3-compatible providers).
        extra_args: Additional CLI arguments to mount-s3.
    """

    # Pydantic discriminator: narrows the parent's `type: str` to a
    # concrete `Literal` subtype. mypy + pyright both accept this
    # (Pydantic's plugin treats the field as the union discriminator);
    # no type-checker suppression is needed.
    type: Literal["mountpoint"] = "mountpoint"
    """Discriminator. Always ``"mountpoint"``."""

    bucket: str
    """S3 bucket name."""

    prefix: str | None = None
    """Optional bucket prefix scoping the mount."""

    endpoint_url: str | None = None
    """Optional custom endpoint (S3-compatible providers)."""

    extra_args: tuple[str, ...] = ()
    """Additional CLI arguments to mount-s3."""


class FuseMountPattern(MountPattern):
    """blobfuse2-driven FUSE mount.

    Targets Azure Blob containers when the sandbox image ships
    ``blobfuse2``.

    Attributes:
        container: Azure Blob container name.
        allow_other: Allow access by users other than the mounter.
        cache_path: Local cache directory inside the sandbox.
        cache_size_mb: Cache size cap in MiB.
        extra_args: Additional CLI arguments to blobfuse2.
    """

    # Pydantic discriminator: narrows the parent's `type: str` to a
    # concrete `Literal` subtype. mypy + pyright both accept this
    # (Pydantic's plugin treats the field as the union discriminator);
    # no type-checker suppression is needed.
    type: Literal["fuse"] = "fuse"
    """Discriminator. Always ``"fuse"``."""

    container: str
    """Azure Blob container name."""

    allow_other: bool = False
    """Allow access by users other than the mounter."""

    cache_path: str | None = None
    """Local cache directory inside the sandbox."""

    cache_size_mb: int | None = None
    """Cache size cap in MiB."""

    extra_args: tuple[str, ...] = ()
    """Additional CLI arguments to blobfuse2."""


class S3FilesMountPattern(MountPattern):
    """S3 Files (NFS-style) mount via ``mount.s3files``.

    Use when the sandbox image ships ``mount.s3files`` and the user
    operates an existing S3 Files mount target.

    Attributes:
        mount_target_id: ID of the S3 Files mount target.
        extra_args: Additional CLI arguments.
    """

    # Pydantic discriminator: narrows the parent's `type: str` to a
    # concrete `Literal` subtype. mypy + pyright both accept this
    # (Pydantic's plugin treats the field as the union discriminator);
    # no type-checker suppression is needed.
    type: Literal["s3files"] = "s3files"
    """Discriminator. Always ``"s3files"``."""

    mount_target_id: str
    """ID of the S3 Files mount target."""

    extra_args: tuple[str, ...] = ()
    """Additional CLI arguments."""


class MountStrategy(BaseModel):
    """Abstract base for mount strategies.

    A strategy answers: *given a Mount entry, how does THIS backend
    actually attach it?* Concrete strategies:
    ``InContainerMountStrategy`` (run a mount tool inside the
    sandbox container), ``DockerVolumeMountStrategy`` (let Docker
    attach a volume-driver-backed mount before the container starts).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    type: str
    """Discriminator string for strategy dispatch."""


class InContainerMountStrategy(MountStrategy):
    """Strategy: run a mount tool inside the sandbox container.

    Supports rclone, mount-s3, blobfuse2, and mount.s3files via the
    ``pattern`` field. Backends that can run privileged commands
    inside the container honor this strategy.

    Attributes:
        pattern: The concrete in-container mount pattern.
    """

    # Pydantic discriminator: narrows the parent's `type: str` to a
    # concrete `Literal` subtype. mypy + pyright both accept this
    # (Pydantic's plugin treats the field as the union discriminator);
    # no type-checker suppression is needed.
    type: Literal["in_container"] = "in_container"
    """Discriminator. Always ``"in_container"``."""

    pattern: RcloneMountPattern | MountpointMountPattern | FuseMountPattern | S3FilesMountPattern = Field(
        discriminator="type"
    )
    """The concrete in-container mount pattern."""


class DockerVolumeMountStrategy(MountStrategy):
    """Strategy: attach a volume-driver-backed mount via Docker.

    Docker-only. The volume driver (e.g., ``rclone``, ``s3fs``) is
    expected to be installed and registered on the Docker host.

    Attributes:
        driver: Volume driver name registered with Docker.
        driver_options: Driver-specific options forwarded to Docker.
    """

    # Pydantic discriminator: narrows the parent's `type: str` to a
    # concrete `Literal` subtype. mypy + pyright both accept this
    # (Pydantic's plugin treats the field as the union discriminator);
    # no type-checker suppression is needed.
    type: Literal["docker_volume"] = "docker_volume"
    """Discriminator. Always ``"docker_volume"``."""

    driver: str
    """Volume driver name registered with Docker."""

    driver_options: dict[str, str] = Field(default_factory=dict)
    """Driver-specific options forwarded to Docker."""


class Mount(BaseEntry):
    """Abstract base for remote-storage mount entries.

    Mounts are ephemeral workspace entries — snapshot flows skip them
    rather than copy remote storage into durable workspace state.

    Attributes:
        mount_path: Workspace-relative destination (None ⇒ entry key
            is the destination).
        read_only: When True (default), the sandbox cannot write back.
        mount_strategy: How this backend attaches the storage.
    """

    ephemeral: bool = True
    """Mounts are always ephemeral; never persisted into snapshots."""

    mount_path: str | None = None
    """Workspace-relative destination (None ⇒ entry key is destination)."""

    read_only: bool = True
    """When True, the sandbox cannot write back to mounted storage."""

    mount_strategy: InContainerMountStrategy | DockerVolumeMountStrategy = Field(discriminator="type")
    """How this backend attaches the storage."""

    @field_validator("mount_path")
    @classmethod
    def _validate_mount_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        raw = value
        if len(raw) == 0:
            raise ValueError("Mount.mount_path must be non-empty when set")
        # Windows drive prefixes ("C:foo", "C:\\foo") are POSIX-absolute only
        # when fully qualified; `PurePosixPath.is_absolute()` treats them as
        # relative. Reject them explicitly so a Windows-authored manifest
        # cannot smuggle a host-rooted destination into the sandbox.
        if len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha():
            raise ValueError(f"Mount.mount_path must be workspace-relative POSIX; got Windows drive path: {raw!r}")
        # Interpret as POSIX regardless of host OS: backslashes are separators
        # (a Windows-authored manifest), not literal filename characters.
        # Without this normalization, `Path` on a POSIX host treats
        # ``..\\..\\x`` as a single part, so the ``..`` traversal guard below
        # never fires and a backslash-separated traversal passes validation.
        p = PurePosixPath(raw.replace("\\", "/"))
        if p.is_absolute() or raw.startswith("/") or raw.startswith("\\"):
            raise ValueError(f"Mount.mount_path must be workspace-relative, got absolute path: {raw!r}")
        if ".." in p.parts:
            raise ValueError(f"Mount.mount_path must not contain '..': {raw!r}")
        return p.as_posix()

    @override
    def is_dir(self) -> bool:
        return True


class S3Mount(Mount):
    """AWS S3 mount.

    Attributes:
        bucket: Bucket name.
        prefix: Optional bucket prefix scoping the mount.
        region: AWS region.
        endpoint_url: Optional custom endpoint (S3-compatible providers).
    """

    # Pydantic discriminator: narrows the parent's `type: str` to a
    # concrete `Literal` subtype. mypy + pyright both accept this
    # (Pydantic's plugin treats the field as the union discriminator);
    # no type-checker suppression is needed.
    type: Literal["s3_mount"] = "s3_mount"
    """Discriminator. Always ``"s3_mount"``."""

    bucket: str
    """S3 bucket name."""

    prefix: str | None = None
    """Optional bucket prefix scoping the mount."""

    region: str | None = None
    """AWS region."""

    endpoint_url: str | None = None
    """Optional custom endpoint (S3-compatible providers)."""


class GCSMount(Mount):
    """Google Cloud Storage mount.

    Attributes:
        bucket: Bucket name.
        prefix: Optional bucket prefix scoping the mount.
        project: GCP project ID.
    """

    # Pydantic discriminator: narrows the parent's `type: str` to a
    # concrete `Literal` subtype. mypy + pyright both accept this
    # (Pydantic's plugin treats the field as the union discriminator);
    # no type-checker suppression is needed.
    type: Literal["gcs_mount"] = "gcs_mount"
    """Discriminator. Always ``"gcs_mount"``."""

    bucket: str
    """GCS bucket name."""

    prefix: str | None = None
    """Optional bucket prefix scoping the mount."""

    project: str | None = None
    """GCP project ID."""


class R2Mount(Mount):
    """Cloudflare R2 mount (S3-compatible).

    Attributes:
        bucket: R2 bucket name.
        account_id: Cloudflare account ID.
        prefix: Optional bucket prefix scoping the mount.
    """

    # Pydantic discriminator: narrows the parent's `type: str` to a
    # concrete `Literal` subtype. mypy + pyright both accept this
    # (Pydantic's plugin treats the field as the union discriminator);
    # no type-checker suppression is needed.
    type: Literal["r2_mount"] = "r2_mount"
    """Discriminator. Always ``"r2_mount"``."""

    bucket: str
    """R2 bucket name."""

    account_id: str
    """Cloudflare account ID."""

    prefix: str | None = None
    """Optional bucket prefix scoping the mount."""


class AzureBlobMount(Mount):
    """Azure Blob Storage mount.

    Attributes:
        account: Storage account name.
        container: Blob container name.
        prefix: Optional blob prefix scoping the mount.
    """

    # Pydantic discriminator: narrows the parent's `type: str` to a
    # concrete `Literal` subtype. mypy + pyright both accept this
    # (Pydantic's plugin treats the field as the union discriminator);
    # no type-checker suppression is needed.
    type: Literal["azure_blob_mount"] = "azure_blob_mount"
    """Discriminator. Always ``"azure_blob_mount"``."""

    account: str
    """Storage account name."""

    container: str
    """Blob container name."""

    prefix: str | None = None
    """Optional blob prefix scoping the mount."""


class BoxMount(Mount):
    """Box.com mount.

    Attributes:
        folder_id: Box folder ID to attach.
    """

    # Pydantic discriminator: narrows the parent's `type: str` to a
    # concrete `Literal` subtype. mypy + pyright both accept this
    # (Pydantic's plugin treats the field as the union discriminator);
    # no type-checker suppression is needed.
    type: Literal["box_mount"] = "box_mount"
    """Discriminator. Always ``"box_mount"``."""

    folder_id: str
    """Box folder ID to attach."""


class S3FilesMount(Mount):
    """S3 Files (NFS-style) mount.

    Attributes:
        mount_target_id: S3 Files mount target ID.
        region: AWS region.
    """

    # Pydantic discriminator: narrows the parent's `type: str` to a
    # concrete `Literal` subtype. mypy + pyright both accept this
    # (Pydantic's plugin treats the field as the union discriminator);
    # no type-checker suppression is needed.
    type: Literal["s3_files_mount"] = "s3_files_mount"
    """Discriminator. Always ``"s3_files_mount"``."""

    mount_target_id: str
    """S3 Files mount target ID."""

    region: str | None = None
    """AWS region."""


# --- Backend mount-spec wire types ------------------------------------
#
# Provider-agnostic dicts emitted by the policy mount-translation
# helpers and consumed by backend clients. They are plain dicts at
# runtime (zero conversion on the create path); the backend client
# discriminates on ``strategy`` to choose its native attach call
# (e.g. the Docker client maps these to ``docker.types.Mount`` /
# ``DriverConfig``). They carry NO provider-SDK imports — the SDK
# boundary stays inside ``sandbox/clients/<backend>/``.


class InContainerMountSpec(TypedDict):
    """A neutral spec for an ``InContainerMountStrategy`` attachment.

    The mount tool (rclone / mount-s3 / blobfuse2 / mount.s3files)
    runs INSIDE the sandbox container; the backend only has to
    provide a writable, FUSE-capable mountpoint at ``target``.
    """

    type: Literal["bind"]
    """Framework classification only — a plain writable mountpoint
    the in-container tool mounts over.

    NOT forwarded to docker-py. The Docker backend materializes this
    strategy as an anonymous ``type="volume"`` mount (``source=None``)
    irrespective of this value, because an in-container FUSE
    mountpoint needs a fresh writable directory, not a host bind. A
    consumer MUST dispatch on ``strategy`` and MUST NOT pass this
    field through to ``docker.types.Mount(type=...)``."""

    target: str
    """Absolute in-container destination path."""

    read_only: bool
    """Whether the mounted storage is read-only to the sandbox."""

    strategy: Literal["in_container"]
    """Strategy discriminator."""

    pattern_type: str
    """The ``MountPattern`` discriminator (rclone/mountpoint/fuse/s3files)."""

    pattern_config: dict[str, str | int | bool | list[str]]
    """Pattern-specific runtime config the in-container tool consumes.

    Heterogeneous because the four ``MountPattern`` subclasses carry
    differently-typed fields (e.g. ``FuseMountPattern.allow_other``
    is ``bool`` / ``cache_size_mb`` is ``int``;
    ``RcloneMountPattern.nfs_mount_options`` is a string sequence).
    The value union is bounded to JSON-serializable leaf types — NOT
    ``Any`` — so the spec round-trips losslessly.

    Translation contract: the mount translator OMITS keys whose
    source pattern field is ``None`` (so a missing key faithfully
    means "unspecified — let the in-container tool apply its
    default"; a present key is always a real value), and emits
    ``tuple`` fields as ``list``.

    Consumption contract: a consumer MUST narrow on ``pattern_type``
    before formatting any value. ``bool`` is an ``int`` subclass and
    ``str(True)`` is ``"True"`` (title-case), so blindly stringifying
    or numifying a value of unknown pattern is a silent-coercion
    hazard — rclone / mount-s3 / blobfuse2 expect lowercase
    ``true``/``false`` or flag presence, never Python's ``repr``.
    Read optional keys via ``dict.get`` (a missing key is the
    documented "unspecified" signal, not an error).
    """


class DockerVolumeMountSpec(TypedDict):
    """A neutral spec for a ``DockerVolumeMountStrategy`` attachment.

    Docker attaches a volume-driver-backed volume before the
    container starts; the backend client creates the named volume
    with ``driver`` + ``driver_options`` then binds it at ``target``.
    """

    type: Literal["volume"]
    """Framework classification only — a named driver-backed volume.

    NOT forwarded to docker-py: the Docker backend hardcodes
    ``type="volume"`` when materializing this strategy and dispatches
    on ``strategy``, never on this field. The value happens to equal
    the materialized kind, but a consumer MUST NOT rely on that —
    treat it as a documentation label, not a wire parameter."""

    target: str
    """Absolute in-container destination path."""

    read_only: bool
    """Whether the mounted storage is read-only to the sandbox."""

    strategy: Literal["docker_volume"]
    """Strategy discriminator."""

    driver: str
    """Volume driver name registered with the Docker host."""

    driver_options: dict[str, str]
    """Driver-specific options forwarded to the Docker volume driver."""


MountSpec = InContainerMountSpec | DockerVolumeMountSpec
"""Discriminated (on ``strategy``) neutral backend mount-spec."""

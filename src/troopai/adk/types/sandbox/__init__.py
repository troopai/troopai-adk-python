"""Sandbox Layer-1 data types.

Pure data classes (Pydantic ``BaseModel`` or ``@dataclass``) with no
provider imports. Re-exports here form the public surface of
``troopai.adk.types.sandbox``.
"""

from __future__ import annotations

from troopai.adk.types.sandbox.cost import (
    SandboxBackendCapabilities,
    SandboxBillingRecord,
    SandboxCostDescriptor,
    SandboxRequirements,
)
from troopai.adk.types.sandbox.entries import (
    BaseEntry,
    Dir,
    File,
    GitRepo,
    LocalDir,
    LocalFile,
    MaterializedFile,
)
from troopai.adk.types.sandbox.exec_result import (
    ExecResult,
    ExposedPortEndpoint,
    PtyHandle,
)
from troopai.adk.types.sandbox.iac import IaCBundle
from troopai.adk.types.sandbox.manifest import (
    EnvEntry,
    Environment,
    Manifest,
    StrEnvValue,
)
from troopai.adk.types.sandbox.mounts import (
    AzureBlobMount,
    BoxMount,
    DockerVolumeMountStrategy,
    FuseMountPattern,
    GCSMount,
    InContainerMountStrategy,
    Mount,
    MountPattern,
    MountpointMountPattern,
    MountStrategy,
    R2Mount,
    RcloneMountPattern,
    S3FilesMount,
    S3FilesMountPattern,
    S3Mount,
)
from troopai.adk.types.sandbox.network import (
    NetworkPolicy,
    PortForwardRule,
)
from troopai.adk.types.sandbox.permissions import (
    FileMode,
    Group,
    Permissions,
    User,
)
from troopai.adk.types.sandbox.resource_limits import SandboxResourceLimits
from troopai.adk.types.sandbox.session_state import SandboxSessionState
from troopai.adk.types.sandbox.snapshot import (
    LocalSnapshotSpec,
    NoopSnapshotSpec,
    RemoteSnapshotSpec,
    SnapshotMetadata,
    SnapshotRef,
    SnapshotSpec,
)
from troopai.adk.types.sandbox.usage import SandboxSingleExecUsage, SandboxUsage
from troopai.adk.types.sandbox.workspace_paths import (
    SandboxPathGrant,
    WorkspacePathPolicy,
)

__all__ = [
    "AzureBlobMount",
    "BaseEntry",
    "BoxMount",
    "Dir",
    "DockerVolumeMountStrategy",
    "EnvEntry",
    "Environment",
    "ExecResult",
    "ExposedPortEndpoint",
    "File",
    "FileMode",
    "FuseMountPattern",
    "GCSMount",
    "GitRepo",
    "Group",
    "IaCBundle",
    "InContainerMountStrategy",
    "LocalDir",
    "LocalFile",
    "LocalSnapshotSpec",
    "Manifest",
    "MaterializedFile",
    "Mount",
    "MountPattern",
    "MountStrategy",
    "MountpointMountPattern",
    "NetworkPolicy",
    "NoopSnapshotSpec",
    "Permissions",
    "PortForwardRule",
    "PtyHandle",
    "R2Mount",
    "RcloneMountPattern",
    "RemoteSnapshotSpec",
    "S3FilesMount",
    "S3FilesMountPattern",
    "S3Mount",
    "SandboxBackendCapabilities",
    "SandboxBillingRecord",
    "SandboxCostDescriptor",
    "SandboxPathGrant",
    "SandboxRequirements",
    "SandboxResourceLimits",
    "SandboxSessionState",
    "SandboxSingleExecUsage",
    "SandboxUsage",
    "SnapshotMetadata",
    "SnapshotRef",
    "SnapshotSpec",
    "StrEnvValue",
    "User",
    "WorkspacePathPolicy",
]

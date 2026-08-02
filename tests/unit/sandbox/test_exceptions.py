"""Tests for the SandboxError hierarchy in ``troopai.adk.exceptions``."""

from __future__ import annotations

import pytest

from troopai.adk.exceptions import (
    ApplyPatchError,
    ExecFailureError,
    ExecNonZeroError,
    ExecTimeoutError,
    ExecTransportError,
    ExposedPortUnavailableError,
    GitArtifactError,
    InvalidCompressionSchemeError,
    InvalidManifestPathError,
    LocalArtifactError,
    MountArtifactError,
    PtySessionNotFoundError,
    SandboxArtifactError,
    SandboxCommandRejected,
    SandboxConcurrencyError,
    SandboxConfigurationError,
    SandboxError,
    SandboxNetworkPolicyViolation,
    SandboxResourceLimitExceeded,
    SandboxRuntimeError,
    SandboxSelectionError,
    SandboxStartFailed,
    SandboxStopFailed,
    SkillsConfigError,
    SnapshotError,
    SnapshotNotRestorableError,
    SnapshotPersistError,
    SnapshotRestoreError,
    TroopAIError,
    UnsupportedSandboxClientError,
    WorkspaceArchiveReadError,
    WorkspaceArchiveWriteError,
    WorkspaceIOError,
    WorkspaceReadNotFoundError,
)


class TestHierarchy:
    def test_sandbox_error_is_troopai_error(self) -> None:
        assert issubclass(SandboxError, TroopAIError)

    @pytest.mark.parametrize(
        "subclass",
        [
            SandboxConfigurationError,
            SandboxRuntimeError,
            SandboxArtifactError,
        ],
    )
    def test_top_branches_inherit_sandbox_error(self, subclass: type) -> None:
        assert issubclass(subclass, SandboxError)

    @pytest.mark.parametrize(
        "subclass",
        [
            InvalidManifestPathError,
            InvalidCompressionSchemeError,
            ApplyPatchError,
            SkillsConfigError,
            UnsupportedSandboxClientError,
        ],
    )
    def test_config_subclasses(self, subclass: type) -> None:
        assert issubclass(subclass, SandboxConfigurationError)

    @pytest.mark.parametrize(
        "subclass",
        [
            SandboxStartFailed,
            SandboxStopFailed,
            ExecFailureError,
            ExposedPortUnavailableError,
            WorkspaceIOError,
            PtySessionNotFoundError,
            SandboxConcurrencyError,
            SandboxNetworkPolicyViolation,
            SandboxCommandRejected,
            SandboxResourceLimitExceeded,
        ],
    )
    def test_runtime_subclasses(self, subclass: type) -> None:
        assert issubclass(subclass, SandboxRuntimeError)

    @pytest.mark.parametrize(
        "subclass",
        [
            ExecNonZeroError,
            ExecTimeoutError,
            ExecTransportError,
        ],
    )
    def test_exec_failure_subclasses(self, subclass: type) -> None:
        assert issubclass(subclass, ExecFailureError)

    @pytest.mark.parametrize(
        "subclass",
        [
            WorkspaceReadNotFoundError,
            WorkspaceArchiveReadError,
            WorkspaceArchiveWriteError,
        ],
    )
    def test_workspace_io_subclasses(self, subclass: type) -> None:
        assert issubclass(subclass, WorkspaceIOError)

    @pytest.mark.parametrize(
        "subclass",
        [
            LocalArtifactError,
            GitArtifactError,
            MountArtifactError,
            SnapshotError,
        ],
    )
    def test_artifact_subclasses(self, subclass: type) -> None:
        assert issubclass(subclass, SandboxArtifactError)

    @pytest.mark.parametrize(
        "subclass",
        [
            SnapshotPersistError,
            SnapshotRestoreError,
            SnapshotNotRestorableError,
        ],
    )
    def test_snapshot_subclasses(self, subclass: type) -> None:
        assert issubclass(subclass, SnapshotError)


class TestRichConstructors:
    def test_sandbox_start_failed_captures_attrs(self) -> None:
        err = SandboxStartFailed(
            backend_id="docker",
            reason="image not found",
            details={"image": "missing:latest"},
        )
        assert err.backend_id == "docker"
        assert err.reason == "image not found"
        assert err.details == {"image": "missing:latest"}
        assert "docker" in str(err)

    def test_sandbox_command_rejected_captures_attrs(self) -> None:
        err = SandboxCommandRejected(command="rm -rf /", reason="not allowlisted")
        assert err.command == "rm -rf /"
        assert err.reason == "not allowlisted"
        assert "not allowlisted" in str(err)

    def test_sandbox_resource_limit_captures_attrs(self) -> None:
        err = SandboxResourceLimitExceeded(
            resource="memory_mb",
            limit=512,
            observed=768,
        )
        assert err.resource == "memory_mb"
        assert err.limit == 512
        assert err.observed == 768
        assert "memory_mb" in str(err)

    def test_sandbox_start_failed_default_details(self) -> None:
        err = SandboxStartFailed(backend_id="k8s_pod", reason="connection refused")
        assert err.details == {}


class TestRaisability:
    def test_can_raise_and_catch_via_branch(self) -> None:
        with pytest.raises(SandboxRuntimeError):
            raise ExecTimeoutError("timed out")

    def test_can_raise_and_catch_via_root(self) -> None:
        with pytest.raises(SandboxError):
            raise SnapshotPersistError("disk full")

    def test_can_raise_and_catch_via_troopai_error(self) -> None:
        with pytest.raises(TroopAIError):
            raise SandboxNetworkPolicyViolation("egress to evil.com")


class TestSandboxSelectionError:
    def test_is_configuration_error(self) -> None:
        err = SandboxSelectionError("no candidate")
        assert isinstance(err, SandboxConfigurationError)
        assert str(err) == "no candidate"
        assert str(SandboxSelectionError()) != ""

    def test_exported_from_package(self) -> None:
        from troopai.adk.exceptions.exceptions import (
            SandboxSelectionError as Canonical,
        )

        assert SandboxSelectionError is Canonical

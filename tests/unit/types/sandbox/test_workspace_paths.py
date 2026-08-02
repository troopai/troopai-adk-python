"""Tests for ``troopai.adk.types.sandbox.workspace_paths``."""

from __future__ import annotations

import pytest

from troopai.adk.types.sandbox.workspace_paths import (
    SandboxPathGrant,
    WorkspacePathPolicy,
)


class TestSandboxPathGrant:
    def test_absolute_path_allowed(self) -> None:
        g = SandboxPathGrant(path="/tmp")
        assert g.path == "/tmp"
        assert g.read_only is False

    def test_rejects_empty_path(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            SandboxPathGrant(path="")

    def test_rejects_relative_path(self) -> None:
        with pytest.raises(ValueError, match="must be absolute"):
            SandboxPathGrant(path="tmp")

    def test_rejects_filesystem_root(self) -> None:
        with pytest.raises(ValueError, match="may not be a filesystem root"):
            SandboxPathGrant(path="/")

    def test_read_only_grant(self) -> None:
        g = SandboxPathGrant(path="/opt/toolchain", read_only=True)
        assert g.read_only is True


class TestWorkspacePathPolicy:
    def test_under_workspace(self) -> None:
        policy = WorkspacePathPolicy(root="/workspace")
        assert policy.is_under_workspace("/workspace/sub/file.txt") is True
        assert policy.is_under_workspace("/tmp/file.txt") is False

    def test_matching_grant_picks_most_specific(self) -> None:
        policy = WorkspacePathPolicy(
            root="/workspace",
            grants=(
                SandboxPathGrant(path="/tmp"),
                SandboxPathGrant(path="/tmp/sub"),
            ),
        )
        match = policy.matching_grant("/tmp/sub/file.txt")
        assert match is not None
        assert match.path == "/tmp/sub"

    def test_matching_grant_returns_none_when_no_overlap(self) -> None:
        policy = WorkspacePathPolicy(
            root="/workspace",
            grants=(SandboxPathGrant(path="/tmp"),),
        )
        assert policy.matching_grant("/etc/hosts") is None

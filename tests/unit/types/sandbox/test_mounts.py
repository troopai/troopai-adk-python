"""Tests for ``troopai.adk.types.sandbox.mounts``."""

from __future__ import annotations

import pytest

from troopai.adk.types.sandbox.mounts import (
    DockerVolumeMountStrategy,
    FuseMountPattern,
    GCSMount,
    InContainerMountStrategy,
    MountpointMountPattern,
    RcloneMountPattern,
    S3Mount,
)


def _rclone_strategy(remote: str = "myremote") -> InContainerMountStrategy:
    return InContainerMountStrategy(
        pattern=RcloneMountPattern(remote_name=remote),
    )


class TestRcloneMountPattern:
    def test_default_mode_fuse(self) -> None:
        p = RcloneMountPattern(remote_name="r")
        assert p.mode == "fuse"
        assert p.type == "rclone"

    def test_nfs_mode_allowed(self) -> None:
        p = RcloneMountPattern(remote_name="r", mode="nfs")
        assert p.mode == "nfs"


class TestInContainerStrategyDiscriminator:
    def test_rclone_pattern_dispatched(self) -> None:
        s = InContainerMountStrategy.model_validate(
            {
                "type": "in_container",
                "pattern": {"type": "rclone", "remote_name": "r"},
            }
        )
        assert isinstance(s.pattern, RcloneMountPattern)

    def test_mountpoint_pattern_dispatched(self) -> None:
        s = InContainerMountStrategy.model_validate(
            {
                "type": "in_container",
                "pattern": {"type": "mountpoint", "bucket": "mybucket"},
            }
        )
        assert isinstance(s.pattern, MountpointMountPattern)
        assert s.pattern.bucket == "mybucket"

    def test_fuse_pattern_dispatched(self) -> None:
        s = InContainerMountStrategy.model_validate(
            {
                "type": "in_container",
                "pattern": {"type": "fuse", "container": "mycontainer"},
            }
        )
        assert isinstance(s.pattern, FuseMountPattern)


class TestDockerVolumeStrategy:
    def test_default_options_empty(self) -> None:
        s = DockerVolumeMountStrategy(driver="rclone")
        assert s.driver_options == {}
        assert s.type == "docker_volume"


class TestS3Mount:
    def test_construction(self) -> None:
        m = S3Mount(
            bucket="my-bucket",
            region="us-east-1",
            mount_strategy=_rclone_strategy("s3remote"),
        )
        assert m.bucket == "my-bucket"
        assert m.ephemeral is True  # mounts are always ephemeral
        assert m.read_only is True  # default
        assert m.is_dir() is True

    def test_mount_path_rejects_dot_dot(self) -> None:
        with pytest.raises(ValueError, match="must not contain '..'"):
            S3Mount(
                bucket="b",
                mount_path="sub/../escape",
                mount_strategy=_rclone_strategy(),
            )

    def test_mount_path_rejects_backslash_dot_dot_traversal(self) -> None:
        # A Windows-authored manifest uses backslashes as separators. On a
        # POSIX host `Path("..\\..\\etc")` collapses to a single opaque part,
        # so a naive `".." in parts` check never fires; the validator must
        # normalize backslashes to forward slashes before the traversal guard.
        with pytest.raises(ValueError, match="must not contain '..'"):
            S3Mount(
                bucket="b",
                mount_path="..\\..\\etc\\passwd",
                mount_strategy=_rclone_strategy(),
            )

    def test_mount_path_rejects_windows_drive_prefix(self) -> None:
        with pytest.raises(ValueError, match="Windows drive path"):
            S3Mount(
                bucket="b",
                mount_path="C:\\Windows\\System32",
                mount_strategy=_rclone_strategy(),
            )

    def test_mount_path_normalizes_backslash_separators(self) -> None:
        # A benign Windows-authored relative path is normalized to POSIX so
        # downstream consumers (target resolution, manifest rendering) see one
        # consistent separator flavor.
        m = S3Mount(
            bucket="b",
            mount_path="sub\\dir\\data",
            mount_strategy=_rclone_strategy(),
        )
        assert m.mount_path == "sub/dir/data"


class TestGCSMount:
    def test_construction(self) -> None:
        m = GCSMount(
            bucket="my-gcs-bucket",
            project="my-project",
            mount_strategy=_rclone_strategy("gcsremote"),
        )
        assert m.bucket == "my-gcs-bucket"
        assert m.project == "my-project"

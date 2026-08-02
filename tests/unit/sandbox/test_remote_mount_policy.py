"""Tests for ``troopai.adk.sandbox.remote_mount_policy``."""

from __future__ import annotations

from troopai.adk.sandbox.remote_mount_policy import (
    REMOTE_MOUNT_POLICY_TEMPLATE,
    build_remote_mount_policy_instructions,
    get_remote_mounts,
)
from troopai.adk.types.sandbox.entries import File
from troopai.adk.types.sandbox.manifest import Manifest
from troopai.adk.types.sandbox.mounts import (
    InContainerMountStrategy,
    RcloneMountPattern,
    S3Mount,
)


def _build_manifest_with_s3(mount_path: str = "data", read_only: bool = True) -> Manifest:
    mount = S3Mount(
        bucket="my-bucket",
        mount_path=mount_path,
        read_only=read_only,
        mount_strategy=InContainerMountStrategy(
            pattern=RcloneMountPattern(remote_name="my-bucket"),
        ),
    )
    return Manifest(
        root="/workspace",
        entries={"data": mount},
    )


def test_no_mounts_returns_none() -> None:
    manifest = Manifest(
        root="/workspace",
        entries={"file.txt": File(content=b"x")},
    )
    assert build_remote_mount_policy_instructions(manifest) is None


def test_with_mount_renders_policy() -> None:
    manifest = _build_manifest_with_s3(read_only=True)
    out = build_remote_mount_policy_instructions(manifest)
    assert out is not None
    assert "Mounted remote storage paths" in out
    assert "data" in out
    assert "read-only" in out


def test_read_write_mount_renders_correctly() -> None:
    manifest = _build_manifest_with_s3(read_only=False)
    out = build_remote_mount_policy_instructions(manifest)
    assert out is not None
    assert "read+write" in out


def test_allowlist_text_renders_in_policy() -> None:
    manifest = _build_manifest_with_s3()
    out = build_remote_mount_policy_instructions(manifest)
    assert out is not None
    # The default allowlist includes ls, cat, head, tail, find, etc.
    assert "`ls`" in out
    assert "`cat`" in out


def test_get_remote_mounts_iterates_mount_entries() -> None:
    manifest = _build_manifest_with_s3()
    mounts = get_remote_mounts(manifest)
    assert len(mounts) == 1
    path, read_only = mounts[0]
    assert path.as_posix() == "data"
    assert read_only is True


def test_get_remote_mounts_ignores_non_mount_entries() -> None:
    manifest = Manifest(
        root="/workspace",
        entries={"file.txt": File(content=b"x")},
    )
    assert get_remote_mounts(manifest) == []


def test_policy_template_includes_required_substrings() -> None:
    assert "{path_lines}" in REMOTE_MOUNT_POLICY_TEMPLATE
    assert "{allowlist_text}" in REMOTE_MOUNT_POLICY_TEMPLATE
    assert "{edit_instructions}" in REMOTE_MOUNT_POLICY_TEMPLATE


def test_policy_includes_apply_patch_instructions() -> None:
    manifest = _build_manifest_with_s3()
    out = build_remote_mount_policy_instructions(manifest)
    assert out is not None
    assert "apply_patch" in out

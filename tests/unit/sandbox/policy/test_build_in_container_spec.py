"""Tests for ``build_in_container_mount_spec`` — Mount +
``InContainerMountStrategy`` → neutral ``InContainerMountSpec``.

Real constructed mount/strategy/pattern objects (no mocks); asserts
on the emitted spec dict and the documented omit-None / tuple→list
``pattern_config`` contract, plus the unsupported subclass×pattern
fail-loud path.
"""

from __future__ import annotations

from typing import Literal

import pytest

from troopai.adk.exceptions import (
    SandboxConfigurationError,
    UnsupportedMountPatternError,
)
from troopai.adk.sandbox.policy.mounts import build_in_container_mount_spec
from troopai.adk.types.sandbox.mounts import (
    AzureBlobMount,
    BoxMount,
    FuseMountPattern,
    GCSMount,
    InContainerMountStrategy,
    MountpointMountPattern,
    R2Mount,
    RcloneMountPattern,
    S3FilesMount,
    S3FilesMountPattern,
    S3Mount,
)


def _rclone_strategy(
    *,
    mode: Literal["fuse", "nfs"] = "fuse",
    extra_args: tuple[str, ...] = (),
    nfs_mount_options: tuple[str, ...] = (),
) -> InContainerMountStrategy:
    return InContainerMountStrategy(
        pattern=RcloneMountPattern(
            remote_name="r",
            mode=mode,
            extra_args=extra_args,
            nfs_mount_options=nfs_mount_options,
        )
    )


class TestRclonePattern:
    def test_s3_mount_rclone_spec_shape(self) -> None:
        strat = _rclone_strategy()
        mount = S3Mount(bucket="b", mount_path="data", read_only=True, mount_strategy=strat)
        spec = build_in_container_mount_spec(mount, strat, "/workspace")
        assert spec["type"] == "bind"
        assert spec["strategy"] == "in_container"
        assert spec["pattern_type"] == "rclone"
        assert spec["target"] == "/workspace/data"
        assert spec["read_only"] is True
        cfg = spec["pattern_config"]
        assert cfg["remote_name"] == "r"
        assert cfg["mode"] == "fuse"
        assert cfg["extra_args"] == []
        assert cfg["nfs_mount_options"] == []

    def test_rclone_serves_every_backing(self) -> None:
        # rclone speaks all six backings — none must raise.
        strat = _rclone_strategy()
        mounts = [
            S3Mount(bucket="b", mount_strategy=strat),
            GCSMount(bucket="b", mount_strategy=strat),
            R2Mount(bucket="b", account_id="a", mount_strategy=strat),
            AzureBlobMount(account="a", container="c", mount_strategy=strat),
            BoxMount(folder_id="f", mount_strategy=strat),
            S3FilesMount(mount_target_id="m", mount_strategy=strat),
        ]
        for m in mounts:
            spec = build_in_container_mount_spec(m, strat, "/ws")
            assert spec["pattern_type"] == "rclone"

    def test_omit_none_and_tuple_to_list_contract(self) -> None:
        # nfs_addr/config_file_path are None by default → keys ABSENT.
        # extra_args/nfs_mount_options tuples → lists, preserved.
        strat = _rclone_strategy(
            mode="nfs",
            extra_args=("--vfs-cache-mode", "full"),
            nfs_mount_options=("ro",),
        )
        mount = S3Mount(bucket="b", mount_strategy=strat)
        cfg = build_in_container_mount_spec(mount, strat, "/ws")["pattern_config"]
        assert "nfs_addr" not in cfg
        assert "config_file_path" not in cfg
        assert cfg["extra_args"] == ["--vfs-cache-mode", "full"]
        assert cfg["nfs_mount_options"] == ["ro"]
        assert cfg["mode"] == "nfs"

    def test_present_optional_keys_included(self) -> None:
        strat = InContainerMountStrategy(
            pattern=RcloneMountPattern(remote_name="r", nfs_addr=":2049", config_file_path="/etc/rclone.conf")
        )
        mount = S3Mount(bucket="b", mount_strategy=strat)
        cfg = build_in_container_mount_spec(mount, strat, "/ws")["pattern_config"]
        assert cfg["nfs_addr"] == ":2049"
        assert cfg["config_file_path"] == "/etc/rclone.conf"


class TestMountpointPattern:
    def test_s3_and_r2_supported(self) -> None:
        strat = InContainerMountStrategy(pattern=MountpointMountPattern(bucket="b"))
        for m in (
            S3Mount(bucket="b", mount_strategy=strat),
            R2Mount(bucket="b", account_id="a", mount_strategy=strat),
        ):
            spec = build_in_container_mount_spec(m, strat, "/ws")
            assert spec["pattern_type"] == "mountpoint"
            assert spec["pattern_config"]["bucket"] == "b"

    def test_gcs_with_mountpoint_raises(self) -> None:
        strat = InContainerMountStrategy(pattern=MountpointMountPattern(bucket="b"))
        mount = GCSMount(bucket="b", mount_strategy=strat)
        with pytest.raises(UnsupportedMountPatternError, match="GCSMount") as ei:
            build_in_container_mount_spec(mount, strat, "/ws")
        assert ei.value.mount_type == "GCSMount"
        assert ei.value.pattern_type == "mountpoint"

    def test_mountpoint_optional_keys_omitted(self) -> None:
        strat = InContainerMountStrategy(pattern=MountpointMountPattern(bucket="b"))
        cfg = build_in_container_mount_spec(S3Mount(bucket="b", mount_strategy=strat), strat, "/ws")["pattern_config"]
        assert "prefix" not in cfg
        assert "endpoint_url" not in cfg


class TestFusePattern:
    def test_azure_blob_supported_bool_int_preserved(self) -> None:
        strat = InContainerMountStrategy(pattern=FuseMountPattern(container="c", allow_other=True, cache_size_mb=256))
        mount = AzureBlobMount(account="a", container="c", read_only=False, mount_strategy=strat)
        spec = build_in_container_mount_spec(mount, strat, "/ws")
        cfg = spec["pattern_config"]
        assert cfg["allow_other"] is True  # bool preserved, not coerced
        assert cfg["cache_size_mb"] == 256  # int preserved
        assert spec["read_only"] is False

    def test_s3_with_fuse_raises(self) -> None:
        strat = InContainerMountStrategy(pattern=FuseMountPattern(container="c"))
        mount = S3Mount(bucket="b", mount_strategy=strat)
        with pytest.raises(UnsupportedMountPatternError) as ei:
            build_in_container_mount_spec(mount, strat, "/ws")
        assert ei.value.mount_type == "S3Mount"
        assert ei.value.pattern_type == "fuse"

    def test_fuse_cache_path_omitted_when_none(self) -> None:
        strat = InContainerMountStrategy(pattern=FuseMountPattern(container="c"))
        cfg = build_in_container_mount_spec(
            AzureBlobMount(account="a", container="c", mount_strategy=strat), strat, "/ws"
        )["pattern_config"]
        assert "cache_path" not in cfg
        assert "cache_size_mb" not in cfg


class TestS3FilesPattern:
    def test_s3files_supported(self) -> None:
        strat = InContainerMountStrategy(pattern=S3FilesMountPattern(mount_target_id="mt-1"))
        mount = S3FilesMount(mount_target_id="mt-1", mount_strategy=strat)
        spec = build_in_container_mount_spec(mount, strat, "/ws")
        assert spec["pattern_type"] == "s3files"
        assert spec["pattern_config"]["mount_target_id"] == "mt-1"

    def test_box_with_s3files_raises(self) -> None:
        strat = InContainerMountStrategy(pattern=S3FilesMountPattern(mount_target_id="mt-1"))
        mount = BoxMount(folder_id="f", mount_strategy=strat)
        with pytest.raises(UnsupportedMountPatternError) as ei:
            build_in_container_mount_spec(mount, strat, "/ws")
        assert ei.value.mount_type == "BoxMount"
        assert ei.value.pattern_type == "s3files"


class TestTargetResolution:
    def test_mount_path_none_uses_fallback(self) -> None:
        strat = _rclone_strategy()
        mount = S3Mount(bucket="b", mount_strategy=strat)  # mount_path None
        spec = build_in_container_mount_spec(mount, strat, "/workspace")
        assert spec["target"] == "/workspace/mount-s3_mount"

    def test_workspace_root_trailing_slash_normalized(self) -> None:
        strat = _rclone_strategy()
        mount = S3Mount(bucket="b", mount_path="sub/dir", mount_strategy=strat)
        spec = build_in_container_mount_spec(mount, strat, "/workspace/")
        assert spec["target"] == "/workspace/sub/dir"

    def test_read_only_propagation(self) -> None:
        strat = _rclone_strategy()
        ro = S3Mount(bucket="b", read_only=True, mount_strategy=strat)
        rw = S3Mount(bucket="b", read_only=False, mount_strategy=strat)
        assert build_in_container_mount_spec(ro, strat, "/ws")["read_only"] is True
        assert build_in_container_mount_spec(rw, strat, "/ws")["read_only"] is False

    def test_non_s3_mount_path_none_fallback(self) -> None:
        # Fallback path on a non-S3 subclass — confirms the
        # wire-discriminator fallback (mount.type == "gcs_mount") is
        # exercised beyond S3Mount, not the Python class name.
        strat = _rclone_strategy()
        mount = GCSMount(bucket="b", mount_strategy=strat)  # mount_path None
        spec = build_in_container_mount_spec(mount, strat, "/workspace")
        assert spec["target"] == "/workspace/mount-gcs_mount"


class TestFailLoudGuards:
    """The lane's flagship guarantee: every unsupported / unknown /
    future pattern fails loud at translation time — never a silent
    drop or a mistyped spec."""

    def test_unknown_pattern_type_raises(self) -> None:
        # An unrecognized pattern.type (e.g. a future 5th pattern, or
        # a tampered discriminator) must hit the empty-tuple gate and
        # raise — NOT silently produce a spec.
        strat = _rclone_strategy()
        mount = S3Mount(bucket="b", mount_strategy=strat)
        # pattern is a frozen pydantic model; simulate an unknown
        # discriminator via the frozen-bypass.
        object.__setattr__(strat.pattern, "type", "totally_unknown_5th")
        with pytest.raises(UnsupportedMountPatternError) as ei:
            build_in_container_mount_spec(mount, strat, "/ws")
        assert ei.value.pattern_type == "totally_unknown_5th"
        assert ei.value.mount_type == "S3Mount"

    def test_future_pattern_subclass_reaching_dispatch_raises(self) -> None:
        # A 5th MountPattern that PASSES the support-table gate (its
        # .type is a known key) but is none of the four dispatched
        # classes must hit the exhaustiveness `else` and raise —
        # proving it is never silently fed to the wrong config
        # builder (the latent silent-corruption this guard closes).
        class _FifthPattern:
            type = "rclone"  # known key → passes the support gate

        strat = _rclone_strategy()
        mount = S3Mount(bucket="b", mount_strategy=strat)
        object.__setattr__(strat, "pattern", _FifthPattern())
        with pytest.raises(UnsupportedMountPatternError) as ei:
            build_in_container_mount_spec(mount, strat, "/ws")
        assert ei.value.pattern_type == "rclone"
        assert ei.value.mount_type == "S3Mount"

    def test_strategy_not_matching_mount_raises(self) -> None:
        # A strategy that is NOT the mount's own (but happens to be
        # subclass-compatible) would silently build the wrong tool's
        # config. The precondition guard must reject it loud rather
        # than emit a mistyped spec.
        own = _rclone_strategy()
        mount = S3Mount(bucket="b", mount_strategy=own)
        other = InContainerMountStrategy(pattern=MountpointMountPattern(bucket="elsewhere"))
        with pytest.raises(SandboxConfigurationError, match="does not match"):
            build_in_container_mount_spec(mount, other, "/ws")

    def test_value_equal_strategy_copy_is_accepted(self) -> None:
        # A distinct-but-value-equal strategy object must pass (the
        # guard checks identity OR equality — a re-constructed
        # equivalent strategy is the same configuration).
        own = _rclone_strategy()
        mount = S3Mount(bucket="b", mount_path="d", mount_strategy=own)
        twin = _rclone_strategy()  # value-equal, different object
        spec = build_in_container_mount_spec(mount, twin, "/ws")
        assert spec["pattern_type"] == "rclone"
        assert spec["target"] == "/ws/d"

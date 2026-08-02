"""Tests for ``build_docker_volume_mount_spec`` — Mount +
``DockerVolumeMountStrategy`` → neutral ``DockerVolumeMountSpec``.

Real constructed mount/strategy objects (no mocks); asserts the
emitted spec shape, verbatim driver/driver_options passthrough (with
defensive copy), target resolution, read_only propagation, and the
shared strategy-mismatch fail-loud guard (including the cross-arm
in-container-vs-docker-volume misuse).
"""

from __future__ import annotations

import pytest

from troopai.adk.exceptions import SandboxConfigurationError
from troopai.adk.sandbox.policy.mounts import build_docker_volume_mount_spec
from troopai.adk.types.sandbox.mounts import (
    DockerVolumeMountStrategy,
    GCSMount,
    InContainerMountStrategy,
    RcloneMountPattern,
    S3Mount,
)


class TestDockerVolumeSpecShape:
    def test_spec_shape_and_driver_passthrough(self) -> None:
        strat = DockerVolumeMountStrategy(
            driver="rclone",
            driver_options={"type": "s3", "s3_provider": "AWS"},
        )
        mount = S3Mount(bucket="b", mount_path="data", read_only=True, mount_strategy=strat)
        spec = build_docker_volume_mount_spec(mount, strat, "/workspace")
        assert spec["type"] == "volume"
        assert spec["strategy"] == "docker_volume"
        assert spec["driver"] == "rclone"
        assert spec["driver_options"] == {"type": "s3", "s3_provider": "AWS"}
        assert spec["target"] == "/workspace/data"
        assert spec["read_only"] is True

    def test_empty_driver_options_passthrough(self) -> None:
        strat = DockerVolumeMountStrategy(driver="local")
        mount = S3Mount(bucket="b", mount_strategy=strat)
        spec = build_docker_volume_mount_spec(mount, strat, "/ws")
        assert spec["driver"] == "local"
        assert spec["driver_options"] == {}

    def test_driver_options_is_defensive_copy(self) -> None:
        # Mutating the spec's dict must NOT mutate the frozen
        # strategy's driver_options.
        strat = DockerVolumeMountStrategy(driver="d", driver_options={"k": "v"})
        mount = S3Mount(bucket="b", mount_strategy=strat)
        spec = build_docker_volume_mount_spec(mount, strat, "/ws")
        # Explicit decoupling intent, not only behavioural.
        assert spec["driver_options"] is not strat.driver_options
        spec["driver_options"]["injected"] = "x"
        assert "injected" not in strat.driver_options
        assert strat.driver_options == {"k": "v"}

    def test_read_only_propagation(self) -> None:
        ro = DockerVolumeMountStrategy(driver="d")
        rw = DockerVolumeMountStrategy(driver="d")
        m_ro = S3Mount(bucket="b", read_only=True, mount_strategy=ro)
        m_rw = S3Mount(bucket="b", read_only=False, mount_strategy=rw)
        assert build_docker_volume_mount_spec(m_ro, ro, "/ws")["read_only"] is True
        assert build_docker_volume_mount_spec(m_rw, rw, "/ws")["read_only"] is False

    def test_target_fallback_when_mount_path_none(self) -> None:
        strat = DockerVolumeMountStrategy(driver="d")
        mount = GCSMount(bucket="b", mount_strategy=strat)  # mount_path None
        spec = build_docker_volume_mount_spec(mount, strat, "/workspace/")
        assert spec["target"] == "/workspace/mount-gcs_mount"


class TestStrategyMatchGuardShared:
    def test_mismatched_docker_volume_strategy_raises(self) -> None:
        own = DockerVolumeMountStrategy(driver="rclone", driver_options={"type": "s3"})
        mount = S3Mount(bucket="b", mount_strategy=own)
        other = DockerVolumeMountStrategy(driver="s3fs", driver_options={"x": "y"})
        with pytest.raises(SandboxConfigurationError, match="does not match"):
            build_docker_volume_mount_spec(mount, other, "/ws")

    def test_value_equal_strategy_copy_accepted(self) -> None:
        own = DockerVolumeMountStrategy(driver="rclone", driver_options={"type": "s3"})
        mount = S3Mount(bucket="b", mount_path="d", mount_strategy=own)
        twin = DockerVolumeMountStrategy(driver="rclone", driver_options={"type": "s3"})
        spec = build_docker_volume_mount_spec(mount, twin, "/ws")
        assert spec["driver"] == "rclone"
        assert spec["target"] == "/ws/d"

    def test_cross_arm_in_container_mount_rejected(self) -> None:
        # The mount is configured for an in-container strategy; calling
        # the docker-volume builder with a docker-volume strategy must
        # fail loud (the shared guard catches the cross-arm mismatch),
        # not silently build a volume for an in-container mount.
        in_c = InContainerMountStrategy(pattern=RcloneMountPattern(remote_name="r"))
        mount = S3Mount(bucket="b", mount_strategy=in_c)
        dvol = DockerVolumeMountStrategy(driver="rclone")
        with pytest.raises(SandboxConfigurationError, match="does not match") as ei:
            build_docker_volume_mount_spec(mount, dvol, "/ws")
        # message names both discriminators so the cross-arm mismatch
        # is unambiguous in a post-mortem.
        assert "docker_volume" in str(ei.value)
        assert "in_container" in str(ei.value)


class TestDriverValidation:
    @pytest.mark.parametrize(
        "bad_driver",
        [
            "",
            "   ",
            "\t",
            "\n ",
            chr(0x200B),  # ZERO WIDTH SPACE — not str.isspace()
            chr(0x2060),  # WORD JOINER — not str.isspace()
            chr(0xFEFF),  # ZW NO-BREAK SPACE / BOM
            "  " + chr(0x200B) + "\t",  # whitespace + zero-width mix
        ],
    )
    def test_empty_or_blank_driver_raises(self, bad_driver: str) -> None:
        # A blank driver is statically invalid on every Docker host;
        # it must fail loud at translation time, not as an opaque
        # daemon error layers later.
        strat = DockerVolumeMountStrategy(driver=bad_driver)
        mount = S3Mount(bucket="b", mount_strategy=strat)
        with pytest.raises(SandboxConfigurationError, match="empty/blank volume driver"):
            build_docker_volume_mount_spec(mount, strat, "/ws")

    def test_nonblank_driver_accepted_even_if_unregistered(self) -> None:
        # An unregistered (but non-blank) driver name is host-dependent
        # and correctly NOT rejected here — it surfaces at the backend.
        strat = DockerVolumeMountStrategy(driver="some-unregistered-driver")
        mount = S3Mount(bucket="b", mount_strategy=strat)
        spec = build_docker_volume_mount_spec(mount, strat, "/ws")
        assert spec["driver"] == "some-unregistered-driver"

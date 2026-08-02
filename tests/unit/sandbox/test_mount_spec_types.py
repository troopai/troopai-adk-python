"""Tests for the backend mount-spec wire types
(``InContainerMountSpec`` / ``DockerVolumeMountSpec`` / ``MountSpec``)
and the two mount-translation exceptions
(``UnsupportedMountStrategyError`` / ``UnsupportedMountPatternError``).

These are the contract the mount-translation helpers and backend
clients build on, so they are pinned with real constructed values
(no mocks).
"""

from __future__ import annotations

import json

import pytest

from troopai.adk.exceptions import (
    SandboxConfigurationError,
    SandboxError,
    TroopAIError,
    UnsupportedMountPatternError,
    UnsupportedMountStrategyError,
)
from troopai.adk.types.sandbox.mounts import (
    DockerVolumeMountSpec,
    InContainerMountSpec,
    MountSpec,
)


class TestInContainerMountSpec:
    def test_construct_and_fields(self) -> None:
        spec: InContainerMountSpec = {
            "type": "bind",
            "target": "/workspace/data",
            "read_only": True,
            "strategy": "in_container",
            "pattern_type": "rclone",
            "pattern_config": {"remote_name": "s3backend", "mode": "ro"},
        }
        assert spec["type"] == "bind"
        assert spec["strategy"] == "in_container"
        assert spec["target"] == "/workspace/data"
        assert spec["read_only"] is True
        assert spec["pattern_type"] == "rclone"
        assert spec["pattern_config"]["remote_name"] == "s3backend"

    def test_json_roundtrip(self) -> None:
        # Plain-dict-at-runtime contract: the spec must serialize so a
        # backend client can hand it across a process boundary.
        spec: InContainerMountSpec = {
            "type": "bind",
            "target": "/workspace/m",
            "read_only": False,
            "strategy": "in_container",
            "pattern_type": "mountpoint",
            "pattern_config": {},
        }
        assert json.loads(json.dumps(spec)) == spec

    def test_pattern_config_holds_heterogeneous_json_safe_values(self) -> None:
        # Pins the widened contract: FuseMountPattern contributes
        # bool (allow_other) + int (cache_size_mb); RcloneMountPattern
        # contributes a string list (nfs_mount_options). dict[str,str]
        # would have forced lossy coercion of these. JSON round-trip
        # must preserve the leaf types exactly.
        spec: InContainerMountSpec = {
            "type": "bind",
            "target": "/workspace/blob",
            "read_only": True,
            "strategy": "in_container",
            "pattern_type": "fuse",
            "pattern_config": {
                "container": "data",
                "allow_other": True,
                "cache_size_mb": 512,
                "nfs_mount_options": ["ro", "noatime"],
            },
        }
        restored = json.loads(json.dumps(spec))
        assert restored == spec
        cfg = spec["pattern_config"]
        assert cfg["allow_other"] is True
        assert cfg["cache_size_mb"] == 512
        assert cfg["nfs_mount_options"] == ["ro", "noatime"]


class TestDockerVolumeMountSpec:
    def test_construct_and_fields(self) -> None:
        spec: DockerVolumeMountSpec = {
            "type": "volume",
            "target": "/workspace/vol",
            "read_only": False,
            "strategy": "docker_volume",
            "driver": "rclone",
            "driver_options": {"type": "s3", "s3_provider": "AWS"},
        }
        assert spec["type"] == "volume"
        assert spec["strategy"] == "docker_volume"
        assert spec["driver"] == "rclone"
        assert spec["driver_options"]["type"] == "s3"
        assert spec["read_only"] is False

    def test_json_roundtrip(self) -> None:
        spec: DockerVolumeMountSpec = {
            "type": "volume",
            "target": "/workspace/v",
            "read_only": True,
            "strategy": "docker_volume",
            "driver": "local",
            "driver_options": {},
        }
        assert json.loads(json.dumps(spec)) == spec


class TestMountSpecUnionDiscriminates:
    def test_strategy_is_the_discriminator(self) -> None:
        # The union is discriminated on "strategy"; a consumer
        # branches on it to choose its native attach call.
        in_c: InContainerMountSpec = {
            "type": "bind",
            "target": "/w/a",
            "read_only": True,
            "strategy": "in_container",
            "pattern_type": "fuse",
            "pattern_config": {},
        }
        dvol: DockerVolumeMountSpec = {
            "type": "volume",
            "target": "/w/b",
            "read_only": True,
            "strategy": "docker_volume",
            "driver": "rclone",
            "driver_options": {},
        }
        specs: list[InContainerMountSpec | DockerVolumeMountSpec] = [in_c, dvol]
        seen = {spec["strategy"] for spec in specs}
        assert seen == {"in_container", "docker_volume"}

    def test_consumer_branches_on_strategy_key(self) -> None:
        # Pins the exact consumption idiom every mount-translation
        # helper and backend client MUST implement. TypedDicts have
        # no isinstance; the discriminator is `spec["strategy"]`. A
        # rename of the key or its literals must break this test
        # BEFORE downstream consumers build on it.
        in_c: MountSpec = {
            "type": "bind",
            "target": "/w/a",
            "read_only": True,
            "strategy": "in_container",
            "pattern_type": "rclone",
            "pattern_config": {"remote_name": "r"},
        }
        dvol: MountSpec = {
            "type": "volume",
            "target": "/w/b",
            "read_only": False,
            "strategy": "docker_volume",
            "driver": "rclone",
            "driver_options": {"type": "s3"},
        }
        for spec in (in_c, dvol):
            if spec["strategy"] == "in_container":
                assert spec["type"] == "bind"
                assert spec["pattern_type"] == "rclone"
            elif spec["strategy"] == "docker_volume":
                assert spec["type"] == "volume"
                assert spec["driver"] == "rclone"
            else:  # pragma: no cover - exhaustive guard
                pytest.fail(f"unhandled strategy {spec['strategy']!r}")


class TestUnsupportedMountStrategyError:
    def test_attributes_and_message(self) -> None:
        err = UnsupportedMountStrategyError(mount_type="S3Mount", strategy_type="docker_volume", backend="k8s")
        assert err.mount_type == "S3Mount"
        assert err.strategy_type == "docker_volume"
        assert err.backend == "k8s"
        msg = str(err)
        assert "S3Mount" in msg
        assert "docker_volume" in msg
        assert "k8s" in msg

    def test_hierarchy(self) -> None:
        err = UnsupportedMountStrategyError(mount_type="GCSMount", strategy_type="docker_volume", backend="local")
        assert isinstance(err, SandboxConfigurationError)
        assert isinstance(err, SandboxError)
        # Reaches the framework-wide base every top-level handler keys
        # on: if SandboxError were reparented off TroopAIError, an
        # `except TroopAIError` handler would silently miss this.
        assert isinstance(err, TroopAIError)

    def test_is_distinct_from_pattern_error(self) -> None:
        # Symmetric to TestUnsupportedMountPatternError's distinctness
        # test: a regression making this subclass the pattern error
        # would let `except UnsupportedMountPatternError` silently
        # also swallow strategy failures.
        strategy_err = UnsupportedMountStrategyError(mount_type="S3Mount", strategy_type="docker_volume", backend="k8s")
        assert not isinstance(strategy_err, UnsupportedMountPatternError)

    def test_raises_and_catchable_as_config_error(self) -> None:
        with pytest.raises(SandboxConfigurationError):
            raise UnsupportedMountStrategyError(mount_type="BoxMount", strategy_type="docker_volume", backend="k8s")


class TestUnsupportedMountPatternError:
    def test_attributes_and_message(self) -> None:
        err = UnsupportedMountPatternError(mount_type="BoxMount", pattern_type="mountpoint")
        assert err.mount_type == "BoxMount"
        assert err.pattern_type == "mountpoint"
        msg = str(err)
        assert "BoxMount" in msg
        assert "mountpoint" in msg

    def test_hierarchy(self) -> None:
        err = UnsupportedMountPatternError(mount_type="GCSMount", pattern_type="mountpoint")
        assert isinstance(err, SandboxConfigurationError)
        assert isinstance(err, SandboxError)
        assert isinstance(err, TroopAIError)

    def test_is_distinct_from_strategy_error(self) -> None:
        # Pattern-incompat and strategy-incompat are different
        # failures; catching one must not catch the other.
        pattern_err = UnsupportedMountPatternError(mount_type="BoxMount", pattern_type="mountpoint")
        assert not isinstance(pattern_err, UnsupportedMountStrategyError)

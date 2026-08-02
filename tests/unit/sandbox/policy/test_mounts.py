"""Tests for Mount → backend translation helpers."""

from __future__ import annotations

import pytest

from troopai.adk.exceptions import UnsupportedMountStrategyError
from troopai.adk.sandbox.policy import (
    apply_mounts_to_docker,
    apply_mounts_to_hosted_bridge,
    apply_mounts_to_k8s_pod,
    describe_mount_for_local,
)
from troopai.adk.types.sandbox.mounts import (
    AzureBlobMount,
    BoxMount,
    DockerVolumeMountStrategy,
    GCSMount,
    InContainerMountStrategy,
    Mount,
    RcloneMountPattern,
    S3FilesMount,
    S3Mount,
)


def _s3() -> S3Mount:
    return S3Mount(
        bucket="b",
        prefix="p",
        mount_path="data",
        mount_strategy=InContainerMountStrategy(
            type="in_container", pattern=RcloneMountPattern(type="rclone", remote_name="r")
        ),
    )


def _gcs() -> GCSMount:
    return GCSMount(
        bucket="gb",
        prefix="gp",
        mount_path="gdata",
        mount_strategy=InContainerMountStrategy(
            type="in_container", pattern=RcloneMountPattern(type="rclone", remote_name="r")
        ),
    )


def _azure() -> AzureBlobMount:
    return AzureBlobMount(
        account="acct",
        container="c1",
        prefix="ap",
        mount_path="adata",
        mount_strategy=InContainerMountStrategy(
            type="in_container", pattern=RcloneMountPattern(type="rclone", remote_name="r")
        ),
    )


def _box() -> BoxMount:
    return BoxMount(
        folder_id="f1",
        mount_path="bdata",
        mount_strategy=InContainerMountStrategy(
            type="in_container", pattern=RcloneMountPattern(type="rclone", remote_name="r")
        ),
    )


def _s3files() -> S3FilesMount:
    return S3FilesMount(
        mount_target_id="t1",
        mount_path="sfdata",
        mount_strategy=InContainerMountStrategy(
            type="in_container", pattern=RcloneMountPattern(type="rclone", remote_name="r")
        ),
    )


class TestDockerMounts:
    def test_empty_passthrough(self) -> None:
        kwargs = {"image": "x"}
        result = apply_mounts_to_docker([], kwargs)
        assert result == {"image": "x"}

    def test_in_container_mount_emits_neutral_spec_and_cap_add(self) -> None:
        kwargs = apply_mounts_to_docker([_s3()], {})
        # New contract: a neutral `mounts` spec list, NOT the old
        # (invalid) URI-keyed `volumes` dict.
        assert "volumes" not in kwargs
        assert "mounts" in kwargs
        spec = kwargs["mounts"][0]
        assert spec["type"] == "bind"
        assert spec["strategy"] == "in_container"
        assert spec["pattern_type"] == "rclone"
        assert spec["target"] == "/workspace/data"
        assert spec["read_only"] is True
        # in-container FUSE tool needs SYS_ADMIN, added once.
        assert kwargs["cap_add"] == ["SYS_ADMIN"]

    def test_read_only_false_propagates(self) -> None:
        mount = S3Mount(
            bucket="b",
            mount_path="rw",
            read_only=False,
            mount_strategy=InContainerMountStrategy(
                type="in_container", pattern=RcloneMountPattern(type="rclone", remote_name="r")
            ),
        )
        kwargs = apply_mounts_to_docker([mount], {})
        assert kwargs["mounts"][0]["read_only"] is False

    def test_docker_volume_strategy_emits_volume_spec_no_cap_add(self) -> None:
        strat = DockerVolumeMountStrategy(driver="rclone", driver_options={"type": "s3"})
        mount = S3Mount(bucket="b", mount_path="d", mount_strategy=strat)
        kwargs = apply_mounts_to_docker([mount], {})
        spec = kwargs["mounts"][0]
        assert spec["type"] == "volume"
        assert spec["strategy"] == "docker_volume"
        assert spec["driver"] == "rclone"
        assert spec["driver_options"] == {"type": "s3"}
        assert spec["target"] == "/workspace/d"
        # No in-container mount → no SYS_ADMIN injected.
        assert "cap_add" not in kwargs

    def test_cap_add_extends_existing_not_replaces(self) -> None:
        kwargs = apply_mounts_to_docker([_s3()], {"cap_add": ["NET_ADMIN"]})
        assert kwargs["cap_add"] == ["NET_ADMIN", "SYS_ADMIN"]

    def test_cap_add_not_duplicated(self) -> None:
        kwargs = apply_mounts_to_docker([_s3()], {"cap_add": ["SYS_ADMIN"]})
        assert kwargs["cap_add"] == ["SYS_ADMIN"]

    def test_workspace_root_override(self) -> None:
        kwargs = apply_mounts_to_docker([_s3()], {}, workspace_root="/custom/ws")
        assert kwargs["mounts"][0]["target"] == "/custom/ws/data"

    def test_mounts_extends_preexisting_list(self) -> None:
        sentinel = {"type": "bind", "target": "/pre", "read_only": True}
        kwargs = apply_mounts_to_docker([_s3()], {"mounts": [sentinel]})
        assert kwargs["mounts"][0] is sentinel
        assert len(kwargs["mounts"]) == 2

    def test_mixed_in_container_and_docker_volume_mounts(self) -> None:
        # Real workspace pattern: a FUSE-mounted bucket alongside a
        # docker-volume-mounted dataset. Pins heterogeneous
        # accumulation + that a single in-container mount triggers
        # the (deduped) SYS_ADMIN injection for the whole call.
        in_c = _s3()  # InContainerMountStrategy(rclone)
        dvol_mount = S3Mount(
            bucket="ds",
            mount_path="dataset",
            mount_strategy=DockerVolumeMountStrategy(driver="rclone", driver_options={"type": "s3"}),
        )
        kwargs = apply_mounts_to_docker([in_c, dvol_mount], {})
        assert len(kwargs["mounts"]) == 2
        assert kwargs["mounts"][0]["strategy"] == "in_container"
        assert kwargs["mounts"][1]["strategy"] == "docker_volume"
        assert kwargs["mounts"][1]["driver"] == "rclone"
        # One in-container mount ⇒ SYS_ADMIN added once for the call.
        assert kwargs["cap_add"] == ["SYS_ADMIN"]

    def test_unsupported_strategy_kind_raises(self) -> None:
        # Defense-in-depth exhaustive-else: a strategy that is neither
        # union arm must fail loud, never silently skip the mount.
        class _AlienStrategy:
            type = "alien"

        mount = _s3()
        object.__setattr__(mount, "mount_strategy", _AlienStrategy())
        with pytest.raises(UnsupportedMountStrategyError) as ei:
            apply_mounts_to_docker([mount], {})
        assert ei.value.strategy_type == "alien"
        assert ei.value.backend == "docker"


class TestK8sMounts:
    def test_empty_passthrough(self) -> None:
        spec = {"containers": [{"name": "c"}]}
        result = apply_mounts_to_k8s_pod([], spec)
        assert result == {"containers": [{"name": "c"}]}

    def test_s3_emits_csi_volume(self) -> None:
        spec = apply_mounts_to_k8s_pod([_s3()], {"containers": [{"name": "c"}]})
        volumes = spec["volumes"]
        assert len(volumes) == 1
        assert volumes[0]["csi"]["driver"] == "s3.csi.aws.com"
        assert volumes[0]["csi"]["volumeAttributes"]["bucketName"] == "b"
        mounts = spec["containers"][0]["volumeMounts"]
        # K8s rejects a relative mountPath — it MUST be absolute, rooted
        # at the container workingDir (default /workspace), matching the
        # Docker backend's /workspace-rooted targets.
        assert mounts[0]["mountPath"] == "/workspace/data"
        assert mounts[0]["readOnly"] is True

    def test_mount_path_absolute_for_every_container(self) -> None:
        # K8s admission rejects any relative volumeMounts[].mountPath. Two
        # containers ⇒ both must receive an absolute, workspace-rooted path.
        spec = apply_mounts_to_k8s_pod([_s3()], {"containers": [{"name": "c1"}, {"name": "c2"}]})
        for container in spec["containers"]:
            mount_path = container["volumeMounts"][0]["mountPath"]
            assert mount_path == "/workspace/data"
            assert mount_path.startswith("/")

    def test_unpathed_mount_path_absolute_via_wire_discriminator(self) -> None:
        # mount_path omitted → fallback "mount-s3_mount" must still be
        # rooted absolute, not the relative bare fallback K8s rejects.
        mount = S3Mount(
            bucket="b",
            mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern(remote_name="r")),
        )
        spec = apply_mounts_to_k8s_pod([mount], {"containers": [{"name": "c"}]})
        assert spec["containers"][0]["volumeMounts"][0]["mountPath"] == "/workspace/mount-s3_mount"

    def test_workspace_root_override_roots_mount_path(self) -> None:
        # The container workingDir flows in as workspace_root so the
        # mountPath lands under the same absolute root as the workspace.
        spec = apply_mounts_to_k8s_pod([_s3()], {"containers": [{"name": "c"}]}, workspace_root="/custom/ws")
        assert spec["containers"][0]["volumeMounts"][0]["mountPath"] == "/custom/ws/data"

    def test_gcs_uses_gcs_csi_driver(self) -> None:
        spec = apply_mounts_to_k8s_pod([_gcs()], {"containers": [{"name": "c"}]})
        assert spec["volumes"][0]["csi"]["driver"] == "gcs.csi.ofek.dev"

    def test_azure_uses_blob_csi_driver(self) -> None:
        spec = apply_mounts_to_k8s_pod([_azure()], {"containers": [{"name": "c"}]})
        assert spec["volumes"][0]["csi"]["driver"] == "blob.csi.azure.com"

    def test_box_falls_back_to_emptydir(self) -> None:
        spec = apply_mounts_to_k8s_pod([_box()], {"containers": [{"name": "c"}]})
        assert spec["volumes"][0]["emptyDir"] == {}

    def test_docker_volume_strategy_rejected(self) -> None:
        # K8s has no Docker daemon — a DockerVolumeMountStrategy must
        # fail loud, never be silently materialized as a CSI volume.
        mount = S3Mount(
            bucket="b",
            mount_path="d",
            mount_strategy=DockerVolumeMountStrategy(driver="rclone", driver_options={"type": "s3"}),
        )
        with pytest.raises(UnsupportedMountStrategyError) as ei:
            apply_mounts_to_k8s_pod([mount], {"containers": [{"name": "c"}]})
        assert ei.value.strategy_type == "docker_volume"
        assert ei.value.backend == "k8s"

    def test_unsupported_strategy_kind_raises(self) -> None:
        # Defense-in-depth exhaustive guard: a strategy that is neither
        # union arm must fail loud per backend, never silently skip.
        class _AlienStrategy:
            type = "alien"

        mount = _s3()
        object.__setattr__(mount, "mount_strategy", _AlienStrategy())
        with pytest.raises(UnsupportedMountStrategyError) as ei:
            apply_mounts_to_k8s_pod([mount], {"containers": [{"name": "c"}]})
        assert ei.value.strategy_type == "alien"
        assert ei.value.backend == "k8s"

    def test_bad_strategy_in_list_leaves_pod_spec_unmutated(self) -> None:
        # All-or-nothing: a valid mount followed by a docker_volume
        # mount must raise WITHOUT having mutated pod_spec — no
        # dangling volumeMounts referencing volumes never appended.
        bad = S3Mount(
            bucket="b",
            mount_path="d",
            mount_strategy=DockerVolumeMountStrategy(driver="rclone", driver_options={"type": "s3"}),
        )
        pod_spec = {"containers": [{"name": "c"}]}
        with pytest.raises(UnsupportedMountStrategyError):
            apply_mounts_to_k8s_pod([_s3(), bad], pod_spec)
        assert "volumes" not in pod_spec  # deferred write never ran
        assert "volumeMounts" not in pod_spec["containers"][0]  # no in-place torn state


class TestLocalMounts:
    def test_s3_uses_rclone(self) -> None:
        spec = describe_mount_for_local(_s3())
        assert spec["tool"] == "rclone"
        assert "s3:b/p" in spec["argv"]

    def test_gcs_uses_gcsfuse(self) -> None:
        spec = describe_mount_for_local(_gcs())
        assert spec["tool"] == "gcsfuse"
        assert "gb" in spec["argv"]

    def test_azure_uses_blobfuse2(self) -> None:
        spec = describe_mount_for_local(_azure())
        assert spec["tool"] == "blobfuse2"

    def test_docker_volume_strategy_rejected(self) -> None:
        # The local backend has no Docker daemon — reject
        # DockerVolumeMountStrategy rather than rclone-mount it as if
        # the strategy had been in-container.
        mount = S3Mount(
            bucket="b",
            mount_path="d",
            mount_strategy=DockerVolumeMountStrategy(driver="rclone", driver_options={"type": "s3"}),
        )
        with pytest.raises(UnsupportedMountStrategyError) as ei:
            describe_mount_for_local(mount)
        assert ei.value.strategy_type == "docker_volume"
        assert ei.value.backend == "local"

    def test_unsupported_strategy_kind_raises(self) -> None:
        class _AlienStrategy:
            type = "alien"

        mount = _s3()
        object.__setattr__(mount, "mount_strategy", _AlienStrategy())
        with pytest.raises(UnsupportedMountStrategyError) as ei:
            describe_mount_for_local(mount)
        assert ei.value.strategy_type == "alien"
        assert ei.value.backend == "local"


class TestHostedBridgeMounts:
    def test_empty_passthrough(self) -> None:
        body = {"x": 1}
        result = apply_mounts_to_hosted_bridge([], "modal", body)
        assert result == {"x": 1}

    @pytest.mark.parametrize(
        "provider,field",
        [
            ("modal", "cloud_bucket_mounts"),
            ("daytona", "buckets"),
            ("cloudflare", "r2_bindings"),
            ("blaxel", "cloud_buckets"),
            ("e2b", "mounts"),
        ],
    )
    def test_provider_field_name(self, provider: str, field: str) -> None:
        body = apply_mounts_to_hosted_bridge([_s3()], provider, {})
        assert field in body
        assert len(body[field]) == 1
        assert body[field][0]["target_path"] == "data"

    def test_s3files_emits_uri(self) -> None:
        body = apply_mounts_to_hosted_bridge([_s3files()], "modal", {})
        item = body["cloud_bucket_mounts"][0]
        assert item["uri"].startswith("s3files://")

    def test_type_is_stable_wire_discriminator_not_class_name(self) -> None:
        # The emitted wire `type` must be the stable discriminator
        # (mount.type == "s3_mount"), NOT type(mount).__name__ ("S3Mount"),
        # so a pure Python class rename cannot break the provider contract.
        body = apply_mounts_to_hosted_bridge([_s3()], "modal", {})
        item = body["cloud_bucket_mounts"][0]
        assert item["type"] == "s3_mount"
        assert item["type"] != type(_s3()).__name__

    @pytest.mark.parametrize(
        "mount,expected_type",
        [
            (_s3(), "s3_mount"),
            (_gcs(), "gcs_mount"),
            (_azure(), "azure_blob_mount"),
            (_box(), "box_mount"),
            (_s3files(), "s3_files_mount"),
        ],
    )
    def test_type_matches_discriminator_for_each_subclass(self, mount: Mount, expected_type: str) -> None:
        body = apply_mounts_to_hosted_bridge([mount], "modal", {})
        assert body["cloud_bucket_mounts"][0]["type"] == expected_type

    def test_mount_target_fallback_uses_wire_discriminator(self) -> None:
        # _mount_target's no-mount_path fallback must derive from the
        # stable wire discriminator (mount.type == "s3_mount"), NOT the
        # Python class name ("s3mount"). Pins the refactor-stable path.
        mount = S3Mount(
            bucket="b",
            mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern(remote_name="r")),
        )  # mount_path omitted → None → fallback path
        body = apply_mounts_to_hosted_bridge([mount], "modal", {})
        assert body["cloud_bucket_mounts"][0]["target_path"] == "mount-s3_mount"

    def test_docker_volume_strategy_accepted_by_design(self) -> None:
        # Asymmetry with the k8s/local strategy guards: a hosted
        # bridge is a managed sandbox — the provider owns mount
        # realization, so mount_strategy is advisory and NOT rejected
        # here. A
        # DockerVolumeMountStrategy must NOT raise; the bucket is still
        # emitted for the provider to mount. By design, not a drop.
        mount = S3Mount(
            bucket="b",
            mount_path="d",
            mount_strategy=DockerVolumeMountStrategy(driver="rclone", driver_options={"type": "s3"}),
        )
        body = apply_mounts_to_hosted_bridge([mount], "modal", {})
        assert len(body["cloud_bucket_mounts"]) == 1
        assert body["cloud_bucket_mounts"][0]["target_path"] == "d"

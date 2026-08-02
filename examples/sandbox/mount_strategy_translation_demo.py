"""One declarative cloud mount → every sandbox backend, via strategy dispatch.

A single ``S3Mount`` + ``mount_strategy`` is translated, with no
agent and no network, into:

* a Docker ``containers.run`` neutral mount-spec (+ ``SYS_ADMIN``),
* a Kubernetes pod CSI volume + volumeMount,
* a local ``rclone`` subprocess argv,
* each hosted provider's cloud-bucket create-body field
  (Modal / Daytona / Cloudflare / Blaxel / E2B).

It also shows the strategy *asymmetry*: a ``DockerVolumeMountStrategy``
materializes on Docker but is rejected loud on k8s / local (no Docker
daemon there), while hosted bridges accept it strategy-agnostically
(the managed provider owns mount realization).

OpenAI's Agents SDK has no sandbox cloud-mount system at all; this
provider-agnostic mount-strategy → backend translation is a feature
it lacks entirely.

No external API key required (pure, synthetic policy translation).
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import logging

from troopai.adk.exceptions import UnsupportedMountStrategyError
from troopai.adk.sandbox.policy import (
    apply_mounts_to_docker,
    apply_mounts_to_hosted_bridge,
    apply_mounts_to_k8s_pod,
    build_in_container_mount_spec,
    describe_mount_for_local,
)
from troopai.adk.types.sandbox.mounts import (
    DockerVolumeMountStrategy,
    InContainerMountStrategy,
    RcloneMountPattern,
    S3Mount,
)

logger = logging.getLogger(__name__)

_HOSTED = (
    ("modal", "cloud_bucket_mounts"),
    ("daytona", "buckets"),
    ("cloudflare", "r2_bindings"),
    ("blaxel", "cloud_buckets"),
    ("e2b", "mounts"),
)


def _verify(condition: bool, detail: str) -> None:
    """Fail loud on a translation regression.

    The bare ``assert`` statement is stripped under ``python -O`` /
    ``PYTHONOPTIMIZE``. This example doubles as a regression canary,
    so its checks raise explicitly and stay armed under every
    interpreter mode — explicit ``if ...: raise`` guards stay active
    even under ``python -O`` / ``PYTHONOPTIMIZE``, which strips
    ``assert``.
    """
    if not condition:
        raise RuntimeError(f"mount-translation regression: {detail}")


def _demo_in_container() -> None:
    """One in-container (rclone FUSE) S3 mount → Docker, K8s, local, each hosted provider."""
    strategy = InContainerMountStrategy(
        pattern=RcloneMountPattern(remote_name="research-corpus"),
    )
    mount = S3Mount(
        bucket="research-corpus",
        prefix="papers/",
        mount_path="data",
        read_only=True,
        mount_strategy=strategy,
    )

    docker_kwargs = apply_mounts_to_docker([mount], {})
    spec = docker_kwargs["mounts"][0]
    logger.info(
        "Docker  : strategy=%s pattern=%s cap_add=%s", spec["strategy"], spec["pattern_type"], docker_kwargs["cap_add"]
    )
    _verify(spec["strategy"] == "in_container", "docker in-container strategy")
    _verify(docker_kwargs["cap_add"] == ["SYS_ADMIN"], "docker cap_add SYS_ADMIN")

    pod = apply_mounts_to_k8s_pod([mount], {"containers": [{"name": "c"}]})
    logger.info(
        "K8s     : csi=%s mountPath=%s",
        pod["volumes"][0]["csi"]["driver"],
        pod["containers"][0]["volumeMounts"][0]["mountPath"],
    )
    _verify(pod["volumes"][0]["csi"]["driver"] == "s3.csi.aws.com", "k8s S3 CSI driver")

    local = describe_mount_for_local(mount)
    logger.info("Local   : tool=%s argv=%s", local["tool"], local["argv"])
    _verify(local["tool"] == "rclone", "local rclone tool")

    for provider, field in _HOSTED:
        body = apply_mounts_to_hosted_bridge([mount], provider, {})
        logger.info("Hosted  : %-10s field=%-18s target=%s", provider, field, body[field][0]["target_path"])
        _verify(body[field][0]["target_path"] == "data", f"hosted {provider} target_path")

    low_level = build_in_container_mount_spec(mount, strategy, "/workspace")
    logger.info("Builder : neutral spec target=%s read_only=%s", low_level["target"], low_level["read_only"])
    _verify(low_level["target"] == "/workspace/data", "builder neutral-spec target")


def _demo_strategy_asymmetry() -> None:
    """DockerVolumeMountStrategy: Docker yes; k8s/local loud-reject; hosted agnostic."""
    mount = S3Mount(
        bucket="datasets",
        mount_path="ds",
        mount_strategy=DockerVolumeMountStrategy(driver="rclone", driver_options={"type": "s3"}),
    )

    spec = apply_mounts_to_docker([mount], {})["mounts"][0]
    logger.info("Docker  : strategy=%s driver=%s (no cap_add)", spec["strategy"], spec["driver"])
    _verify(spec["strategy"] == "docker_volume", "docker docker_volume strategy")

    try:
        apply_mounts_to_k8s_pod([mount], {"containers": [{"name": "c"}]})
        raise RuntimeError("k8s should have rejected docker_volume")
    except UnsupportedMountStrategyError as exc:
        logger.info("k8s     : correctly rejected — backend=%s strategy=%s", exc.backend, exc.strategy_type)
        _verify(exc.backend == "k8s", "k8s rejection backend")
        _verify(exc.strategy_type == "docker_volume", "k8s rejection strategy_type")

    try:
        describe_mount_for_local(mount)
        raise RuntimeError("local should have rejected docker_volume")
    except UnsupportedMountStrategyError as exc:
        logger.info("local   : correctly rejected — backend=%s strategy=%s", exc.backend, exc.strategy_type)
        _verify(exc.backend == "local", "local rejection backend")
        _verify(exc.strategy_type == "docker_volume", "local rejection strategy_type")

    body = apply_mounts_to_hosted_bridge([mount], "modal", {})
    logger.info(
        "Hosted  : modal accepts strategy-agnostically — target=%s", body["cloud_bucket_mounts"][0]["target_path"]
    )
    _verify(len(body["cloud_bucket_mounts"]) == 1, "hosted modal emits one item")


def main() -> None:
    logger.info("=== One in-container S3 mount → Docker, K8s, local, hosted ===")
    _demo_in_container()
    logger.info("=== Strategy asymmetry: docker_volume across backends ===")
    _demo_strategy_asymmetry()
    logger.info("All translations verified.")


if __name__ == "__main__":
    main()

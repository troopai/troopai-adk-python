"""GKE target — build + push the image, then apply manifests to a cluster.

Reuses the Kubernetes manifests; the deploy action builds and pushes the
image, fetches GKE cluster credentials via gcloud, and applies the
Kustomize set with kubectl.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from troopai.adk.deploy.commands import require_tool, run_checked
from troopai.adk.deploy.targets.docker import DockerTarget
from troopai.adk.deploy.targets.k8s import K8sTarget

if TYPE_CHECKING:
    from pathlib import Path

    from troopai.adk.deploy.commands import CommandRunner
    from troopai.adk.deploy.context import DeployContext


@dataclass(frozen=True)
class GKETarget:
    """Builds/pushes the image and applies manifests to a GKE cluster."""

    key: ClassVar[str] = "gke"
    required_tools: ClassVar[tuple[str, ...]] = ("gcloud", "docker", "kubectl")

    def generate(self, ctx: DeployContext) -> dict[str, str]:
        """Render the same artifacts as the Kubernetes target.

        Args:
            ctx: The deploy context.

        Returns:
            Map of relative path to file content.
        """
        return K8sTarget().generate(ctx)

    def deploy(
        self,
        ctx: DeployContext,
        runner: CommandRunner,
        *,
        project: str,
        region: str,
        cluster: str,
        context_dir: Path,
        push: bool = True,
    ) -> None:
        """Build/push the image, fetch credentials, and apply the manifests.

        Args:
            ctx: The deploy context (supplies the image).
            runner: The command runner.
            project: GCP project id.
            region: Cluster region/location.
            cluster: GKE cluster name.
            context_dir: Directory the manifests were written under.
            push: Whether to push the image after building.

        Raises:
            DeployToolMissing: If gcloud, docker, or kubectl is missing.
            DeployCommandFailed: If any command exits non-zero.
        """
        require_tool(runner, "gcloud")
        require_tool(runner, "kubectl")
        DockerTarget().build(ctx, runner, context_dir=context_dir, push=push)
        run_checked(
            runner,
            [
                "gcloud",
                "container",
                "clusters",
                "get-credentials",
                cluster,
                "--region",
                region,
                "--project",
                project,
            ],
        )
        run_checked(runner, ["kubectl", "apply", "-k", str(context_dir / "deploy" / "k8s")])

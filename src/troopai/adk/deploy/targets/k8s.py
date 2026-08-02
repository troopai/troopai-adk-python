"""Kubernetes target — manifests plus ``kubectl apply`` to any cluster.

Generates the container artifacts and a Kustomize manifest set
(Deployment + Service + HPA + ConfigMap + example Secret), then applies
them to the operator's current (or named) cluster context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from troopai.adk.deploy.commands import require_tool, run_checked
from troopai.adk.deploy.k8s_manifests import render_k8s_manifests
from troopai.adk.deploy.targets.docker import DockerTarget

if TYPE_CHECKING:
    from pathlib import Path

    from troopai.adk.deploy.commands import CommandRunner
    from troopai.adk.deploy.context import DeployContext


@dataclass(frozen=True)
class K8sTarget:
    """Renders Kubernetes manifests and applies them with kubectl."""

    key: ClassVar[str] = "k8s"
    required_tools: ClassVar[tuple[str, ...]] = ("kubectl",)

    def generate(self, ctx: DeployContext) -> dict[str, str]:
        """Render the container artifacts plus the Kubernetes manifest set.

        Args:
            ctx: The deploy context.

        Returns:
            Map of relative path to file content.
        """
        files = dict(DockerTarget().generate(ctx))
        files.update(render_k8s_manifests(ctx))
        return files

    def apply(self, runner: CommandRunner, *, context_dir: Path, kube_context: str | None = None) -> None:
        """Apply the generated manifests with ``kubectl apply -k``.

        Args:
            runner: The command runner.
            context_dir: Directory the manifests were written under.
            kube_context: Optional kubeconfig context to target.

        Raises:
            DeployToolMissing: If kubectl is not installed.
            DeployCommandFailed: If kubectl exits non-zero.
        """
        require_tool(runner, "kubectl")
        args = ["kubectl"]
        if kube_context is not None:
            args.extend(["--context", kube_context])
        args.extend(["apply", "-k", str(context_dir / "deploy" / "k8s")])
        run_checked(runner, args)

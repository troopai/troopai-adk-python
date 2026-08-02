"""Helm target — render a chart and ``helm upgrade --install`` it.

Generates the container artifacts and a Helm chart, then installs/upgrades
the release on the operator's current cluster context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from troopai.adk.deploy.commands import require_tool, run_checked
from troopai.adk.deploy.helm_chart import render_helm_chart
from troopai.adk.deploy.targets.docker import DockerTarget

if TYPE_CHECKING:
    from pathlib import Path

    from troopai.adk.deploy.commands import CommandRunner
    from troopai.adk.deploy.context import DeployContext


@dataclass(frozen=True)
class HelmTarget:
    """Renders a Helm chart and installs it with helm."""

    key: ClassVar[str] = "helm"
    required_tools: ClassVar[tuple[str, ...]] = ("helm",)

    def generate(self, ctx: DeployContext) -> dict[str, str]:
        """Render the container artifacts plus a Helm chart.

        Args:
            ctx: The deploy context.

        Returns:
            Map of relative path to file content.
        """
        files = dict(DockerTarget().generate(ctx))
        files.update(render_helm_chart(ctx))
        return files

    def install(
        self,
        ctx: DeployContext,
        runner: CommandRunner,
        *,
        context_dir: Path,
        namespace: str | None = None,
    ) -> None:
        """Install/upgrade the release with ``helm upgrade --install``.

        Args:
            ctx: The deploy context (supplies the release/chart name).
            runner: The command runner.
            context_dir: Directory the chart was written under.
            namespace: Optional namespace (created if missing).

        Raises:
            DeployToolMissing: If helm is not installed.
            DeployCommandFailed: If helm exits non-zero.
        """
        require_tool(runner, "helm")
        chart_dir = context_dir / "deploy" / "helm" / ctx.app_name
        args = ["helm", "upgrade", "--install", ctx.app_name, str(chart_dir)]
        if namespace is not None:
            args.extend(["--namespace", namespace, "--create-namespace"])
        run_checked(runner, args)

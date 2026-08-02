"""GCP Cloud Run target — build from source via Cloud Build and deploy.

``deploy cloud-run`` runs ``gcloud run deploy --source`` so no local docker
daemon is needed; Cloud Build builds the generated Dockerfile and Cloud Run
serves it. API keys map to Secret Manager secrets of the same name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from troopai.adk.deploy.cloudrun_manifests import render_cloudrun
from troopai.adk.deploy.commands import require_tool, run_checked
from troopai.adk.deploy.targets.docker import DockerTarget

if TYPE_CHECKING:
    from pathlib import Path

    from troopai.adk.deploy.commands import CommandRunner
    from troopai.adk.deploy.context import DeployContext


@dataclass(frozen=True)
class CloudRunTarget:
    """Renders Cloud Run artifacts and deploys via gcloud."""

    key: ClassVar[str] = "cloudrun"
    required_tools: ClassVar[tuple[str, ...]] = ("gcloud",)

    def generate(self, ctx: DeployContext) -> dict[str, str]:
        """Render the container artifacts plus a Knative ``service.yaml``.

        Args:
            ctx: The deploy context.

        Returns:
            Map of relative path to file content.
        """
        files = dict(DockerTarget().generate(ctx))
        files.update(render_cloudrun(ctx))
        return files

    def deploy(
        self,
        ctx: DeployContext,
        runner: CommandRunner,
        *,
        project: str,
        region: str,
        source_dir: Path,
        allow_unauthenticated: bool = False,
        min_instances: int = 0,
    ) -> None:
        """Deploy to Cloud Run with ``gcloud run deploy --source``.

        Args:
            ctx: The deploy context (supplies the service name and port).
            runner: The command runner.
            project: GCP project id.
            region: Cloud Run region.
            source_dir: Build context (must contain the generated Dockerfile).
            allow_unauthenticated: Make the service publicly invokable;
                defaults to requiring authentication.
            min_instances: Warm instances to keep (``0`` scales to zero).

        Raises:
            DeployToolMissing: If gcloud is not installed.
            DeployCommandFailed: If gcloud exits non-zero.
        """
        require_tool(runner, "gcloud")
        args = [
            "gcloud",
            "run",
            "deploy",
            ctx.app_name,
            "--source",
            str(source_dir),
            "--project",
            project,
            "--region",
            region,
            "--port",
            str(ctx.port),
        ]
        args.append("--allow-unauthenticated" if allow_unauthenticated else "--no-allow-unauthenticated")
        if min_instances > 0:
            args.extend(["--min-instances", str(min_instances)])
        for key in ctx.env_keys:
            args.extend(["--set-secrets", f"{key}={key}:latest"])
        run_checked(runner, args)

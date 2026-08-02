"""Docker target — the universal container image every platform consumes.

The same image runs on Cloud Run, GKE/Kubernetes, ECS, and App Runner;
the other targets layer orchestration on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from troopai.adk.deploy.commands import require_tool, run_checked
from troopai.adk.deploy.templates import (
    render_dockerfile,
    render_dockerignore,
    render_requirements,
)

if TYPE_CHECKING:
    from pathlib import Path

    from troopai.adk.deploy.commands import CommandRunner
    from troopai.adk.deploy.context import DeployContext


@dataclass(frozen=True)
class DockerTarget:
    """Renders the container artifacts and builds/pushes the image."""

    key: ClassVar[str] = "docker"
    required_tools: ClassVar[tuple[str, ...]] = ("docker",)

    def generate(self, ctx: DeployContext) -> dict[str, str]:
        """Render the Dockerfile, ``.dockerignore``, and starter requirements.

        Args:
            ctx: The deploy context.

        Returns:
            The three container artifacts keyed by filename.
        """
        return {
            "Dockerfile": render_dockerfile(ctx),
            ".dockerignore": render_dockerignore(),
            "requirements.txt": render_requirements(ctx),
        }

    def build(
        self,
        ctx: DeployContext,
        runner: CommandRunner,
        *,
        context_dir: Path,
        push: bool = False,
    ) -> None:
        """Build the image with ``docker build`` and optionally push it.

        Args:
            ctx: The deploy context (supplies the image tag).
            runner: The command runner.
            context_dir: Docker build context directory.
            push: Whether to ``docker push`` after a successful build.

        Raises:
            DeployToolMissing: If docker is not installed.
            DeployCommandFailed: If a docker command exits non-zero.
        """
        require_tool(runner, "docker")
        run_checked(runner, ["docker", "build", "-t", ctx.image, str(context_dir)], cwd=context_dir)
        if push:
            run_checked(runner, ["docker", "push", ctx.image])

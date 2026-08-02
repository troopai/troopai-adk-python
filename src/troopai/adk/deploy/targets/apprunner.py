"""AWS App Runner target — the simplest container-to-URL path on AWS.

Renders an ``aws apprunner create-service`` input document and creates the
service from an ECR image. App Runner manages the load balancer, TLS, and
scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from troopai.adk.deploy.aws_ecr import build_and_push_to_ecr
from troopai.adk.deploy.aws_manifests import render_apprunner, render_apprunner_create_service
from troopai.adk.deploy.commands import require_tool, run_checked
from troopai.adk.deploy.targets.docker import DockerTarget

if TYPE_CHECKING:
    from pathlib import Path

    from troopai.adk.deploy.commands import CommandRunner
    from troopai.adk.deploy.context import DeployContext


@dataclass(frozen=True)
class AppRunnerTarget:
    """Renders an App Runner create-service input and creates the service."""

    key: ClassVar[str] = "apprunner"
    required_tools: ClassVar[tuple[str, ...]] = ("aws",)

    def generate(self, ctx: DeployContext) -> dict[str, str]:
        """Render the container artifacts plus an App Runner create-service input.

        Args:
            ctx: The deploy context.

        Returns:
            Map of relative path to file content.
        """
        files = dict(DockerTarget().generate(ctx))
        files.update(render_apprunner(ctx))
        return files

    def deploy(
        self,
        ctx: DeployContext,
        runner: CommandRunner,
        *,
        region: str,
        access_role_arn: str,
        push: bool = False,
        context_dir: Path | None = None,
    ) -> None:
        """Create the App Runner service from the ECR image.

        Args:
            ctx: The deploy context.
            runner: The command runner.
            region: AWS region.
            access_role_arn: IAM role ARN App Runner uses to pull from ECR.
            push: Log in to ECR, build, and push ``ctx.image`` first.
            context_dir: Build context; required when ``push`` is set.

        Raises:
            ValueError: If ``push`` is set without ``context_dir``.
            DeployToolMissing: If the aws (or docker, when pushing) CLI is missing.
            DeployCommandFailed: If the command exits non-zero.
        """
        if push:
            if context_dir is None:
                raise ValueError("push=True requires context_dir (the build context).")
            build_and_push_to_ecr(ctx, runner, region=region, context_dir=context_dir)
        require_tool(runner, "aws")
        document = render_apprunner_create_service(ctx, access_role_arn=access_role_arn)
        run_checked(
            runner,
            ["aws", "apprunner", "create-service", "--region", region, "--cli-input-json", document],
        )

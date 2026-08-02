"""AWS Lambda target — container image with the Lambda Web Adapter.

Renders a Lambda-flavored Dockerfile (the same ``troopai serve`` HTTP
app behind the AWS Lambda Web Adapter) and updates an existing function's
image. Only fits spiky/intermittent traffic — sustained multi-turn agents
belong on ECS/Cloud Run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from troopai.adk.deploy.aws_ecr import build_and_push_to_ecr
from troopai.adk.deploy.aws_manifests import render_lambda_dockerfile
from troopai.adk.deploy.commands import require_tool, run_checked
from troopai.adk.deploy.templates import render_dockerignore, render_requirements

if TYPE_CHECKING:
    from pathlib import Path

    from troopai.adk.deploy.commands import CommandRunner
    from troopai.adk.deploy.context import DeployContext


@dataclass(frozen=True)
class LambdaTarget:
    """Renders a Lambda Web Adapter image and updates a function."""

    key: ClassVar[str] = "lambda"
    required_tools: ClassVar[tuple[str, ...]] = ("aws",)

    def generate(self, ctx: DeployContext) -> dict[str, str]:
        """Render the Lambda Dockerfile, ``.dockerignore``, and requirements.

        Args:
            ctx: The deploy context.

        Returns:
            Map of relative path to file content.
        """
        return {
            "deploy/aws-lambda/Dockerfile": render_lambda_dockerfile(ctx),
            "requirements.txt": render_requirements(ctx),
            ".dockerignore": render_dockerignore(),
        }

    def deploy(
        self,
        ctx: DeployContext,
        runner: CommandRunner,
        *,
        region: str,
        function_name: str | None = None,
        push: bool = False,
        context_dir: Path | None = None,
    ) -> None:
        """Update the function's image with ``aws lambda update-function-code``.

        Args:
            ctx: The deploy context (supplies the image URI).
            runner: The command runner.
            region: AWS region.
            function_name: Target function name; defaults to the app name.
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
        name = function_name if function_name is not None else ctx.app_name
        run_checked(
            runner,
            [
                "aws",
                "lambda",
                "update-function-code",
                "--function-name",
                name,
                "--image-uri",
                ctx.image,
                "--region",
                region,
            ],
        )

"""AWS ECS Fargate target — the idiomatic long-running-agent path on AWS.

Renders a Fargate task definition and registers it with the aws CLI;
optionally forces a new deployment on an existing service. Pushing the
image to ECR and wiring the cluster/ALB are operator concerns (see the
deploy docs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from troopai.adk.deploy.aws_ecr import build_and_push_to_ecr
from troopai.adk.deploy.aws_manifests import render_ecs, render_ecs_task_definition
from troopai.adk.deploy.commands import require_tool, run_checked
from troopai.adk.deploy.targets.docker import DockerTarget

if TYPE_CHECKING:
    from pathlib import Path

    from troopai.adk.deploy.commands import CommandRunner
    from troopai.adk.deploy.context import DeployContext


@dataclass(frozen=True)
class ECSTarget:
    """Renders an ECS task definition and registers it via aws."""

    key: ClassVar[str] = "ecs"
    required_tools: ClassVar[tuple[str, ...]] = ("aws",)

    def generate(self, ctx: DeployContext) -> dict[str, str]:
        """Render the container artifacts plus an ECS task definition.

        Args:
            ctx: The deploy context.

        Returns:
            Map of relative path to file content.
        """
        files = dict(DockerTarget().generate(ctx))
        files.update(render_ecs(ctx))
        return files

    def deploy(
        self,
        ctx: DeployContext,
        runner: CommandRunner,
        *,
        region: str,
        execution_role_arn: str,
        cluster: str | None = None,
        service: str | None = None,
        push: bool = False,
        context_dir: Path | None = None,
    ) -> None:
        """Register the task definition and optionally roll the service.

        Args:
            ctx: The deploy context.
            runner: The command runner.
            region: AWS region.
            execution_role_arn: Task execution role ARN (ECR + secrets).
            cluster: Existing ECS cluster to roll (optional).
            service: Existing ECS service to roll (requires *cluster*).
            push: Log in to ECR, build, and push ``ctx.image`` first.
            context_dir: Build context; required when ``push`` is set.

        Raises:
            ValueError: If ``push`` is set without ``context_dir``.
            DeployToolMissing: If the aws (or docker, when pushing) CLI is missing.
            DeployCommandFailed: If a command exits non-zero.
        """
        if push:
            if context_dir is None:
                raise ValueError("push=True requires context_dir (the build context).")
            build_and_push_to_ecr(ctx, runner, region=region, context_dir=context_dir)
        require_tool(runner, "aws")
        task_json = render_ecs_task_definition(ctx, region=region, execution_role_arn=execution_role_arn)
        run_checked(
            runner,
            ["aws", "ecs", "register-task-definition", "--region", region, "--cli-input-json", task_json],
        )
        if cluster is not None and service is not None:
            run_checked(
                runner,
                [
                    "aws",
                    "ecs",
                    "update-service",
                    "--region",
                    region,
                    "--cluster",
                    cluster,
                    "--service",
                    service,
                    "--force-new-deployment",
                ],
            )

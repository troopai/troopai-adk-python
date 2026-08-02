"""ECR authentication + image push for the AWS targets.

The aws CLI mints a short-lived ECR password; it is piped to
``docker login --password-stdin`` (never placed in argv), then the image
is built and pushed. Used by the AWS targets when ``push=True``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from troopai.adk.deploy.commands import require_tool, run_checked
from troopai.adk.deploy.targets.docker import DockerTarget

if TYPE_CHECKING:
    from pathlib import Path

    from troopai.adk.deploy.commands import CommandRunner
    from troopai.adk.deploy.context import DeployContext


def ecr_registry(image: str) -> str:
    """Return the registry host of an image URI (text before the first ``/``).

    Args:
        image: A container image URI (e.g. ``acct.dkr.ecr.r.amazonaws.com/app:1``).

    Returns:
        The registry host (e.g. ``acct.dkr.ecr.r.amazonaws.com``).
    """
    return image.split("/")[0]


def ecr_login(runner: CommandRunner, image: str, *, region: str) -> None:
    """Authenticate docker to the image's ECR registry.

    Args:
        runner: The command runner.
        image: The ECR image URI (its host is the registry to log in to).
        region: AWS region.

    Raises:
        DeployToolMissing: If aws or docker is not installed.
        DeployCommandFailed: If a command exits non-zero.
    """
    require_tool(runner, "aws")
    require_tool(runner, "docker")
    password = run_checked(runner, ["aws", "ecr", "get-login-password", "--region", region]).stdout.strip()
    run_checked(
        runner,
        ["docker", "login", "--username", "AWS", "--password-stdin", ecr_registry(image)],
        input_text=password,
    )


def build_and_push_to_ecr(ctx: DeployContext, runner: CommandRunner, *, region: str, context_dir: Path) -> None:
    """Log in to ECR, then build and push ``ctx.image``.

    Args:
        ctx: The deploy context (supplies the image).
        runner: The command runner.
        region: AWS region.
        context_dir: Docker build context directory.

    Raises:
        DeployToolMissing: If aws or docker is not installed.
        DeployCommandFailed: If a command exits non-zero.
    """
    ecr_login(runner, ctx.image, region=region)
    DockerTarget().build(ctx, runner, context_dir=context_dir, push=True)
